#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""A/B comparison of FINAL task outputs: official Python ultralytics vs cpp_ggml.

Two comparison levels:

  --same-input  ENGINE level. The Python letterbox tensor is dumped once and fed
                verbatim to the C++ runtime (--input-f32), so both engines see a
                bit-identical input and only the engine + postprocessing are
                compared. Detect/pose/obb/segment support this; Python results
                are mapped to letterbox-canvas coordinates for the diff.
  default       END-TO-END level. Both sides read the image themselves. Two
                independent image decoders (stb_image vs OpenCV) and resamplers
                differ by up to one 1/255 quantization step on ~10% of pixels,
                which the yolo26 end2end heads amplify near thresholds (measured:
                the SAME torch model on two such inputs moves conf by up to 0.09
                and reorders detections). Depth/classify absorb that noise well;
                near-threshold box tasks need --same-input for a tight verdict.

Per task:
  detect    boxes (xyxy), conf, cls                    C++: detect  --dets-json
  segment   boxes + full-canvas mask RLE               C++: detect  --dets-json
  pose      boxes + COCO-17 keypoints                  C++: pose    --dets-json
  obb       rotated boxes (cx, cy, w, h, angle)        C++: obb     --dets-json
  classify  full softmax probability vector            C++: classify --raw
  semantic  orig-size class map                        C++: semantic --raw
  depth     orig-size metric depth map                 C++: depth    --raw
  track     ids/boxes vs the Python tracker            scripts/verify_track_parity.py

The .pt and the GGUF must come from the same checkpoint — models/pytorch/*.pt
are the canonical conversion inputs (e.g. yolo26n.pt <-> yolo26n-f32.gguf). F32
GGUFs give the tightest comparison; F16/Q8_0 drift more — raise the thresholds.
"""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

NDIMS = {b"YINP0001": 3, b"YCLS0001": 1, b"YDEP0001": 2, b"YSEM0001": 2}

# Engine-level thresholds (--same-input, bit-identical input tensors): only the
# ggml-vs-torch numeric differences remain, so these are tight.
ENGINE_THRESHOLDS = {
    "conf": 2e-3,  # max |conf_cpp - conf_py| over matched detections
    "coord": 0.5,  # max box coordinate diff, px (obb: cx/cy/w/h)
    "angle": 0.01,  # max |angle_cpp - angle_py|, radians (obb)
    "kpt": 1.0,  # max keypoint x/y diff, px
    # min per-detection mask IoU (segment): soft-mask boundary cells sit at
    # sigmoid ~ 0.5 and flip under the ~1e-4 mask-coefficient differences
    # between the two engines; 0.94-0.98 is typical, weak masks lower.
    "mask_iou": 0.75,
    "prob": 5e-3,  # max |prob_cpp - prob_py| (classify, end-to-end)
    "depth_rel_p99": 0.02,  # max relative p99 over the depth map
    "sem_agree": 0.985,  # min class-map pixel agreement (semantic); f16 noise grows with scale (n 99.85% -> m 98.64%)
}

# End-to-end thresholds (both sides read the image themselves): two independent
# image decoders/resamplers differ by up to one 1/255 step on ~10% of the
# pixels, and the yolo26 end2end heads amplify that near thresholds — the SAME
# torch model on two such inputs moves conf by up to 0.09 and reorders boxes.
# These gates measure the integration envelope, not engine parity.
E2E_THRESHOLDS = {
    **ENGINE_THRESHOLDS,
    "conf": 0.1,
    "coord": 2.0,
    "angle": 0.05,
    "kpt": 2.0,
}


def read_bin(path: str):
    with open(path, "rb") as f:
        magic = f.read(8)
        nd = NDIMS[magic]
        dims = struct.unpack(f"<{nd}i", f.read(4 * nd))
        data = np.fromfile(f, dtype=np.float32)
    return dims, data.reshape(dims)


def write_bin(path: str, magic: bytes, dims, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack(f"<{len(dims)}i", *dims))
        data.astype(np.float32).tofile(f)


def rle_decode(rle: str, w: int, h: int) -> np.ndarray:
    out = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for run in (int(x) for x in rle.split(",")):
        if val:
            out[pos : pos + run] = True
        pos += run
        val = not val
    return out.reshape(h, w)


def iou_xyxy(a, b) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(cpp, py, sim_fn, sim_min):
    """Greedy best-similarity pairing between cpp and py detections (cls-aware)."""
    cands = sorted(
        ((sim_fn(c, p), i, j) for i, c in enumerate(cpp) for j, p in enumerate(py) if c["cls"] == p["cls"]),
        key=lambda t: -t[0],
    )
    pairs, used_c, used_p = [], set(), set()
    for sim, i, j in cands:
        if sim < sim_min or i in used_c or j in used_p:
            continue
        pairs.append((i, j, sim))
        used_c.add(i)
        used_p.add(j)
    return pairs


def run_cpp(args):
    out = subprocess.run(args, capture_output=True, text=True, cwd=REPO, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"cpp failed ({out.returncode}): {' '.join(args)}\n{out.stderr[-2000:]}")


def load_cpp_json(path):
    return json.loads(Path(path).read_text())


def cpp_imgsz(cli, gguf):
    info = subprocess.run([cli, "info", "--model", gguf], capture_output=True, text=True, cwd=REPO, check=False).stdout
    return next(int(line.split(":")[1]) for line in info.splitlines() if line.startswith("imgsz"))


def dump_letterbox_input(pt, img, imgsz, out_path):
    """Dump the exact letterbox tensor Python predict feeds the model (YINP0001)."""
    import cv2

    from ultralytics.data.augment import LetterBox

    model_stride = 32
    im0 = cv2.imread(img)[:, :, ::-1]  # BGR -> RGB, exactly like the predict pipeline
    lb = LetterBox(imgsz, auto=True, stride=model_stride)(image=im0)
    chw = np.ascontiguousarray(lb.transpose(2, 0, 1)) / 255.0
    write_bin(out_path, b"YINP0001", (3, *chw.shape[1:]), chw)
    return chw.shape[1], chw.shape[2]  # canvas_h, canvas_w


def yolo_predict(pt, img, conf, iou, imgsz=None):
    from ultralytics import YOLO

    model = YOLO(pt)
    # device=cpu matches the CPU C++ build the comparison targets; the CUDA
    # kernels round differently and would pollute the engine-level diff.
    kwargs = {"conf": conf, "iou": iou, "verbose": False, "device": "cpu"}
    if imgsz:
        kwargs["imgsz"] = imgsz
    return model.predict(img, **kwargs)[0]


# ---- per-task comparisons -----------------------------------------------------


def compare_boxlike(task, cpp_dets, py, thr, mask_fn=None):
    if task == "obb":
        from ultralytics.utils.metrics import batch_probiou

        def sim(c, p):
            return float(
                batch_probiou(
                    np.array([[c["cx"], c["cy"], c["w"], c["h"], c["angle"]]], dtype=np.float32),
                    np.array([[p["cx"], p["cy"], p["w"], p["h"], float(p["angle"])]], dtype=np.float32),
                )[0, 0]
            )

        pairs = greedy_match(cpp_dets, py, sim, 0.5)
    else:
        pairs = greedy_match(cpp_dets, py, lambda c, p: iou_xyxy(c["xyxy"], p["xyxy"]), 0.5)
    m = {"cpp_n": len(cpp_dets), "py_n": len(py), "matched": len(pairs)}
    max_coord = max_conf = max_angle = max_kpt = 0.0
    min_mask_iou = 1.0
    cls_mismatch = 0
    for i, j, _ in pairs:
        c, p = cpp_dets[i], py[j]
        if c["cls"] != p["cls"]:
            cls_mismatch += 1
        max_conf = max(max_conf, abs(c["conf"] - p["conf"]))
        if task == "obb":
            max_coord = max(
                max_coord, abs(c["cx"] - p["cx"]), abs(c["cy"] - p["cy"]), abs(c["w"] - p["w"]), abs(c["h"] - p["h"])
            )
            max_angle = max(max_angle, abs(c["angle"] - float(p["angle"])))
        else:
            max_coord = max(max_coord, max(abs(c["xyxy"][k] - float(p["xyxy"][k])) for k in range(4)))
        if c.get("kpts") is not None:
            ck = np.asarray(c["kpts"], dtype=np.float32)
            pk = np.asarray(p["kpts"], dtype=np.float32).reshape(-1)
            if ck.size == pk.size:
                max_kpt = max(max_kpt, float(np.abs(ck - pk).max()))
        if mask_fn is not None:
            min_mask_iou = min(min_mask_iou, mask_fn(i, j))
    cpp_ids = {p[0] for p in pairs}
    py_ids = {p[1] for p in pairs}
    m.update(
        max_coord_diff=max_coord,
        max_conf_diff=max_conf,
        max_angle_diff=max_angle,
        max_kpt_diff=max_kpt,
        cls_mismatch=cls_mismatch,
        min_mask_iou=None if mask_fn is None else min_mask_iou,
        unmatched_cpp=[cpp_dets[i]["cls"] for i in range(len(cpp_dets)) if i not in cpp_ids],
        unmatched_py=[py[j]["cls"] for j in range(len(py)) if j not in py_ids],
    )
    ok = (
        len(cpp_dets) == len(py)
        and len(pairs) == len(py)
        and cls_mismatch == 0
        and max_conf <= thr["conf"]
        and max_coord <= thr["coord"]
        and max_angle <= thr["angle"]
        and max_kpt <= thr["kpt"]
        and (mask_fn is None or min_mask_iou >= thr["mask_iou"])
    )
    return m, ok


def cmp_detect(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    imgsz = cpp_imgsz(cli, gguf)  # run both engines at the GGUF graph resolution
    dets_json = f"{work}/{task}_ab.json"
    args = [cli, "detect", "--model", gguf, "--conf", str(conf), "--iou", str(iou), "--dets-json", dets_json]
    r = yolo_predict(pt, img, conf, iou, imgsz=imgsz)
    if same:
        in_bin = f"{work}/{task}_in.bin"
        _canvas_h, _canvas_w = dump_letterbox_input(pt, img, imgsz, in_bin)
        # --img-size makes the C++ report boxes in original-image coordinates
        # (unscale + the Python-pipeline boundary clip), matching Results.
        args += ["--input-f32", in_bin, "--img-size", f"{r.orig_shape[1]},{r.orig_shape[0]}"]
    else:
        args += ["--source", img]
    run_cpp(args)
    cpp = load_cpp_json(dets_json)
    py = [
        {"cls": int(c), "conf": float(cf), "xyxy": [float(v) for v in xyxy]}
        for c, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]
    mask_fn = None
    if cpp.get("detections") and "mask" in cpp["detections"][0]:
        cw, ch = cpp["canvas"]
        rles = [d.get("mask") for d in cpp["detections"]]

        def mask_fn(i, j):
            cm = rle_decode(rles[i], cw, ch)
            pm = r.masks.data[j].cpu().numpy() > 0
            if pm.shape != cm.shape:
                return 0.0
            inter = np.logical_and(cm, pm).sum()
            union = np.logical_or(cm, pm).sum()
            return inter / union if union else 1.0

    m, ok = compare_boxlike(task, cpp["detections"], py, thr, mask_fn)
    m["task"] = task
    m["level"] = "engine" if same else "end-to-end"
    return m, ok


def cmp_pose(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    imgsz = cpp_imgsz(cli, gguf)  # run both engines at the GGUF graph resolution
    dets_json = f"{work}/pose_ab.json"
    args = [cli, "pose", "--model", gguf, "--conf", str(conf), "--iou", str(iou), "--dets-json", dets_json]
    r = yolo_predict(pt, img, conf, iou, imgsz=imgsz)
    if same:
        in_bin = f"{work}/pose_in.bin"
        dump_letterbox_input(pt, img, imgsz, in_bin)
        args += ["--input-f32", in_bin, "--img-size", f"{r.orig_shape[1]},{r.orig_shape[0]}"]
    else:
        args += ["--source", img]
    run_cpp(args)
    cpp = load_cpp_json(dets_json)
    py = []
    for c, cf, xyxy, k in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist(), r.keypoints.data):
        py.append(
            {
                "cls": int(c),
                "conf": float(cf),
                "xyxy": [float(v) for v in xyxy],
                "kpts": k.float().cpu().numpy().flatten().tolist(),
            }
        )
    m, ok = compare_boxlike("pose", cpp["detections"], py, thr)
    m["task"] = "pose"
    m["level"] = "engine" if same else "end-to-end"
    return m, ok


def cmp_obb(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    imgsz = cpp_imgsz(cli, gguf)  # run both engines at the GGUF graph resolution
    dets_json = f"{work}/obb_ab.json"
    args = [cli, "obb", "--model", gguf, "--conf", str(conf), "--iou", str(iou), "--dets-json", dets_json]
    r = yolo_predict(pt, img, conf, iou, imgsz=imgsz)
    if same:
        in_bin = f"{work}/obb_in.bin"
        dump_letterbox_input(pt, img, imgsz, in_bin)
        # OBB boxes are unclipped in both pipelines (rotated boxes cannot be
        # clipped); --img-size still maps them back to original coordinates.
        args += ["--input-f32", in_bin, "--img-size", f"{r.orig_shape[1]},{r.orig_shape[0]}"]
    else:
        args += ["--source", img]
    run_cpp(args)
    cpp = load_cpp_json(dets_json)
    py = []
    for c, cf, v in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xywhr.tolist()):
        py.append(
            {
                "cls": int(c),
                "conf": float(cf),
                "cx": float(v[0]),
                "cy": float(v[1]),
                "w": float(v[2]),
                "h": float(v[3]),
                "angle": float(v[4]),
            }
        )
    m, ok = compare_boxlike("obb", cpp["detections"], py, thr)
    m["task"] = "obb"
    m["level"] = "engine" if same else "end-to-end"
    return m, ok


def cmp_segment(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    return cmp_detect(task, gguf, pt, img, cli, work, conf, iou, thr, same)


def cmp_classify(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    raw = f"{work}/cls_ab.bin"
    run_cpp([cli, "classify", "--model", gguf, "--source", img, "--raw", raw])
    _dims, cpp = read_bin(raw)
    r = yolo_predict(pt, img, conf, iou)
    py = r.probs.data.float().cpu().numpy()
    m = {
        "task": "classify",
        "level": "end-to-end",
        "classes": int(cpp.size),
        "max_prob_diff": float(np.abs(cpp - py).max()),
        "l1_diff": float(np.abs(cpp - py).sum()),
        "argmax_cpp": int(cpp.argmax()),
        "argmax_py": int(py.argmax()),
    }
    ok = m["max_prob_diff"] <= thr["prob"] and m["argmax_cpp"] == m["argmax_py"]
    return m, ok


def cmp_semantic(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    raw = f"{work}/sem_ab.bin"
    imgsz = cpp_imgsz(cli, gguf)  # both engines run at the GGUF graph resolution
    run_cpp([cli, "semantic", "--model", gguf, "--source", img, "--raw", raw])
    _dims, cpp = read_bin(raw)
    r = yolo_predict(pt, img, conf, iou, imgsz=imgsz)
    py = r.semantic_mask.data.cpu().numpy().astype(np.int32)
    cppi = cpp.astype(np.int32)
    m = {
        "task": "semantic",
        "level": "end-to-end",
        "shape": list(cpp.shape),
        "pixel_agreement": float((cppi == py).mean()),
        "cpp_classes": sorted(np.unique(cppi).tolist()),
        "py_classes": sorted(np.unique(py).tolist()),
    }
    ok = m["pixel_agreement"] >= thr["sem_agree"] and m["cpp_classes"] == m["py_classes"]
    return m, ok


def cmp_depth(task, gguf, pt, img, cli, work, conf, iou, thr, same):
    raw = f"{work}/depth_ab.bin"
    imgsz = cpp_imgsz(cli, gguf)
    run_cpp([cli, "depth", "--model", gguf, "--source", img, "--raw", raw])
    _dims, cpp = read_bin(raw)
    r = yolo_predict(pt, img, conf, iou, imgsz=imgsz)
    py = r.depth.data.float().cpu().numpy()
    diff = np.abs(cpp - py)
    rel = diff / np.maximum(np.abs(py), 1e-3)
    m = {
        "task": "depth",
        "level": "end-to-end",
        "shape": list(cpp.shape),
        "abs_mean": float(diff.mean()),
        "abs_p99": float(np.quantile(diff, 0.99)),
        "abs_max": float(diff.max()),
        "rel_p99": float(np.quantile(rel, 0.99)),
    }
    ok = m["rel_p99"] <= thr["depth_rel_p99"]
    return m, ok


SAME_INPUT_SUPPORTED = {"detect", "segment", "pose", "obb"}

COMPARERS = {
    "detect": cmp_detect,
    "segment": cmp_segment,
    "pose": cmp_pose,
    "obb": cmp_obb,
    "classify": cmp_classify,
    "semantic": cmp_semantic,
    "depth": cmp_depth,
}

# (pt, gguf) defaults: the canonical conversion pairs for every task at n scale.
DEFAULT_PAIRS = {
    "detect": ("yolo26n.pt", "yolo26n-f32.gguf"),
    "segment": ("yolo26n-seg.pt", "yolo26n-seg-f32.gguf"),
    "pose": ("yolo26n-pose.pt", "yolo26n-pose-f32.gguf"),
    "obb": ("yolo26n-obb.pt", "yolo26n-obb-f32.gguf"),
    "classify": ("yolo26n-cls.pt", "yolo26n-cls-f32.gguf"),
    "semantic": ("yolo26n-sem.pt", "yolo26n-sem-f32.gguf"),
    "depth": ("yolo26n-depth.pt", "yolo26n-depth-f32.gguf"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", choices=[*COMPARERS, "all"])
    ap.add_argument("--pt", default=None, help="default: models/pytorch/<task pair>")
    ap.add_argument("--gguf", default=None, help="default: models/gguf/<task pair f32>")
    ap.add_argument("--img", default="ultralytics/assets/bus.jpg")
    ap.add_argument("--conf", type=float, default=None, help="default: 0.25 (0.01 for obb)")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--cli", default="cpp_ggml/build-track/bin/yolo-cli")
    ap.add_argument("--workdir", default="/tmp/yolo_ab")
    ap.add_argument("--same-input", action="store_true", help="engine-level A/B on a bit-identical input tensor")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()

    tasks = list(COMPARERS) if args.task == "all" else [args.task]
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    conf = args.conf
    report, all_ok = {}, True
    for task in tasks:
        pt = args.pt or str(REPO / "cpp_ggml/models/pytorch" / DEFAULT_PAIRS[task][0])
        gguf = args.gguf or str(REPO / "cpp_ggml/models/gguf" / DEFAULT_PAIRS[task][1])
        task_conf = conf if conf is not None else (0.01 if task == "obb" else 0.25)
        same = args.same_input and task in SAME_INPUT_SUPPORTED
        try:
            m, ok = COMPARERS[task](
                task,
                gguf,
                pt,
                args.img,
                args.cli,
                args.workdir,
                task_conf,
                args.iou,
                ENGINE_THRESHOLDS if same else E2E_THRESHOLDS,
                same,
            )
        except Exception as e:  # noqa: BLE001 - report the failure, keep the matrix running
            m, ok = {"task": task, "error": str(e)}, False
        report[task] = m
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        summary = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items() if k not in ("task",)}
        print(f"[{status}] {task:<9} {summary}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
