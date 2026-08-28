#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dataset-level mAP for the cpp_ggml engine, scored by the COCO reference toolkit.

Nothing in this repository could answer "is the engine accurate": `parity_reference.py`
compares single tensors, `bench_all.sh` records latency, and the visual parity grids
prove only that two engines disagree on the same pixels. The metric is not ours to
reimplement either - pycocotools defines COCO mAP and is what ultralytics itself uses
for its published COCO numbers - so this script owns only the glue: run the engine over
a dataset split, translate its detections into COCO result JSON, and score.

Protocol matches ultralytics val: conf 0.001, NMS IoU 0.7, max_det 300, and the
letterbox the GGUF was converted for. One caveat is structural: ultralytics val feeds
rectangular batches while this engine letterboxes each image to a square, so on small
splits the absolute mAP can sit well below `model.val()` even for a bit-exact engine.
`parity_reference.py` is the engine-accuracy proof (identical inputs, identical raw
tensors); this script scores what the deployed engine actually produces. On coco8 val,
yolov8n-f16 scores 0.207 here while the same weights under `predict` + this scorer
score 0.207 too, against `model.val()` at 0.63 - the whole gap is the input protocol.
An open-vocabulary run maps the class names the engine reported (`--dets-json` carries
them) onto dataset categories by name; names with
no dataset counterpart are dropped and counted, which is how a prompt list narrower or
wider than the dataset labels behaves. Categories without predictions are excluded from
the mean by the COCO definition, so a narrow vocabulary is scored over the classes it
actually covers.

Usage:
    python3 scripts/val_map.py --data coco8.yaml --model models/gguf/yolov8s-world-f32.gguf
    python3 scripts/val_map.py --data coco8.yaml --model models/gguf/yolov8n-f16.gguf --pt yolo26n.pt
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # repo-local ultralytics
from ultralytics.data.utils import check_det_dataset, img2label_paths  # noqa: E402

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def ground_truth(label_file: str, wh: tuple[int, int]) -> list[dict]:
    """Read one YOLO label file into COCO annotations (pixel xywh).

    A 5-field row is a box; a longer row is a segmentation polygon, whose bounding box
    is what detection mAP needs.
    """
    w, h = wh
    anns = []
    for line in Path(label_file).read_text().splitlines():
        fields = [float(v) for v in line.split()]
        if len(fields) < 5:
            continue
        cls, values = int(fields[0]), fields[1:]
        if len(values) == 4:
            cx, cy, bw, bh = values
            x, y, bw, bh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
        else:
            pts = np.asarray(values, dtype=np.float32).reshape(-1, 2) * np.array([w, h], dtype=np.float32)
            lo, hi = pts.min(0), pts.max(0)
            x, y, bw, bh = (*lo, *(hi - lo))
        if bw <= 0 or bh <= 0:
            continue
        anns.append(
            {
                "category_id": cls,
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0,
            }
        )
    return anns


def run_engine(cli: Path, model: str, img: Path, out: Path, args: argparse.Namespace) -> dict:
    """One `yolo-cli detect` for one image, returning its `--dets-json` payload."""
    cmd = [
        str(cli), "detect", "--model", str(model), "--source", str(img),
        "--conf", str(args.conf), "--iou", str(args.iou), "--max-det", str(args.max_det),
        "--dets-json", str(out),
    ]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    for opt in ("classes", "text_embed", "clip_model"):
        value = getattr(args, opt)
        if value:
            cmd += [f"--{opt.replace('_', '-')}", str(value)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"yolo-cli failed on {img}:\n{proc.stderr}")
    return json.loads(out.read_text()), proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset YAML (ultralytics cfg/datasets name or path)")
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--build", default="build-cpu", help="build directory holding bin/yolo-cli")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--conf", type=float, default=0.001, help="keep everything: val uses 0.001")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--classes", help="open vocabulary, e.g. 'person,bus,' (trailing empty = background)")
    ap.add_argument("--text-embed", help="pre-encoded YTXT0002 blob")
    ap.add_argument("--clip-model", help="CLIP GGUF used with --classes")
    ap.add_argument("--limit", type=int, help="score only the first N images (smoke runs)")
    ap.add_argument("--pt", help="also run the official ultralytics val on the same split, for comparison")
    ap.add_argument("--json", dest="json_out", help="append the machine-readable record here")
    args = ap.parse_args()

    data = check_det_dataset(args.data, split=args.split)
    images = sorted(p for g in IMAGE_GLOBS for p in Path(data["val"]).rglob(g))
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"no images under {data['val']}")
    labels = img2label_paths([str(p) for p in images])

    from PIL import Image

    cli = ROOT / args.build / "bin" / "yolo-cli"
    # Category ids must be the dataset's own: label files carry those ids verbatim,
    # and enumerate(sorted(names)) only coincides with them when keys are contiguous.
    categories = [{"id": int(k), "name": n} for k, n in sorted(data["names"].items())]
    cat_by_name = {c["name"]: c["id"] for c in categories}
    gt, preds, sizes, vocab, backend = [], [], [], None, "?"
    with tempfile.TemporaryDirectory() as td:
        det_json, log = Path(td) / "dets.json", None
        for ni, (img, lab) in enumerate(zip(images, labels)):
            with Image.open(img) as im:
                wh = im.size
            sizes.append(wh)
            result, log = run_engine(cli, args.model, img, det_json, args)
            if vocab is None:
                vocab = result["vocabulary"]
                backend = (re.search(r"backend=(\S+?)\)", log) or [None, "?"])[1]
            gt += [dict(a, image_id=ni, id=len(gt) + 1) for a in ground_truth(lab, wh)]
            for d in result["detections"]:
                name = vocab[d["cls"]] if d["cls"] < len(vocab) else None
                if name not in cat_by_name:
                    continue
                x, y, x2, y2 = d["xyxy"]
                preds.append(
                    {
                        "image_id": ni,
                        "category_id": cat_by_name[name],
                        "score": d["conf"],
                        "bbox": [x, y, x2 - x, y2 - y],
                    }
                )

        covered = {a["category_id"] for a in gt}
        gt_json, res_json = Path(td) / "gt.json", Path(td) / "res.json"
        gt_json.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": i, "file_name": p.name, "width": wh[0], "height": wh[1]}
                        for i, (p, wh) in enumerate(zip(images, sizes))
                    ],
                    "annotations": gt,
                    "categories": [c for c in categories if c["id"] in covered],
                }
            )
        )
        res_json.write_text(json.dumps(preds))
        # pycocotools reads from disk; it does not accept an in-memory dict here.
        coco = COCO(str(gt_json))
        ev = COCOeval(coco, coco.loadRes(str(res_json)), "bbox")
        ev.params.iouThrs = np.linspace(0.5, 0.95, 10)
        ev.params.maxDets = [1, 10, 100]  # the canonical buckets; COCOeval indexes stats by position
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        m = ev.stats
    print(
        f"\nmodel={Path(args.model).name} build={args.build} data={Path(args.data).stem} split={args.split} "
        f"images={len(images)} gt={len(gt)} preds={len(preds)} categories={len(covered)} backend={backend}"
    )
    print(f"ggml      mAP50-95={m[0]:.4f} mAP50={m[1]:.4f} mAP75={m[2]:.4f} (large={m[5]:.4f})")

    if args.pt:
        from ultralytics import YOLO

        model = YOLO(args.pt)
        if vocab is not None and args.classes:
            model.set_classes(vocab)
        metrics = model.val(
            data=args.data,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            plots=False,
        )
        print(f"pytorch   mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}")

    record = {
        "model": Path(args.model).name,
        "backend": backend,
        "data": Path(args.data).stem,
        "split": args.split,
        "images": len(images),
        "gt": len(gt),
        "preds": len(preds),
        "categories": len(covered),
        "vocabulary": vocab[:3] + ["..."] if vocab and len(vocab) > 3 else vocab,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "eval_max_dets": [1, 10, 100],
        "map50_95": round(float(m[0]), 5),
        "map50": round(float(m[1]), 5),
        "map75": round(float(m[2]), 5),
    }
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[json] appended to {args.json_out}")


if __name__ == "__main__":
    main()
