# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""True end-to-end tracking A/B: both runtimes run the full pipeline from raw
frames -- decode -> letterbox -> inference -> detection decode -> tracker --
independently, and every frame's track ids and boxes are compared.

Unlike verify_track_parity.py (which replays the C++ detections into the
Python tracker and proves the tracker algorithms), this closes the last gap:
the detection-envelope sensitivity of the whole chain, where a small decode /
kernel difference may flip an id or drift a track.

Examples:
    python cpp_ggml/scripts/e2e_track_ab.py --pt cpp_ggml/models/pytorch/yolo26n.pt \\
        --gguf cpp_ggml/models/gguf/yolo26n-f32.gguf --frames /tmp/e2e_pan
    python cpp_ggml/scripts/e2e_track_ab.py --pt ... --gguf ... --frames ... --tracker botsort
    python cpp_ggml/scripts/e2e_track_ab.py --task obb --pt ...-obb.pt --gguf ...-obb-f16.gguf \\
        --frames /tmp/e2e_pan --imgsz 1024
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def list_frames(path):
    p = Path(path)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = (
        sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        if p.is_dir()
        else sorted(line.strip() for line in p.read_text().splitlines() if line.strip())
    )
    if not files:
        sys.exit(f"no frames under {path}")
    # The C++ --source accepts a directory (its natural input) or a comma list.
    source = str(p) if p.is_dir() else ",".join(files)
    return files, source


def run_cpp_tracks(cli, gguf, source, n_frames, tracker, conf):
    out = Path("/tmp/e2e_track_cpp.jsonl")
    cmd = [
        cli,
        "track",
        "--model",
        str(gguf),
        "--source",
        source,
        "--tracker",
        tracker,
        "--conf",
        str(conf),
        "--tracks-json",
        str(out),
    ]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    if len(rows) != n_frames:
        sys.exit(f"cpp produced {len(rows)} frames for {n_frames} inputs")
    return rows


def run_py_tracks(pt, frames, tracker, conf, imgsz):
    from ultralytics import YOLO

    model = YOLO(str(pt))
    kwargs = {
        "conf": conf,
        "iou": 0.7,
        "device": "cpu",
        "persist": True,
        "stream": True,
        "verbose": False,
        "tracker": str(REPO / "ultralytics/cfg/trackers" / f"{tracker}.yaml"),
    }
    if imgsz:
        kwargs["imgsz"] = imgsz
    py = []
    for r in model.track([str(f) for f in frames], **kwargs):
        if r.obb is not None and len(r.obb):
            ids = r.obb.id.int().tolist() if r.obb.id is not None else []
            boxes = r.obb.xywhr.cpu().numpy()
            py.append({"angled": True, "tracks": {i: b[:5].tolist() for i, b in zip(ids, boxes)}})
        elif r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
            boxes = r.boxes.xywh.cpu().numpy()
            py.append({"angled": False, "tracks": {i: b[:4].tolist() for i, b in zip(ids, boxes)}})
        else:
            py.append({"angled": False, "tracks": {}})
    return py


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--tracker", default="tracktrack")
    ap.add_argument("--task", default="detect", choices=["detect", "obb"])
    ap.add_argument("--imgsz", type=int, default=None, help="pass explicitly for obb/sem graphs")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=2.5, help="max center/size drift in px (end-to-end envelope)")
    args = ap.parse_args()

    frames, source = list_frames(args.frames)
    cpp = run_cpp_tracks(
        REPO / "cpp_ggml/build-cpu/bin/yolo-cli", args.gguf, source, len(frames), args.tracker, args.conf
    )
    py = run_py_tracks(args.pt, frames, args.tracker, args.conf, args.imgsz)

    # Track ids are arbitrary labels: when two detections have near-equal
    # confidences, the +-0.01 end-to-end score envelope can swap their order
    # in the first frame's detection list, and every downstream id assignment
    # swaps with it. So compare up to a label permutation: recover the cpp->py
    # id mapping from the frame-by-frame box sequences (least mean drift),
    # require it to be a bijection that is consistent over every frame, and
    # then compare boxes under that mapping.
    cpp_ids = sorted({t["id"] for c in cpp for t in c["tracks"]})
    py_ids = sorted({i for p in py for i in p["tracks"]})
    if len(cpp_ids) != len(py_ids):
        print(f"FAIL: distinct track count differs: cpp {len(cpp_ids)} vs py {len(py_ids)}")
        return 1

    def seq(rows, angled, tid):
        return [next((t for t in c["tracks"] if t["id"] == tid), None) for c in rows]

    def drift(cbox, pbox):
        """Max component drift between a cpp track row and a python box row."""
        if cbox is None or pbox is None:
            return None  # missing on one side only: mismatched presence
        return (
            max(
                abs(cbox["cx"] - pbox[0]),
                abs(cbox["cy"] - pbox[1]),
                abs(cbox["w"] - pbox[2]),
                abs(cbox["h"] - pbox[3]),
                abs(cbox.get("angle", 0.0) - (pbox[4] if angled else 0.0)),
            )
            if angled
            else max(
                abs(cbox["cx"] - pbox[0]), abs(cbox["cy"] - pbox[1]), abs(cbox["w"] - pbox[2]), abs(cbox["h"] - pbox[3])
            )
        )

    # The decode/kernel envelope scales with box size (a 778-px bus moves a few
    # px where a 120-px person moves a fraction), so a track passes when every
    # frame stays within max(--tol px, 1% of the box extent).
    def within(d, cbox):
        return d <= max(args.tol, 0.01 * max(cbox["w"], cbox["h"]))

    angled = bool(cpp[0]["tracks"]) and "angle" in cpp[0]["tracks"][0]
    mapping, id_flips = {}, 0
    for cid in cpp_ids:
        cseq = seq(cpp, angled, cid)
        best, best_cost = None, float("inf")
        for pid in py_ids:
            # Frames where both sides lack the track are consistent (skip);
            # drift() returns None only when exactly one side has it.
            costs = []
            consistent = True
            for cbox, p in zip(cseq, py):
                d = drift(cbox, p["tracks"].get(pid))
                if d is None:
                    if cbox is not None:  # present here, absent there
                        consistent = False
                        break
                    continue  # absent on both sides
                costs.append(d)
            if not consistent or not costs:
                continue
            cost = max(costs)
            if cost < best_cost:
                best, best_cost = pid, cost
        if best is None or not within(best_cost, cseq[0]):
            id_flips += 1
            print(f"cpp id {cid}: no consistent python counterpart within {args.tol}px")
        else:
            mapping[cid] = best
    if len(set(mapping.values())) != len(mapping):
        id_flips += 1
        print(f"FAIL: mapping is not a bijection: {mapping}")

    box_drifts = 0
    max_drift = 0.0
    for k, (c, p) in enumerate(zip(cpp, py)):
        for t in c["tracks"]:
            pid = mapping.get(t["id"])
            pbox = p["tracks"].get(pid)
            if pbox is None:
                box_drifts += 1
                print(f"frame {k}: cpp id {t['id']} has no mapped python track")
                continue
            cbox = (t["cx"], t["cy"], t["w"], t["h"], t.get("angle"))
            d = max(abs(cbox[i] - pbox[i]) for i in range(5 if angled else 4))
            max_drift = max(max_drift, d)
            if not within(d, t):
                box_drifts += 1
                print(f"frame {k} cpp id {t['id']} (py id {pid}): drift {d:.2f}px")
    relabeled = "(ids permuted)" if any(k != v for k, v in mapping.items()) and not id_flips else ""
    total = len(cpp)
    if id_flips == 0 and box_drifts == 0:
        print(
            f"OK: {total} frames e2e ({args.tracker}, {args.task}), tracks match python up to "
            f"id labels {relabeled} (max drift {max_drift:.3f}px)"
        )
        return 0
    print(f"FAIL: {total} frames, {id_flips} unmatched ids, {box_drifts} box drifts > {args.tol}px")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
