# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Replay the detections recorded by `yolo-cli track` into the official Python
trackers and compare track IDs box-for-box.

This is the algorithm-level alignment check for cpp_ggml/src/tracker.cpp: the
C++ CLI writes every frame's tracker input (detections, optional loose-NMS
recoveries) and output (tracks with ids) to a JSONL file via --tracks-json.
This script feeds the identical detections into the Python tracker of the same
type/config and reports any id or box mismatch.

GMC and ReID must be off on both sides for a deterministic comparison:
run the CLI with --no-gmc (Python then sees img=None) and the default
with_reid=False of every official tracker YAML. To exercise GMC instead,
record frames with the CLI (--frames) and pass the same directory here:
the Python tracker then receives img as a BGR ndarray, and the GMC warps
match because both runtimes call the same OpenCV routines.

Examples:
    yolo-cli track --model m.gguf --source frames/ --tracker bytetrack \
        --no-gmc --tracks-json tracks.jsonl
    python scripts/verify_track_parity.py --jsonl tracks.jsonl --tracker bytetrack
    yolo-cli track --dets-jsonl d.jsonl --frames frames/ --tracker botsort \
        --tracker-config botsort.yaml --tracks-json t.jsonl
    python scripts/verify_track_parity.py --jsonl t.jsonl --tracker botsort --frames frames/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace


class DetResults:
    """Minimal Results stand-in exposing xywh/xywhr, conf, cls, and boolean indexing."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.xywhr = None  # absent (not None) on the non-OBB path: parse_bboxes checks hasattr
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    @classmethod
    def from_box(cls, entries):
        out = cls(
            [[e["cx"], e["cy"], e["w"], e["h"]] for e in entries],
            [e["score"] for e in entries],
            [e["cls"] for e in entries],
        )
        del out.xywhr  # non-OBB: parse_bboxes switches on hasattr(results, "xywhr")
        return out

    @classmethod
    def from_obb(cls, entries):
        out = cls(
            [[e["cx"], e["cy"], e["w"], e["h"]] for e in entries],
            [e["score"] for e in entries],
            [e["cls"] for e in entries],
        )
        out.xywhr = np.asarray(
            [[e["cx"], e["cy"], e["w"], e["h"], e["angle"]] for e in entries], dtype=np.float32
        ).reshape(-1, 5)
        return out

    def __len__(self):
        return len(self.conf)

    @property
    def xyxy(self):
        """Axis-aligned corners, mirroring Results.xyxy (BOTSORT's GMC mask input)."""
        return np.stack(
            [
                self.xywh[:, 0] - self.xywh[:, 2] / 2,
                self.xywh[:, 1] - self.xywh[:, 3] / 2,
                self.xywh[:, 0] + self.xywh[:, 2] / 2,
                self.xywh[:, 1] + self.xywh[:, 3] / 2,
            ],
            axis=1,
        )

    def __getitem__(self, mask):
        mask = np.asarray(mask, dtype=bool)
        sub = DetResults(self.xywh[mask], self.conf[mask], self.cls[mask])
        sub.__dict__.pop("xywhr", None)
        if getattr(self, "xywhr", None) is not None:
            sub.xywhr = self.xywhr[mask]
        return sub


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="tracks JSONL written by yolo-cli track --tracks-json")
    ap.add_argument("--tracker", required=True, help="tracker name (bytetrack, botsort, ocsort, ...)")
    ap.add_argument("--yaml", default=None, help="optional tracker YAML (defaults to the official one)")
    ap.add_argument("--tol", type=float, default=1e-3, help="box center/size tolerance in pixels")
    ap.add_argument(
        "--frames", default=None, help="frame directory to feed the tracker img (enables GMC on both sides)"
    )
    args = ap.parse_args()

    cfg_path = args.yaml or str(REPO / "ultralytics/cfg/trackers" / f"{args.tracker}.yaml")
    cfg = IterableSimpleNamespace(**YAML.load(cfg_path))
    assert cfg.tracker_type == args.tracker, f"YAML tracker_type {cfg.tracker_type} != {args.tracker}"
    tracker = TRACKER_MAP[args.tracker](cfg)

    frame_files = []
    if args.frames:
        import cv2

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        p = Path(args.frames)
        frame_files = (
            sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            if p.is_dir()
            else sorted(line.strip() for line in p.read_text().splitlines() if line.strip())
        )

    frames = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines() if line.strip()]
    if args.frames and len(frame_files) != len(frames):
        print(f"FAIL: --frames has {len(frame_files)} files but --jsonl has {len(frames)} frames")
        return 1
    id_mismatches = box_mismatches = 0
    for k, fr in enumerate(frames):
        img = cv2.imread(str(frame_files[k])) if args.frames else None  # BGR, like the predict pipeline
        angled = bool(fr["detections"]) and "angle" in fr["detections"][0]
        results = DetResults.from_obb(fr["detections"]) if angled else DetResults.from_box(fr["detections"])
        dets_del = None
        if fr.get("detections_del"):
            del_angled = "angle" in fr["detections_del"][0]
            sub = DetResults.from_obb(fr["detections_del"]) if del_angled else DetResults.from_box(fr["detections_del"])
            boxes = sub.xywhr if del_angled else sub.xywh
            dets_del = (boxes, sub.conf, sub.cls)
        if args.tracker == "tracktrack":
            out = tracker.update(results, img=img, dets_del=dets_del)
        else:
            out = tracker.update(results, img=img)
        # box rows are [x1, y1, x2, y2, id, score, cls, idx]; OBB rows insert the
        # angle after the box: [cx, cy, w, h, angle, id, score, cls, idx].
        width = 9 if angled else 8
        id_col = 5 if angled else 4
        py = {int(row[id_col]): row for row in np.asarray(out, dtype=np.float32).reshape(-1, width)}
        cpp = {t["id"]: t for t in fr["tracks"]}
        if set(py) != set(cpp):
            id_mismatches += 1
            print(f"frame {fr['frame']}: id sets differ: python={sorted(py)} cpp={sorted(cpp)}")
            continue
        for tid, row in py.items():
            t = cpp[tid]
            if not angled:
                dx = abs(row[0] - (t["cx"] - t["w"] / 2)) + abs(row[2] - (t["cx"] + t["w"] / 2))
                dy = abs(row[1] - (t["cy"] - t["h"] / 2)) + abs(row[3] - (t["cy"] + t["h"] / 2))
            else:  # OBB rows are xywhr
                dx = abs(row[0] - t["cx"]) + abs(row[2] - t["w"])
                dy = abs(row[1] - t["cy"]) + abs(row[3] - t["h"]) + abs(row[4] - t["angle"])
            if max(dx, dy) > args.tol:
                box_mismatches += 1
                print(f"frame {fr['frame']} id {tid}: box differs (python={row[:4].tolist()}, cpp={t})")
    total = len(frames)
    if id_mismatches == 0 and box_mismatches == 0:
        print(f"OK: {total} frames, track ids and boxes match the Python {args.tracker} implementation")
        return 0
    print(f"FAIL: {total} frames, {id_mismatches} id-set mismatches, {box_mismatches} box mismatches")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
