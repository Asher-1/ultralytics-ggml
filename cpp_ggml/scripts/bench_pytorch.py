#!/usr/bin/env python3
"""PyTorch e2e reference benchmark for the ggml comparison report.

Mirrors yolo-cli bench timing: warmup, then timed iterations of a full
predict() on one image (preprocess + GPU forward + postprocess), reported
as mean / min / p50 / p90 e2e milliseconds.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
from ultralytics import YOLO


def parse_imgsz(text: str):
    """ "480,640" -> (480, 640) tuple; plain "640" -> 640."""
    if "," in text:
        h, w = (int(v) for v in text.split(","))
        return (h, w)
    return int(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            # yolov8 detection + segmentation
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolov8n-seg",
            "yolov8s-seg",
            "yolov8m-seg",
            "yolov8l-seg",
            "yolov8x-seg",
            # yolo26 detection
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
            # yolo26 segmentation
            "yolo26n-seg",
            "yolo26s-seg",
            "yolo26m-seg",
            "yolo26l-seg",
            "yolo26x-seg",
            # yolo26 depth
            "yolo26n-depth",
            "yolo26s-depth",
            "yolo26m-depth",
            "yolo26l-depth",
            "yolo26x-depth",
            # yolo26 pose
            "yolo26n-pose",
            "yolo26s-pose",
            "yolo26m-pose",
            "yolo26l-pose",
            "yolo26x-pose",
            # yolo26 obb
            "yolo26n-obb",
            "yolo26s-obb",
            "yolo26m-obb",
            "yolo26l-obb",
            "yolo26x-obb",
            # yolo26 semantic
            "yolo26n-sem",
            "yolo26s-sem",
            "yolo26m-sem",
            "yolo26l-sem",
            "yolo26x-sem",
            # yolo26 classify
            "yolo26n-cls",
            "yolo26s-cls",
            "yolo26m-cls",
            "yolo26l-cls",
            "yolo26x-cls",
        ],
    )
    ap.add_argument("--source", default=str(ROOT.parent / "ultralytics/assets/bus.jpg"))
    ap.add_argument(
        "--imgsz", type=parse_imgsz, default="480,640", help="'h,w' tuple matching the ggml letterbox canvas, or int"
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cooldown", type=float, default=3.0)
    args = ap.parse_args()
    image = cv2.imread(args.source)
    if image is None:
        raise SystemExit(f"cannot read image: {args.source}")

    for name in args.models:
        model = YOLO(ROOT / "models" / "pytorch" / f"{name}.pt")
        model.to(args.device)
        # Match the ggml canvas per task: depth checkpoints run at 768,
        # classify at 224 (checkpoint transforms), everything else 480x640.
        imgsz = 768 if model.task == "depth" else (224 if model.task == "classify" else args.imgsz)
        for _ in range(args.warmup):
            model.predict(image, imgsz=imgsz, device=args.device, verbose=False)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model.predict(image, imgsz=imgsz, device=args.device, verbose=False)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        n = len(times)
        size = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        print(
            json.dumps(
                {
                    "backend": f"pytorch-{args.device}",
                    "model": name,
                    "task": model.task,
                    "dtype": "f32",
                    "imgsz": list(size),
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "e2e_ms": {
                        "mean": statistics.mean(times),
                        "min": times[0],
                        "p50": times[n // 2],
                        "p90": times[int(n * 0.9)],
                        "max": times[-1],
                    },
                }
            )
        )
        time.sleep(args.cooldown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
