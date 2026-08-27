#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Encode YOLOE class names into the detector's post-reprta YTXT0001 blob.

Example:
    python scripts/encode_mobileclip_text.py --detector models/pytorch/yoloe-26n-seg.pt \
        --classes "bus,person,car" --output classes.ytxt

YOLOE does not consume a raw MobileCLIP vector. Its detection head first applies
the checkpoint-specific ``reprta`` residual block and then L2-normalises the
result. The emitted YTXT therefore is tied to ``--detector`` and is exactly the
tensor passed to the head by ``YOLOEModel.predict``.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.nn.text_model import build_text_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True, help="source YOLOE .pt checkpoint")
    parser.add_argument("--classes", required=True, help="comma-separated class names")
    parser.add_argument("--output", type=Path, required=True, help="YTXT0001 output path")
    parser.add_argument("--text-model", help="MobileCLIP variant; defaults to detector metadata")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    names = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not names:
        parser.error("--classes must contain at least one name")
    if not args.detector.is_file():
        parser.error(f"--detector not found: {args.detector}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    detector = YOLO(args.detector)
    if not isinstance(detector.model, YOLOEModel) or not isinstance(detector.model.model[-1], YOLOEDetect):
        parser.error("--detector must be a YOLOE detect or segment checkpoint")
    detector.model.to(device).eval()
    head = detector.model.model[-1]
    text_model_name = args.text_model or detector.model.text_model
    text_model = build_text_model(text_model_name, device=device)
    with torch.inference_mode():
        tokens = text_model.tokenize(names)
        raw_embeddings = text_model.encode_text(tokens)
        dtype = next(head.parameters()).dtype
        embeddings = head.get_tpe(raw_embeddings.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0)
        embeddings = embeddings.float().cpu().numpy().astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise RuntimeError(f"MobileCLIP returned {embeddings.shape}; YOLOE requires [nc, 512]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(b"YTXT0001")
        f.write(struct.pack("<ii", *embeddings.shape))
        f.write(np.ascontiguousarray(embeddings).tobytes())
    print(
        f"wrote {args.output}: {len(names)} classes, {embeddings.shape[1]} dims, "
        f"detector={args.detector.name}, text_model={text_model_name}, device={device}"
    )


if __name__ == "__main__":
    main()
