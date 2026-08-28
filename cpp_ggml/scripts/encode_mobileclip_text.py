#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Encode class names into a YTXT0002 blob of raw MobileCLIP text features.

Example:
    python scripts/encode_mobileclip_text.py \
        --classes "bus,person,car" --output classes.ytxt

The blob is the pre-reprta graph text input of an op-graph v4 YOLOE detector:
the checkpoint-specific ``reprta`` residual runs inside the detector graph
itself, so one blob serves every YOLOE checkpoint sharing the same text tower
(the MobileCLIP output is already L2-normalised).
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

from ultralytics.nn.text_model import build_text_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", required=True, help="comma-separated class names")
    parser.add_argument("--output", type=Path, required=True, help="YTXT0002 output path")
    parser.add_argument("--text-model", default="mobileclip2:b", help="MobileCLIP variant")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    names = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not names:
        parser.error("--classes must contain at least one name")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    text_model = build_text_model(args.text_model, device=device)
    with torch.inference_mode():
        tokens = text_model.tokenize(names)
        embeddings = text_model.encode_text(tokens).float().cpu().numpy().astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise RuntimeError(f"MobileCLIP returned {embeddings.shape}; YOLOE requires [nc, 512]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(b"YTXT0002")
        f.write(struct.pack("<ii", *embeddings.shape))
        f.write(np.ascontiguousarray(embeddings).tobytes())
    print(
        f"wrote {args.output}: {len(names)} classes, {embeddings.shape[1]} dims, "
        f"text_model={args.text_model}, device={device}"
    )


if __name__ == "__main__":
    main()
