#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render the CLIP ViT-B/32 validation and architecture charts.

Inputs:
    models/gguf/clip-ViT-B-32-f16.ref.npz      (PyTorch reference, gen_clip_ref.py)
    cpp_txt_embed.txt                          (yolo-similarity --text "a photo of a bus" --dump_embed)
    cpp_img_embed.txt                          (yolo-similarity --image bus.jpg --dump_embed)

Outputs:
    benchmarks/clip_validation.png             cosine similarity + first-200-dims overlay
    benchmarks/clip_architecture.png           text/image encoder architecture table
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmarks"


def parse_embed(path: Path, key: str) -> np.ndarray:
    """Parse a `key = [a, b, ...]` line from a yolo-similarity --dump_embed dump."""
    txt = path.read_text()
    m = re.search(key + r" = \[(.*?)\]", txt)
    if not m:
        raise RuntimeError(f"{key} not found in {path}")
    return np.array([float(x) for x in m.group(1).split(",")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, default=ROOT / "models" / "gguf" / "clip-ViT-B-32-f16.ref.npz")
    ap.add_argument("--cpp-text", type=Path, required=True, help="yolo-similarity text dump")
    ap.add_argument("--cpp-image", type=Path, required=True, help="yolo-similarity image dump")
    args = ap.parse_args()
    ref = np.load(args.ref)
    cpp_txt = parse_embed(args.cpp_text, "text_embed")
    cpp_img = parse_embed(args.cpp_image, "image_embed")
    pt_txt = ref["text_embed"][0]
    pt_img = ref["image_embed"][0]

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_text, cos_img = cos(cpp_txt, pt_txt), cos(cpp_img, pt_img)
    print(f"cos(text)  = {cos_text:.9f}")
    print(f"cos(image) = {cos_img:.9f}")

    # ---- Chart 1: validation (similarity bars + first-200-dims overlay) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.7]})

    ax = axes[0]
    labels = ["Text\nEmbedding", "Image\nEmbedding"]
    values = [cos_text, cos_img]
    colors = ["#2ca02c", "#1f77b4"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v - 6e-5, f"{v:.7f}", ha="center", va="top", fontsize=9)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="perfect match")
    ax.set_ylim(0.99985, 1.00005)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("CLIP ViT-B/32: C++ vs PyTorch embedding cosine", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    n = 200
    ax.plot(range(n), pt_txt[:n], color="#d62728", lw=1.6, label="PyTorch (ref)")
    ax.plot(range(n), cpp_txt[:n], color="#1f77b4", lw=1.1, ls="--", alpha=0.85, label="ggml C++")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.set_title("Text embedding, first 200 dims (\"a photo of a bus\")", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("CLIP ViT-B/32 — full C++ (BPE tokenizer + text/image encoders) vs PyTorch", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RES / "clip_validation.png", dpi=150)
    print("wrote benchmarks/clip_validation.png")

    # ---- Chart 2: architecture table ----
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    rows = [
        ("Component", "Input → Output", "Architecture"),
        ("Text encoder", "77 tokens → 512-d L2-normalized",
         "BPE tokenizer (49,408 vocab, 77 ctx)\n"
         "token_embedding + positional_embedding\n"
         "12× Transformer blocks · 8 heads · 512 hidden · causal attention\n"
         "ln_final + EOT-position gather + text_projection (512×512)"),
        ("Image encoder", "224×224 RGB → 512-d L2-normalized",
         "conv2d patch embedding 32×32 (768 out), stride 32 → 7×7 grid\n"
         "class token + positional_embedding (50 tokens)\n"
         "12× Transformer blocks · 12 heads · 768 hidden · bidirectional\n"
         "ln_pre/ln_post + cls-token gather + visual_projection (768×512)"),
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.16, 0.27, 0.57])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.6)
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, 3):
        table[i, 0].set_facecolor("#eef4fb")
    ax.set_title("CLIP ViT-B/32 GGML Architecture (ggml graph ops only: get_rows, conv_2d, mul_mat, soft_max, gelu_quick)", fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(RES / "clip_architecture.png", dpi=150)
    print("wrote benchmarks/clip_architecture.png")


if __name__ == "__main__":
    main()
