#!/usr/bin/env python3
"""Refresh the model-family montage with real OBB and open-vocabulary outputs."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="existing montage to update")
    parser.add_argument("--obb", type=Path, required=True, help="rendered OBB image")
    parser.add_argument("--world", type=Path, required=True, help="rendered World multi-class image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    obb = Image.open(args.obb).convert("RGB")
    # draw_obb writes exact class-dependent RGB line colors. Requiring 32
    # pixels avoids accepting a plain source image or JPEG/PNG re-encoding.
    palette = {
        (min(255, class_id * 53 + 30), max(0, 220 - class_id * 37), min(255, class_id * 91 + 60))
        for class_id in range(15)
    }
    annotated = sum(1 for px in obb.get_flattened_data() if px in palette)
    if annotated < 32:
        raise ValueError("OBB render lacks rotated-box annotation pixels; choose an image with an OBB detection")

    montage = Image.open(args.base).convert("RGB")
    header = "YOLO Model Families - ggml inference visualization (bus.jpg / zidane.jpg / boats.jpg)"
    header_box = (400, 0, montage.width - 400, 55)
    ImageDraw.Draw(montage).rectangle(header_box, fill="white")
    ImageDraw.Draw(montage).text((560, 12), header, fill="black")
    # The montage is a fixed 3x3, 2700x1950 canvas. Keep the existing labels
    # and replace only the previously empty OBB and detect-only World slots.
    montage.paste(fit(args.world, (420, 560)), (2040, 160))
    montage.paste(fit(args.obb, (420, 560)), (2040, 760))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)


if __name__ == "__main__":
    main()
