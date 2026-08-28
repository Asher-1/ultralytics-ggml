#!/usr/bin/env python3
"""Compose the model-family montage from live yolo-cli renders.

Usage:
    python scripts/render_model_family_overview.py --build build-cuda \
        [--output benchmarks/model_family_overview.png]

Renders one tile per supported task family with yolo-cli --out (detect, World
and YOLOE open-vocabulary, depth, pose, obb, segment, semantic). Classify has
no pixel output, so its tile overlays the top-5 prediction text on bus.jpg.
boats.jpg (OBB ships) is fetched into .cache/ on first use.
"""

import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(get_data_path()) / "fonts" / "ttf" / name), size)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "ultralytics" / "assets"
CACHE = Path(__file__).resolve().parent.parent / ".cache"
GGUF = Path(__file__).resolve().parent.parent / "models" / "gguf"
YTXT = Path(__file__).resolve().parent.parent / "benchmarks" / "ytxt"
CLASSES = "person,bus,car,truck"

SLOT = (480, 600)
MARGIN = 20
LABEL_H = 60
COLS = 5


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def boats_path() -> Path:
    path = ASSETS / "boats.jpg"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / "boats.jpg"
    if not path.exists():
        urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/boats.jpg", path)
    return path


def render_tile(cli: Path, spec: dict) -> Image.Image:
    """Run one yolo-cli command and return its labeled tile image."""
    source = spec["source"]
    if source.name == "boats.jpg":
        source = boats_path()
    if spec["cmd"] == "classify":
        out = subprocess.run(
            [cli, "classify", "--model", str(spec["model"]), "--source", str(source), "--topk", "5"],
            capture_output=True, text=True, check=True,
        ).stdout
        tile = fit(source, SLOT)
        draw = ImageDraw.Draw(tile)
        lines = [m.group(0).strip() for m in re.finditer(r"\d+\. \S+ +\d\.\d+", out)]
        draw.rectangle((8, 8, 320, 44 + 30 * len(lines[:5])), fill="white")
        for i, line in enumerate(lines[:5]):
            draw.text((16, 16 + 30 * i), line, fill="black", font=font(20, mono=True))
        return tile
    tmp = CACHE / f"_overview_{spec['label'].replace(' ', '_')}.png"
    cmd = [cli, spec["cmd"], "--model", str(spec["model"]), "--source", str(source), "--out", str(tmp)]
    cmd += spec.get("extra", [])
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return fit(tmp, SLOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build-cuda", help="build directory providing bin/yolo-cli")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "benchmarks" / "model_family_overview.png")
    args = parser.parse_args()
    cli = Path(__file__).resolve().parent.parent / args.build / "bin" / "yolo-cli"

    g = lambda name: GGUF / f"{name}-f16.gguf"  # noqa: E731
    tiles = [
        {"label": "YOLOv8n Detect", "cmd": "detect", "model": g("yolov8n"), "source": ASSETS / "bus.jpg"},
        {"label": "YOLO26n Detect", "cmd": "detect", "model": g("yolo26n"), "source": ASSETS / "bus.jpg"},
        {"label": "YOLOv8s-World", "cmd": "detect", "model": g("yolov8s-world"), "source": ASSETS / "bus.jpg",
         "extra": ["--classes", CLASSES, "--clip-model", str(GGUF / "clip-ViT-B-32-f16.gguf")]},
        {"label": "YOLOE-26n-Seg", "cmd": "detect", "model": g("yoloe-26n-seg"), "source": ASSETS / "bus.jpg",
         "extra": ["--classes", CLASSES, "--text-embed", str(YTXT / "yoloe-26n-seg.ytxt")]},
        {"label": "YOLO26s-Depth", "cmd": "depth", "model": g("yolo26s-depth"), "source": ASSETS / "bus.jpg"},
        {"label": "YOLO26n-Pose", "cmd": "pose", "model": g("yolo26n-pose"), "source": ASSETS / "zidane.jpg"},
        {"label": "YOLO26n-OBB", "cmd": "obb", "model": g("yolo26n-obb"), "source": ASSETS / "boats.jpg"},
        {"label": "YOLOv8n Instance-Seg", "cmd": "detect", "model": g("yolov8n-seg"), "source": ASSETS / "bus.jpg"},
        {"label": "YOLO26n Semantic-Seg", "cmd": "semantic", "model": g("yolo26n-sem"), "source": ASSETS / "bus.jpg"},
        {"label": "YOLO26n-Cls", "cmd": "classify", "model": g("yolo26n-cls"), "source": ASSETS / "bus.jpg"},
    ]

    rows = (len(tiles) + COLS - 1) // COLS
    cell_w, cell_h = SLOT[0] + 2 * MARGIN, SLOT[1] + LABEL_H + 2 * MARGIN
    header_h = 70
    canvas = Image.new("RGB", (COLS * cell_w, header_h + rows * cell_h), "whitesmoke")
    draw = ImageDraw.Draw(canvas)
    header = "YOLO Model Families - ggml inference visualization (bus.jpg / zidane.jpg / boats.jpg)"
    draw.text((canvas.width // 2, 24), header, fill="black", anchor="mm", font=font(24))
    for i, spec in enumerate(tiles):
        print(f"[render] {spec['label']}", flush=True)
        tile = render_tile(cli, spec)
        x = (i % COLS) * cell_w + MARGIN
        y = header_h + (i // COLS) * cell_h + MARGIN
        draw.text((x + SLOT[0] // 2, y + 18), spec["label"], fill="black", anchor="mm", font=font(28))
        canvas.paste(tile, (x, y + LABEL_H))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    sys.exit(main())
