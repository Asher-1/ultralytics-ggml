"""Regenerate clip-ViT-B-32.ref.npz.

The image tensor is preprocessed with EXACTLY the same bilinear short-edge
resize + center-crop pipeline as the C++ side (clip_preprocess_image in
src/clip_graph.cpp), so the cosine check isolates the transformer graph
computation from interpolation differences. Text is "a photo of a bus",
matching yolo-similarity --text.

Usage:
    python scripts/gen_clip_ref.py [--image PATH] [--out PATH] [--text TEXT]

Paths default to repo-relative locations: the ultralytics assets bus.jpg and
cpp_ggml/models/gguf/clip-ViT-B-32.ref.npz, discovered from this script's
location (works from any cwd).
"""
import argparse
from pathlib import Path
import numpy as np, torch, sys, cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = REPO_ROOT / "ultralytics" / "assets" / "bus.jpg"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "gguf" / "clip-ViT-B-32.ref.npz"

ap = argparse.ArgumentParser(description="Regenerate clip-ViT-B-32.ref.npz")
ap.add_argument("--image", default=str(DEFAULT_IMAGE), help="source image (default: ultralytics/assets/bus.jpg)")
ap.add_argument("--out", default=str(DEFAULT_OUT), help="output .npz path")
ap.add_argument("--text", default="a photo of a bus", help="reference text prompt")
args = ap.parse_args()

sys.path.insert(0, str(REPO_ROOT))
import clip

TARGET = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])

img = cv2.imread(args.image)
if img is None:
    raise SystemExit(f"cannot read image: {args.image}")
img = img[:, :, ::-1]  # BGR -> RGB (CLIP expects RGB; stb in C++ loads RGB)
h, w = img.shape[:2]
# C++ clip_preprocess_image: short edge -> 224, direct bilinear sampling
if w < h:
    scale = TARGET / w; new_w = TARGET; new_h = int(h * scale + 0.5)
else:
    scale = TARGET / h; new_h = TARGET; new_w = int(w * scale + 0.5)
crop_x = (new_w - TARGET) // 2
crop_y = (new_h - TARGET) // 2
out = np.zeros((3, TARGET, TARGET), dtype=np.float32)
for y in range(TARGET):
    for x in range(TARGET):
        sx = (x + crop_x) / scale
        sy = (y + crop_y) / scale
        ix, iy = int(sx), int(sy)
        fx, fy = sx - ix, sy - iy
        ix0 = max(0, min(ix, w - 1)); ix1 = max(0, min(ix + 1, w - 1))
        iy0 = max(0, min(iy, h - 1)); iy1 = max(0, min(iy + 1, h - 1))
        for c in range(3):
            ch = img[:, :, c]
            v00 = ch[iy0, ix0]; v10 = ch[iy0, ix1]
            v01 = ch[iy1, ix0]; v11 = ch[iy1, ix1]
            v = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy
            out[c, y, x] = (v / 255.0 - MEAN[c]) / STD[c]
# NOTE: C++ reads BGR (stb) but CLIP expects RGB. Check yolo-similarity usage.

device = "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()
with torch.no_grad():
    ref_text = clip.tokenize([args.text])
    ref_text_embed = model.encode_text(ref_text)
    ref_text_embed = ref_text_embed / ref_text_embed.norm(dim=-1, keepdim=True)
    ref_img_t = torch.from_numpy(out).unsqueeze(0)
    ref_img_embed = model.encode_image(ref_img_t)
    ref_img_embed = ref_img_embed / ref_img_embed.norm(dim=-1, keepdim=True)

np.savez_compressed(args.out,
    text_ids=ref_text.numpy(), text_embed=ref_text_embed.numpy(),
    image_tensor=ref_img_t.numpy(), image_embed=ref_img_embed.numpy())
print("ref saved to", args.out)
print("image_tensor mean/std:", out.mean(), out.std())
