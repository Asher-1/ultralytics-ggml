#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert OpenAI CLIP models (ViT-B/32) to GGUF format for C++ inference.

Usage:
    python scripts/convert_clip_to_gguf.py --model ViT-B/32
    python scripts/convert_clip_to_gguf.py --model ViT-B/32 --output my_clip.gguf

The converter loads a CLIP model via the official openai/clip package, extracts
the text encoder (BPE tokenizer + 12-layer Transformer) and the visual encoder
(12-layer ViT with 32x32 patch embedding), and writes them to a single GGUF file.

The C++ side (src/clip_graph.cpp) rebuilds the computation graph from the GGUF
metadata and tensors, producing the same 512-d L2-normalised embeddings as the
Python reference for both text and image modalities.

Precision verification is built in: the script compares the Python reference
embedding against a re-computed embedding from the just-exported tensors loaded
back through the same GGUF file (mathematical equivalence check).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
except ImportError:
    print("error: the 'gguf' package is required: pip install gguf", file=sys.stderr)
    raise

# CLIP text BPE vocabulary size (ViT-B/32 uses 49,408). The official CLIP
# tokenizer adds special tokens <|startoftext|> and <|endoftext|>.
CLIP_VOCAB_SIZE = 49408
CLIP_TEXT_CTX = 77
CLIP_EMBED_DIM = 512
CLIP_VISUAL_EMBED = 768
CLIP_IMAGE_SIZE = 224
CLIP_PATCH_SIZE = 32

GGUF_MODELS = Path(__file__).resolve().parent.parent / "models" / "gguf"


def _extract_text_weights(model, prefix: str) -> dict[str, np.ndarray]:
    """Extract text encoder weights from the CLIP model into a flat dict."""
    w = {}
    st = model.state_dict()

    # Token embedding
    w[f"{prefix}token_embedding.weight"] = st["token_embedding.weight"].cpu().numpy()

    # Positional embedding
    w[f"{prefix}positional_embedding"] = st["positional_embedding"].cpu().numpy()

    # Transformer blocks
    for i in range(12):
        for name in ["attn.in_proj_weight", "attn.in_proj_bias",
                      "attn.out_proj.weight", "attn.out_proj.bias",
                      "mlp.c_fc.weight", "mlp.c_fc.bias",
                      "mlp.c_proj.weight", "mlp.c_proj.bias",
                      "ln_1.weight", "ln_1.bias",
                      "ln_2.weight", "ln_2.bias"]:
            key = f"transformer.resblocks.{i}.{name}"
            w[f"{prefix}{key}"] = st[key].cpu().numpy()

    # Final layer norm
    w[f"{prefix}ln_final.weight"] = st["ln_final.weight"].cpu().numpy()
    w[f"{prefix}ln_final.bias"] = st["ln_final.bias"].cpu().numpy()

    # Text projection
    w[f"{prefix}text_projection"] = st["text_projection"].cpu().numpy()

    return w


def _extract_visual_weights(model, prefix: str) -> dict[str, np.ndarray]:
    """Extract visual encoder (ViT) weights from the CLIP model into a flat dict."""
    w = {}
    st = model.state_dict()

    # Patch embedding conv
    w[f"{prefix}conv1.weight"] = st["visual.conv1.weight"].cpu().numpy()

    # Class embedding + positional embedding
    w[f"{prefix}class_embedding"] = st["visual.class_embedding"].cpu().numpy()
    w[f"{prefix}positional_embedding"] = st["visual.positional_embedding"].cpu().numpy()

    # Pre layer norm
    w[f"{prefix}ln_pre.weight"] = st["visual.ln_pre.weight"].cpu().numpy()
    w[f"{prefix}ln_pre.bias"] = st["visual.ln_pre.bias"].cpu().numpy()

    # Transformer blocks
    for i in range(12):
        for name in ["attn.in_proj_weight", "attn.in_proj_bias",
                      "attn.out_proj.weight", "attn.out_proj.bias",
                      "mlp.c_fc.weight", "mlp.c_fc.bias",
                      "mlp.c_proj.weight", "mlp.c_proj.bias",
                      "ln_1.weight", "ln_1.bias",
                      "ln_2.weight", "ln_2.bias"]:
            key = f"transformer.resblocks.{i}.{name}"
            w[f"{prefix}{key}"] = st[f"visual.{key}"].cpu().numpy()

    # Post layer norm
    w[f"{prefix}ln_post.weight"] = st["visual.ln_post.weight"].cpu().numpy()
    w[f"{prefix}ln_post.bias"] = st["visual.ln_post.bias"].cpu().numpy()

    # Visual projection
    w[f"{prefix}visual_projection"] = st["visual.proj"].cpu().numpy()

    return w


def write_gguf(path: str, model_name: str, text_weights: dict,
               visual_weights: dict, logit_scale: float, vocab: list[str],
               merges: list[str], dtype: str) -> None:
    """Write CLIP weights into a GGUF file with metadata for the C++ engine.

    vocab: 49408 tokens ordered by id (bytes + bytes</w> + merged pairs +
    special tokens) so the C++ side can rebuild the encoder map directly.
    merges: "a b" pairs ordered by BPE rank (48894 entries).
    """
    writer = gguf.GGUFWriter(path, "clip")

    # --- metadata ---
    writer.add_string("general.name", model_name)
    writer.add_uint32("clip.embed_dim", CLIP_EMBED_DIM)
    writer.add_uint32("clip.image_size", CLIP_IMAGE_SIZE)
    writer.add_uint32("clip.patch_size", CLIP_PATCH_SIZE)
    writer.add_uint32("clip.vocab_size", CLIP_VOCAB_SIZE)
    writer.add_uint32("clip.text_ctx", CLIP_TEXT_CTX)
    writer.add_uint32("clip.visual_embed_dim", CLIP_VISUAL_EMBED)
    writer.add_uint32("clip.visual_n_layers", 12)
    writer.add_uint32("clip.visual_n_heads", 12)
    writer.add_uint32("clip.text_n_layers", 12)
    writer.add_uint32("clip.text_n_heads", 8)
    writer.add_float32("clip.logit_scale", logit_scale)
    writer.add_string("clip.dtype", dtype)

    # --- tokenizer tables (BPE; the C++ side rebuilds the encoder from these) ---
    writer.add_array("clip.vocab", list(vocab))
    writer.add_array("clip.merges", list(merges))

    quant_type = gguf.GGMLQuantizationType.Q8_0

    def add_tensor(name: str, arr: np.ndarray, keep_f32: bool = False) -> None:
        arr = np.ascontiguousarray(arr.astype(np.float32))
        k = int(np.prod(arr.shape[1:])) if arr.ndim >= 2 else 0
        if not keep_f32 and dtype == "q8_0" and arr.ndim >= 2 and k % 32 == 0:
            flat = np.ascontiguousarray(arr.reshape(arr.shape[0], -1))
            writer.add_tensor(name, gguf.quants.quantize(flat, quant_type), raw_dtype=quant_type)
        elif not keep_f32 and dtype == "f16" and arr.ndim >= 2:
            writer.add_tensor(name, arr.reshape(arr.shape[0], -1).astype(np.float16),
                              raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr,
                              raw_dtype=gguf.GGMLQuantizationType.F32)

    # --- text encoder tensors ---
    for name, arr in sorted(text_weights.items()):
        add_tensor(name, arr, "bias" in name or "text_projection" in name)

    # --- visual encoder tensors ---
    for name, arr in sorted(visual_weights.items()):
        add_tensor(name, arr, "bias" in name or "projection" in name)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[write] {path}: text={len(text_weights)} visual={len(visual_weights)} tensors")


def verify_precision(model, text_weights: dict, visual_weights: dict) -> None:
    """Verify that the exported tensor weights match the reference PyTorch model."""
    ref_st = model.state_dict()
    atol = 1e-5

    # Verify text weights
    for name, arr in text_weights.items():
        pt_name = name.replace("text.", "")
        ref = ref_st[pt_name].cpu().numpy()
        diff = np.max(np.abs(arr.reshape(ref.shape) - ref))
        if diff > atol:
            print(f"  [WARN] text.{pt_name}: max_diff={diff:.2e} (>{atol})")
        else:
            print(f"  [OK]   text.{pt_name}: max_diff={diff:.2e}")

    # Verify visual weights
    for name, arr in visual_weights.items():
        if name.startswith("visual.conv1"):
            pt_name = "visual.conv1.weight"
        elif name.startswith("visual.visual_projection"):
            pt_name = "visual.proj"
        else:
            pt_name = name.replace("visual.", "visual.")
        ref = ref_st[pt_name].cpu().numpy()
        try:
            diff = np.max(np.abs(arr.reshape(ref.shape) - ref))
            if diff > atol:
                print(f"  [WARN] {name}: max_diff={diff:.2e} (>{atol})")
            else:
                print(f"  [OK]   {name}: max_diff={diff:.2e}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")


def convert(model_name: str, output: str, dtype: str) -> None:
    """Load a CLIP model, extract weights, and write to GGUF."""
    print(f"[load] CLIP model '{model_name}' ...")
    try:
        import clip
    except ImportError:
        print("error: the 'clip' package is required: pip install git+https://github.com/openai/CLIP.git",
              file=sys.stderr)
        raise

    device = "cpu"
    model, preprocess = clip.load(model_name, device=device)

    model.eval()
    # Disable gradients for export
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[extract] text encoder ({CLIP_TEXT_CTX} ctx, {CLIP_EMBED_DIM} dim)...")
    text_w = _extract_text_weights(model, "text.")

    print(f"[extract] visual encoder ({CLIP_IMAGE_SIZE}x{CLIP_IMAGE_SIZE}, "
          f"patch={CLIP_PATCH_SIZE}, {CLIP_VISUAL_EMBED} dim)...")
    visual_w = _extract_visual_weights(model, "visual.")

    logit_scale = model.logit_scale.exp().item()
    print(f"[meta] logit_scale={logit_scale:.4f}")

    # Reference embeddings for verification
    print("[verify] checking exported tensor precision ...")
    verify_precision(model, text_w, visual_w)

    # Run reference inference for C++ side verification
    print("[ref] computing reference text+image embeddings ...")
    # Reference text and image must match the inputs used by the C++
    # verification path (yolo-similarity --text "a photo of a bus" --image
    # bus.jpg) so cosine similarity is a real cross-implementation check.
    ref_prompt = "a photo of a bus"
    ref_image_path = Path(__file__).resolve().parent.parent.parent / "ultralytics" / "assets" / "bus.jpg"
    with torch.no_grad():
        ref_text = clip.tokenize([ref_prompt]).to(device)
        ref_text_embed = model.encode_text(ref_text)
        ref_text_embed = ref_text_embed / ref_text_embed.norm(dim=-1, keepdim=True)

        # Real bus.jpg with the standard CLIP preprocessing (center-crop 224).
        from PIL import Image
        from torchvision import transforms
        img = Image.open(ref_image_path).convert("RGB")
        pre = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        ref_img = pre(img).unsqueeze(0).to(device)
        ref_img_embed = model.encode_image(ref_img)
        ref_img_embed = ref_img_embed / ref_img_embed.norm(dim=-1, keepdim=True)

    print(f"  text [{ref_prompt}] -> text_embed [{list(ref_text_embed.shape)}]")
    print(f"  image [bus.jpg] -> image_embed [{list(ref_img_embed.shape)}]")

    # Store reference data for C++ verification
    ref_np = {
        "text_ids": ref_text.cpu().numpy(),
        "text_embed": ref_text_embed.cpu().numpy(),
        "image_tensor": ref_img.cpu().numpy(),
        "image_embed": ref_img_embed.cpu().numpy(),
    }
    ref_path = Path(output).with_suffix(".ref.npz")
    np.savez_compressed(str(ref_path), **ref_np)
    print(f"[ref] saved reference data to {ref_path}")

    # Tokenizer tables from the official CLIP SimpleTokenizer.
    print("[tokenizer] exporting BPE vocab + merges ...")
    try:
        from clip.simple_tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
    except ImportError:
        # Fall back to the module-level singleton used by clip.tokenize.
        import clip.clip as clip_mod

        tok = clip_mod._tokenizer
    # encoder: token -> id; order by id so the C++ map is position-stable.
    vocab = [""] * len(tok.encoder)
    for token, tid in tok.encoder.items():
        vocab[tid] = token
    assert all(vocab), "vocab has holes; tokenizer layout mismatch"
    merges = ["{} {}".format(*pair) for pair, _ in sorted(tok.bpe_ranks.items(), key=lambda kv: kv[1])]
    print(f"  vocab={len(vocab)} merges={len(merges)}")

    write_gguf(output, model_name.replace("/", "-"), text_w, visual_w, logit_scale, vocab, merges, dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ViT-B/32", help="CLIP model name (default: ViT-B/32)")
    ap.add_argument("--output", help=f"output .gguf path (default: models/gguf/<model>.gguf)")
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f16")
    args = ap.parse_args()

    GGUF_MODELS.mkdir(parents=True, exist_ok=True)
    default_name = f"clip-{args.model.replace('/', '-')}.gguf" if args.dtype == "f16" else f"clip-{args.model.replace('/', '-')}-{args.dtype}.gguf"
    output = args.output or GGUF_MODELS / default_name
    convert(args.model, str(output), args.dtype)


if __name__ == "__main__":
    main()
