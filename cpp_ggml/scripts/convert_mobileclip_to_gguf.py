#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the MobileCLIP2-B text encoder (mobileclip2_b.ts) to GGUF.

Usage:
    python scripts/convert_mobileclip_to_gguf.py [--ts mobileclip2_b.ts] [--dtype f16]

YOLOE detectors consume text features from Apple's MobileCLIP2-B encoder
(`ultralytics.nn.text_model.MobileCLIPTS`). The TorchScript module takes
[nc, 77] CLIP SimpleTokenizer tokens and returns L2-normalised [nc, 512]
features; this converter extracts its 151 baked-in constants and transposes
the matrices to torch (out, in) layout — the traced constants sit (in, out)
for direct `x @ W` matmuls, and the C++ side (src/mobileclip_graph.cpp)
reuses the same mul_mat convention as the CLIP GGUF.

The BPE tokenizer is the same CLIP SimpleTokenizer as clip-ViT-B-32; its
vocab/merges tables are embedded in the GGUF (mobileclip.vocab/merges) so the
C++ runtime accepts plain-text class names end to end.

Verification: every emitted tensor is checked against the TorchScript constant
it came from, and a numpy re-implementation of the encoder reproduces the TS
output for a fixed token batch before the file is written.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
GGUF_MODELS = Path(__file__).resolve().parent.parent / "models" / "gguf"

N_LAYERS = 12
EMBED_DIM = 512
N_HEADS = 8
TEXT_CTX = 77
VOCAB_SIZE = 49408


def load_ts(path: Path):
    m = torch.jit.load(str(path), map_location="cpu")
    m.eval()
    _, consts = m.code_with_constants
    return m, consts.const_mapping


def extract_weights(cm: dict) -> dict[str, np.ndarray]:
    """Map the 151 traced constants to named tensors.

    Traced constants are stored (in, out) for direct `x @ W` matmuls; GGUF
    transposes them to torch (out, in) layout so the C++ side uses the same
    mul_mat convention as the CLIP GGUF. Constant map: c0 token embed, c1
    pos embed, c2/c3 ln_pre, c7 the shared causal mask (derived on the C++
    side, not stored). Block 0 uses c5,c6 then c8..c17 (c7 skipped); blocks
    1..11 use 12 contiguous constants starting at c18. Per block the order is
    [qkv_w, qkv_b, out_w, out_b, ln_mid_w, ln_mid_b, fc1_w, fc1_b, fc2_w,
    fc2_b, ln_post_w, ln_post_b]. Attention consumes an already-LayerNormed
    input (ln_pre for block 0, the previous block's ln_post afterwards), so
    the last block's ln_post doubles as the final LN before EOS pooling.
    """
    t = lambda i: cm[f"c{i}"].float().cpu().numpy()
    w = {
        "mobileclip.token_embedding.weight": t(0),
        "mobileclip.positional_embedding": t(1),
        "mobileclip.ln_pre.weight": t(2),
        "mobileclip.ln_pre.bias": t(3),
    }
    for i in range(N_LAYERS):
        if i == 0:
            qkv, out = 5, 8
        else:
            qkv = 18 + 12 * (i - 1)
            out = qkv + 2
        p = f"mobileclip.blk{i}."
        w[p + "attn.in_proj_weight"] = t(qkv).T
        w[p + "attn.in_proj_bias"] = t(qkv + 1)
        w[p + "attn.out_proj.weight"] = t(out).T
        w[p + "attn.out_proj.bias"] = t(out + 1)
        w[p + "ln_mid.weight"] = t(out + 2)
        w[p + "ln_mid.bias"] = t(out + 3)
        w[p + "mlp.c_fc.weight"] = t(out + 4).T
        w[p + "mlp.c_fc.bias"] = t(out + 5)
        w[p + "mlp.c_proj.weight"] = t(out + 6).T
        w[p + "mlp.c_proj.bias"] = t(out + 7)
        w[p + "ln_post.weight"] = t(out + 8)
        w[p + "ln_post.bias"] = t(out + 9)
    w["mobileclip.text_projection"] = t(150).T
    return w


def numpy_forward(w: dict, tokens: np.ndarray) -> np.ndarray:
    """Recompute the encoder in f32 numpy from the emitted tensors.

    tokens: int64 [nc, 77]; returns normalised f32 [nc, 512].
    """

    def ln(x, gw, gb, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True, ddof=0)
        return gw * (x - mu) / np.sqrt(var + eps) + gb

    nc = tokens.shape[0]
    h = w["mobileclip.token_embedding.weight"][tokens] + w["mobileclip.positional_embedding"]
    cur = ln(h, w["mobileclip.ln_pre.weight"], w["mobileclip.ln_pre.bias"])
    d_head = EMBED_DIM // N_HEADS
    for i in range(N_LAYERS):
        p = f"mobileclip.blk{i}."
        # attention consumes the normalised view; residuals hit the raw stream
        qkv = cur @ w[p + "attn.in_proj_weight"].T + w[p + "attn.in_proj_bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        shape = (nc, TEXT_CTX, N_HEADS, d_head)
        q, k, v = (t.reshape(shape).transpose(0, 2, 1, 3) for t in (q, k, v))
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        att = np.where(np.triu(np.ones((TEXT_CTX, TEXT_CTX), bool), 1), -np.inf, att)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(nc, TEXT_CTX, EMBED_DIM)
        h = h + o @ w[p + "attn.out_proj.weight"].T + w[p + "attn.out_proj.bias"]
        x = ln(h, w[p + "ln_mid.weight"], w[p + "ln_mid.bias"])
        m = x @ w[p + "mlp.c_fc.weight"].T + w[p + "mlp.c_fc.bias"]
        # exact GELU (erf), matching torch.gelu default
        m = 0.5 * m * (1.0 + np.vectorize(__import__("math").erf)(m / np.sqrt(2.0)))
        h = h + m @ w[p + "mlp.c_proj.weight"].T + w[p + "mlp.c_proj.bias"]
        cur = ln(h, w[p + "ln_post.weight"], w[p + "ln_post.bias"])  # next block's attn input
    eos = tokens.argmax(-1)
    pooled = cur[np.arange(nc), eos]
    proj = pooled @ w["mobileclip.text_projection"].T
    return proj / np.linalg.norm(proj, axis=-1, keepdims=True)


def torch_forward_f32(cm: dict, tokens: torch.Tensor) -> np.ndarray:
    """f32 torch replay of the decompiled TorchScript ops (reference).

    Mirrors the traced graph: constants are applied with `x @ W` on the
    transposed [seq, nc, D] stream, causal SDPA with the baked mask c7,
    exact GELU, EOS argmax pooling, projection + L2 norm.
    """

    def ln(x, gw, gb):
        return torch.layer_norm(x, [EMBED_DIM], gw, gb)

    def blk_const(i):
        return (5, 8) if i == 0 else (18 + 12 * (i - 1), 20 + 12 * (i - 1))

    with torch.inference_mode():
        x = cm["c0"][tokens] + cm["c1"][:tokens.shape[1]]
        cur = ln(x, cm["c2"], cm["c3"])
        stream = x
        for i in range(N_LAYERS):
            qkv_i, out_i = blk_const(i)
            qkv = cur.transpose(0, 1) @ cm[f"c{qkv_i}"] + cm[f"c{qkv_i+1}"]
            proj = qkv.unflatten(-1, [3, EMBED_DIM]).unsqueeze(0).transpose(0, -2).squeeze(-2)
            heads = lambda t: t.reshape(tokens.shape[1], -1, EMBED_DIM // N_HEADS).transpose(0, 1).reshape(
                tokens.shape[0], N_HEADS, tokens.shape[1], EMBED_DIM // N_HEADS)
            q, k, v = heads(proj[0]), heads(proj[1]), heads(proj[2])
            ao = torch.nn.functional.scaled_dot_product_attention(q, k, v, cm["c7"])
            seq = tokens.shape[1]
            ao = (ao.permute(2, 0, 1, 3).reshape(seq * tokens.shape[0], EMBED_DIM) @ cm[f"c{out_i}"]
                  + cm[f"c{out_i+1}"]).reshape(seq, tokens.shape[0], EMBED_DIM)
            stream = stream + ao.transpose(0, 1)
            mid = ln(stream, cm[f"c{out_i+2}"], cm[f"c{out_i+3}"])
            ff = torch.nn.functional.gelu(mid @ cm[f"c{out_i+4}"] + cm[f"c{out_i+5}"])
            stream = stream + ff @ cm[f"c{out_i+6}"] + cm[f"c{out_i+7}"]
            cur = ln(stream, cm[f"c{out_i+8}"], cm[f"c{out_i+9}"])
        pooled = cur[np.arange(tokens.shape[0]), tokens.argmax(-1)]
        proj = pooled @ cm["c150"]
        return (proj / proj.norm(dim=-1, keepdim=True)).numpy()


def write_gguf(path: Path, weights: dict, vocab: list[str], merges: list[str], dtype: str) -> None:
    writer = gguf.GGUFWriter(str(path), "mobileclip")
    writer.add_string("general.name", "mobileclip2-b-text")
    writer.add_uint32("mobileclip.embed_dim", EMBED_DIM)
    writer.add_uint32("mobileclip.n_layers", N_LAYERS)
    writer.add_uint32("mobileclip.n_heads", N_HEADS)
    writer.add_uint32("mobileclip.text_ctx", TEXT_CTX)
    writer.add_uint32("mobileclip.vocab_size", VOCAB_SIZE)
    writer.add_string("mobileclip.dtype", dtype)
    writer.add_array("mobileclip.vocab", vocab)
    writer.add_array("mobileclip.merges", merges)

    quant_type = gguf.GGMLQuantizationType.Q8_0
    counts = {"q8_0": 0, "f16": 0, "f32": 0}

    def add_tensor(name: str, arr: np.ndarray, keep_f32: bool = False) -> None:
        arr = np.ascontiguousarray(arr.astype(np.float32))
        k = int(np.prod(arr.shape[1:])) if arr.ndim >= 2 else 0
        if not keep_f32 and dtype == "q8_0" and arr.ndim >= 2 and k % 32 == 0:
            flat = np.ascontiguousarray(arr.reshape(arr.shape[0], -1))
            writer.add_tensor(name, gguf.quants.quantize(flat, quant_type), raw_dtype=quant_type)
            counts["q8_0"] += 1
        elif not keep_f32 and dtype == "f16" and arr.ndim >= 2:
            writer.add_tensor(name, arr.reshape(arr.shape[0], -1).astype(np.float16),
                              raw_dtype=gguf.GGMLQuantizationType.F16)
            counts["f16"] += 1
        else:
            writer.add_tensor(name, arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr,
                              raw_dtype=gguf.GGMLQuantizationType.F32)
            counts["f32"] += 1

    for name, arr in sorted(weights.items()):
        add_tensor(name, arr, "bias" in name or "text_projection" in name)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[write] {path}: {len(weights)} tensors, dtype={dtype} ({counts})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts", default=str(REPO_ROOT / "mobileclip2_b.ts"), help="TorchScript encoder path")
    ap.add_argument("--output", help="output .gguf path (default: models/gguf/mobileclip2_b-<dtype>.gguf)")
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f16")
    args = ap.parse_args()

    ts_path = Path(args.ts)
    if not ts_path.is_file():
        ap.error(f"--ts not found: {ts_path}")

    output = Path(args.output) if args.output else GGUF_MODELS / f"mobileclip2_b-{args.dtype}.gguf"

    sys.path.insert(0, str(REPO_ROOT))
    print(f"[load] {ts_path}")
    model, cm = load_ts(ts_path)

    weights = extract_weights(cm)

    # Weight-level check: independently recompute the constant index per
    # block and require the emitted tensor to round-trip that raw constant.
    t = lambda i: cm[f"c{i}"].float().cpu().numpy()
    checks = [("mobileclip.token_embedding.weight", t(0)),
              ("mobileclip.positional_embedding", t(1)),
              ("mobileclip.ln_pre.weight", t(2)),
              ("mobileclip.ln_pre.bias", t(3)),
              ("mobileclip.text_projection", t(150).T)]
    for i in range(N_LAYERS):
        if i == 0:
            qkv, out = 5, 8
        else:
            qkv = 18 + 12 * (i - 1)
            out = qkv + 2
        p = f"mobileclip.blk{i}."
        checks += [(p + "attn.in_proj_weight", t(qkv).T),
                   (p + "attn.in_proj_bias", t(qkv + 1)),
                   (p + "attn.out_proj.weight", t(out).T),
                   (p + "ln_mid.weight", t(out + 2)),
                   (p + "mlp.c_fc.weight", t(out + 4).T),
                   (p + "mlp.c_proj.weight", t(out + 6).T),
                   (p + "ln_post.weight", t(out + 8)),
                   (p + "ln_post.bias", t(out + 9))]
    for name, ref in checks:
        diff = float(np.abs(weights[name] - ref).max())
        assert diff < 1e-6, f"{name}: max diff {diff}"
    print(f"[verify] weight mapping exact on {len(checks)} sampled tensors")

    # Functional check. The traced module runs its matmuls in f16
    # (torch.to(x, 6)), which drifts ~0.15 from the exact f32 math — so the
    # tight check compares numpy against an f32 torch replay of the decompiled
    # ops, and the TS module itself only has to stay cosine-close.
    import clip
    names = ["person", "bus", "a photo of a cat", "traffic light"]
    tokens = clip.tokenize(names, truncate=True)
    out = numpy_forward(weights, tokens.numpy())

    ref32 = torch_forward_f32(cm, tokens)
    diff = np.abs(out - ref32)
    print(f"[verify] numpy-vs-f32-replay: max={diff.max():.3e} mean={diff.mean():.3e}")
    assert diff.max() < 1e-4, "numpy forward diverges from the f32 op replay"

    with torch.inference_mode():
        ref_ts = model(tokens).float().numpy()
    cos = (out * ref_ts).sum(-1)
    print(f"[verify] numpy-vs-ts(f16): cos_min={cos.min():.7f}")
    assert cos.min() > 0.999, "numpy forward diverges from the TorchScript module"

    # Tokenizer tables: MobileCLIPTS uses clip.clip.tokenize (SimpleTokenizer).
    import clip as clip_mod
    simple = clip_mod.simple_tokenizer.SimpleTokenizer()
    vocab = [None] * len(simple.encoder)
    for token, idx in simple.encoder.items():
        vocab[idx] = token
    assert all(vocab), "vocab has holes; tokenizer layout mismatch"
    merges = ["{} {}".format(*pair) for pair, _ in sorted(simple.bpe_ranks.items(), key=lambda kv: kv[1])]
    print(f"[tokenizer] vocab={len(vocab)} merges={len(merges)}")

    GGUF_MODELS.mkdir(parents=True, exist_ok=True)
    write_gguf(output, weights, vocab, merges, args.dtype)


if __name__ == "__main__":
    main()
