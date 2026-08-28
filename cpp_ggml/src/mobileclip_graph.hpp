// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once
// MobileCLIP2-B text encoder inference on ggml (YOLOE's text tower).
//
// Usage:
//   MobileclipSession* s = mobileclip_create_session("mobileclip2_b-f16.gguf");
//   float embed[512];
//   mobileclip_encode_string(s, "person", embed);  // BPE + encode, end to end
//   mobileclip_free_session(s);
//
// The GGUF (scripts/convert_mobileclip_to_gguf.py) carries the encoder
// weights plus the CLIP SimpleTokenizer vocab/merges, so the runtime accepts
// plain-text class names. Output is the L2-normalised 512-d text feature —
// the pre-reprta embedding that the YOLOE detector graph feeds into its
// reprta projection.
//
// Precision: all internal computation runs in F32; weight matrices are stored
// as F16 (or Q8_0) and cast to F32 at graph build time. The TorchScript
// reference computes in f16, so the C++ f32 result tracks it to cosine ~1.0.

#include "clip_graph.hpp"

#include <cstdint>
#include <string>

namespace mobileclip {

// MobileCLIP2-B constants (mirror scripts/convert_mobileclip_to_gguf.py).
constexpr int EMBED_DIM = 512;
constexpr int TEXT_CTX = 77;
constexpr int N_LAYERS = 12;
constexpr int N_HEADS = 8;
constexpr int VOCAB_SIZE = 49408;

struct MobileclipSession {
    ggml_context* wctx = nullptr;  // weight tensor structs
    yolo::BackendCtx backend;

    ggml_tensor* embed_table = nullptr;  // token_embedding.weight [49408, 512]
    ggml_tensor* pos_embed = nullptr;    // positional_embedding   [77, 512]
    ggml_tensor* ln_pre_w = nullptr;     // [512]
    ggml_tensor* ln_pre_b = nullptr;     // [512]
    ggml_tensor* proj = nullptr;         // text_projection [512, 512]

    // MobileCLIP block: attention consumes the LayerNormed view of the
    // stream, residuals hit the raw stream, ln_mid sits between the two
    // sub-blocks and ln_post feeds the next block (or the EOS pooling).
    struct Block {
        ggml_tensor* attn_in_w;   // [1536, 512] (qkv combined)
        ggml_tensor* attn_in_b;   // [1536]
        ggml_tensor* attn_out_w;  // [512, 512]
        ggml_tensor* attn_out_b;  // [512]
        ggml_tensor* ln_mid_w;    // [512]
        ggml_tensor* ln_mid_b;    // [512]
        ggml_tensor* mlp_fc_w;    // [2048, 512]
        ggml_tensor* mlp_fc_b;    // [2048]
        ggml_tensor* mlp_proj_w;  // [512, 2048]
        ggml_tensor* mlp_proj_b;  // [512]
        ggml_tensor* ln_post_w;   // [512]
        ggml_tensor* ln_post_b;   // [512]
    };
    Block blocks[N_LAYERS];

    // BPE tokenizer tables (same CLIP SimpleTokenizer as clip-ViT-B-32).
    clip::ClipBpe bpe;

    ggml_cgraph* graph = nullptr;
    ggml_tensor* input_tokens = nullptr;   // int32 [TEXT_CTX]
    ggml_tensor* eot_idx = nullptr;        // int32 [1] EOS position (dynamic)
    ggml_tensor* output_embed = nullptr;   // float [EMBED_DIM]
    ggml_context* gctx = nullptr;          // graph context (freed on destroy)
};

// Create a session from a MobileCLIP GGUF file. Returns nullptr on error.
MobileclipSession* mobileclip_create_session(const std::string& gguf_path, int threads = 0);

// Free a session previously created with mobileclip_create_session.
void mobileclip_free_session(MobileclipSession* s);

// Encode tokenized text (TEXT_CTX int32 token IDs, zero-padded) into a 512-d
// L2-normalised embedding.
bool mobileclip_encode_tokens(MobileclipSession* s, const int32_t* tokens, float* embed);

// Convenience: BPE-tokenize a plain string, then encode it.
bool mobileclip_encode_string(MobileclipSession* s, const char* text, float* embed);

}  // namespace mobileclip
