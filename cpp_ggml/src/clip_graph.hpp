// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once
// CLIP (ViT-B/32) inference on ggml: text encoder + visual encoder.
//
// Usage:
//   ClipSession* s = clip_create_session("clip-vit-b-32.gguf", opts);
//   // Text encoding
//   int32_t tokens[77] = { ... };
//   float txt_embed[512];
//   clip_encode_text(s, tokens, txt_embed);
//   // Image encoding
//   float img[224*224*3];
//   float img_embed[512];
//   clip_encode_image(s, img, img_embed);
//   // Cosine similarity
//   float sim = clip_cosine_similarity(txt_embed, img_embed, 512);
//   clip_free_session(s);
//
// Precision: all internal computation runs in F32 for numerical stability;
// weight matrices are stored as F16 in the GGUF and cast to F32 at graph
// build time (identical to the PyTorch CLIP forward pass).

#include "backend.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace clip {

// CLIP ViT-B/32 constants (mirror scripts/convert_clip_to_gguf.py).
constexpr int EMBED_DIM = 512;
constexpr int VISUAL_EMBED = 768;
constexpr int IMAGE_SIZE = 224;
constexpr int PATCH_SIZE = 32;
constexpr int TEXT_CTX = 77;
constexpr int VOCAB_SIZE = 49408;
constexpr int TEXT_N_LAYERS = 12;
constexpr int TEXT_N_HEADS = 8;
constexpr int VISUAL_N_LAYERS = 12;
constexpr int VISUAL_N_HEADS = 12;
constexpr int VISUAL_N_PATCHES = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);  // 49
constexpr int VISUAL_N_TOKENS = VISUAL_N_PATCHES + 1;  // 50 (patches + cls)

// BPE tokenizer tables rebuilt from the GGUF KV arrays; shared by CLIP and
// MobileCLIP (both embed the same CLIP SimpleTokenizer tables).
struct ClipBpe {
    std::vector<std::string> vocab;                          // tokens by id
    std::vector<std::pair<std::string, std::string>> merges; // merges by rank
    std::unordered_map<std::string, int> encoder;            // token -> id
    std::unordered_map<std::string, std::string> bpe_cache;  // token -> merged
};

struct ClipSession {
    ggml_context* wctx = nullptr;              // weight tensor structs
    ggml_backend_buffer_t wbuf = nullptr;
    yolo::BackendCtx backend;

    // Text encoder tensors
    ggml_tensor* text_embed_table = nullptr;   // token_embedding.weight  [49408, 512]
    ggml_tensor* text_pos_embed = nullptr;     // positional_embedding   [77, 512]
    // 12 text blocks: weights stored in model_state map

    // Visual encoder tensors
    ggml_tensor* visual_conv1 = nullptr;       // conv1.weight  [768, 3, 32, 32]
    ggml_tensor* visual_cls = nullptr;         // class_embedding [768]
    ggml_tensor* visual_pos = nullptr;         // positional_embedding [50, 768]
    ggml_tensor* visual_ln_pre_w = nullptr;    // ln_pre.weight [768]
    ggml_tensor* visual_ln_pre_b = nullptr;    // ln_pre.bias   [768]
    ggml_tensor* visual_ln_post_w = nullptr;   // ln_post.weight [768]
    ggml_tensor* visual_ln_post_b = nullptr;   // ln_post.bias   [768]
    ggml_tensor* visual_proj = nullptr;        // visual_proj [768, 512]

    // Text encoder state (pointer into the model_state map below)
    struct TextBlock {
        ggml_tensor* attn_in_w;   // [1536, 512]   (qkv combined)
        ggml_tensor* attn_in_b;   // [1536]
        ggml_tensor* attn_out_w;  // [512, 512]
        ggml_tensor* attn_out_b;  // [512]
        ggml_tensor* mlp_fc_w;    // [2048, 512]
        ggml_tensor* mlp_fc_b;    // [2048]
        ggml_tensor* mlp_proj_w;  // [512, 2048]
        ggml_tensor* mlp_proj_b;  // [512]
        ggml_tensor* ln1_w;       // [512]
        ggml_tensor* ln1_b;       // [512]
        ggml_tensor* ln2_w;       // [512]
        ggml_tensor* ln2_b;       // [512]
    };
    TextBlock text_blocks[TEXT_N_LAYERS];
    ggml_tensor* text_ln_final_w;   // [512]
    ggml_tensor* text_ln_final_b;   // [512]
    ggml_tensor* text_proj;         // [512, 512]

    struct VisualBlock {
        ggml_tensor* attn_in_w;   // [2304, 768]
        ggml_tensor* attn_in_b;   // [2304]
        ggml_tensor* attn_out_w;  // [768, 768]
        ggml_tensor* attn_out_b;  // [768]
        ggml_tensor* mlp_fc_w;    // [3072, 768]
        ggml_tensor* mlp_fc_b;    // [3072]
        ggml_tensor* mlp_proj_w;  // [768, 3072]
        ggml_tensor* mlp_proj_b;  // [768]
        ggml_tensor* ln1_w;       // [768]
        ggml_tensor* ln1_b;       // [768]
        ggml_tensor* ln2_w;       // [768]
        ggml_tensor* ln2_b;       // [768]
    };
    VisualBlock visual_blocks[VISUAL_N_LAYERS];

    // Per-frame scratch for text embedding (avoid reallocation in hot loop).
    std::vector<float> text_embed_scratch;
    std::vector<float> image_embed_scratch;

    // BPE tokenizer tables (loaded from the GGUF; rebuilt by clip_tokenize_text).
    ClipBpe bpe;

    // Encoder graphs (text and image are separate ggml compute graphs).
    ggml_cgraph* text_graph = nullptr;
    ggml_tensor* text_input_tokens = nullptr;   // int32 [TEXT_CTX]
    ggml_tensor* text_eot_idx = nullptr;       // int32 [1] EOT position (dynamic)
    ggml_tensor* text_output_embed = nullptr;   // float [EMBED_DIM]
    ggml_context* text_gctx = nullptr;          // text graph context (freed on destroy)

    // Image encoder graph
    ggml_cgraph* image_graph = nullptr;
    ggml_tensor* image_input = nullptr;          // float [224, 224, 3]
    ggml_tensor* image_output_embed = nullptr;   // float [EMBED_DIM]
    ggml_context* image_gctx = nullptr;          // image graph context (freed on destroy)
};

struct ClipSessionOptions {
    int threads = 0;  // <= 0: hardware default
    // Offload to a GPU backend when available. Safe ONLY for standalone CLIP
    // processes (yolo-similarity): the YOLO session may already own the shared
    // GPU device, and a second backend init/free pair corrupts its state, so
    // callers embedding CLIP into a YOLO process must keep this false.
    bool use_gpu = false;
};

// Create a session from a CLIP GGUF file. Returns nullptr on error.
ClipSession* clip_create_session(const std::string& gguf_path, const ClipSessionOptions& opts = {});

// Free a session previously created with clip_create_session.
void clip_free_session(ClipSession* s);

// Encode tokenized text (TEXT_CTX int32 token IDs) into a 512-d L2-normalised
// embedding. `tokens` must be TEXT_CTX values; pad unused positions with 0.
bool clip_encode_text(ClipSession* s, const int32_t* tokens, float* embed);

// Tokenize a single text string into TEXT_CTX token ids exactly like the
// Python reference (clip.tokenize): lowercase -> BPE -> [sot, ..., eot] with
// truncation at TEXT_CTX. Returns the number of tokens written (<= TEXT_CTX).
int clip_tokenize_text(ClipSession* s, const char* text, int32_t* tokens);

// Convenience: clip_tokenize_text + clip_encode_text in one call.
bool clip_encode_string(ClipSession* s, const char* text, float* embed);

// Encode an RGB image (IMAGE_SIZE * IMAGE_SIZE * 3 floats, CHW layout, pixel
// values normalised to z-scores using CLIP's mean/std) into a 512-d L2-
// normalised embedding.
bool clip_encode_image(ClipSession* s, const float* image, float* embed);

// Helper: CLIP image preprocessing (equivalent to torchvision.Compose):
//   resize(short_edge=224) -> center_crop(224,224) -> to_tensor -> normalize
// Input: raw RGBA/BGR bytes from stb_image [w, h, channels].
// Output: preallocated `out` float array [3, 224, 224] in CHW layout.
void clip_preprocess_image(const unsigned char* pixels, int w, int h, int channels,
                           float* out);

// Cosine similarity between two L2-normalised vectors.
inline float clip_cosine_similarity(const float* a, const float* b, int d) {
    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += a[i] * b[i];
    return dot;
}

// ---------------------------------------------------------------------------
// Shared encoder building blocks (also used by mobileclip_graph.cpp)
// ---------------------------------------------------------------------------

// LayerNorm over ne0 with fixed eps=1e-5.
ggml_tensor* clip_layer_norm(ggml_context* ctx, ggml_tensor* x,
                             ggml_tensor* weight, ggml_tensor* bias);

// Multi-head self-attention; x is [D, S], causal enables the lower-triangular
// mask. Weights may be F16/Q8_0 (cast to F32 internally).
ggml_tensor* clip_self_attention(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* in_proj_w, ggml_tensor* in_proj_b,
                                 ggml_tensor* out_proj_w, ggml_tensor* out_proj_b,
                                 int n_heads, int d_head, bool causal);

// Two-layer MLP; exact_gelu selects erf-GELU (MobileCLIP) over QuickGELU
// (CLIP).
ggml_tensor* clip_mlp_block(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* fc_w, ggml_tensor* fc_b,
                            ggml_tensor* proj_w, ggml_tensor* proj_b,
                            bool exact_gelu);

// L2-normalise the last dimension.
ggml_tensor* clip_l2_norm(ggml_context* ctx, ggml_tensor* x);

// Load vocab/merges KV arrays written by the converters.
bool clip_bpe_load(ClipBpe& bpe, const gguf_context* g,
                   const char* vocab_key, const char* merges_key, int expect_vocab);

// Tokenize `text` into ctx_len token ids exactly like clip.tokenize:
// lowercase -> BPE -> [sot, ..., eot], zero-padded, truncating at ctx_len.
// Returns the number of tokens written (<= ctx_len).
int clip_bpe_tokenize(ClipBpe& bpe, const char* text, int ctx_len, int32_t* tokens);

}  // namespace clip