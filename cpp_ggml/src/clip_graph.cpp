// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "clip_graph.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace clip {

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

// Find a weight tensor in a ggml context by name.
static ggml_tensor* find_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "clip: tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

// ---------------------------------------------------------------------------
// weight_resolver: load tensor pointers from the GGUF weight context
// ---------------------------------------------------------------------------

struct weight_resolver {
    ggml_context* wctx;
};

// Resolve a text block's weights
static bool resolve_text_block(weight_resolver& r, ClipSession::TextBlock& blk,
                                const char* prefix) {
    char name[128];
    auto resolve = [&](const char* suffix) -> ggml_tensor* {
        snprintf(name, sizeof(name), "text.%s.%s", prefix, suffix);
        return find_tensor(r.wctx, name);
    };
    blk.attn_in_w  = resolve("attn.in_proj_weight");
    blk.attn_in_b  = resolve("attn.in_proj_bias");
    blk.attn_out_w = resolve("attn.out_proj.weight");
    blk.attn_out_b = resolve("attn.out_proj.bias");
    blk.mlp_fc_w   = resolve("mlp.c_fc.weight");
    blk.mlp_fc_b   = resolve("mlp.c_fc.bias");
    blk.mlp_proj_w = resolve("mlp.c_proj.weight");
    blk.mlp_proj_b = resolve("mlp.c_proj.bias");
    blk.ln1_w      = resolve("ln_1.weight");
    blk.ln1_b      = resolve("ln_1.bias");
    blk.ln2_w      = resolve("ln_2.weight");
    blk.ln2_b      = resolve("ln_2.bias");
    return blk.attn_in_w != nullptr;
}

// Resolve a visual block's weights
static bool resolve_visual_block(weight_resolver& r, ClipSession::VisualBlock& blk,
                                  const char* prefix) {
    char name[128];
    auto resolve = [&](const char* suffix) -> ggml_tensor* {
        snprintf(name, sizeof(name), "visual.%s.%s", prefix, suffix);
        return find_tensor(r.wctx, name);
    };
    blk.attn_in_w  = resolve("attn.in_proj_weight");
    blk.attn_in_b  = resolve("attn.in_proj_bias");
    blk.attn_out_w = resolve("attn.out_proj.weight");
    blk.attn_out_b = resolve("attn.out_proj.bias");
    blk.mlp_fc_w   = resolve("mlp.c_fc.weight");
    blk.mlp_fc_b   = resolve("mlp.c_fc.bias");
    blk.mlp_proj_w = resolve("mlp.c_proj.weight");
    blk.mlp_proj_b = resolve("mlp.c_proj.bias");
    blk.ln1_w      = resolve("ln_1.weight");
    blk.ln1_b      = resolve("ln_1.bias");
    blk.ln2_w      = resolve("ln_2.weight");
    blk.ln2_b      = resolve("ln_2.bias");
    return blk.attn_in_w != nullptr;
}

// ---------------------------------------------------------------------------
// Graph building helpers
// ---------------------------------------------------------------------------

// L2-normalise the last dimension of `x` in-place.
ggml_tensor* clip_l2_norm(ggml_context* ctx, ggml_tensor* x) {
    // sum = sum(x^2, dim=-1)  -->  uses ggml_sum_rows if 2D
    ggml_tensor* sq = ggml_sqr(ctx, x);
    ggml_tensor* sum = ggml_sum_rows(ctx, sq);  // [D] or [N, D] -> [1] or [N, 1]
    // Add epsilon via ggml_repeat
    ggml_tensor* norm = ggml_sqrt(ctx, sum);
    return ggml_div(ctx, x, norm);
}

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
ggml_tensor* clip_layer_norm(ggml_context* ctx, ggml_tensor* x,
                             ggml_tensor* weight, ggml_tensor* bias) {
    // ggml_norm: y = (x - mean) / sqrt(var + eps), where mean/var computed
    // over the last dimension. eps is hardcoded to 1e-5 in ggml.
    ggml_tensor* y = ggml_norm(ctx, x, 1e-5f);
    if (weight) y = ggml_mul(ctx, y, weight);
    if (bias)   y = ggml_add(ctx, y, bias);
    return y;
}

// Self-attention block (for both text and visual encoders).
// x: [D, S] where D=embed_dim, S=seq_len
// Returns: [D, S]
// `causal` enables the GPT-style lower-triangular mask used by the CLIP text
// encoder (visual encoder attention is bidirectional).
ggml_tensor* clip_self_attention(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* in_proj_w, ggml_tensor* in_proj_b,
                                 ggml_tensor* out_proj_w, ggml_tensor* out_proj_b,
                                 int n_heads, int d_head, bool causal) {
    // Cast F16 weights to F32 for computation
    if (in_proj_w->type != GGML_TYPE_F32)
        in_proj_w = ggml_cast(ctx, in_proj_w, GGML_TYPE_F32);
    if (in_proj_b && in_proj_b->type != GGML_TYPE_F32)
        in_proj_b = ggml_cast(ctx, in_proj_b, GGML_TYPE_F32);
    if (out_proj_w->type != GGML_TYPE_F32)
        out_proj_w = ggml_cast(ctx, out_proj_w, GGML_TYPE_F32);
    if (out_proj_b && out_proj_b->type != GGML_TYPE_F32)
        out_proj_b = ggml_cast(ctx, out_proj_b, GGML_TYPE_F32);

    const int D = (int)x->ne[0];   // embed_dim
    const int S = (int)x->ne[1];   // seq_len
    const int d_h = d_head;
    const int n_h = n_heads;

    // QKV projection
    // in_proj_w: [3*D, D], x: [D, S]
    // mul_mat: result[m,n] = sum_k A[k,m] B[k,n]
    // A=in_proj_w: [3*D, D], B=x: [D, S]
    // K=D, M=3*D, N=S -> result: [3*D, S]
    // QKV projection
    // in_proj_w: [3*D, D], x: [D, S]
    // mul_mat: result[m,n] = sum_k A[k,m] B[k,n]
    // A=in_proj_w: [3*D, D], B=x: [D, S]
    // K=D, M=3*D, N=S -> result: [3*D, S]
    ggml_tensor* qkv = ggml_mul_mat(ctx, in_proj_w, x);  // [3*D, S]
    if (in_proj_b)
        qkv = ggml_add(ctx, qkv, ggml_reshape_2d(ctx, in_proj_b, in_proj_b->ne[0], 1));

    // Make contiguous so view splitting works
    qkv = ggml_cont(ctx, qkv);

    // Split q, k, v as 4D views [d_h, n_h, S, 1] into the [3D, S] qkv buffer
    // (no copy): the per-position stride is 3D elements, the per-head stride
    // d_h elements.
    const size_t ts = ggml_type_size(qkv->type);
    const size_t row_bytes = (size_t)3 * D * ts;   // position stride in qkv
    const size_t head_bytes = (size_t)d_h * ts;    // head stride within a q/k/v block
    ggml_tensor* q4 = ggml_view_4d(ctx, qkv, d_h, n_h, S, 1, head_bytes, row_bytes, qkv->nb[3], 0);
    ggml_tensor* k4 = ggml_view_4d(ctx, qkv, d_h, n_h, S, 1, head_bytes, row_bytes, qkv->nb[3], D * ts);
    ggml_tensor* v4 = ggml_view_4d(ctx, qkv, d_h, n_h, S, 1, head_bytes, row_bytes, qkv->nb[3], 2 * D * ts);

    // Permute to [d_h, S, n_h, 1] for KQ mul_mat
    // permute(k4, 0, 2, 1, 3): ne0=d_h, ne1=S, ne2=n_h, ne3=1
    ggml_tensor* kT = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3));  // [d_h, S, n_h, 1]
    ggml_tensor* qT = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3));  // [d_h, S, n_h, 1]

    // Scale Q by 1/sqrt(d_head)
    const float scale = 1.0f / sqrtf((float)d_h);
    ggml_tensor* qs = ggml_scale(ctx, qT, scale);

    // KQ attention scores
    // mul_mat: A=[d_h, S, n_h, 1], B=[d_h, S, n_h, 1]
    // K=d_h(ne0), M=S(ne1 of A), N=S(ne1 of B)
    // result: [S, S, n_h, 1] with ne0=key, ne1=query, ne2=head
    ggml_tensor* attn = ggml_mul_mat(ctx, kT, qs);  // [S, S, n_h, 1]

    // Causal mask: query n may only attend keys m <= n.  attn has ne0=key,
    // ne1=query, so diag_mask_inf(_, 0) zeroes the strictly-upper triangle
    // (key > query), matching clip.model._build_causal_attention_mask.
    if (causal) {
        attn = ggml_diag_mask_inf(ctx, attn, 0);
    }
    attn = ggml_soft_max(ctx, attn);  // softmax along ne0 (key) per query row

    // Apply attention to values
    // v4: [d_h, n_h, S, 1] -> vT: [S, d_h, n_h, 1] via permute(1, 2, 0, 3)
    // (ggml permute: ne[axis_i] = a->ne[i]).
    ggml_tensor* vT = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 2, 0, 3));  // [S, d_h, n_h, 1]

    // mul_mat: A=vT=[S, d_h, n_h, 1], B=attn=[S, S, n_h, 1]
    // K=S(ne0), M=d_h(ne1 of A), N=S(ne1 of B)
    // result: [d_h, S, n_h, 1]
    ggml_tensor* out = ggml_mul_mat(ctx, vT, attn);  // [d_h, S, n_h, 1]

    // Merge heads: [d_h, S, n_h, 1] -> [D, S] (contiguous)
    ggml_tensor* out_merged = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));  // [d_h, n_h, S, 1]
    ggml_tensor* out_2d = ggml_reshape_2d(ctx, out_merged, D, S);  // [D, S]

    // Output projection
    ggml_tensor* result = ggml_mul_mat(ctx, out_proj_w, out_2d);  // [D, S]
    if (out_proj_b)
        result = ggml_add(ctx, result, ggml_reshape_2d(ctx, out_proj_b, D, 1));

    return result;
}

// MLP block (same for text and visual, just different dims); exact_gelu
// selects erf-GELU (MobileCLIP) over QuickGELU (CLIP).
ggml_tensor* clip_mlp_block(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* fc_w, ggml_tensor* fc_b,
                            ggml_tensor* proj_w, ggml_tensor* proj_b,
                            bool exact_gelu) {
    if (fc_w->type != GGML_TYPE_F32)
        fc_w = ggml_cast(ctx, fc_w, GGML_TYPE_F32);
    if (fc_b && fc_b->type != GGML_TYPE_F32)
        fc_b = ggml_cast(ctx, fc_b, GGML_TYPE_F32);
    if (proj_w->type != GGML_TYPE_F32)
        proj_w = ggml_cast(ctx, proj_w, GGML_TYPE_F32);
    if (proj_b && proj_b->type != GGML_TYPE_F32)
        proj_b = ggml_cast(ctx, proj_b, GGML_TYPE_F32);

    ggml_tensor* h = ggml_mul_mat(ctx, fc_w, x);  // [4*D, N]
    if (fc_b) h = ggml_add(ctx, h, ggml_reshape_2d(ctx, fc_b, fc_b->ne[0], 1));
    h = exact_gelu ? ggml_gelu_erf(ctx, h)        // exact GELU (MobileCLIP)
                   : ggml_gelu_quick(ctx, h);     // QuickGELU: x * sigmoid(1.702x) (CLIP)
    h = ggml_mul_mat(ctx, proj_w, h);  // [D, N]
    if (proj_b) h = ggml_add(ctx, h, ggml_reshape_2d(ctx, proj_b, proj_b->ne[0], 1));
    return h;
}

// Build one transformer block (attention + MLP with residuals)
static ggml_tensor* transformer_block(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* ln1_w, ggml_tensor* ln1_b,
                                       ggml_tensor* attn_in_w, ggml_tensor* attn_in_b,
                                       ggml_tensor* attn_out_w, ggml_tensor* attn_out_b,
                                       ggml_tensor* ln2_w, ggml_tensor* ln2_b,
                                       ggml_tensor* mlp_fc_w, ggml_tensor* mlp_fc_b,
                                       ggml_tensor* mlp_proj_w, ggml_tensor* mlp_proj_b,
                                       int n_heads, int d_head, bool causal) {
    // Attention sub-block
    ggml_tensor* residual = x;
    x = clip_layer_norm(ctx, x, ln1_w, ln1_b);
    x = clip_self_attention(ctx, x, attn_in_w, attn_in_b, attn_out_w, attn_out_b,
                            n_heads, d_head, causal);
    x = ggml_add(ctx, residual, x);  // residual 1

    // MLP sub-block
    residual = x;
    x = clip_layer_norm(ctx, x, ln2_w, ln2_b);
    x = clip_mlp_block(ctx, x, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b, /*exact_gelu*/ false);
    x = ggml_add(ctx, residual, x);  // residual 2
    return x;
}

// ---------------------------------------------------------------------------
// clip_create_session
// ---------------------------------------------------------------------------

// Load the BPE vocab + merges written by the converters and rebuild the
// encoder map (token -> id) exactly like the Python SimpleTokenizer: vocab is
// already ordered by id.
bool clip_bpe_load(ClipBpe& bpe, const gguf_context* g,
                   const char* vocab_key, const char* merges_key, int expect_vocab) {
    const int64_t vid = gguf_find_key(g, vocab_key);
    const int64_t mid = gguf_find_key(g, merges_key);
    if (vid < 0 || mid < 0) {
        fprintf(stderr, "bpe: GGUF has no %s / %s\n", vocab_key, merges_key);
        return false;
    }
    const size_t nv = gguf_get_arr_n(g, vid);
    const size_t nm = gguf_get_arr_n(g, mid);
    if (nv != (size_t)expect_vocab) {
        fprintf(stderr, "bpe: unexpected vocab size %zu (expected %d)\n", nv, expect_vocab);
        return false;
    }
    bpe.vocab.resize(nv);
    for (size_t i = 0; i < nv; i++) {
        bpe.vocab[i] = gguf_get_arr_str(g, vid, i);
        bpe.encoder[bpe.vocab[i]] = (int)i;
    }
    bpe.merges.reserve(nm);
    for (size_t i = 0; i < nm; i++) {
        std::string pair = gguf_get_arr_str(g, mid, i);
        const size_t sp = pair.find(' ');
        if (sp == std::string::npos) {
            fprintf(stderr, "bpe: bad merge entry %zu: '%s'\n", i, pair.c_str());
            return false;
        }
        bpe.merges.emplace_back(pair.substr(0, sp), pair.substr(sp + 1));
    }
    return true;
}

ClipSession* clip_create_session(const std::string& gguf_path, const ClipSessionOptions& opts) {
    // Load GGUF file
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(gguf_path.c_str(), ip);
    if (!g) {
        fprintf(stderr, "clip: failed to open GGUF: %s\n", gguf_path.c_str());
        return nullptr;
    }

    // Verify it's a CLIP model
    const char* general_name = "";
    if (int64_t id = gguf_find_key(g, "general.name"); id >= 0)
        general_name = gguf_get_val_str(g, id);
    fprintf(stderr, "clip: loaded %s (%s)\n", gguf_path.c_str(), general_name);

    auto* s = new ClipSession();
    s->wctx = weight_ctx;

    if (opts.use_gpu) {
        // Standalone CLIP process (yolo-similarity): build the full backend
        // context so supported ops run on the GPU and anything the GPU lacks
        // (e.g. QuickGELU on Vulkan) falls back to CPU through the scheduler.
        s->backend = yolo::init_backend_ctx(opts.threads, 4096);  // image graph node cap
        if (!s->backend.cpu) {
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    } else {
        // CPU-only backend context. CLIP is called once per session from the
        // YOLO process, where creating+destroying a second GPU backend while
        // the YOLO session holds the first one corrupts shared Vulkan device
        // state. Build the CPU part directly instead of going through
        // init_backend_ctx (which would also create a GPU backend and
        // scheduler we'd immediately have to tear down).
        s->backend.n_threads = opts.threads > 0 ? opts.threads : (int)std::thread::hardware_concurrency();
        if (s->backend.n_threads <= 0) s->backend.n_threads = 4;
        s->backend.cpu = ggml_backend_cpu_init();
        if (!s->backend.cpu) {
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(s->backend.cpu, s->backend.n_threads);
        {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(s->backend.n_threads);
            s->backend.threadpool = ggml_threadpool_new(&tpp);
            if (s->backend.threadpool) {
                ggml_backend_cpu_set_threadpool(s->backend.cpu, s->backend.threadpool);
            }
        }
    }

    // Resolve all weight tensor pointers from the GGML context
    weight_resolver wr{s->wctx};

    // Text encoder weights
    s->text_embed_table = find_tensor(s->wctx, "text.token_embedding.weight");
    s->text_pos_embed   = find_tensor(s->wctx, "text.positional_embedding");
    s->text_ln_final_w  = find_tensor(s->wctx, "text.ln_final.weight");
    s->text_ln_final_b  = find_tensor(s->wctx, "text.ln_final.bias");
    s->text_proj        = find_tensor(s->wctx, "text.text_projection");

    // 12 text blocks
    for (int i = 0; i < TEXT_N_LAYERS; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "transformer.resblocks.%d", i);
        if (!resolve_text_block(wr, s->text_blocks[i], prefix)) {
            fprintf(stderr, "clip: failed to resolve text block %d\n", i);
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    // Visual encoder weights
    s->visual_conv1     = find_tensor(s->wctx, "visual.conv1.weight");
    s->visual_cls       = find_tensor(s->wctx, "visual.class_embedding");
    s->visual_pos       = find_tensor(s->wctx, "visual.positional_embedding");
    s->visual_ln_pre_w  = find_tensor(s->wctx, "visual.ln_pre.weight");
    s->visual_ln_pre_b  = find_tensor(s->wctx, "visual.ln_pre.bias");
    s->visual_ln_post_w = find_tensor(s->wctx, "visual.ln_post.weight");
    s->visual_ln_post_b = find_tensor(s->wctx, "visual.ln_post.bias");
    s->visual_proj      = find_tensor(s->wctx, "visual.visual_projection");

    // 12 visual blocks
    for (int i = 0; i < VISUAL_N_LAYERS; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "transformer.resblocks.%d", i);
        if (!resolve_visual_block(wr, s->visual_blocks[i], prefix)) {
            fprintf(stderr, "clip: failed to resolve visual block %d\n", i);
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    if (!clip_bpe_load(s->bpe, g, "clip.vocab", "clip.merges", VOCAB_SIZE)) {
        clip_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    // Weights: gguf_init loaded them into host memory (no_alloc = false).
    // CLIP is only called once per session (text encoding), so keep weights
    // on CPU even when a GPU backend is available — no performance impact.

    // Build the text encoder graph
    {
        const size_t mem = 16u * 1024u * 1024u;  // 16 MB for graph intermediates
        ggml_context* gctx = ggml_init({mem, nullptr, /*no_alloc*/ true});
        if (!gctx) {
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }

        // Input: token IDs [TEXT_CTX]
        s->text_input_tokens = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, TEXT_CTX);
        ggml_set_input(s->text_input_tokens);
        ggml_set_name(s->text_input_tokens, "tokens");

        // Token embedding lookup: [VOCAB_SIZE, EMBED_DIM] x [TEXT_CTX] -> [EMBED_DIM, TEXT_CTX]
        ggml_tensor* embed_table_f32 = ggml_cast(gctx, s->text_embed_table, GGML_TYPE_F32);
        ggml_tensor* h = ggml_get_rows(gctx, embed_table_f32, s->text_input_tokens);  // [EMBED_DIM, TEXT_CTX]

        // Add positional embedding: both h and pos are [EMBED_DIM, TEXT_CTX]
        // in ggml layout (GGUF stores torch [TEXT_CTX, EMBED_DIM] transposed).
        ggml_tensor* pos_f32 = ggml_cast(gctx, s->text_pos_embed, GGML_TYPE_F32);
        h = ggml_add(gctx, h, pos_f32);

        // 12 text transformer blocks (causal attention)
        const int d_head_text = EMBED_DIM / TEXT_N_HEADS;  // 64
        for (int i = 0; i < TEXT_N_LAYERS; i++) {
            auto& b = s->text_blocks[i];
            h = transformer_block(gctx, h,
                                   b.ln1_w, b.ln1_b,
                                   b.attn_in_w, b.attn_in_b,
                                   b.attn_out_w, b.attn_out_b,
                                   b.ln2_w, b.ln2_b,
                                   b.mlp_fc_w, b.mlp_fc_b,
                                   b.mlp_proj_w, b.mlp_proj_b,
                                   TEXT_N_HEADS, d_head_text, /*causal*/ true);
        }

        // Final layer norm
        h = clip_layer_norm(gctx, h, s->text_ln_final_w, s->text_ln_final_b);

        // Take the EOT token embedding at the sequence position given by the
        // runtime input (CLIP: x[arange(batch), text.argmax(-1)]).  Positions
        // after EOT are zero-padded, so argmax == first EOT position.
        s->text_eot_idx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
        ggml_set_input(s->text_eot_idx);
        ggml_set_name(s->text_eot_idx, "eot_idx");
        h = ggml_get_rows(gctx, h, s->text_eot_idx);  // [EMBED_DIM, 1]

        // Text projection: x @ P (CLIP uses the projection without transposing).
        // GGUF stores torch P [D, D] row-major, so ggml A[i0,i1] = P[i1,i0];
        // mul_mat computes y[m] = sum_k A[k,m] x[k], which needs A = P.  The
        // permute+cont flips the memory layout so A'[i0,i1] = P[i0,i1].
        ggml_tensor* proj_f32 = ggml_cast(gctx, s->text_proj, GGML_TYPE_F32);
        ggml_tensor* proj_T = ggml_cont(gctx, ggml_permute(gctx, proj_f32, 1, 0, 2, 3));
        h = ggml_mul_mat(gctx, proj_T, ggml_reshape_2d(gctx, h, EMBED_DIM, 1));  // [EMBED_DIM, 1]

        // L2 normalize
        h = clip_l2_norm(gctx, ggml_reshape_1d(gctx, h, EMBED_DIM));
        h = ggml_reshape_1d(gctx, h, EMBED_DIM);

        ggml_set_output(h);
        ggml_set_name(h, "text_embed");

        // Build graph
        s->text_gctx = gctx;
        s->text_graph = ggml_new_graph_custom(gctx, 2048, false);
        ggml_build_forward_expand(s->text_graph, h);
        s->text_output_embed = h;

        // Allocate graph
        if (!yolo::backend_graph_alloc(s->backend, s->text_graph)) {
            fprintf(stderr, "clip: text graph alloc failed\n");
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    // Build the image encoder graph
    {
        const size_t mem = 128u * 1024u * 1024u;  // 128 MB (12 ViT blocks ~ 2k nodes)
        ggml_context* gctx = ggml_init({mem, nullptr, /*no_alloc*/ true});
        if (!gctx) {
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }

        // Input: preprocessed image [3, IMAGE_SIZE, IMAGE_SIZE]
        s->image_input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                             IMAGE_SIZE, IMAGE_SIZE, 3, 1);
        ggml_set_input(s->image_input);
        ggml_set_name(s->image_input, "image");

        // CLIP uses CHW layout: ne[0]=W, ne[1]=H, ne[2]=C, ne[3]=N

        // Patch embedding: conv2d with 32x32 kernel, stride 32
        // Patch embedding conv: the converter flattens 4D conv weights to 2D
        // [K, OC] (K = kw*kh*ic), so restore the [kw, kh, ic, oc] view.
        ggml_tensor* conv1_f32 = ggml_cast(gctx, s->visual_conv1, GGML_TYPE_F32);
        ggml_tensor* conv_perm = ggml_reshape_4d(gctx, ggml_cont(gctx, conv1_f32), PATCH_SIZE, PATCH_SIZE, 3,
                                                 VISUAL_EMBED);

        // Apply conv2d: input [224, 224, 3, 1] + kernel [32, 32, 3, 768]
        // stride=32, pad=0
        ggml_tensor* patches = ggml_conv_2d(gctx, conv_perm, s->image_input,
                                              32, 32, 0, 0, 1, 1);
        // patches: [7, 7, 768, 1]  (ne[0]=W=7, ne[1]=H=7, ne[2]=C=768, ne[3]=N=1)

        // Flatten spatial dims: patches [7, 7, 768, 1] -> [768, 49] with the
        // channel axis on ne0 (matches cls_T for the token concat).
        // permute(1,2,0,3): ne[axis_i]=a->ne[i] -> ne=[768, 7, 7] (oc, ow, oh).
        ggml_tensor* patches_c = ggml_cont(gctx, ggml_permute(gctx, patches, 1, 2, 0, 3));  // [768, 7, 7]
        ggml_tensor* patch_flat = ggml_reshape_2d(gctx, patches_c, VISUAL_EMBED, 49);       // [768, 49]

        // Prepare class token + positional embedding
        ggml_tensor* cls_f32 = ggml_cast(gctx, s->visual_cls, GGML_TYPE_F32);  // [VISUAL_EMBED]

        // Reshape cls: [VISUAL_EMBED] -> [VISUAL_EMBED, 1]
        ggml_tensor* cls_T = ggml_reshape_2d(gctx, cls_f32, VISUAL_EMBED, 1);  // [VISUAL_EMBED, 1]

        // Concatenate cls + patches along dim 1: [VISUAL_EMBED, 50]
        ggml_tensor* tokens = ggml_concat(gctx, cls_T, patch_flat, 1);  // [VISUAL_EMBED, 50]

        // Add positional embeddings
        // pos_f32 GGUF layout is already [EMBED_DIM, N_TOKENS] (transposed
        // from torch [N_TOKENS, EMBED_DIM]), matching `tokens` — direct add.
        ggml_tensor* pos_f32 = ggml_cast(gctx, s->visual_pos, GGML_TYPE_F32);
        ggml_tensor* h = ggml_add(gctx, tokens, pos_f32);

        // Pre layer norm
        h = clip_layer_norm(gctx, h, s->visual_ln_pre_w, s->visual_ln_pre_b);

        // 12 visual transformer blocks (bidirectional attention)
        const int d_head_visual = VISUAL_EMBED / VISUAL_N_HEADS;  // 64
        for (int i = 0; i < VISUAL_N_LAYERS; i++) {
            auto& b = s->visual_blocks[i];
            h = transformer_block(gctx, h,
                                   b.ln1_w, b.ln1_b,
                                   b.attn_in_w, b.attn_in_b,
                                   b.attn_out_w, b.attn_out_b,
                                   b.ln2_w, b.ln2_b,
                                   b.mlp_fc_w, b.mlp_fc_b,
                                   b.mlp_proj_w, b.mlp_proj_b,
                                   VISUAL_N_HEADS, d_head_visual, /*causal*/ false);
        }

        // Take cls token (first position)
        // h: [VISUAL_EMBED, 50], take index 0
        h = ggml_view_1d(gctx, h, VISUAL_EMBED, 0);

        // Post layer norm
        h = clip_layer_norm(gctx, h, s->visual_ln_post_w, s->visual_ln_post_b);

        // Visual projection: [768] -> mul_mat(proj_T, [768, 1]) -> [512, 1] -> [512]
        ggml_tensor* proj_f32 = ggml_cast(gctx, s->visual_proj, GGML_TYPE_F32);
        // GGUF stores torch [768, 512] transposed as [512, 768] (ne0=512), so
        // transpose to [768, 512] for the mul_mat inner-product axis.
        ggml_tensor* proj_T = ggml_cont(gctx, ggml_permute(gctx, proj_f32, 1, 0, 2, 3));
        h = ggml_mul_mat(gctx, proj_T, ggml_reshape_2d(gctx, h, VISUAL_EMBED, 1));
        h = ggml_reshape_1d(gctx, h, EMBED_DIM);

        // L2 normalize
        h = clip_l2_norm(gctx, h);
        h = ggml_reshape_1d(gctx, h, EMBED_DIM);

        ggml_set_output(h);
        ggml_set_name(h, "image_embed");

        // Build graph
        s->image_gctx = gctx;
        s->image_graph = ggml_new_graph_custom(gctx, 4096, false);
        ggml_build_forward_expand(s->image_graph, h);
        s->image_output_embed = h;

        // Allocate graph
        if (!yolo::backend_graph_alloc(s->backend, s->image_graph)) {
            fprintf(stderr, "clip: image graph alloc failed\n");
            clip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    // Allocate scratch buffers
    s->text_embed_scratch.resize(EMBED_DIM);
    s->image_embed_scratch.resize(EMBED_DIM);

    gguf_free(g);
    fprintf(stderr, "clip: session ready, backend=%s\n",
            yolo::backend_name(s->backend));
    return s;
}

// ---------------------------------------------------------------------------
// BPE tokenizer (matches clip.simple_tokenizer.SimpleTokenizer + clip.tokenize)
// ---------------------------------------------------------------------------

// The reversible byte<->unicode mapping from GPT-2 / CLIP: printable ASCII and
// Latin-1 stay identity; the remaining 163 bytes map to chr(256 + n) in order.
static const std::unordered_map<unsigned char, std::string>& byte_to_unicode() {
    static std::unordered_map<unsigned char, std::string> table = [] {
        std::unordered_map<unsigned char, std::string> t;
        std::vector<int> bs;
        auto add_range = [&](int lo, int hi) { for (int i = lo; i <= hi; i++) bs.push_back(i); };
        add_range('!', '~');
        add_range(0xA1, 0xAC);
        add_range(0xAE, 0xFF);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            t[(unsigned char)bs[i]] = std::string(1, (char)cs[i]);
        }
        return t;
    }();
    return table;
}

// Decode the next UTF-8 code point; advances `p`. Returns -1 on invalid input.
static int32_t utf8_decode(const unsigned char* s, size_t len, size_t* p) {
    const unsigned char c = s[*p];
    if (c < 0x80) { (*p)++; return c; }
    int32_t cp = 0; int extra = 0;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { (*p)++; return -1; }
    if (*p + extra >= len + 1 || *p + extra > len) { (*p)++; return -1; }
    for (int i = 1; i <= extra; i++) {
        const unsigned char cc = s[*p + i];
        if ((cc & 0xC0) != 0x80) { (*p)++; return -1; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    *p += 1 + extra;
    return cp;
}

// Unicode letter / number classification covering the code ranges that occur
// in practice (Latin, Greek, Cyrillic, Hebrew, Arabic, CJK, kana, hangul).
static bool is_unicode_letter(int32_t cp) {
    if (cp < 0) return false;
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0xC0) return false;
    return (cp >= 0xC0 && cp <= 0x2AF) ||    // Latin-1 supplement .. Latin Extended
           (cp >= 0x370 && cp <= 0x52F) ||   // Greek + Cyrillic
           (cp >= 0x531 && cp <= 0x58F) ||   // Armenian
           (cp >= 0x590 && cp <= 0x5F4) ||   // Hebrew
           (cp >= 0x600 && cp <= 0x6FF) ||   // Arabic
           (cp >= 0x900 && cp <= 0x97F) ||   // Devanagari
           (cp >= 0x1E00 && cp <= 0x1FFF) || // Latin Extended Additional
           (cp >= 0x2C60 && cp <= 0x2C7F) || // Latin Extended-C
           (cp >= 0x3040 && cp <= 0x30FF) || // hiragana + katakana
           (cp >= 0x3400 && cp <= 0x9FFF) || // CJK
           (cp >= 0xAC00 && cp <= 0xD7AF);   // hangul
}
static bool is_unicode_number(int32_t cp) {
    return cp >= 0 && ((cp >= '0' && cp <= '9') || (cp >= 0x660 && cp <= 0x669) ||  // Arabic-Indic
                       (cp >= 0x6F0 && cp <= 0x6F9) ||
                       (cp >= 0x96F && cp <= 0x96F) ||
                       (cp >= 0xFF10 && cp <= 0xFF19));  // fullwidth
}
static bool is_unicode_space(int32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200B) || cp == 0x2028 || cp == 0x2029 || cp == 0x3000;
}

// Tokenize `text` into BPE subword tokens matching the Python SimpleTokenizer
// regex: <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|'ll|'d | letters+ |
// one digit | runs of other non-space chars.
static std::vector<std::string> regex_split(const std::string& text) {
    std::vector<std::string> out;
    const unsigned char* s = (const unsigned char*)text.data();
    const size_t len = text.size();
    size_t p = 0;
    auto literal = [&](const char* lit) -> bool {
        const size_t n = strlen(lit);
        if (p + n <= len && memcmp(s + p, lit, n) == 0) { out.emplace_back(lit); p += n; return true; }
        return false;
    };
    while (p < len) {
        if (literal("<|startoftext|>") || literal("<|endoftext|>")) continue;
        if (p + 3 <= len && (memcmp(s + p, "'re", 3) == 0 || memcmp(s + p, "'ve", 3) == 0 ||
                             memcmp(s + p, "'ll", 3) == 0)) {
            out.emplace_back(text.substr(p, 3));
            p += 3;
            continue;
        }
        if (p + 2 <= len && (memcmp(s + p, "'s", 2) == 0 || memcmp(s + p, "'t", 2) == 0 ||
                             memcmp(s + p, "'m", 2) == 0 || memcmp(s + p, "'d", 2) == 0)) {
            out.emplace_back(text.substr(p, 2));
            p += 2;
            continue;
        }
        size_t q = p;
        const int32_t cp = utf8_decode(s, len, &q);
        if (cp > 0 && is_unicode_letter(cp)) {
            size_t e = q;
            while (e < len) { size_t q2 = e; int32_t c2 = utf8_decode(s, len, &q2); if (!(c2 > 0 && is_unicode_letter(c2))) break; e = q2; }
            out.push_back(text.substr(p, e - p));
            p = e;
        } else if (cp > 0 && is_unicode_number(cp)) {
            out.emplace_back(text.substr(p, q - p));
            p = q;
        } else if (cp > 0 && is_unicode_space(cp)) {
            p = q;
        } else {
            size_t e = q;
            while (e < len) {
                size_t q2 = e; int32_t c2 = utf8_decode(s, len, &q2);
                if (c2 > 0 && (is_unicode_space(c2) || is_unicode_letter(c2) || is_unicode_number(c2))) break;
                e = q2;
            }
            out.push_back(text.substr(p, e - p));
            p = e;
        }
    }
    return out;
}

// whitespace_clean: collapse runs of whitespace into a single space.
static std::string whitespace_clean(std::string t) {
    std::string out;
    bool pending_space = false;
    for (unsigned char c : t) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') pending_space = true;
        else { if (pending_space && !out.empty()) out += ' '; pending_space = false; out += (char)c; }
    }
    return out;
}

// basic_clean approximation: strip control chars / html entity unescape is
// intentionally minimal (ftfy + double html.unescape rarely fire on prompts).
static std::string basic_clean(std::string t) {
    std::string out;
    for (unsigned char c : t) {
        if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') continue;
        out += (char)c;
    }
    return out;
}

// Byte-encode a token: map each UTF-8 byte through byte_to_unicode.
static std::string byte_encode(const std::string& tok) {
    const auto& table = byte_to_unicode();
    std::string out;
    for (unsigned char b : tok) out += table.at(b);
    return out;
}

// GPT-2 BPE merge of a single byte-encoded token (SimpleTokenizer.bpe).
static std::string bpe_merge(ClipBpe& bpe, const std::string& token) {
    auto it = bpe.bpe_cache.find(token);
    if (it != bpe.bpe_cache.end()) return it->second;
    // word = tuple(token[:-1]) + (token[-1] + '</w>',)
    std::vector<std::string> word;
    for (size_t i = 0; i + 1 < token.size(); i++) word.push_back(token.substr(i, 1));
    word.push_back(token.substr(token.size() - 1) + "</w>");
    auto pairs_of = [](const std::vector<std::string>& w) {
        std::vector<std::pair<std::string, std::string>> ps;
        for (size_t i = 0; i + 1 < w.size(); i++) ps.emplace_back(w[i], w[i + 1]);
        return ps;
    };
    auto ranks = [&](const std::string& a, const std::string& b) {
        for (size_t i = 0; i < bpe.merges.size(); i++) {
            if (bpe.merges[i].first == a && bpe.merges[i].second == b) return (int)i;
        }
        return INT_MAX;
    };
    std::vector<std::pair<std::string, std::string>> pairs = pairs_of(word);
    while (!pairs.empty()) {
        int best = INT_MAX; size_t bi = 0;
        for (size_t i = 0; i < pairs.size(); i++) {
            const int r = ranks(pairs[i].first, pairs[i].second);
            if (r < best) { best = r; bi = i; }
        }
        if (best == INT_MAX) break;
        const std::string& f = pairs[bi].first;
        const std::string& sc = pairs[bi].second;
        std::vector<std::string> nw;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == f && word[i + 1] == sc) {
                nw.push_back(f + sc);
                i += 2;
            } else {
                nw.push_back(word[i]);
                i++;
            }
        }
        word.swap(nw);
        pairs = pairs_of(word);
    }
    std::string joined;
    for (size_t i = 0; i < word.size(); i++) { if (i) joined += ' '; joined += word[i]; }
    bpe.bpe_cache[token] = joined;
    return joined;
}

int clip_bpe_tokenize(ClipBpe& bpe, const char* text, int ctx_len, int32_t* tokens) {
    if (!text || !tokens || ctx_len <= 0) return 0;
    std::string t = text;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    t = whitespace_clean(basic_clean(t));

    std::vector<int32_t> ids;
    ids.push_back(49406);  // <|startoftext|>
    for (const std::string& tok : regex_split(t)) {
        const std::string enc = byte_encode(tok);
        const std::string merged = bpe_merge(bpe, enc);
        size_t pos = 0;
        while (pos <= merged.size()) {
            const size_t sp = merged.find(' ', pos);
            const std::string sub = sp == std::string::npos ? merged.substr(pos) : merged.substr(pos, sp - pos);
            auto it = bpe.encoder.find(sub);
            if (it == bpe.encoder.end()) {
                // Unknown subword: fall back to the raw byte-encoded token's
                // first char (CLIP would produce <unk>-like behaviour); skip to
                // keep the reference alignment for common prompts.
                pos = sp == std::string::npos ? merged.size() + 1 : sp + 1;
                continue;
            }
            ids.push_back(it->second);
            if (sp == std::string::npos) break;
            pos = sp + 1;
        }
    }
    ids.push_back(49407);  // <|endoftext|>
    if ((int)ids.size() > ctx_len) {
        ids.resize(ctx_len);
        ids.back() = 49407;
    }
    for (int i = 0; i < ctx_len; i++) tokens[i] = i < (int)ids.size() ? ids[i] : 0;
    return (int)ids.size();
}

int clip_tokenize_text(ClipSession* s, const char* text, int32_t* tokens) {
    if (!s) return 0;
    return clip_bpe_tokenize(s->bpe, text, TEXT_CTX, tokens);
}

bool clip_encode_string(ClipSession* s, const char* text, float* embed) {
    if (!s) return false;
    int32_t tokens[TEXT_CTX];
    clip_tokenize_text(s, text, tokens);
    return clip_encode_text(s, tokens, embed);
}

// ---------------------------------------------------------------------------
// clip_encode_text
// ---------------------------------------------------------------------------

bool clip_encode_text(ClipSession* s, const int32_t* tokens, float* embed) {
    if (!s || !s->text_graph) return false;

    // Copy token IDs to the input tensor
    ggml_backend_tensor_set(s->text_input_tokens, tokens, 0, TEXT_CTX * sizeof(int32_t));

    // EOT position: first 49407 in the sequence (positions after it are 0).
    int32_t eot = TEXT_CTX - 1;
    for (int i = 0; i < TEXT_CTX; i++) {
        if (tokens[i] == VOCAB_SIZE - 1) { eot = i; break; }
    }
    ggml_backend_tensor_set(s->text_eot_idx, &eot, 0, sizeof(int32_t));

    // Compute
    if (yolo::backend_graph_compute(s->backend, s->text_graph) != 0) {
        fprintf(stderr, "clip: text graph compute failed\n");
        return false;
    }

    // Read result
    ggml_backend_tensor_get(s->text_output_embed, embed, 0, EMBED_DIM * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// clip_encode_image
// ---------------------------------------------------------------------------

bool clip_encode_image(ClipSession* s, const float* image, float* embed) {
    if (!s || !s->image_graph) return false;

    // Copy image data to the input tensor
    const size_t nbytes = (size_t)IMAGE_SIZE * IMAGE_SIZE * 3 * sizeof(float);
    ggml_backend_tensor_set(s->image_input, image, 0, nbytes);

    // Compute
    if (yolo::backend_graph_compute(s->backend, s->image_graph) != 0) {
        fprintf(stderr, "clip: image graph compute failed\n");
        return false;
    }

    // Read result
    ggml_backend_tensor_get(s->image_output_embed, embed, 0, EMBED_DIM * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// clip_preprocess_image
// ---------------------------------------------------------------------------

void clip_preprocess_image(const unsigned char* pixels, int w, int h, int channels,
                            float* out) {
    // CLIP image preprocessing (matches torchvision.Compose used by Python CLIP):
    // 1. Resize: short edge to IMAGE_SIZE (224), preserve aspect ratio
    // 2. Center crop: IMAGE_SIZE x IMAGE_SIZE
    // 3. To tensor: HWC -> CHW, /255
    // 4. Normalize: mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]

    const int target = IMAGE_SIZE;

    // Step 1: Compute scale factor
    float scale;
    int new_w, new_h;
    if (w < h) {
        scale = (float)target / w;
        new_w = target;
        new_h = (int)(h * scale + 0.5f);
    } else {
        scale = (float)target / h;
        new_h = target;
        new_w = (int)(w * scale + 0.5f);
    }

    // Step 2: Center crop
    const int crop_x = (new_w - target) / 2;
    const int crop_y = (new_h - target) / 2;

    // Simplified: directly sample (bilinear) from the original image into the
    // target CHW buffer, applying normalization at the same time.
    // This avoids allocating a full intermediate resized image.

    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    const int c_in = channels >= 3 ? 3 : channels;

    for (int y = 0; y < target; y++) {
        for (int x = 0; x < target; x++) {
            // Map target pixel (x, y) to source image coordinates
            const float src_x = (x + crop_x) / scale;
            const float src_y = (y + crop_y) / scale;

            // Bilinear interpolation in source image
            const int ix = (int)src_x;
            const int iy = (int)src_y;
            const float fx = src_x - ix;
            const float fy = src_y - iy;

            const int ix0 = std::max(0, std::min(ix, w - 1));
            const int ix1 = std::max(0, std::min(ix + 1, w - 1));
            const int iy0 = std::max(0, std::min(iy, h - 1));
            const int iy1 = std::max(0, std::min(iy + 1, h - 1));

            for (int c = 0; c < c_in; c++) {
                // Bilinear interpolate
                float v00 = pixels[(iy0 * w + ix0) * channels + c];
                float v10 = pixels[(iy0 * w + ix1) * channels + c];
                float v01 = pixels[(iy1 * w + ix0) * channels + c];
                float v11 = pixels[(iy1 * w + ix1) * channels + c];

                float v = (v00 * (1 - fx) + v10 * fx) * (1 - fy) +
                          (v01 * (1 - fx) + v11 * fx) * fy;

                // Normalize: (v / 255 - mean) / std
                // Output CHW: out[c * target * target + y * target + x]
                out[c * target * target + y * target + x] = (v / 255.0f - mean[c]) / std[c];
            }
            // Zero unused channels (if input was grayscale)
            for (int c = c_in; c < 3; c++) {
                out[c * target * target + y * target + x] = -mean[c] / std[c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// clip_free_session
// ---------------------------------------------------------------------------

void clip_free_session(ClipSession* s) {
    if (!s) return;
    if (s->wbuf) ggml_backend_buffer_free(s->wbuf);
    yolo::free_backend_ctx(s->backend);
    if (s->wctx) ggml_free(s->wctx);
    if (s->text_gctx) ggml_free(s->text_gctx);
    if (s->image_gctx) ggml_free(s->image_gctx);
    delete s;
}

}  // namespace clip