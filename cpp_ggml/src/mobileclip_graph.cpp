// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "mobileclip_graph.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <thread>

namespace mobileclip {

static ggml_tensor* find_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "mobileclip: tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

static bool resolve_block(ggml_context* wctx, MobileclipSession::Block& blk, int i) {
    char name[128];
    auto resolve = [&](const char* suffix) -> ggml_tensor* {
        snprintf(name, sizeof(name), "mobileclip.blk%d.%s", i, suffix);
        return find_tensor(wctx, name);
    };
    blk.attn_in_w  = resolve("attn.in_proj_weight");
    blk.attn_in_b  = resolve("attn.in_proj_bias");
    blk.attn_out_w = resolve("attn.out_proj.weight");
    blk.attn_out_b = resolve("attn.out_proj.bias");
    blk.ln_mid_w   = resolve("ln_mid.weight");
    blk.ln_mid_b   = resolve("ln_mid.bias");
    blk.mlp_fc_w   = resolve("mlp.c_fc.weight");
    blk.mlp_fc_b   = resolve("mlp.c_fc.bias");
    blk.mlp_proj_w = resolve("mlp.c_proj.weight");
    blk.mlp_proj_b = resolve("mlp.c_proj.bias");
    blk.ln_post_w  = resolve("ln_post.weight");
    blk.ln_post_b  = resolve("ln_post.bias");
    return blk.attn_in_w != nullptr;
}

MobileclipSession* mobileclip_create_session(const std::string& gguf_path, int threads) {
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(gguf_path.c_str(), ip);
    if (!g) {
        fprintf(stderr, "mobileclip: failed to open GGUF: %s\n", gguf_path.c_str());
        return nullptr;
    }

    const char* general_name = "";
    if (int64_t id = gguf_find_key(g, "general.name"); id >= 0)
        general_name = gguf_get_val_str(g, id);
    fprintf(stderr, "mobileclip: loaded %s (%s)\n", gguf_path.c_str(), general_name);

    auto* s = new MobileclipSession();
    s->wctx = weight_ctx;

    // CPU-only backend context: MobileCLIP runs inside the YOLO process,
    // where a second GPU backend init/free pair would corrupt the shared
    // device state held by the YOLO session (same constraint as CLIP).
    s->backend.n_threads = threads > 0 ? threads : (int)std::thread::hardware_concurrency();
    if (s->backend.n_threads <= 0) s->backend.n_threads = 4;
    s->backend.cpu = ggml_backend_cpu_init();
    if (!s->backend.cpu) {
        mobileclip_free_session(s);
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

    s->embed_table = find_tensor(s->wctx, "mobileclip.token_embedding.weight");
    s->pos_embed   = find_tensor(s->wctx, "mobileclip.positional_embedding");
    s->ln_pre_w    = find_tensor(s->wctx, "mobileclip.ln_pre.weight");
    s->ln_pre_b    = find_tensor(s->wctx, "mobileclip.ln_pre.bias");
    s->proj        = find_tensor(s->wctx, "mobileclip.text_projection");
    if (!s->embed_table || !s->pos_embed || !s->ln_pre_w || !s->ln_pre_b || !s->proj) {
        mobileclip_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    for (int i = 0; i < N_LAYERS; i++) {
        if (!resolve_block(s->wctx, s->blocks[i], i)) {
            fprintf(stderr, "mobileclip: failed to resolve block %d\n", i);
            mobileclip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    if (!clip::clip_bpe_load(s->bpe, g, "mobileclip.vocab", "mobileclip.merges", VOCAB_SIZE)) {
        mobileclip_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    // Build the encoder graph. Weights stay on CPU (encoded once per
    // session, same as CLIP), so no GPU upload is needed.
    {
        const size_t mem = 16u * 1024u * 1024u;  // 16 MB for graph intermediates
        ggml_context* gctx = ggml_init({mem, nullptr, /*no_alloc*/ true});
        if (!gctx) {
            mobileclip_free_session(s);
            gguf_free(g);
            return nullptr;
        }

        // Input: token IDs [TEXT_CTX]
        s->input_tokens = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, TEXT_CTX);
        ggml_set_input(s->input_tokens);
        ggml_set_name(s->input_tokens, "tokens");

        // Token embedding lookup: [VOCAB, D] x [CTX] -> [D, CTX]
        ggml_tensor* embed_f32 = ggml_cast(gctx, s->embed_table, GGML_TYPE_F32);
        ggml_tensor* stream = ggml_get_rows(gctx, embed_f32, s->input_tokens);  // [D, CTX]

        // Positional embedding: GGUF stores torch [CTX, D] transposed, so the
        // ggml layout is already [D, CTX] — direct add.
        ggml_tensor* pos_f32 = ggml_cast(gctx, s->pos_embed, GGML_TYPE_F32);
        stream = ggml_add(gctx, stream, pos_f32);

        // Attention consumes the LayerNormed view; residuals hit the raw
        // stream. Block 0's pre-attn LN is ln_pre; every later block reuses
        // the previous block's ln_post output.
        ggml_tensor* cur = clip::clip_layer_norm(gctx, stream, s->ln_pre_w, s->ln_pre_b);
        const int d_head = EMBED_DIM / N_HEADS;  // 64
        for (int i = 0; i < N_LAYERS; i++) {
            auto& b = s->blocks[i];
            ggml_tensor* a = clip::clip_self_attention(gctx, cur,
                                                       b.attn_in_w, b.attn_in_b,
                                                       b.attn_out_w, b.attn_out_b,
                                                       N_HEADS, d_head, /*causal*/ true);
            stream = ggml_add(gctx, stream, a);  // residual 1
            ggml_tensor* mid = clip::clip_layer_norm(gctx, stream, b.ln_mid_w, b.ln_mid_b);
            ggml_tensor* ff = clip::clip_mlp_block(gctx, mid,
                                                   b.mlp_fc_w, b.mlp_fc_b,
                                                   b.mlp_proj_w, b.mlp_proj_b,
                                                   /*exact_gelu*/ true);
            stream = ggml_add(gctx, stream, ff);  // residual 2
            cur = clip::clip_layer_norm(gctx, stream, b.ln_post_w, b.ln_post_b);
        }

        // EOS gather: pool the LN'd stream at the runtime EOT position
        // (MobileCLIPTS: x[arange(batch), text.argmax(-1)]).
        s->eot_idx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
        ggml_set_input(s->eot_idx);
        ggml_set_name(s->eot_idx, "eot_idx");
        ggml_tensor* h = ggml_get_rows(gctx, cur, s->eot_idx);  // [D, 1]

        // Text projection: pooled @ P. GGUF stores torch P (out, in), so
        // the ggml tensor is P transposed and mul_mat(proj, pooled)
        // computes sum_k P[out,k] pooled[k] directly — no layout flip.
        ggml_tensor* proj_f32 = ggml_cast(gctx, s->proj, GGML_TYPE_F32);
        h = ggml_mul_mat(gctx, proj_f32, ggml_reshape_2d(gctx, h, EMBED_DIM, 1));  // [D, 1]

        // L2 normalize
        h = clip::clip_l2_norm(gctx, ggml_reshape_1d(gctx, h, EMBED_DIM));
        h = ggml_reshape_1d(gctx, h, EMBED_DIM);

        ggml_set_output(h);
        ggml_set_name(h, "text_embed");

        s->gctx = gctx;
        s->graph = ggml_new_graph_custom(gctx, 2048, false);
        ggml_build_forward_expand(s->graph, h);
        s->output_embed = h;

        if (!yolo::backend_graph_alloc(s->backend, s->graph)) {
            fprintf(stderr, "mobileclip: graph alloc failed\n");
            mobileclip_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    gguf_free(g);
    fprintf(stderr, "mobileclip: session ready (%d threads)\n", s->backend.n_threads);
    return s;
}

bool mobileclip_encode_tokens(MobileclipSession* s, const int32_t* tokens, float* embed) {
    if (!s || !s->graph || !tokens || !embed) return false;

    ggml_backend_tensor_set(s->input_tokens, tokens, 0, TEXT_CTX * sizeof(int32_t));

    // EOS position: first EOT id in the sequence (positions after are 0).
    int32_t eot = TEXT_CTX - 1;
    for (int i = 0; i < TEXT_CTX; i++) {
        if (tokens[i] == VOCAB_SIZE - 1) { eot = i; break; }
    }
    ggml_backend_tensor_set(s->eot_idx, &eot, 0, sizeof(int32_t));

    if (yolo::backend_graph_compute(s->backend, s->graph) != 0) {
        fprintf(stderr, "mobileclip: graph compute failed\n");
        return false;
    }
    ggml_backend_tensor_get(s->output_embed, embed, 0, EMBED_DIM * sizeof(float));
    return true;
}

bool mobileclip_encode_string(MobileclipSession* s, const char* text, float* embed) {
    if (!s || !text) return false;
    int32_t tokens[TEXT_CTX];
    clip::clip_bpe_tokenize(s->bpe, text, TEXT_CTX, tokens);
    return mobileclip_encode_tokens(s, tokens, embed);
}

void mobileclip_free_session(MobileclipSession* s) {
    if (!s) return;
    yolo::free_backend_ctx(s->backend);
    if (s->wctx) ggml_free(s->wctx);
    if (s->gctx) ggml_free(s->gctx);
    delete s;
}

}  // namespace mobileclip
