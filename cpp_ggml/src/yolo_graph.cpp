// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "yolo_graph.hpp"
#include "common.hpp"

#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>

namespace yolo {

namespace {

// Q8_0 block layout (binary-compatible with ggml's block_q8_0): a per-32-
// element fp16 scale followed by 32 int8 deltas. Defined locally so the host
// can dequantize weights for backends with no Q8 conv path (vulkan) without
// pulling ggml-common.h out of the ggml src tree.
constexpr int QK8_0 = 32;
struct block_q8_0 { ggml_fp16_t d; int8_t qs[QK8_0]; };
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "block_q8_0 layout mismatch");

ggml_tensor* dup_host_tensor(ggml_context* ctx, const HostTensor& ht) {
    ggml_tensor* t = ggml_new_tensor(ctx, ht.type, 4, ht.ne);
    ggml_set_name(t, ht.name.c_str());
    return t;
}

struct GraphBuilder {
    ggml_context* gctx;                // graph tensors
    ggml_context* wctx;                // weight tensors (data lives in wbuf)
    const ModelDef& model;
    std::vector<ggml_tensor*> weights;  // tensors whose data lives in wbuf
    // CUDA: every quantized weight conforms to the igemm Q8_0 path (K
    // 32-aligned), so quantized convs join the f16 direct flow and the CUDA
    // kernel dequantizes weight blocks inside the implicit-GEMM B tile.
    // vulkan: the same flag triggers a one-shot host dequant of Q8_0 weights
    // to f16 at load time, since vulkan has no Q8 conv shader — the graph
    // then runs the plain f16 direct-conv path.
    bool q8_direct = false;

    ggml_tensor* w(const std::string& prefix, const char* suffix) {
        auto it = model.tensors.find(prefix + "." + suffix);
        if (it == model.tensors.end()) return nullptr;
        ggml_tensor* t = dup_host_tensor(wctx, it->second);
        weights.push_back(t);
        return t;
    }

    // Restore the 4D conv kernel view for 2D-stored weights (Q8_0 blocks, or
    // F16 weights dequantized on the host for vulkan). F16/F32 weights that
    // the converter already stored 4D skip the reshape.
    ggml_tensor* kernel4d(const OpDef& op, ggml_tensor* wT) {
        if (wT->ne[2] != 1 || wT->ne[3] != 1 || wT->type == GGML_TYPE_F32) {
            return wT;
        }
        const int64_t kh = op.ai("k", 0), kw = op.ai("k", 1);
        const int64_t out = wT->ne[1];
        const int64_t in = wT->ne[0] / (kh * kw);
        return ggml_reshape_4d(gctx, wT, kw, kh, in, out);
    }

    ggml_tensor* add_bias_act(const OpDef& op, const std::string& prefix, ggml_tensor* out) {
        if (ggml_tensor* b = w(prefix, "b")) {
            if (b->type != out->type) b = ggml_cast(gctx, b, out->type);
            out = ggml_add(gctx, out, ggml_reshape_4d(gctx, b, 1, 1, b->ne[0], 1));
        }
        auto act = op.sparams.find("act");
        if (act != op.sparams.end() && act->second == "silu") {
            out = ggml_silu(gctx, out);
        }
        return out;
    }

    ggml_tensor* conv2d(const OpDef& op, const std::string& prefix, ggml_tensor* x) {
        ggml_tensor* wT = w(prefix, "w");
        GGML_ASSERT(wT && "conv without weight");
        const bool depthwise = op.type == "dwconv";
        [[maybe_unused]] ggml_tensor* bias = w(prefix, "b");
        const auto act = op.sparams.find("act");
        [[maybe_unused]] const bool silu = act != op.sparams.end() && act->second == "silu";
        [[maybe_unused]] const bool direct_types =
            wT->type == x->type && (wT->type == GGML_TYPE_F32 || wT->type == GGML_TYPE_F16);
        ggml_tensor* out;
#if defined(YOLO_USE_VULKAN)
        if (direct_types) {
            ggml_tensor* w4d = kernel4d(op, wT);
            if (!depthwise && bias && silu) {
                return ggml_conv_2d_direct_bias_silu(gctx, w4d, x, bias,
                                                     (int)op.ai("s", 0), (int)op.ai("s", 1),
                                                     (int)op.ai("p", 0), (int)op.ai("p", 1),
                                                     (int)op.ai("d", 0), (int)op.ai("d", 1));
            }
            out = depthwise
                ? ggml_conv_2d_dw_direct(gctx, w4d, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                                         (int)op.ai("p", 0), (int)op.ai("p", 1),
                                         (int)op.ai("d", 0), (int)op.ai("d", 1))
                : ggml_conv_2d_direct(gctx, w4d, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                                      (int)op.ai("p", 0), (int)op.ai("p", 1),
                                      (int)op.ai("d", 0), (int)op.ai("d", 1));
        } else
#elif defined(YOLO_USE_CUDA)
        // CUDA has direct kernels for matching F32 and F16 tensors. Q8_0
        // joins via the plan-time direct path below.
        // The F32 direct kernel uses Ampere TF32 WMMA internally.  Keep the
        // tensor-core path for F16, but route F32 through the standard ggml
        // convolution/GEMM implementation to preserve the F32 reference.
        const bool cuda_direct_types = direct_types && wT->type != GGML_TYPE_F32;
        if (!depthwise && cuda_direct_types && bias && silu) {
            return ggml_conv_2d_direct_bias_silu(gctx, wT, x, bias,
                                                 (int)op.ai("s", 0), (int)op.ai("s", 1),
                                                 (int)op.ai("p", 0), (int)op.ai("p", 1),
                                                 (int)op.ai("d", 0), (int)op.ai("d", 1));
        } else if (cuda_direct_types) {
            out = depthwise
                ? ggml_conv_2d_dw_direct(gctx, wT, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                                         (int)op.ai("p", 0), (int)op.ai("p", 1),
                                         (int)op.ai("d", 0), (int)op.ai("d", 1))
                : ggml_conv_2d_direct(gctx, wT, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                                      (int)op.ai("p", 0), (int)op.ai("p", 1),
                                      (int)op.ai("d", 0), (int)op.ai("d", 1));
        } else
#endif
        if (!depthwise && ggml_is_quantized(wT->type)) {
#if defined(YOLO_USE_CUDA)
            if (q8_direct) {
                // PSA attention feeds its proj conv F32 (mul_mat/softmax chain);
                // the igemm Q8 path takes F16 activations.
                if (x->type != GGML_TYPE_F16) x = ggml_cast(gctx, x, GGML_TYPE_F16);
                ggml_tensor* w4d = kernel4d(op, wT);
                if (bias && silu) {
                    return ggml_conv_2d_direct_bias_silu(gctx, w4d, x, bias, (int)op.ai("s", 0),
                                                         (int)op.ai("s", 1), (int)op.ai("p", 0),
                                                         (int)op.ai("p", 1), (int)op.ai("d", 0), (int)op.ai("d", 1));
                }
                out = ggml_conv_2d_direct(gctx, w4d, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                                          (int)op.ai("p", 0), (int)op.ai("p", 1),
                                          (int)op.ai("d", 0), (int)op.ai("d", 1));
            } else
#endif
            out = conv2d_q(wT, kernel4d(op, wT), x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                           (int)op.ai("p", 0), (int)op.ai("p", 1), (int)op.ai("d", 0), (int)op.ai("d", 1));
        } else {
            ggml_tensor* w4d = kernel4d(op, wT);
            if (!depthwise && w4d->ne[2] != x->ne[2]) {
                YOLO_LOG_ERROR("conv %s type=%s: kernel ne=[%lld,%lld,%lld,%lld] vs input ne=[%lld,%lld,%lld,%lld]",
                               prefix.c_str(), op.type.c_str(), (long long)w4d->ne[0], (long long)w4d->ne[1],
                               (long long)w4d->ne[2], (long long)w4d->ne[3], (long long)x->ne[0], (long long)x->ne[1],
                               (long long)x->ne[2], (long long)x->ne[3]);
            }
            out = depthwise
                ? ggml_conv_2d_dw(gctx, w4d, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                                  (int)op.ai("p", 0), (int)op.ai("p", 1),
                                  (int)op.ai("d", 0), (int)op.ai("d", 1))
                : ggml_conv_2d(gctx, w4d, x, (int)op.ai("s", 0), (int)op.ai("s", 1),
                               (int)op.ai("p", 0), (int)op.ai("p", 1),
                               (int)op.ai("d", 0), (int)op.ai("d", 1));
        }
        return add_bias_act(op, prefix, out);
    }

    ggml_tensor* conv_transpose(const OpDef& op, const std::string& prefix, ggml_tensor* x) {
        ggml_tensor* wT = w(prefix, "w");
        GGML_ASSERT(wT && "transpose conv without weight");
        ggml_tensor* out = ggml_conv_transpose_2d_p0(gctx, wT, x, (int)op.ip("s"));
        return add_bias_act(op, prefix, out);
    }

    // Quantized conv: ggml_conv_2d would build mul_mat(F16 im2col, Q8 kernel) which
    // asserts on CPU (src1 must be F32 or the kernel dtype). Mirror llama.cpp instead:
    // mul_mat(Q8 kernel [K,OC], F32 im2col) — src1 F32 gets dynamically quantized to
    // the kernel's vec_dot type. w4d only lends KH/KW/IC shape metadata to im2col;
    // mul_mat consumes wT itself so quant blocks stay contiguous.
    ggml_tensor* conv2d_q(ggml_tensor* wT, ggml_tensor* w4d, ggml_tensor* x,
                          int s0, int s1, int p0, int p1, int d0, int d1) {
        ggml_tensor* im2 = ggml_im2col(gctx, w4d, x, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F32); // [K, OW, OH, N]
        const int64_t P = im2->ne[1] * im2->ne[2] * im2->ne[3];
        ggml_tensor* dst = ggml_mul_mat(gctx, wT, ggml_reshape_2d(gctx, im2, im2->ne[0], P));      // [OC, P]
        dst = ggml_reshape_4d(gctx, dst, wT->ne[1], im2->ne[1], im2->ne[2], im2->ne[3]);            // (N, OH, OW, OC)
        // permute semantics: ne[axis_i] = old ne[i] — send OC to slot 2, W/H to 0/1.
        return ggml_cont(gctx, ggml_permute(gctx, dst, 2, 0, 1, 3));                              // [W, H, OC, N]
    }

    // 1x1 / depthwise convs inside psa_attention (no act). Quantized weights are
    // stored 2D [K, out]; the 4D view only lends shape metadata for im2col.
    ggml_tensor* attention_conv(const std::string& prefix, const char* tag, ggml_tensor* x, int64_t k = 1) {
        ggml_tensor* wT = w(prefix, (std::string(tag) + "_w").c_str());
        GGML_ASSERT(wT && "attention weight missing");
        ggml_tensor* out;
        if (ggml_is_quantized(wT->type) && k == 1) {
            ggml_tensor* w4d = ggml_reshape_4d(gctx, wT, 1, 1, wT->ne[0], wT->ne[1]);
#if defined(YOLO_USE_CUDA)
            if (q8_direct) {
                if (x->type != GGML_TYPE_F16) x = ggml_cast(gctx, x, GGML_TYPE_F16);
                out = ggml_conv_2d_direct(gctx, w4d, x, 1, 1, 0, 0, 1, 1);
            } else
#endif
            out = conv2d_q(wT, w4d, x, 1, 1, 0, 0, 1, 1);
        } else {
            if (wT->ne[2] == 1 && wT->ne[3] == 1) {
                wT = ggml_reshape_4d(gctx, wT, k, k, wT->ne[0] / (k * k), wT->ne[1]);
            }
#if defined(YOLO_USE_CUDA) || defined(YOLO_USE_VULKAN)
            if (wT->type == x->type && (wT->type == GGML_TYPE_F32 || wT->type == GGML_TYPE_F16)) {
                out = k > 1
                    ? ggml_conv_2d_dw_direct(gctx, wT, x, 1, 1, (int)(k / 2), (int)(k / 2), 1, 1)
                    : ggml_conv_2d_direct(gctx, wT, x, 1, 1, 0, 0, 1, 1);
            } else {
                out = k > 1
                    ? ggml_conv_2d_dw(gctx, wT, x, 1, 1, (int)(k / 2), (int)(k / 2), 1, 1)
                    : ggml_conv_2d(gctx, wT, x, 1, 1, 0, 0, 1, 1);
            }
#else
            out = k > 1
                ? ggml_conv_2d_dw(gctx, wT, x, 1, 1, (int)(k / 2), (int)(k / 2), 1, 1)
                : ggml_conv_2d(gctx, wT, x, 1, 1, 0, 0, 1, 1);
#endif
        }
        if (ggml_tensor* b = w(prefix, (std::string(tag) + "_b").c_str())) {
            out = ggml_add(gctx, out, ggml_reshape_4d(gctx, b, 1, 1, b->ne[0], 1));
        }
        return out;
    }

    ggml_tensor* psa_attention(const OpDef& op, const std::string& prefix, ggml_tensor* x) {
        const int64_t nh = op.ip("nh"), kd = op.ip("kd"), hd = op.ip("hd");
        const float scale = op.fparams.count("scale") ? (float)op.fparams.at("scale") : 1.0f;
        const int64_t W = x->ne[0], H = x->ne[1], N = x->ne[3];
        const int64_t HW = W * H, k2d = 2 * kd + hd, C = nh * hd;

        ggml_tensor* qkv = attention_conv(prefix, "qkv", x);  // [W, H, nh*k2d, N]
        // torch: qkv.view(B, nh, k2d, N) — token dim innermost, channel outer.
        qkv = ggml_reshape_4d(gctx, qkv, HW, k2d, nh, N);     // [tokens, k2d, nh, N]
        auto view = [&](int64_t start, int64_t len) {
            return ggml_cont(gctx, ggml_view_4d(gctx, qkv, HW, len, nh, N,
                                                qkv->nb[1], qkv->nb[2], qkv->nb[3], start * qkv->nb[1]));
        };
        ggml_tensor* q = ggml_scale(gctx, view(0, kd), scale);  // [HW, kd, nh, N]
        ggml_tensor* k = view(kd, kd);
        ggml_tensor* v = view(2 * kd, hd);                      // [HW, hd, nh, N]

        // torch: attn = softmax((q*scale)^T @ k, dim=-1); x = v @ attn^T.
        // ggml dst[m,n] = sum_k A[k,m]B[k,n] with ne0 from A->ne1, so mul_mat(kT, qT)
        // puts k-tokens on ne0 — ggml_soft_max then normalizes over keys exactly like
        // torch dim=-1 (llama.cpp KQ pattern). mul_mat needs non-transposed contiguous
        // operands, hence the cont(permute)s.
        ggml_tensor* qT = ggml_cont(gctx, ggml_permute(gctx, q, 1, 0, 2, 3));       // [kd, HW, nh, N]
        ggml_tensor* kT = ggml_cont(gctx, ggml_permute(gctx, k, 1, 0, 2, 3));       // [kd, HW, nh, N]
        ggml_tensor* attn = ggml_soft_max(gctx, ggml_mul_mat(gctx, kT, qT));        // [k_tok, q_tok, nh, N]
        ggml_tensor* out = ggml_mul_mat(gctx, v, attn);                             // [hd, q_tok, nh, N]
        out = ggml_reshape_4d(gctx, ggml_cont(gctx, ggml_permute(gctx, out, 1, 0, 2, 3)), W, H, C, N);

        // pe: depthwise 3x3 on v, residual, proj 1x1
        ggml_tensor* v_img = ggml_reshape_4d(gctx, ggml_cont(gctx, v), W, H, C, N);
        ggml_tensor* pe = attention_conv(prefix, "pe", v_img, 3);
        ggml_tensor* sum = ggml_add(gctx, out, pe);
        return attention_conv(prefix, "proj", sum);
    }

    // ------------------------------------------------------------------
    // YOLO-World ops (open-vocabulary detection, op-graph v3)
    // ------------------------------------------------------------------

    // Exact AdaptiveMaxPool2d(k, k): out(i, j) = max over the window
    // [floor(i*H/k), ceil((i+1)*H/k)) x [floor(j*W/k), ceil((j+1)*W/k)).
    // Windows overlap when dim % k != 0, so each output cell is a separate
    // non-overlapping view_4d + max_pool_2d pair, then tiles are assembled
    // with concat/permute into a [C, k*k] (channel, patch) tensor.
    ggml_tensor* adaptive_max_pool2d(ggml_tensor* x, int k) {
        const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
        GGML_ASSERT(N == 1);
        auto win = [&](int i, int64_t dim) {
            // ATen adaptive pooling window edges (AdaptivePooling.h):
            //   start = (a/b)*c + ((a%b)*c)/b  == floor(a*c/b)  (multiply FIRST)
            //   end   = 1 + ((a+1)*c-1)/b      == ceil((a+1)*c/b)
            // The naive (i/k)*dim differs whenever i % k != 0, shifting every
            // pooled window and corrupting the image-aware text update (drives
            // the world confidence drift vs PyTorch).
            const int64_t s = ((int64_t)i * dim) / k;
            const int64_t e = ((int64_t)(i + 1) * dim + k - 1) / k;
            return std::make_pair((int)s, (int)e);
        };
        std::vector<ggml_tensor*> rows;
        for (int i = 0; i < k; i++) {
            auto [h0, h1] = win(i, H);
            std::vector<ggml_tensor*> cols;
            for (int j = 0; j < k; j++) {
                auto [w0, w1] = win(j, W);
                ggml_tensor* v = ggml_view_4d(gctx, x, w1 - w0, h1 - h0, C, 1, x->nb[1], x->nb[2], x->nb[3],
                                              ((size_t)h0 * W + w0) * x->nb[0]);
                // ggml_pool_2d inherits the input type, but the CPU pool kernel
                // unconditionally writes F32; an F16 dst would overflow its
                // buffer by 2x (heap corruption on split graphs). Cast up so
                // the pool output is F32-sized.
                if (v->type != GGML_TYPE_F32) v = ggml_cast(gctx, v, GGML_TYPE_F32);
                cols.push_back(ggml_pool_2d(gctx, v, GGML_OP_POOL_MAX, w1 - w0, h1 - h0, w1 - w0, h1 - h0, 0, 0));
            }
            ggml_tensor* row = cols[0];
            for (int j = 1; j < k; j++) row = ggml_concat(gctx, row, cols[j], 0);  // [k, 1, C]
            rows.push_back(ggml_cont(gctx, ggml_permute(gctx, row, 1, 0, 2, 3)));  // [1, k, C]
        }
        ggml_tensor* grid = rows[0];
        for (int i = 1; i < k; i++) grid = ggml_concat(gctx, grid, rows[i], 0);  // [k, k, C] (row i, col j, chan)
        // permute(2,1,0,3): ne[axis_i]=a->ne[i] -> [C, k, k] (chan, col j, row i).
        // Reshape only preserves the channel-contiguous layout while flattening
        // (j, i) row-major: p = j + i*k, matching PyTorch's
        // adaptive_max_pool2d(...).view(B, C, -1).
        grid = ggml_cont(gctx, ggml_permute(gctx, grid, 2, 1, 0, 3));            // [C, k, k]
        return ggml_reshape_2d(gctx, grid, C, k * k);                            // [C, k*k]
    }

    // LayerNorm(ct) + Linear(ct -> ec) over the ne0 (column) axis of `src`.
    // ggml_norm/scale are F32-only on the CPU backend, so F16 inputs are cast
    // up for the norm+linear math and cast back to the input type on exit.
    ggml_tensor* ln_linear(const std::string& prefix, const char* tag, ggml_tensor* src, float eps) {
        const enum ggml_type in_type = src->type;
        if (src->type != GGML_TYPE_F32) src = ggml_cast(gctx, src, GGML_TYPE_F32);
        ggml_tensor* y = ggml_norm(gctx, src, eps);
        if (ggml_tensor* w_ = w(prefix, (std::string(tag) + "_ln_w").c_str())) {
            if (w_->type != GGML_TYPE_F32) w_ = ggml_cast(gctx, w_, GGML_TYPE_F32);
            y = ggml_mul(gctx, y, w_);
        }
        if (ggml_tensor* b_ = w(prefix, (std::string(tag) + "_ln_b").c_str())) {
            if (b_->type != GGML_TYPE_F32) b_ = ggml_cast(gctx, b_, GGML_TYPE_F32);
            y = ggml_add(gctx, y, b_);
        }
        if (ggml_tensor* w_ = w(prefix, (std::string(tag) + "_w").c_str())) {
            if (w_->type != GGML_TYPE_F32) w_ = ggml_cast(gctx, w_, GGML_TYPE_F32);
            y = ggml_mul_mat(gctx, w_, y);
        }
        if (ggml_tensor* b_ = w(prefix, (std::string(tag) + "_b").c_str())) {
            y = ggml_add(gctx, y, ggml_reshape_2d(gctx, b_, b_->ne[0], 1));
        }
        if (in_type != GGML_TYPE_F32) y = ggml_cast(gctx, y, in_type);
        return y;
    }

    // Element-wise max without a native ggml op: max(a, b) == b + relu(a - b).
    ggml_tensor* max2(ggml_tensor* a, ggml_tensor* b) {
        return ggml_add(gctx, b, ggml_relu(gctx, ggml_sub(gctx, a, b)));
    }

    // MaxSigmoidAttnBlock gate: out = proj_conv(x) * sigmoid(max_n(embed . guide_n)/sqrt(hc) + bias).
    // embed [w,h,ec] and proj [w,h,c2] come from plain conv ops; text is [512, nc]
    // (nc rows of 512-d CLIP text embeddings). The max over nc classes is a
    // static tree of max2 nodes because nc is fixed at session creation.
    ggml_tensor* max_sigmoid_attn(const OpDef& op, const std::string& prefix, ggml_tensor* embed,
                                  ggml_tensor* proj, ggml_tensor* text) {
        const int64_t nh = op.ip("nh"), hc = op.ip("hc");
        const int64_t W = embed->ne[0], H = embed->ne[1];
        const int64_t HW = W * H, c2 = proj->ne[2], nc = text->ne[1];

        ggml_tensor* guide = ggml_mul_mat(gctx, w(prefix, "gl_w"), text);  // [ec, nc]
        // mul_mat returns F32. Keep F32 reference graphs in F32, while F16
        // deployment graphs use the native F16 CUDA GEMM contract.
        if (text->type == GGML_TYPE_F16) guide = ggml_cast(gctx, guide, GGML_TYPE_F16);
        if (ggml_tensor* b = w(prefix, "gl_b")) guide = ggml_add(gctx, guide, ggml_reshape_2d(gctx, b, b->ne[0], 1));

        // embed -> [ec, HW] with the channel axis on ne0 (mul_mat weight side).
        // permute(1,2,0,3): ne[axis_i]=a->ne[i] -> [ec, W, H] from [W, H, ec].
        ggml_tensor* eT = ggml_cont(gctx, ggml_permute(gctx, embed, 1, 2, 0, 3));  // [ec, W, H]
        ggml_tensor* e2 = ggml_view_2d(gctx, eT, embed->ne[2], HW, eT->nb[1], 0);   // [ec, HW]
        // proj -> [c2, HW]
        ggml_tensor* pT = ggml_cont(gctx, ggml_permute(gctx, proj, 1, 2, 0, 3));
        ggml_tensor* p2 = ggml_view_2d(gctx, pT, c2, HW, pT->nb[1], 0);

        // Head bias is a per-head scalar constant baked into the graph from the
        // host copy (model.tensors), never a runtime tensor.
        auto bias_it = model.tensors.find(prefix + ".bias");
        const float* bias_data =
            bias_it != model.tensors.end() ? (const float*)bias_it->second.data.data() : nullptr;
        std::vector<ggml_tensor*> head_outs;
        for (int64_t m = 0; m < nh; m++) {
            ggml_tensor* g_m = ggml_view_2d(gctx, guide, hc, nc, guide->nb[1], m * hc * ggml_element_size(guide));
            ggml_tensor* e_m = ggml_view_2d(gctx, e2, hc, HW, e2->nb[1], m * hc * ggml_element_size(e2));
            // CUDA requires matching F32/F16 operands for mul_mat. The fused
            // F32 guide can meet an F16 activation even in an F32 model, so
            // choose the activation type from the actual guide tensor rather
            // than the model's nominal dtype. Keeping F32 here preserves the
            // reference path; F16 models retain their native F16 GEMM.
            if (e_m->type != guide->type) e_m = ggml_cast(gctx, e_m, guide->type);
            ggml_tensor* aw = ggml_mul_mat(gctx, g_m, e_m);  // [nc, HW]
            // tree-max over the nc rows (torch aw.max(dim=-1))
            std::vector<ggml_tensor*> rows;
            // aw is [nc, HW], so a fixed class is a strided column over ne1,
            // not a contiguous row. Keep that stride in a [1, HW] view before
            // reducing classes; view_1d would read unrelated spatial scores.
            for (int64_t n = 0; n < nc; n++) {
                rows.push_back(ggml_view_2d(gctx, aw, 1, HW, aw->nb[1], n * ggml_element_size(aw)));
            }
            while (rows.size() > 1) {
                std::vector<ggml_tensor*> nxt;
                for (size_t j = 0; j + 1 < rows.size(); j += 2) nxt.push_back(max2(rows[j], rows[j + 1]));
                if (rows.size() % 2) nxt.push_back(rows.back());
                rows.swap(nxt);
            }
            ggml_tensor* p_m = ggml_view_2d(gctx, p2, hc, HW, p2->nb[1], m * hc * ggml_element_size(p2));
            // torch: aw / sqrt(hc) + bias[m] -> sigmoid (folded into one scale_bias)
            // ggml scale_bias is F32-only on CPU: cast up and back for F16 graphs.
            ggml_tensor* aw1 = ggml_scale_bias(gctx, ggml_cast(gctx, rows[0], GGML_TYPE_F32),
                                               1.0f / std::sqrt((float)hc),
                                               bias_data ? bias_data[m] : 0.0f);
            aw1 = ggml_sigmoid(gctx, aw1);
            if (p_m->type != GGML_TYPE_F32) aw1 = ggml_cast(gctx, aw1, p_m->type);
            aw1 = ggml_reshape_2d(gctx, aw1, 1, HW);  // [1, HW] broadcast along channels
            head_outs.push_back(ggml_mul(gctx, p_m, aw1));  // [hc, HW] x [1, HW] broadcast
        }
        ggml_tensor* out = head_outs[0];
        // concat along channels (axis 0): [hc, HW] x nh -> [c2, HW]
        for (size_t m = 1; m < head_outs.size(); m++) out = ggml_concat(gctx, out, head_outs[m], 0);
        return ggml_reshape_3d(gctx, ggml_cont(gctx, ggml_permute(gctx, out, 1, 0, 2, 3)), W, H, c2);
    }

    // ImagePoolingAttn: image tokens attend into the text embedding (residual).
    // Each input is a [w,h,ec] 1x1-projected feature map; text is [512, nc].
    ggml_tensor* image_pooling_attn(const OpDef& op, const std::string& prefix,
                                    const std::vector<ggml_tensor*>& feats, ggml_tensor* text) {
        const int64_t nh = op.ip("nh"), hc = op.ip("hc"), k = op.ip("k", 3);
        const int64_t nc = text->ne[1];
        // ggml_pool_2d outputs F32, so the whole attention math runs in F32;
        // the result is cast back to the text type for the residual.
        ggml_tensor* text32 = text->type == GGML_TYPE_F32 ? text : ggml_cast(gctx, text, GGML_TYPE_F32);
        // 1. adaptive max pool each level -> [ec, k*k] patches, concat over patches.
        ggml_tensor* xcat = adaptive_max_pool2d(feats[0], (int)k);  // [ec, k*k]
        for (size_t f = 1; f < feats.size(); f++) {
            xcat = ggml_concat(gctx, xcat, adaptive_max_pool2d(feats[f], (int)k), 1);
        }
        // 2. q = query(text), k/v = key/value(x); LayerNorm over channels.
        ggml_tensor* xT = xcat;  // [ec, P]
        ggml_tensor* kT = ln_linear(prefix, "key", xT, 1e-5f);    // [ec, P]
        ggml_tensor* vT = ln_linear(prefix, "value", xT, 1e-5f);  // [ec, P]
        ggml_tensor* q = ln_linear(prefix, "query", text32, 1e-5f); // [ec, nc]
        const int64_t P = xT->ne[1];
        // 3. per-head scaled dot-product attention (llama.cpp KQ pattern).
        std::vector<ggml_tensor*> head_outs;
        for (int64_t m = 0; m < nh; m++) {
            ggml_tensor* q_m = ggml_view_2d(gctx, q, hc, nc, q->nb[1], m * hc * ggml_element_size(q));      // [hc, nc]
            ggml_tensor* k_m = ggml_view_2d(gctx, kT, hc, P, kT->nb[1], m * hc * ggml_element_size(kT));     // [hc, P]
            ggml_tensor* v_m = ggml_view_2d(gctx, vT, hc, P, vT->nb[1], m * hc * ggml_element_size(vT));     // [hc, P]
            ggml_tensor* aw = ggml_mul_mat(gctx, k_m, q_m);                     // [P, nc] keys on ne0
            aw = ggml_scale(gctx, aw, 1.0f / std::sqrt((float)hc));
            aw = ggml_soft_max(gctx, aw);                                       // over keys (dim=-1 in torch)
            ggml_tensor* vT_m = ggml_cont(gctx, ggml_permute(gctx, v_m, 1, 0, 2, 3));  // [P, hc]
            head_outs.push_back(ggml_mul_mat(gctx, vT_m, aw));                  // [hc, nc]
        }
        ggml_tensor* out = head_outs[0];
        // concat along channels (axis 0): [hc, nc] x nh -> [ec, nc]
        for (size_t m = 1; m < head_outs.size(); m++) out = ggml_concat(gctx, out, head_outs[m], 0);
        ggml_tensor* pw = w(prefix, "proj_w");
        if (pw->type != GGML_TYPE_F32) pw = ggml_cast(gctx, pw, GGML_TYPE_F32);
        out = ggml_mul_mat(gctx, pw, out);  // [ct, nc]
        if (ggml_tensor* b = w(prefix, "proj_b")) out = ggml_add(gctx, out, ggml_reshape_2d(gctx, b, b->ne[0], 1));
        out = ggml_add(gctx, out, text32);  // residual (scale is 1.0 in YOLO-World)
        if (text->type != GGML_TYPE_F32) out = ggml_cast(gctx, out, text->type);
        return out;                        // [512, nc]
    }

    // WorldDetect and YOLOE: contrastive embedding branch + plain detect decode.
    // feats alternate [box0, emb0, box1, emb1, ...]; text is [512, nc].
    ggml_tensor* world_detect(const OpDef& op, const std::string& prefix,
                              const std::vector<ggml_tensor*>& feats, ggml_tensor* text, int64_t nc) {
        const int64_t rm = op.ip("reg_max", 16);
        // L2-normalisation needs F32 (ggml_sum_rows is F32-only); cast the
        // F16 graph text/embedding back for the contrastive head math.
        ggml_tensor* text32 = text->type == GGML_TYPE_F32 ? text : ggml_cast(gctx, text, GGML_TYPE_F32);
        // text -> L2-normalized [512, nc]
        ggml_tensor* sq = ggml_sqr(gctx, text32);
        ggml_tensor* t_norm = ggml_div(gctx, text32, ggml_sqrt(gctx, ggml_sum_rows(gctx, sq)));
        ggml_tensor* out = nullptr;
        const bool has_masks = op.ip("has_masks", 0) != 0;
        const bool bn_contrastive = op.ip("bn_contrastive", 0) != 0;
        const size_t stride = has_masks ? 3 : 2;
        const size_t n_levels = feats.size() / stride;
        for (size_t l = 0; l < n_levels; l++) {
            ggml_tensor* box = feats[stride * l];      // [w, h, 4*rm]
            ggml_tensor* emb = feats[stride * l + 1];  // [w, h, embed]
            const int64_t W = box->ne[0], H = box->ne[1], HW = W * H;
            // World normalizes image embeddings. YOLOE's BNContrastiveHead
            // instead applies its folded BatchNorm affine transform.
            ggml_tensor* eT = ggml_cont(gctx, ggml_permute(gctx, emb, 1, 2, 0, 3));  // [embed, W, H]
            ggml_tensor* eT32 = eT->type == GGML_TYPE_F32 ? eT : ggml_cast(gctx, eT, GGML_TYPE_F32);
            ggml_tensor* e2 = ggml_view_2d(gctx, eT32, emb->ne[2], HW, eT32->nb[1], 0);  // [embed, HW]
            if (bn_contrastive) {
                ggml_tensor* scale = w(prefix, ("cv4_" + std::to_string(l) + "_bn_scale").c_str());
                ggml_tensor* shift = w(prefix, ("cv4_" + std::to_string(l) + "_bn_shift").c_str());
                GGML_ASSERT(scale && shift && "YOLOE BN contrastive head missing affine tensors");
                if (scale->type != GGML_TYPE_F32) scale = ggml_cast(gctx, scale, GGML_TYPE_F32);
                if (shift->type != GGML_TYPE_F32) shift = ggml_cast(gctx, shift, GGML_TYPE_F32);
                e2 = ggml_mul(gctx, e2, ggml_reshape_2d(gctx, scale, scale->ne[0], 1));
                e2 = ggml_add(gctx, e2, ggml_reshape_2d(gctx, shift, shift->ne[0], 1));
            } else {
                e2 = ggml_div(gctx, e2, ggml_sqrt(gctx, ggml_sum_rows(gctx, ggml_sqr(gctx, e2))));
            }
            ggml_tensor* scores = ggml_mul_mat(gctx, t_norm, e2);                    // [nc, HW]
            // scores = scores * logit_scale.exp() + bias (per-level ContrastiveHead)
            auto ls_it = model.tensors.find(prefix + ".cv4_" + std::to_string(l) + "_logit_scale");
            auto bs_it = model.tensors.find(prefix + ".cv4_" + std::to_string(l) + "_bias");
            const float ls = ls_it != model.tensors.end() ? ((const float*)ls_it->second.data.data())[0] : 1.0f;
            const float bs = bs_it != model.tensors.end() ? ((const float*)bs_it->second.data.data())[0] : 0.0f;
            scores = ggml_scale_bias(gctx, scores, ls, bs);
            ggml_tensor* s4 = ggml_reshape_3d(gctx, ggml_cont(gctx, ggml_permute(gctx, scores, 1, 0, 2, 3)), W, H, nc);
            if (s4->type != box->type) s4 = ggml_cast(gctx, s4, box->type);  // concat needs matching types
            ggml_tensor* level = ggml_concat(gctx, box, s4, 2);  // [w, h, 4*rm + nc]
            if (has_masks) {
                ggml_tensor* mask = feats[stride * l + 2];
                level = ggml_concat(gctx, level, mask, 2);
            }
            const int64_t level_no = 4 * rm + nc + (has_masks ? op.ip("nm", 0) : 0);
            ggml_tensor* r = ggml_reshape_2d(gctx, level, HW, level_no);
            out = out ? ggml_concat(gctx, out, r, 0) : r;
        }
        return out;  // [A, 4*rm + nc]
    }
};

}  // namespace

Session* create_session(const std::string& gguf_path, const SessionOptions& opts) {
    auto model = load_gguf(gguf_path);
    if (!model) return nullptr;

    Session* s = new Session();
    s->model = std::move(*model);
    s->input_w = opts.input_w > 0 ? opts.input_w : s->model.meta.imgsz;
    s->input_h = opts.input_h > 0 ? opts.input_h : s->model.meta.imgsz;
    s->profile_gaps = opts.profile_gaps;
    s->backend = init_backend_ctx(opts.threads);
    if (!s->backend.cpu) {
        free_session(s);
        return nullptr;
    }
    if (opts.profile_ops) {
        backend_enable_op_profile(s->backend);
    }

    ModelMeta& meta = s->model.meta;
    // YOLO-World: the class count is a runtime knob (set_classes). It fixes
    // the text-input shape and every nc-dependent tensor in the graph, so the
    // session must be recreated when the class list changes.
    const int64_t world_nc = opts.world_nc > 0 ? opts.world_nc : meta.nc;
    s->world_nc = (int)world_nc;
    if (s->model.has_text_input) meta.nc = (int)world_nc;
    // Per-anchor output channels: detect=4*rm+nc, segment=+nm, pose=+nk, obb=+ne.
    const int no = 4 * meta.reg_max + meta.nc + meta.nm + meta.nk + meta.ne;

    // Weight context: tensor structs only; data goes to the backend buffer.
    s->wctx = ggml_init({(size_t)(s->model.tensors.size() * ggml_tensor_overhead() + 1024 * 1024),
                         nullptr, /*no_alloc*/ true});

    // Graph context: intermediate tensor structs (data lives in galloc/sched).
    size_t g_size = (size_t)(s->model.ops.size() * 12 + 512) * ggml_tensor_overhead() + (32u << 20);
    s->gctx = ggml_init({g_size, nullptr, /*no_alloc*/ true});
    if (!s->wctx || !s->gctx) {
        YOLO_LOG_ERROR("ggml context allocation failed");
        free_session(s);
        return nullptr;
    }

    GraphBuilder gb{s->gctx, s->wctx, s->model, {}};
    // Route quantized convs through the direct flow only when every quantized
    // tensor conforms; a single violation falls the whole model back to the
    // F32 conv2d_q path, whose im2col requires F32 activations. K 32-alignment
    // is the hard constraint (CUDA igemm block addressing; vulkan / cpu host
    // dequant block walk). CUDA igemm also needs OC % 8 == 0 but writes sub-8
    // tails scalars; vulkan / cpu run the plain f16 path so OC is free.
    for (const auto& [name, ht] : s->model.tensors) {
        if (!ggml_is_quantized(ht.type)) continue;
        if (ht.type != GGML_TYPE_Q8_0 || ht.ne[0] % 32 != 0) {
            gb.q8_direct = false;
            break;
        }
        gb.q8_direct = true;
    }
    // Backends without a native Q8 conv path (vulkan, cpu) expand Q8_0 weights
    // to f16 on the host once so the whole graph runs the f16 direct-conv path
    // instead of the dequant-f32 mul_mat route (2-5x faster measured on vulkan).
    if (gb.q8_direct) {
#if !defined(YOLO_USE_CUDA)
        for (auto& [name, ht] : s->model.tensors) {
            if (ht.type != GGML_TYPE_Q8_0) continue;
            const int64_t n = ht.ne[0] * ht.ne[1] * ht.ne[2] * ht.ne[3];
            std::vector<uint8_t> f16(n * sizeof(ggml_fp16_t));
            const block_q8_0 * src = reinterpret_cast<const block_q8_0 *>(ht.data.data());
            ggml_fp16_t * dst = reinterpret_cast<ggml_fp16_t *>(f16.data());
            for (int64_t i = 0; i < n; ++i) {
                const block_q8_0 * blk = src + i / QK8_0;
                dst[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(blk->d) * (float) blk->qs[i % QK8_0]);
            }
            ht.data = std::move(f16);
            ht.type = GGML_TYPE_F16;
        }
#endif
    }
    std::vector<ggml_tensor*> values(s->model.ops.size(), nullptr);

    // Debug aid: YOLO_STOP_OP=N truncates the graph at op N so a unified
    // post-compute dump (--dump-ops) sees N as the last computed op — its
    // inputs are then guaranteed not to have been overwritten by buffer
    // reuse (gallocr aliases non-overlapping lifetimes).
    const char* stop_env = getenv("YOLO_STOP_OP");
    const int stop_op = stop_env ? atoi(stop_env) : -1;

    // The input tensor is always F32; F16-model graphs insert an on-device
    // cast node (below). A host-side scalar f32->f16 round-trip measured
    // ~2.1ms/frame at 640x480 -- pure host overhead on every GPU frame.
    s->input = ggml_new_tensor_4d(s->gctx, GGML_TYPE_F32, s->input_w, s->input_h, 3, 1);
    ggml_set_input(s->input);  // allocated before compute nodes
    ggml_set_name(s->input, "image");

    ggml_tensor* graph_text = nullptr;
    if (s->model.has_text_input) {
        // External [512, nc] F32 text embedding (CLIP text encoder output).
        s->text_input = ggml_new_tensor_2d(s->gctx, GGML_TYPE_F32, 512, world_nc);
        ggml_set_input(s->text_input);
        ggml_set_name(s->text_input, "text");
        graph_text = s->text_input;
    }

    ggml_tensor* graph_input = s->input;
#if defined(YOLO_USE_CUDA)
    // F32 is the numerical reference format. Do not silently lower it to F16
    // on CUDA: its output is used for PyTorch parity. F16/Q8 deployment models
    // retain the tensor-core input flow.
    if (meta.dtype != "f32") {
        graph_input = ggml_cast(s->gctx, s->input, GGML_TYPE_F16);
        if (graph_text) graph_text = ggml_cast(s->gctx, s->text_input, GGML_TYPE_F16);
    }
#else
    // CPU and Vulkan: cast input to F16 when model weights are F16 (native or
    // dequantised from Q8_0) so the im2col+mul_mat pipeline stays all-F16.
    if (meta.dtype == "f16" || gb.q8_direct) {
        graph_input = ggml_cast(s->gctx, s->input, GGML_TYPE_F16);
        if (graph_text) graph_text = ggml_cast(s->gctx, s->text_input, GGML_TYPE_F16);
    }
#endif

    for (size_t i = 0; i < s->model.ops.size(); i++) {
        const OpDef& op = s->model.ops[i];
        const std::string prefix = "op." + std::to_string(i);
        auto in0 = [&]() {
            const int idx = op.inputs.empty() ? -1 : op.inputs[0];
            return idx < 0 ? graph_input : values[idx];
        };
        auto in_text = [&]() {
            const int64_t ti = op.ip("text_in", -1);
            return ti < 0 ? graph_text : values[ti];
        };
        ggml_tensor* out = nullptr;

        if (op.type == "max_sigmoid_attn") {
            out = gb.max_sigmoid_attn(op, prefix, values[op.inputs[0]], values[op.inputs[1]], in_text());
        } else if (op.type == "image_pooling_attn") {
            std::vector<ggml_tensor*> feats;
            for (int j : op.inputs) feats.push_back(values[j]);
            out = gb.image_pooling_attn(op, prefix, feats, in_text());
        } else if (op.type == "world_detect" || op.type == "world_segment") {
            std::vector<ggml_tensor*> feats;
            const bool has_masks = op.type == "world_segment";
            const size_t n = has_masks ? op.inputs.size() - 1 : op.inputs.size();
            for (size_t j = 0; j < n; j++) feats.push_back(values[op.inputs[j]]);
            if (has_masks) s->output_proto = values[op.inputs.back()];
            out = gb.world_detect(op, prefix, feats, in_text(), meta.nc);
        } else if (op.type == "conv" || op.type == "dwconv") {
            out = gb.conv2d(op, prefix, in0());
        } else if (op.type == "maxpool") {
            const int k = (int)op.ip("k"), st = (int)op.ip("s"), p = (int)op.ip("p");
            out = ggml_pool_2d(s->gctx, in0(), GGML_OP_POOL_MAX, k, k, st, st, (float)p, (float)p);
        } else if (op.type == "concat") {
            out = values[op.inputs[0]];
            for (size_t j = 1; j < op.inputs.size(); j++) {
                out = ggml_concat(s->gctx, out, values[op.inputs[j]], 2);
            }
        } else if (op.type == "upsample") {
            out = ggml_upscale(s->gctx, in0(), (int)op.ip("sf"), GGML_SCALE_MODE_NEAREST);
        } else if (op.type == "interpolate") {
            ggml_tensor* x = in0();
            const int64_t sf = op.ip("sf", 1);
            const uint32_t mode = GGML_SCALE_MODE_BILINEAR |
                                  (op.ip("align_corners") ? GGML_SCALE_FLAG_ALIGN_CORNERS : 0);
#if defined(YOLO_USE_CUDA)
            const bool f16 = x->type == GGML_TYPE_F16;
            if (f16) x = ggml_cast(s->gctx, x, GGML_TYPE_F32);
            out = ggml_interpolate(s->gctx, x, x->ne[0] * sf, x->ne[1] * sf, x->ne[2], x->ne[3], mode);
            if (f16) out = ggml_cast(s->gctx, out, GGML_TYPE_F16);
#else
            out = ggml_interpolate(s->gctx, x, x->ne[0] * sf, x->ne[1] * sf, x->ne[2], x->ne[3], mode);
#endif
        } else if (op.type == "conv_transpose") {
            out = gb.conv_transpose(op, prefix, in0());
        } else if (op.type == "add") {
            out = ggml_add(s->gctx, values[op.inputs[0]], values[op.inputs[1]]);
        } else if (op.type == "slice") {
            // The channel slice is a contiguous sub-block view: the nb[0..2]
            // chain matches a dense tensor and ne[3]==1 skips the nb[3] check,
            // so ggml_is_contiguous(view) holds. Every consumer (concat, direct
            // conv) addresses it exactly like a dense tensor — the cont copy
            // would be a redundant kernel per C2f block.
            ggml_tensor* x = in0();
            const int64_t start = op.ip("start"), end = op.ip("end");
            out = ggml_view_4d(s->gctx, x, x->ne[0], x->ne[1], end - start, x->ne[3],
                               x->nb[1], x->nb[2], x->nb[3], start * x->nb[2]);
        } else if (op.type == "psa_attention") {
            out = gb.psa_attention(op, prefix, in0());
        } else if (op.type == "detect" || op.type == "segment" || op.type == "pose" || op.type == "obb") {
            const size_t n_feats = op.inputs.size() - (op.type == "segment" ? 1 : 0);
            for (size_t j = 0; j < n_feats; j++) {
                ggml_tensor* t = values[op.inputs[j]];
                const int64_t HW = t->ne[0] * t->ne[1];
                ggml_tensor* r = ggml_reshape_2d(s->gctx, t, HW, no);
                out = out ? ggml_concat(s->gctx, out, r, 0) : r;
            }
            if (op.type == "segment") s->output_proto = values[op.inputs.back()];
        } else if (op.type == "semantic") {
            // Identity marker: the head convs already emitted [W/8, H/8, nc, 1]
            // logits; the task just declares the readback layout for argmax.
            out = in0();
        } else if (op.type == "avgpool") {
            // Classify: AdaptiveAvgPool2d(1) — a global average pool whose kernel
            // equals the input extent (imgsz/32), so k0/k1 are runtime values.
            ggml_tensor* x = in0();
            out = ggml_pool_2d(s->gctx, x, GGML_OP_POOL_AVG, x->ne[0], x->ne[1], 1, 1, 0, 0);
        } else if (op.type == "linear") {
            // Classify: y = x @ W^T + b. W is stored [in, out]; the pooled
            // [1,1,C,1] feature is flattened to [C,1] for mul_mat.
            ggml_tensor* x = in0();
            const int64_t c = x->ne[0] * x->ne[1] * x->ne[2];
            ggml_tensor* flat = ggml_reshape_2d(s->gctx, x, c, 1);
            ggml_tensor* wT = gb.w(prefix, "w");
            GGML_ASSERT(wT && "linear without weight");
            out = ggml_mul_mat(s->gctx, wT, flat);  // [out, 1]
            if (ggml_tensor* b = gb.w(prefix, "b")) {
                out = ggml_add(s->gctx, out, ggml_reshape_2d(s->gctx, b, b->ne[0], 1));
            }
            out = ggml_reshape_1d(s->gctx, out, out->ne[0]);
        } else if (op.type == "classify") {
            // Identity marker on the [nc] logits; softmax/topk run in postprocess.
            out = in0();
        } else if (op.type == "depth") {
            const float cal_a = (float)(op.fparams.count("cal_a") ? op.fparams.at("cal_a") : 1.0);
            const float cal_b = (float)(op.fparams.count("cal_b") ? op.fparams.at("cal_b") : 0.0);
            out = ggml_exp(s->gctx, ggml_scale_bias(s->gctx, ggml_clamp(s->gctx, in0(), -4.0f, 5.0f), cal_a,
                                                    cal_b));
        } else {
            YOLO_LOG_ERROR("unknown op type '%s' at index %zu", op.type.c_str(), i);
            free_session(s);
            return nullptr;
        }
        values[i] = out;
        s->op_values.push_back(out);
        if (stop_op >= 0 && (int)i >= stop_op) break;
    }

    s->output = (stop_op >= 0 && (size_t)stop_op < values.size()) ? values[stop_op] : values.back();
    GGML_ASSERT(s->output && "graph produced no output");
    // GPU backends: cast F16 head outputs to F32 on-device. The host-side
    // scalar f16->f32 round-trip measured ~1.6ms/frame (output) plus ~1.3ms
    // (proto) -- pure host overhead before postprocess.
    if (s->output->type == GGML_TYPE_F16 && s->backend.gpu) {
        s->output = ggml_cast(s->gctx, s->output, GGML_TYPE_F32);
    }
    if (s->output_proto && s->output_proto->type == GGML_TYPE_F16 && s->backend.gpu) {
        s->output_proto = ggml_cast(s->gctx, s->output_proto, GGML_TYPE_F32);
    }
    if (s->output->type == GGML_TYPE_F16) s->output_f16.resize(ggml_nelements(s->output));
    if (s->output_proto && s->output_proto->type == GGML_TYPE_F16) {
        s->output_proto_f16.resize(ggml_nelements(s->output_proto));
    }

    // Text-conditioned max trees add O(world_nc) graph nodes. The old fixed
    // detect-head estimate overflowed for the default 80-class World session.
    const size_t graph_size = s->model.ops.size() * 12 + 512 +
                              (s->model.has_text_input ? (size_t)world_nc * 64 : 0);
    s->graph = ggml_new_graph_custom(s->gctx, graph_size, /*grads*/ false);
    if (opts.keep_all_ops) {
        // OUTPUT tensors are never freed by gallocr (ggml-alloc.c free_node),
        // so every op's data survives the full compute for --dump-ops.
        for (ggml_tensor* t : s->op_values) {
            ggml_set_output(t);
            ggml_build_forward_expand(s->graph, t);
        }
        // GPU F16 heads append a cast after op_values was collected. Keep the
        // public output live as well so session_read_output remains valid.
        ggml_set_output(s->output);
        ggml_build_forward_expand(s->graph, s->output);
    } else {
        ggml_set_output(s->output);
        ggml_build_forward_expand(s->graph, s->output);
        if (s->output_proto) {  // proto feeds no op; keep it a live graph leaf
            ggml_set_output(s->output_proto);
            ggml_build_forward_expand(s->graph, s->output_proto);
        }
    }

    // Allocate and upload weights to the primary backend.
    ggml_backend_buffer_type_t buft = backend_weight_buft(s->backend);
    s->wbuf = ggml_backend_alloc_ctx_tensors_from_buft(s->wctx, buft);
    if (!s->wbuf) {
        YOLO_LOG_ERROR("weight allocation failed");
        free_session(s);
        return nullptr;
    }
    for (ggml_tensor* t : gb.weights) {
        const HostTensor& ht = s->model.tensors.at(t->name);
        ggml_backend_tensor_set(t, ht.data.data(), 0, ht.data.size());
    }

    if (s->backend.sched && s->backend.gpu) {
        ggml_backend_sched_set_tensor_backend(s->backend.sched, s->input, s->backend.gpu);
        if (s->text_input) ggml_backend_sched_set_tensor_backend(s->backend.sched, s->text_input, s->backend.gpu);
        ggml_backend_sched_set_tensor_backend(s->backend.sched, s->output, s->backend.gpu);
        if (s->output_proto) {
            ggml_backend_sched_set_tensor_backend(s->backend.sched, s->output_proto, s->backend.gpu);
        }
    }
    if (!backend_graph_alloc(s->backend, s->graph)) {
        free_session(s);
        return nullptr;
    }

    if (meta.task == "detect" || meta.task == "segment" || meta.task == "pose" || meta.task == "obb") {
        // Box-anchored tasks share the anchor grid; depth/semantic/classify decode
        // dense or vector outputs and never touch it.
        // Postprocess constants (mirrors ultralytics make_anchors with 0.5 offset).
        for (int l = 0; l < meta.nl; l++) {
            const int stride = (int)meta.strides[l];
            const int fw = s->input_w / stride, fh = s->input_h / stride;
            for (int y = 0; y < fh; y++)
                for (int x = 0; x < fw; x++) {
                    s->anchors.push_back(x + 0.5f);
                    s->anchors.push_back(y + 0.5f);
                    s->anchor_strides.push_back((float)stride);
                }
        }
        s->anchor_total = (int)s->anchor_strides.size();
        for (int i = 0; i < meta.reg_max; i++) s->dfl_proj.push_back((float)i);
    }

    YOLO_LOG_INFO("session ready: backend=%s, %d ops, input=%dx%d, anchors=%d",
                  backend_name(s->backend), (int)s->model.ops.size(), s->input_w, s->input_h, s->anchor_total);
    return s;
}

bool session_run(Session* s, const float* chw_image) {
    // profile_gaps splits session_run on stderr into the input-upload portion
    // and the graph-compute portion (record + submit + any internal
    // synchronization) so the blocking point of an async backend is visible.
    const auto t0 = std::chrono::steady_clock::now();
    const size_t input_bytes = ggml_nbytes(s->input);
    if (s->backend.gpu) {
        ggml_backend_tensor_set_async(s->backend.gpu, s->input, chw_image, 0, input_bytes);
    } else {
        ggml_backend_tensor_set(s->input, chw_image, 0, input_bytes);
    }

    // Graph allocation reuses the text leaf's backing storage between runs.
    // Keep the host embedding for the session and upload it before every graph
    // execution, just as the image input is uploaded on every frame.
    if (!s->text_pending.empty()) {
        const size_t nbytes = s->text_pending.size() * sizeof(float);
        if (s->backend.gpu) {
            ggml_backend_tensor_set_async(s->backend.gpu, s->text_input, s->text_pending.data(), 0, nbytes);
        } else {
            ggml_backend_tensor_set(s->text_input, s->text_pending.data(), 0, nbytes);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int st = backend_graph_compute(s->backend, s->graph);
    if (s->profile_gaps) {
        s->gap_comp_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        s->gap_up_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (++s->gap_frames % 30 == 0) {
            fprintf(stderr, "[gap-prof2] upload=%.3fms compute=%.3fms (frames=%d)\n", s->gap_up_ms / s->gap_frames,
                    s->gap_comp_ms / s->gap_frames, s->gap_frames);
        }
    }
    if (st != GGML_STATUS_SUCCESS) {
        YOLO_LOG_ERROR("graph compute failed: %d", st);
        return false;
    }
    return true;
}

bool session_set_text(Session* s, const float* text_embed) {
    if (!s->text_input) {
        YOLO_LOG_ERROR("session_set_text requires a YOLO-World model");
        return false;
    }
    // Host input is row-major [nc, 512]; ggml stores [512, nc] column-major,
    // i.e. the exact same memory layout, so a plain byte copy suffices.
    // Queue the update and upload it at the start of the next graph run. This
    // keeps CPU and GPU backends on the same input-buffer lifecycle and avoids
    // writing through a tensor before its allocator has attached a buffer.
    s->text_pending.assign(text_embed, text_embed + ggml_nelements(s->text_input));
    return true;
}

bool session_read_output(Session* s, std::vector<float>& out, int& no, int& na) {
    if (s->model.meta.task != "detect" && s->model.meta.task != "segment" && s->model.meta.task != "pose" &&
        s->model.meta.task != "obb") {
        YOLO_LOG_ERROR("session_read_output requires a detect, segment, pose or obb model, got %s",
                       s->model.meta.task.c_str());
        return false;
    }
    // output layout: ne[0] = anchors, ne[1] = channels; element (a, c) at a + c*na.
    na = (int)s->output->ne[0];
    no = (int)s->output->ne[1];
    out.resize((size_t)na * no);
    const auto tr0 = std::chrono::steady_clock::now();
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0, s->output_f16.size() * sizeof(ggml_fp16_t));
        const auto trc = std::chrono::steady_clock::now();
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
        s->gap_cast_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trc).count();
        s->gap_get_ms += std::chrono::duration<double, std::milli>(trc - tr0).count();
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0, out.size() * sizeof(float));
        s->gap_get_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    }
    if (s->profile_gaps && ++s->gap_rframes % 30 == 0) {
        fprintf(stderr, "[gap-prof3] tensor_get=%.3fms cast_out=%.3fms (frames=%d)\n", s->gap_get_ms / s->gap_rframes,
                s->gap_cast_ms / s->gap_rframes, s->gap_rframes);
    }
    return true;
}

bool session_read_proto(Session* s, std::vector<float>& out, int& nm, int& w, int& h) {
    if (!s->output_proto) {
        YOLO_LOG_ERROR("session_read_proto requires a segment model");
        return false;
    }
    // proto layout: ne[0]=W, ne[1]=H, ne[2]=nm on the canvas/4 grid.
    w = (int)s->output_proto->ne[0];
    h = (int)s->output_proto->ne[1];
    nm = (int)s->output_proto->ne[2];
    out.resize((size_t)w * h * nm);
    if (s->output_proto->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output_proto, s->output_proto_f16.data(), 0,
                                s->output_proto_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_proto_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output_proto, out.data(), 0, out.size() * sizeof(float));
    }
    return true;
}

bool session_read_depth(Session* s, std::vector<float>& out, int& width, int& height) {
    if (s->model.meta.task != "depth" || s->output->ne[2] != 1 || s->output->ne[3] != 1) {
        YOLO_LOG_ERROR("session_read_depth requires a single-channel depth model");
        return false;
    }
    width = (int)s->output->ne[0];
    height = (int)s->output->ne[1];
    out.resize((size_t)width * height);
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0, s->output_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0, out.size() * sizeof(float));
    }
    return true;
}

bool session_read_semantic(Session* s, std::vector<float>& out, int& nc, int& w, int& h) {
    if (s->model.meta.task != "semantic") {
        YOLO_LOG_ERROR("session_read_semantic requires a semantic model, got %s", s->model.meta.task.c_str());
        return false;
    }
    // logits layout: ne[0]=W, ne[1]=H, ne[2]=nc on the canvas/8 grid.
    w = (int)s->output->ne[0];
    h = (int)s->output->ne[1];
    nc = (int)s->output->ne[2];
    out.resize((size_t)w * h * nc);
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0, s->output_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0, out.size() * sizeof(float));
    }
    return true;
}

bool session_read_logits(Session* s, std::vector<float>& out) {
    if (s->model.meta.task != "classify") {
        YOLO_LOG_ERROR("session_read_logits requires a classify model, got %s", s->model.meta.task.c_str());
        return false;
    }
    const int64_t n = ggml_nelements(s->output);
    out.resize(n);
    if (s->output->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(s->output, s->output_f16.data(), 0, s->output_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(s->output_f16.data(), out.data(), out.size());
    } else {
        ggml_backend_tensor_get(s->output, out.data(), 0, out.size() * sizeof(float));
    }
    return true;
}

bool session_dump_ops(const Session* s, const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    auto write = [&](const char* name, const ggml_tensor* t) {
        const size_t n = ggml_nelements(t);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.bin", dir.c_str(), name);
        FILE* f = fopen(path, "wb");
        if (!f) return false;
        fwrite("YLYR0001", 1, 8, f);
        const int32_t dims[4] = {(int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2], (int32_t)t->ne[3]};
        fwrite(dims, sizeof(int32_t), 4, f);
        fwrite(buf.data(), sizeof(float), n, f);
        fclose(f);
        return true;
    };
    if (s->input && s->input->type == GGML_TYPE_F32 && !write("input", s->input)) return false;
    if (s->text_input && s->text_input->type == GGML_TYPE_F32 && !write("text", s->text_input)) return false;
    for (size_t i = 0; i < s->op_values.size(); i++) {
        const ggml_tensor* t = s->op_values[i];
        if (!t || t->type != GGML_TYPE_F32) continue;
        char name[32];
        snprintf(name, sizeof(name), "op%03zu_%s", i, s->model.ops[i].type.c_str());
        if (!write(name, t)) return false;
    }
    return true;
}

void free_session(Session* s) {
    if (!s) return;
    if (s->wbuf) ggml_backend_buffer_free(s->wbuf);
    free_backend_ctx(s->backend);
    if (s->wctx) ggml_free(s->wctx);
    if (s->gctx) ggml_free(s->gctx);  // cgraph + tensor structs live in gctx
    delete s;
}

}  // namespace yolo
