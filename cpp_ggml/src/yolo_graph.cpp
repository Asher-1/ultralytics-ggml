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
        // All CUDA dtypes run the f16 direct flow: f32 weights are cast to
        // f16 once at load (see create_session) — same 10-bit mantissa as
        // cuBLAS TF32 mode — and Q8_0 joins via the plan-time dequant.
        if (!depthwise && direct_types && bias && silu) {
            return ggml_conv_2d_direct_bias_silu(gctx, wT, x, bias,
                                                 (int)op.ai("s", 0), (int)op.ai("s", 1),
                                                 (int)op.ai("p", 0), (int)op.ai("p", 1),
                                                 (int)op.ai("d", 0), (int)op.ai("d", 1));
        } else if (direct_types) {
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
            if (depthwise) w4d = dw_kernel(w4d);
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

    // ggml 0.18.1 conv_2d_dw builds mul_mat(F32 kernel, F16 im2col) which asserts on
    // CPU (src1 must be F32 when src0 is F32). Cast F32 kernels to F16 — matching
    // the im2col F16 dtype — so both operands are F16 like every other conv.
    ggml_tensor* dw_kernel(ggml_tensor* wT) {
        return wT->type == GGML_TYPE_F32 ? ggml_cast(gctx, wT, GGML_TYPE_F16) : wT;
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
                    ? ggml_conv_2d_dw(gctx, dw_kernel(wT), x, 1, 1, (int)(k / 2), (int)(k / 2), 1, 1)
                    : ggml_conv_2d(gctx, wT, x, 1, 1, 0, 0, 1, 1);
            }
#else
            out = k > 1
                ? ggml_conv_2d_dw(gctx, dw_kernel(wT), x, 1, 1, (int)(k / 2), (int)(k / 2), 1, 1)
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

    const ModelMeta& meta = s->model.meta;
    const int no = 4 * meta.reg_max + meta.nc + meta.nm;  // segment appends nm mask coeffs

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
#if defined(YOLO_USE_CUDA) || defined(YOLO_USE_VULKAN)
    // Route quantized convs through the direct flow only when every quantized
    // tensor conforms; a single violation falls the whole model back to the
    // F32 conv2d_q path, whose im2col requires F32 activations. K 32-alignment
    // is the hard constraint (CUDA igemm block addressing; vulkan host dequant
    // block walk). CUDA igemm also needs OC % 8 == 0 but writes sub-8 tails
    // scalars; vulkan runs the plain f16 path so OC is free.
    for (const auto& [name, ht] : s->model.tensors) {
        if (!ggml_is_quantized(ht.type)) continue;
        if (ht.type != GGML_TYPE_Q8_0 || ht.ne[0] % 32 != 0) {
            gb.q8_direct = false;
            break;
        }
        gb.q8_direct = true;
    }
#endif
#if defined(YOLO_USE_VULKAN)
    // vulkan has no Q8 conv shader and no Q8->f16 cast pipeline; expand Q8_0
    // weights to f16 on the host once so the whole graph runs the f16 direct-
    // conv path (2-5x faster than the dequant-f32 mul_mat route).
    if (gb.q8_direct) {
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
    }
#endif
#if defined(YOLO_USE_CUDA)
    // CUDA f32 models: cast weights to f16 once on the host. The igemm f16
    // path runs fp16 tensor cores (10-bit mantissa, the same precision class
    // as cuBLAS TF32 mode and PyTorch's default cudnn TF32 convs) at 2x the
    // TF32 mma throughput and skips the im2col materialization entirely.
    if (meta.dtype == "f32") for (auto& [name, ht] : s->model.tensors) {
        if (ht.type != GGML_TYPE_F32) continue;
        // Biases stay F32: the fused conv kernels read the bias pointer as
        // fp32 (contract asserted in conv2d.cu), and every add_bias/attention
        // path already casts it where needed.
        if (name.size() > 2 && name.compare(name.size() - 2, 2, ".b") == 0) continue;
        const int64_t n = ht.ne[0] * ht.ne[1] * ht.ne[2] * ht.ne[3];
        std::vector<uint8_t> f16(n * sizeof(ggml_fp16_t));
        const float * src = reinterpret_cast<const float *>(ht.data.data());
        ggml_fp16_t * dst = reinterpret_cast<ggml_fp16_t *>(f16.data());
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = ggml_fp32_to_fp16(src[i]);
        }
        ht.data = std::move(f16);
        ht.type = GGML_TYPE_F16;
    }
#endif
    std::vector<ggml_tensor*> values(s->model.ops.size(), nullptr);

    // The input tensor is always F32; F16-model graphs insert an on-device
    // cast node (below). A host-side scalar f32->f16 round-trip measured
    // ~2.1ms/frame at 640x480 -- pure host overhead on every GPU frame.
    s->input = ggml_new_tensor_4d(s->gctx, GGML_TYPE_F32, s->input_w, s->input_h, 3, 1);
    ggml_set_input(s->input);  // allocated before compute nodes
    ggml_set_name(s->input, "image");

    ggml_tensor* graph_input = s->input;
#if defined(YOLO_USE_CUDA)
    graph_input = ggml_cast(s->gctx, s->input, GGML_TYPE_F16);  // every CUDA dtype runs the f16 flow
#elif defined(YOLO_USE_VULKAN)
    if (meta.dtype == "f16" || gb.q8_direct) graph_input = ggml_cast(s->gctx, s->input, GGML_TYPE_F16);
#endif

    for (size_t i = 0; i < s->model.ops.size(); i++) {
        const OpDef& op = s->model.ops[i];
        const std::string prefix = "op." + std::to_string(i);
        auto in0 = [&]() {
            const int idx = op.inputs.empty() ? -1 : op.inputs[0];
            return idx < 0 ? graph_input : values[idx];
        };
        ggml_tensor* out = nullptr;

        if (op.type == "conv" || op.type == "dwconv") {
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
        } else if (op.type == "detect" || op.type == "segment") {
            // Per-level conv output ne=[W,H,no,N] is already CHW-ordered in memory
            // (c outer, h middle, w inner) — a plain reshape_2d matches torch's
            // x.view(B, no, H*W); concat along the anchor dim. No permute needed.
            // segment's last input is the proto map, kept as a second output.
            const size_t n_feats = op.inputs.size() - (op.type == "segment" ? 1 : 0);
            for (size_t j = 0; j < n_feats; j++) {
                ggml_tensor* t = values[op.inputs[j]];
                const int64_t HW = t->ne[0] * t->ne[1];
                ggml_tensor* r = ggml_reshape_2d(s->gctx, t, HW, no);
                out = out ? ggml_concat(s->gctx, out, r, 0) : r;
            }
            if (op.type == "segment") s->output_proto = values[op.inputs.back()];
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
    }

    s->output = values.back();
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

    s->graph = ggml_new_graph_custom(s->gctx, s->model.ops.size() * 12 + 512, /*grads*/ false);
    if (opts.keep_all_ops) {
        // OUTPUT tensors are never freed by gallocr (ggml-alloc.c free_node),
        // so every op's data survives the full compute for --dump-ops.
        for (ggml_tensor* t : s->op_values) {
            ggml_set_output(t);
            ggml_build_forward_expand(s->graph, t);
        }
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
        ggml_backend_sched_set_tensor_backend(s->backend.sched, s->output, s->backend.gpu);
        if (s->output_proto) {
            ggml_backend_sched_set_tensor_backend(s->backend.sched, s->output_proto, s->backend.gpu);
        }
    }
    if (!backend_graph_alloc(s->backend, s->graph)) {
        free_session(s);
        return nullptr;
    }

    if (meta.task != "depth") {  // detect and segment share the anchor grid
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

bool session_read_output(Session* s, std::vector<float>& out, int& no, int& na) {
    if (s->model.meta.task != "detect" && s->model.meta.task != "segment") {
        YOLO_LOG_ERROR("session_read_output requires a detect or segment model, got %s",
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
