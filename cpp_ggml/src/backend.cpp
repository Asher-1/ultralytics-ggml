// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "backend.hpp"
#include "common.hpp"

#include "ggml-cpu.h"

#if defined(YOLO_USE_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(YOLO_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(YOLO_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace yolo {

/* Per-op timing via the sched eval callback, installed with
 * backend_enable_op_profile(): every node is dispatched and synchronized
 * individually, so each measurement includes a fixed per-node sync overhead
 * (~tens of us). Useful for relative shares, not absolute latency. Conv-family
 * nodes are keyed by their weight tensor's name ("op.N.w"), which pins the
 * measurement to one model layer. */
namespace {
struct OpStat {
    int count = 0;
    double total_ms = 0.0;
};
std::unordered_map<std::string, OpStat> g_op_stats;
bool g_op_profiling = false;
int g_op_iters = 0;
std::chrono::steady_clock::time_point g_op_t0;

std::string op_profile_key(const ggml_tensor* t) {
    std::string k = ggml_op_desc(t);
    if (t->src[0] && t->src[0]->name[0]) {
        k += ' ';
        k += t->src[0]->name;
    }
    char buf[48];
    std::snprintf(buf, sizeof buf, " [%lldx%lldx%lld]",
                  (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
    k += buf;
    return k;
}

bool op_profile_eval_cb(ggml_tensor* t, bool ask, void*) {
    if (ask) return true;  // observe every node: per-node dispatch + sync
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - g_op_t0).count();
    g_op_t0 = now;
    OpStat& st = g_op_stats[op_profile_key(t)];
    st.count++;
    st.total_ms += ms;
    return true;  // false would abort the remaining graph
}
}  // namespace

static void forward_ggml_log(enum ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    std::fputs(text, stderr);
    std::fflush(stderr);
}

/* Try to create a GPU backend if one was compiled in and a device exists.
 * Returns nullptr (not an error) when no GPU backend is built or no device
 * is present — the caller falls back to CPU-only. */
static ggml_backend_t try_init_gpu_backend() {
#if defined(YOLO_USE_CUDA)
    int n = ggml_backend_cuda_get_device_count();
    if (n > 0) {
        ggml_backend_t b = ggml_backend_cuda_init(0);
        if (b) {
            YOLO_LOG_INFO("GPU backend: CUDA device 0 (%d available)", n);
            return b;
        }
        YOLO_LOG_WARN("ggml_backend_cuda_init(0) failed; using CPU");
    }
    return nullptr;
#elif defined(YOLO_USE_METAL)
    ggml_backend_t b = ggml_backend_metal_init();
    if (b) {
        YOLO_LOG_INFO("GPU backend: Metal");
        return b;
    }
    YOLO_LOG_WARN("ggml_backend_metal_init failed; using CPU");
    return nullptr;
#elif defined(YOLO_USE_VULKAN)
    int n = ggml_backend_vk_get_device_count();
    if (n > 0) {
        ggml_backend_t b = ggml_backend_vk_init(0);
        if (b) {
            YOLO_LOG_INFO("GPU backend: Vulkan device 0 (%d available)", n);
            return b;
        }
        YOLO_LOG_WARN("ggml_backend_vk_init(0) failed; using CPU");
    }
    return nullptr;
#else
    return nullptr;  // CPU-only build
#endif
}

BackendCtx init_backend_ctx(int n_threads) {
    ggml_log_set(forward_ggml_log, nullptr);
    BackendCtx ctx{};
    if (n_threads <= 0) {
        // Hardware default. The backend and the persistent threadpool below
        // MUST agree on the thread count: ggml_graph_plan takes n_threads from
        // the backend while compute indexes threadpool->workers[omp_thread],
        // so any mismatch overruns the workers array.
        n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    ctx.n_threads = n_threads;

    ctx.cpu = ggml_backend_cpu_init();
    if (!ctx.cpu) {
        YOLO_LOG_ERROR("ggml_backend_cpu_init returned null");
        return ctx;
    }
    ggml_backend_cpu_set_n_threads(ctx.cpu, n_threads);

    /* Attach a persistent threadpool to the CPU backend — amortizes the
     * disposable-threadpool setup ggml would otherwise pay per call. */
    {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        ctx.threadpool = ggml_threadpool_new(&tpp);
        if (ctx.threadpool) {
            ggml_backend_cpu_set_threadpool(ctx.cpu, ctx.threadpool);
        } else {
            YOLO_LOG_WARN("ggml_threadpool_new failed; per-call threadpool");
        }
    }

    /* Try a GPU backend. If present, build a scheduler spanning [gpu, cpu]
     * so ops the GPU can't run fall back to CPU automatically. */
    ctx.gpu = try_init_gpu_backend();
    if (ctx.gpu) {
        std::vector<ggml_backend_t> backends = {ctx.gpu, ctx.cpu};
        std::vector<ggml_backend_buffer_type_t> bufts = {
            ggml_backend_get_default_buffer_type(ctx.gpu),
            ggml_backend_get_default_buffer_type(ctx.cpu),
        };
        ctx.sched = ggml_backend_sched_new(
            backends.data(), bufts.data(), (int)backends.size(),
            /*graph_size*/ 8192, /*parallel*/ false, /*op_offload*/ true);
        if (!ctx.sched) {
            YOLO_LOG_WARN("ggml_backend_sched_new failed; CPU-only");
            ggml_backend_free(ctx.gpu);
            ctx.gpu = nullptr;
        }
    }
    return ctx;
}

void backend_enable_op_profile(BackendCtx& ctx) {
    if (!ctx.sched || !ctx.gpu || g_op_profiling) return;
    g_op_profiling = true;
    ggml_backend_sched_set_eval_callback(ctx.sched, op_profile_eval_cb, nullptr);
}

ggml_backend_buffer_type_t backend_weight_buft(const BackendCtx& ctx) {
    if (ctx.gpu) {
        return ggml_backend_get_default_buffer_type(ctx.gpu);
    }
    return ggml_backend_get_default_buffer_type(ctx.cpu);
}

void free_backend_ctx(BackendCtx& ctx) {
    if (g_op_profiling && !g_op_stats.empty()) {
        std::vector<std::pair<std::string, OpStat>> v(g_op_stats.begin(), g_op_stats.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });
        std::fprintf(stderr, "\n[op profile] %d graph computes (incl. warmup); per-op wall time incl. per-node sync overhead\n", g_op_iters);
        std::fprintf(stderr, "%12s %7s %10s  %s\n", "total_ms", "calls", "avg_us", "op");
        for (const auto& [key, st] : v) {
            if (st.total_ms < 0.2) break;  // only the significant entries
            std::fprintf(stderr, "%12.2f %7d %10.0f  %s\n", st.total_ms, st.count, st.total_ms * 1000.0 / st.count, key.c_str());
        }
        std::fflush(stderr);
    }
    if (ctx.galloc) {
        ggml_gallocr_free(ctx.galloc);
        ctx.galloc = nullptr;
    }
    if (ctx.sched) {
        ggml_backend_sched_free(ctx.sched);
        ctx.sched = nullptr;
    }
    if (ctx.gpu) {
        ggml_backend_free(ctx.gpu);
        ctx.gpu = nullptr;
    }
    if (ctx.cpu) {
        ggml_backend_free(ctx.cpu);
        ctx.cpu = nullptr;
    }
    if (ctx.threadpool) {
        ggml_threadpool_free(ctx.threadpool);
        ctx.threadpool = nullptr;
    }
}

bool backend_graph_alloc(BackendCtx& ctx, ggml_cgraph* graph) {
    if (ctx.sched) {
        if (!ggml_backend_sched_alloc_graph(ctx.sched, graph)) {
            YOLO_LOG_ERROR("backend_graph_alloc: sched alloc failed");
            return false;
        }
        return true;
    }
    /* CPU path: persistent gallocr. */
    if (!ctx.galloc) {
        ctx.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.cpu));
        if (!ctx.galloc) {
            YOLO_LOG_ERROR("backend_graph_alloc: gallocr_new failed");
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(ctx.galloc, graph)) {
        YOLO_LOG_ERROR("backend_graph_alloc: gallocr_alloc_graph failed");
        return false;
    }
    return true;
}

int backend_graph_compute(BackendCtx& ctx, ggml_cgraph* graph) {
    if (ctx.sched) {
        if (g_op_profiling) {
            g_op_t0 = std::chrono::steady_clock::now();
            g_op_iters++;
        }
        ggml_status st = ggml_backend_sched_graph_compute(ctx.sched, graph);
        return (int)st;
    }
    ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
    return (int)st;
}

const char* backend_name(const BackendCtx& ctx) {
    if (ctx.gpu) return ggml_backend_name(ctx.gpu);
    if (ctx.cpu) return ggml_backend_name(ctx.cpu);
    return "none";
}

}  // namespace yolo
