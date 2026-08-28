// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace yolo {

/* Backend context: persistent CPU threadpool + optional GPU backend spanned
 * by a scheduler so unsupported ops fall back to CPU automatically. */
struct BackendCtx {
    int n_threads = 1;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t galloc = nullptr;

    bool has_gpu() const { return gpu != nullptr; }
};

// n_threads <= 0 keeps the hardware default. graph_nodes is the largest
// ggml_cgraph the caller will build: the scheduler sizes its node hash set
// once, at creation.
BackendCtx init_backend_ctx(int n_threads, size_t graph_nodes);
// Install the per-op profiling callback on the scheduler (GPU builds).
// Every node is then dispatched and synced individually: the printed table
// is useful for relative shares, not absolute latency.
void backend_enable_op_profile(BackendCtx& ctx);
void free_backend_ctx(BackendCtx& ctx);
ggml_backend_buffer_type_t backend_weight_buft(const BackendCtx& ctx);
bool backend_graph_alloc(BackendCtx& ctx, ggml_cgraph* graph);
int backend_graph_compute(BackendCtx& ctx, ggml_cgraph* graph);
const char* backend_name(const BackendCtx& ctx);

}  // namespace yolo
