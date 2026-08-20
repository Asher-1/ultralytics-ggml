// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "backend.hpp"
#include "gguf_loader.hpp"

#include <string>
#include <vector>

namespace yolo {

/* An inference session: parsed GGUF model + built ggml graph on a backend. */
struct Session {
    ModelDef model;
    BackendCtx backend;
    ggml_context* gctx = nullptr;              // graph tensors
    ggml_context* wctx = nullptr;              // weight tensors
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_tensor* input = nullptr;              // external [W, H, 3] F32 input; F16 models cast on-device
    ggml_tensor* output = nullptr;             // raw detect [A,no] or metric depth [W,H,1,1]
    ggml_tensor* output_proto = nullptr;       // segment protos [W,H,nm,1] on the H/4 grid
    ggml_cgraph* graph = nullptr;
    std::vector<ggml_fp16_t> output_f16;        // CPU-backend F16 readback scratch, allocated once per session
    std::vector<ggml_fp16_t> output_proto_f16;  // segment proto F16 readback scratch
    int input_w = 640;                          // letterboxed canvas dims (stride-multiple,
    int input_h = 640;                          // non-square under LetterBox auto=True)

    // Postprocess constants (mirrors ultralytics make_anchors).
    std::vector<float> anchors;                // [A*2] (x+0.5, y+0.5) per anchor
    std::vector<float> anchor_strides;         // [A]
    int anchor_total = 0;
    std::vector<float> dfl_proj;               // [reg_max]

    // Debug: per-op output tensors, parallel to model.ops (parity testing).
    std::vector<ggml_tensor*> op_values;

    // Gap-profiler state (only touched when SessionOptions::profile_gaps).
    bool profile_gaps = false;
    double gap_up_ms = 0.0, gap_comp_ms = 0.0;   // session_run: upload vs compute
    double gap_get_ms = 0.0, gap_cast_ms = 0.0;  // session_read_output: download vs cast
    int gap_frames = 0, gap_rframes = 0;
};

/* Session creation options. Defaults mirror the plain inference path; the
 * profiling flags are diagnostics that add per-node sync (profile_ops) or
 * per-stage stderr traces (profile_gaps) and are off in normal runs. */
struct SessionOptions {
    int threads = 0;             // <= 0: hardware default
    int input_w = 0;             // 0: square imgsz stored in the GGUF metadata
    int input_h = 0;
    bool keep_all_ops = false;   // keep every op output alive for --dump-ops (debug, extra memory)
    bool profile_ops = false;    // per-op wall-time table via sched eval callback (adds per-node sync)
    bool profile_gaps = false;   // per-stage (upload/compute/readback) running averages on stderr
};

// Create a session for a GGUF model. See SessionOptions for the knobs.
Session* create_session(const std::string& gguf_path, const SessionOptions& opts = {});
void free_session(Session* s);

// Copy a CHW float image into the input tensor, run the graph.
bool session_run(Session* s, const float* chw_image);

// Read back the raw output [no, A] (row-major: no rows x A anchors). For
// segment models `no` additionally carries nm mask-coefficient rows.
bool session_read_output(Session* s, std::vector<float>& out, int& no, int& na);

// Read back the segment mask prototypes [nm, H, W] (row-major, canvas/4 grid).
bool session_read_proto(Session* s, std::vector<float>& out, int& nm, int& w, int& h);

// Read back a metric depth map in meters, row-major [height, width].
bool session_read_depth(Session* s, std::vector<float>& out, int& width, int& height);

// Dump every per-op output tensor to `dir` as YLYR0001 bins (4x i32 dims + f32).
bool session_dump_ops(const Session* s, const std::string& dir);

}  // namespace yolo
