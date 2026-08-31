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
    ggml_tensor* text_input = nullptr;         // YOLO-World: external [512, nc] F32 text embedding
    std::vector<float> text_pending;            // host cache for set_text before backend alloc
    int world_nc = 0;                          // YOLO-World: class count driving the text input shape
    // YOLOE visual prompts (SAVPE): active when SessionOptions::visual_count
    // > 0. The external [W3, H3, Q] F32 mask leaf replaces the text input;
    // vp_pending is the host-side letterbox-aware rasterization, uploaded
    // before every run just like text_pending.
    ggml_tensor* vp_input = nullptr;
    std::vector<float> vp_pending;
    std::vector<float> visual_boxes;  // prompted boxes, original-image pixels
    ggml_tensor* savpe_out = nullptr;  // debug: the [512, Q] vpe node (visual mode)
    ggml_tensor* savpe_fpn_dbg[3] = {nullptr, nullptr, nullptr};  // debug: savpe inputs
    bool visual_mode() const { return vp_input != nullptr; }
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
    int world_nc = 0;            // YOLO-World class count (0: use the GGUF yolo.nc default)
    // YOLOE visual prompts: Q example boxes as a flat [x1,y1,x2,y2, ...] array
    // in original-image pixels; visual_count > 0 switches the session to the
    // savpe mask input (requires a GGUF with yolo.savpe = 1).
    int visual_count = 0;
    std::vector<float> visual_boxes;
};

// Create a session for a GGUF model. See SessionOptions for the knobs.
Session* create_session(const std::string& gguf_path, const SessionOptions& opts = {});
void free_session(Session* s);

// Copy a CHW float image into the input tensor, run the graph.
bool session_run(Session* s, const float* chw_image);

// YOLO-World: copy an [nc, 512] row-major text embedding into the text input
// (column-major [512, nc] in ggml) before session_run.
bool session_set_text(Session* s, const float* text_embed);

// YOLOE visual prompts: rasterize the session's prompted boxes (original-
// image pixels, stored in SessionOptions::visual_boxes) onto the letterboxed
// P3 mask grid and queue the upload. Call after create_session and before
// every session_run; the session must be in visual mode.
ggml_tensor* yolo_debug_emb();  // debug: level-0 embed matrix (YOLO_EMB_DUMP)

bool session_prepare_visual_masks(Session* s, const LetterboxInfo& info);

// Read back the raw output [no, A] (row-major: no rows x A anchors). For
// segment models `no` additionally carries nm mask-coefficient rows.
bool session_read_output(Session* s, std::vector<float>& out, int& no, int& na);

// Read back the segment mask prototypes [nm, H, W] (row-major, canvas/4 grid).
bool session_read_proto(Session* s, std::vector<float>& out, int& nm, int& w, int& h);

// Read back a metric depth map in meters, row-major [height, width].
bool session_read_depth(Session* s, std::vector<float>& out, int& width, int& height);

// Read back semantic logits [nc, H, W] on the canvas/8 grid (row-major channels).
bool session_read_semantic(Session* s, std::vector<float>& out, int& nc, int& w, int& h);

// Read back classify logits [nc] (row-major).
bool session_read_logits(Session* s, std::vector<float>& out);

// Dump every per-op output tensor to `dir` as YLYR0001 bins (4x i32 dims + f32).
bool session_dump_ops(const Session* s, const std::string& dir);

}  // namespace yolo
