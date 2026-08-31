// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "common.hpp"

#include "ggml.h"
#include "gguf.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

// One node of the op-graph vocabulary written by scripts/convert_yolo_to_gguf.py.
struct OpDef {
    std::string type;  // primitive ops plus detect/segment/world_detect/world_segment task heads
    std::vector<int> inputs;        // op indices; -1 = graph input image
    std::map<std::string, int64_t> iparams;    // ints
    std::map<std::string, double> fparams;     // floats
    std::map<std::string, std::vector<int64_t>> aparams;  // int arrays (s/p/d)
    std::map<std::string, std::string> sparams; // strings (act)
    std::vector<std::string> tensor_names;      // w, b, qkv_w, ...

    int64_t ip(const std::string& k, int64_t def = 0) const {
        auto it = iparams.find(k);
        return it == iparams.end() ? def : it->second;
    }
    int64_t ai(const std::string& k, int idx, int64_t def = 0) const {
        auto it = aparams.find(k);
        if (it == aparams.end() || idx >= (int)it->second.size()) return def;
        return it->second[idx];
    }
};

// Host-side weight: ggml_type + raw block data + logical shape (torch order).
struct HostTensor {
    std::vector<uint8_t> data;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};  // ggml order: ne[0] fastest
    std::string name;
};

struct ModelDef {
    ModelMeta meta;
    std::vector<OpDef> ops;
    std::map<std::string, HostTensor> tensors;

    // Flattened per-level head info, taken from the single detect op.
    bool has_detect = false;
    int detect_op_index = -1;
    // YOLO-World: the graph consumes a runtime text-embedding input ([512, nc])
    // alongside the image. nc is fixed at session creation (set_classes).
    bool has_text_input = false;
    // YOLO-World offline vocabulary carried by the checkpoint: row-major
    // [nc, 512] f32, used as the text input when the caller sets none.
    std::vector<float> vocab_txt;
    // YOLOE visual-prompt support (KV yolo.savpe = 1): the GGUF carries the
    // head's SAVPE conv weights so example boxes can drive the head instead
    // of a text embedding. savpe_fpn_ops lists the op indices producing the
    // FPN features [P3, P4, P5] the savpe encoder consumes (resolved at load
    // time by walking the head's box/embed branches back to their shared
    // producer); empty when the GGUF has no savpe.
    bool has_savpe = false;
    std::vector<int> savpe_fpn_ops;
};

// Read only the metadata header (cheap: no tensor data).
ModelMeta read_gguf_meta(const std::string& path);

// Load and parse a GGUF file. Returns nullptr and logs on failure.
std::unique_ptr<ModelDef> load_gguf(const std::string& path);

}  // namespace yolo
