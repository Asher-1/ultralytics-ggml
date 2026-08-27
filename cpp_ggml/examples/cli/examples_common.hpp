// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once
// Shared inference plumbing for the end-to-end scenario examples.
//
// Each example in this directory is a single-file program pairing one task
// model with one official Ultralytics solution scenario (region counting,
// object cropping, workout monitoring, ...). This header owns the
// load -> letterbox -> session -> run -> readback pipeline that every
// example needs; the examples own only the scenario logic on top.

#include "../../src/yolo_graph.hpp"
#include "../../src/postprocess.hpp"
#include "../../src/image_io.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolo_examples {

struct Options {
    int threads = 0;      // <= 0: hardware default
    float conf = 0.25f;
    float iou = 0.7f;
    int max_det = 300;
    int topk = 5;
};

using SessionPtr = std::unique_ptr<yolo::Session, decltype(&yolo::free_session)>;

// Minimal --key value / --flag argument map (mirrors yolo-cli's parse_args).
using Args = std::unordered_map<std::string, std::string>;

inline Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || k.size() <= 2) {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return {};
        }
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            args[k] = argv[++i];
        } else {
            args[k] = "1";
        }
    }
    return args;
}

inline std::string arg_s(const Args& a, const char* k, const std::string& def = "") {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

inline int arg_i(const Args& a, const char* k, int def) {
    auto it = a.find(k);
    return it == a.end() ? def : atoi(it->second.c_str());
}

inline float arg_f(const Args& a, const char* k, float def) {
    auto it = a.find(k);
    return it == a.end() ? def : (float)atof(it->second.c_str());
}

// Load the image, letterbox it, create a session sized to the canvas, and run
// the graph. `sess` receives the live session (caller keeps it until readback).
bool run_letterbox(const std::string& gguf, const yolo::Image& img, const Options& opt, SessionPtr& sess,
                   yolo::LetterboxInfo& info, std::vector<float>& input);

// ---- per-task pipelines (each mirrors the verified cli.cpp path) ----

// detect/segment: boxes in original pixels, masks in canvas coordinates.
bool run_boxes(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
               std::vector<yolo::Detection>& dets, std::vector<yolo::SegMask>& masks, yolo::LetterboxInfo& info,
               std::vector<std::string>& names);

bool run_pose(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
              std::vector<yolo::PoseDetection>& poses);

bool run_obb(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
             std::vector<yolo::OBBDetection>& obbs);

bool run_semantic(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
                  std::vector<uint8_t>& classes, int& gw, int& gh, std::vector<std::string>& names);

bool run_classify(const std::string& gguf, const std::string& source, const Options& opt,
                  std::vector<float>& probs, std::vector<std::string>& names);

bool run_depth(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
               std::vector<float>& depth);

}  // namespace yolo_examples
