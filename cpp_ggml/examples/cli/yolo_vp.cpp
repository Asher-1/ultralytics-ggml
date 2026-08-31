// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-vp — YOLOE visual-prompt scenario on the ggml C++ stack. One example
// box per target drives the checkpoint's SAVPE encoder (official visual-
// prompt semantics); the head localizes every prompted target and results
// are labeled object0..objectN-1. Requires a YOLOE GGUF carrying savpe
// weights (yolo.savpe = 1) — produce one from a non-prompt-free checkpoint
// with scripts/convert_yoloe_savpe_gguf.py.
//
// Usage:
//   yolo-vp --model yoloe-26s-seg-f16.gguf --source IMG \
//       --boxes "x1,y1,x2,y2[,x1,y1,x2,y2,...]" [--conf 0.25] [--iou 0.7] \
//       [--max-det 300] [--threads N] [--dets-json OUT.json]

#include "../../src/yolo_graph.hpp"
#include "../../src/postprocess.hpp"
#include "../../src/image_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
    // Minimal --key value / --flag argument map (mirrors yolo-cli).
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || k.size() <= 2) {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return 1;
        }
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            args[k] = argv[++i];
        } else {
            args[k] = "1";
        }
    }
    auto arg_s = [&](const char* k) {
        auto it = args.find(k);
        return it == args.end() ? std::string() : it->second;
    };

    const std::string model_path = arg_s("model");
    const std::string source = arg_s("source");
    const std::string boxes_spec = arg_s("boxes");
    if (model_path.empty() || source.empty() || boxes_spec.empty()) {
        fprintf(stderr,
                "usage: yolo-vp --model yoloe-26N-seg-DTYPE.gguf --source IMG \\\n"
                "           --boxes \"x1,y1,x2,y2[,x1,y1,x2,y2,...]\" [--conf 0.25] [--iou 0.7] \\\n"
                "           [--max-det 300] [--threads N] [--dets-json OUT.json]\n");
        return 1;
    }

    // Flat [x1,y1,x2,y2, ...] example boxes in original-image pixels; the box
    // count Q becomes the detection class count (object0..objectQ-1).
    std::vector<float> boxes;
    size_t pos = 0;
    while (pos <= boxes_spec.size()) {
        const size_t comma = boxes_spec.find(',', pos);
        const std::string tok =
            boxes_spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) boxes.push_back(strtof(tok.c_str(), nullptr));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (boxes.empty() || boxes.size() % 4 != 0) {
        fprintf(stderr, "--boxes must be one or more x1,y1,x2,y2 quadruples\n");
        return 1;
    }

    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "segment" && meta.task != "detect") {
        fprintf(stderr, "yolo-vp requires a yoloe detect or segment model, got task=%s\n", meta.task.c_str());
        return 1;
    }

    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);

    yolo::SessionOptions sopts;
    sopts.threads = atoi(arg_s("threads").c_str());
    if (sopts.threads <= 0) sopts.threads = 0;
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.visual_count = (int)(boxes.size() / 4);
    sopts.visual_boxes = boxes;

    std::unique_ptr<yolo::Session, decltype(&yolo::free_session)> session(yolo::create_session(model_path, sopts),
                                                                          yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    // Rasterize the example boxes onto the letterboxed P3 grid; re-run for
    // every new canvas before session_run.
    if (!yolo::session_prepare_visual_masks(s, info)) {
        fprintf(stderr, "failed to rasterize the visual prompt boxes\n");
        return 1;
    }
    if (!yolo::session_run(s, input.data())) return 1;

    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;

    yolo::PostprocConfig cfg;
    cfg.conf_thres = arg_s("conf").empty() ? 0.25f : strtof(arg_s("conf").c_str(), nullptr);
    cfg.iou_thres = arg_s("iou").empty() ? 0.7f : strtof(arg_s("iou").c_str(), nullptr);
    cfg.max_det = arg_s("max-det").empty() ? s->model.meta.max_det : atoi(arg_s("max-det").c_str());
    std::vector<yolo::Detection> dets =
        yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    std::vector<yolo::SegMask> masks;
    if (meta.task == "segment") {
        std::vector<float> proto;
        int nm = 0, pw = 0, ph = 0;
        if (!yolo::session_read_proto(s, proto, nm, pw, ph)) return 1;
        masks = yolo::compose_masks(dets, raw, na, s->model.meta, proto, pw, ph, info.imgsz_w, info.imgsz_h);
    }
    yolo::unscale_boxes(dets, info);

    printf("%d detection%s (%s, %s, %dx%d, backend=%s)\n", (int)dets.size(), dets.size() == 1 ? "" : "s",
           s->model.meta.name.c_str(), s->model.meta.dtype.c_str(), info.imgsz_w, info.imgsz_h,
           yolo::backend_name(s->backend));
    for (size_t i = 0; i < dets.size(); i++) {
        const auto& d = dets[i];
        const std::string cname =
            "object" + std::to_string(d.class_id < sopts.visual_count ? d.class_id : 0);
        printf("  %-12s %.2f  [%.1f, %.1f, %.1f, %.1f]", cname.c_str(), d.score, d.x1, d.y1, d.x2, d.y2);
        if (i < masks.size() && masks[i].w > 0) {
            size_t bits = 0;
            for (uint8_t b : masks[i].bits) bits += b;
            printf("  mask=%zu", bits);
        }
        printf("\n");
    }

    const std::string dets_json = arg_s("dets-json");
    if (!dets_json.empty()) {
        FILE* f = fopen(dets_json.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "failed to write --dets-json %s\n", dets_json.c_str());
            return 1;
        }
        fputs("{\"vocabulary\":[", f);
        for (int i = 0; i < sopts.visual_count; i++) {
            fprintf(f, "%s\"object%d\"", i ? "," : "", i);
        }
        fputs("],\"detections\":[", f);
        for (size_t i = 0; i < dets.size(); i++) {
            const auto& d = dets[i];
            fprintf(f, "%s{\"cls\":%d,\"conf\":%.6f,\"xyxy\":[%.3f,%.3f,%.3f,%.3f}]}", i ? "," : "", d.class_id,
                    d.score, d.x1, d.y1, d.x2, d.y2);
        }
        fputs("]}\n", f);
        fclose(f);
    }
    return 0;
}
