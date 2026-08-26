// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-region-count — official "Region Counting" solution scenario
// (https://docs.ultralytics.com/solutions/region-counting) on the ggml C++
// stack. Runs a detect model, counts how many detection centers fall inside
// each user-defined rectangle ROI, and renders the detections.
//
// Usage:
//   yolo-region-count --model yolo26n-f16.gguf --source bus.jpg \
//       --roi 0,0,320,640 --roi 320,0,640,640 [--conf 0.25] [--out out.png]
//
// Each --roi is x1,y1,x2,y2 in original-image pixels; repeat for more regions.

#include "examples_common.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Roi {
    float x1, y1, x2, y2;
};

// Parse one "x1,y1,x2,y2" ROI spec; false on malformed input.
bool parse_roi(const std::string& spec, Roi& roi) {
    float v[4] = {0, 0, 0, 0};
    int n = 0;
    for (size_t i = 0; i < spec.size() && n < 4;) {
        char* end = nullptr;
        v[n] = strtof(spec.c_str() + i, &end);
        if (end == spec.c_str() + i) break;
        n++;
        i = (size_t)(end - spec.c_str());
        if (i < spec.size() && spec[i] == ',') i++;
    }
    if (n != 4) {
        fprintf(stderr, "bad --roi '%s', expected x1,y1,x2,y2\n", spec.c_str());
        return false;
    }
    roi = {v[0], v[1], v[2], v[3]};
    return true;
}

bool inside(const Roi& r, float cx, float cy) {
    return cx >= r.x1 && cx <= r.x2 && cy >= r.y1 && cy <= r.y2;
}

}  // namespace

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-region-count --model M.gguf --source IMG [--roi x1,y1,x2,y2]... "
                        "[--conf F] [--out OUT.png] [--threads N]\n");
        return 1;
    }
    // parse_args keeps the last value of a repeated key, so collect the
    // repeatable --roi flags by rescanning argv.
    std::vector<Roi> rois;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--roi") {
            Roi roi;
            if (!parse_roi(argv[i + 1], roi)) return 1;
            rois.push_back(roi);
        }
    }
    if (rois.empty()) {
        fprintf(stderr, "at least one --roi x1,y1,x2,y2 is required\n");
        return 1;
    }

    yolo_examples::Options opt;
    opt.conf = yolo_examples::arg_f(args, "conf", 0.25f);
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    yolo::Image img;
    std::vector<yolo::Detection> dets;
    std::vector<yolo::SegMask> masks;
    yolo::LetterboxInfo info;
    std::vector<std::string> names;
    if (!yolo_examples::run_boxes(model, source, opt, img, dets, masks, info, names)) return 1;

    printf("region counting on %dx%d image: %zu detection%s across %zu ROI%s\n", img.w, img.h, dets.size(),
           dets.size() == 1 ? "" : "s", rois.size(), rois.size() == 1 ? "" : "s");
    for (size_t r = 0; r < rois.size(); r++) {
        const Roi& roi = rois[r];
        int total = 0;
        std::vector<int> per_class(names.size(), 0);
        for (const auto& d : dets) {
            const float cx = (d.x1 + d.x2) * 0.5f;
            const float cy = (d.y1 + d.y2) * 0.5f;
            if (inside(roi, cx, cy)) {
                total++;
                if (d.class_id >= 0 && d.class_id < (int)per_class.size()) per_class[d.class_id]++;
            }
        }
        printf("  ROI %zu [%.0f, %.0f, %.0f, %.0f]: %d object%s", r + 1, roi.x1, roi.y1, roi.x2, roi.y2, total,
               total == 1 ? "" : "s");
        for (size_t c = 0; c < per_class.size(); c++) {
            if (per_class[c]) printf(", %s=%d", c < names.size() ? names[c].c_str() : "?", per_class[c]);
        }
        printf("\n");
    }

    const std::string out = yolo_examples::arg_s(args, "out");
    if (!out.empty() && !yolo::draw_detections(out, img, dets, names, nullptr, nullptr)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}
