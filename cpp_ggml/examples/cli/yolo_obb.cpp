// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-obb — oriented object detection scenario (DOTA-style) on the ggml C++
// stack. Runs an OBB model, prints each oriented box with its angle, exports
// a CSV, and renders the rotated boxes over the image.
//
// Usage:
//   yolo-obb --model yolo26n-obb-f16.gguf --source IMG \
//       [--conf 0.25] [--csv out.csv] [--out out.png] [--threads N]

#include "examples_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-obb --model M-obb.gguf --source IMG "
                        "[--conf F] [--csv OUT.csv] [--out OUT.png] [--threads N]\n");
        return 1;
    }

    yolo_examples::Options opt;
    opt.conf = yolo_examples::arg_f(args, "conf", 0.25f);
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    yolo::Image img;
    std::vector<yolo::OBBDetection> obbs;
    if (!yolo_examples::run_obb(model, source, opt, img, obbs)) return 1;

    printf("oriented detection on %dx%d image: %zu object%s\n", img.w, img.h, obbs.size(),
           obbs.size() == 1 ? "" : "s");
    printf("  %-10s %6s %8s %8s %6s %6s %7s\n", "class", "score", "cx", "cy", "w", "h", "angle");
    for (const auto& o : obbs) {
        printf("  %-10s %6.2f %8.1f %8.1f %6.1f %6.1f %7.1f\n", "object", o.score, o.cx, o.cy, o.w, o.h,
               o.angle * 180.0f / (float)M_PI);
    }

    const std::string csv = yolo_examples::arg_s(args, "csv");
    if (!csv.empty()) {
        FILE* f = fopen(csv.c_str(), "w");
        if (!f) {
            fprintf(stderr, "failed to open %s\n", csv.c_str());
            return 1;
        }
        fprintf(f, "class,score,cx,cy,w,h,angle_deg\n");
        for (const auto& o : obbs) {
            fprintf(f, "%s,%.4f,%.1f,%.1f,%.1f,%.1f,%.2f\n", "object", o.score, o.cx, o.cy, o.w, o.h,
                    o.angle * 180.0f / (float)M_PI);
        }
        fclose(f);
        printf("wrote %zu oriented box%s to %s\n", obbs.size(), obbs.size() == 1 ? "" : "es", csv.c_str());
    }

    const std::string out = yolo_examples::arg_s(args, "out");
    if (!out.empty() && !yolo::draw_obb(out, img, obbs, {"object"})) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}
