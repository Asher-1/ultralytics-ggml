// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-distance — official "Distance Calculation" solution scenario
// (https://docs.ultralytics.com/solutions/distance-calculation) on the ggml
// C++ stack. Runs a depth model, restores the per-pixel metric depth map to
// the source resolution, and reports distances at the image center, at an
// optional --px/--py point, and per 3x3 grid cell.
//
// Usage:
//   yolo-distance --model yolo26n-depth-f16.gguf --source IMG \
//       [--px X --py Y] [--out depth.png] [--threads N]

#include "examples_common.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Mean of the depth cells inside a rectangle of the restored map.
float cell_mean(const std::vector<float>& depth, int w, int h, int x0, int y0, int x1, int y1) {
    double sum = 0;
    int n = 0;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            const float v = depth[(size_t)y * w + x];
            if (v > 0.0f) {  // skip invalid/zero depth
                sum += v;
                n++;
            }
        }
    }
    return n ? (float)(sum / n) : 0.0f;
}

}  // namespace

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-distance --model M-depth.gguf --source IMG "
                        "[--px X --py Y] [--out OUT.png] [--threads N]\n");
        return 1;
    }

    yolo_examples::Options opt;
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    yolo::Image img;
    std::vector<float> depth;
    if (!yolo_examples::run_depth(model, source, opt, img, depth)) return 1;
    const int w = img.w, h = img.h;

    const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
    const double mean = std::accumulate(depth.begin(), depth.end(), 0.0) / depth.size();
    printf("distance map %dx%d meters (min=%.3f mean=%.3f max=%.3f)\n", w, h, *lo, mean, *hi);

    // Center point distance (official scenario's default probe).
    const int cx = w / 2, cy = h / 2;
    printf("  center (%d, %d):          %.3f m\n", cx, cy, depth[(size_t)cy * w + cx]);

    // Optional explicit probe point.
    const int px = yolo_examples::arg_i(args, "px", -1);
    const int py = yolo_examples::arg_i(args, "py", -1);
    if (px >= 0 && py >= 0 && px < w && py < h) {
        printf("  point (%d, %d):           %.3f m\n", px, py, depth[(size_t)py * w + px]);
    }

    // 3x3 grid cell averages.
    printf("  3x3 grid cell means (meters):\n");
    for (int gy = 0; gy < 3; gy++) {
        printf("    ");
        for (int gx = 0; gx < 3; gx++) {
            const int x0 = gx * w / 3, x1 = (gx + 1) * w / 3;
            const int y0 = gy * h / 3, y1 = (gy + 1) * h / 3;
            printf("  %6.2f", cell_mean(depth, w, h, x0, y0, x1, y1));
        }
        printf("\n");
    }

    const std::string out = yolo_examples::arg_s(args, "out");
    if (!out.empty() && !yolo::write_depth_png(out, depth, w, h, 0.0f)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}
