// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-city — semantic segmentation scenario (Cityscapes-style) on the ggml
// C++ stack. Runs a semantic model, prints the per-class pixel histogram with
// area share, and renders the class map blended over the image.
//
// Usage:
//   yolo-city --model yolo26n-sem-f16.gguf --source IMG \
//       [--out out.png] [--threads N]

#include "examples_common.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-city --model M-sem.gguf --source IMG [--out OUT.png] [--threads N]\n");
        return 1;
    }

    yolo_examples::Options opt;
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    yolo::Image img;
    std::vector<uint8_t> classes;
    int gw = 0, gh = 0;
    std::vector<std::string> names;
    if (!yolo_examples::run_semantic(model, source, opt, img, classes, gw, gh, names)) return 1;

    // Per-class histogram on the grid, sorted by pixel count.
    std::vector<size_t> hist(256, 0);
    for (uint8_t c : classes) hist[c]++;
    std::vector<int> order(256);
    for (int i = 0; i < 256; i++) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                      [&](int a, int b) { return hist[a] > hist[b]; });
    const size_t total = (size_t)gw * gh;
    printf("semantic segmentation: %dx%d grid, %d classes, top-10 by area:\n", gw, gh, (int)names.size());
    printf("  %-16s %10s %7s\n", "class", "pixels", "share");
    int shown = 0;
    for (int i = 0; i < 256 && shown < 10; i++) {
        const int c = order[i];
        if (hist[c] == 0) continue;
        const char* cname = c < (int)names.size() ? names[c].c_str() : "?";
        printf("  %-16s %10zu %6.2f%%\n", cname, hist[c], 100.0 * hist[c] / total);
        shown++;
    }

    const std::string out = yolo_examples::arg_s(args, "out");
    if (!out.empty() && !yolo::draw_semantic(out, img, classes, gw, gh, 256)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}
