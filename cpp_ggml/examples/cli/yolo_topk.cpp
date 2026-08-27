// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-topk — image classification scenario on the ggml C++ stack. Runs a
// classify model, prints the top-k classes with softmax probabilities, and a
// compact probability bar per rank.
//
// Usage:
//   yolo-topk --model yolo26n-cls-f16.gguf --source IMG \
//       [--topk 5] [--threads N]

#include "examples_common.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-topk --model M-cls.gguf --source IMG [--topk N] [--threads N]\n");
        return 1;
    }

    yolo_examples::Options opt;
    opt.topk = yolo_examples::arg_i(args, "topk", 5);
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    std::vector<float> probs;
    std::vector<std::string> names;
    if (!yolo_examples::run_classify(model, source, opt, probs, names)) return 1;

    const int topk = std::clamp(opt.topk, 1, (int)probs.size());
    std::vector<size_t> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [&](size_t a, size_t b) { return probs[a] > probs[b]; });

    printf("classification top-%d (%.0f classes):\n", topk, (double)probs.size());
    for (int i = 0; i < topk; i++) {
        const size_t c = idx[i];
        const char* cname = c < names.size() ? names[c].c_str() : "?";
        const int bar = (int)lroundf(probs[c] * 40.0f);
        printf("  %2d. %-28s %.4f  |%s%s|\n", i + 1, cname, probs[c], std::string(bar, '#').c_str(),
               std::string(40 - bar, ' ').c_str());
    }
    return 0;
}
