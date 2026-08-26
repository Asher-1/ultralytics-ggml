// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-crop — official "Object Cropping" solution scenario
// (https://docs.ultralytics.com/solutions/object-cropping) on the ggml C++
// stack. Runs a segment model, crops each instance out of the image using its
// binary mask, and writes it as a transparent-background PNG.
//
// Usage:
//   yolo-crop --model yolo26n-seg-f16.gguf --source bus.jpg \
//       [--dir crops] [--conf 0.25] [--threads N]
//
// Outputs crops/k000_<class>_<score>.png per detection (transparent outside
// the instance mask), plus a per-crop summary line on stdout.

#include "examples_common.hpp"

// stbi_write_png is implemented in libyologgml (src/image_io.cpp); only the
// header declarations are needed here.
#include "../../third_party/stb/stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Map a canvas-space mask window back to original pixels and crop the masked
// region. Returns false when the mask contains no original-image pixels.
bool crop_masked(const yolo::Image& img, const yolo::SegMask& mask, const yolo::LetterboxInfo& info,
                 std::vector<uint8_t>& rgba, int& ow, int& oh, int& ox0, int& oy0) {
    // Pass 1: bounding box of mask pixels that map inside the original image.
    int min_x = img.w, min_y = img.h, max_x = -1, max_y = -1;
    for (int wy = 0; wy < mask.h; wy++) {
        for (int wx = 0; wx < mask.w; wx++) {
            if (!mask.bits[(size_t)wy * mask.w + wx]) continue;
            const int ox = (int)lroundf((mask.x + wx - info.pad_w) / info.scale);
            const int oy = (int)lroundf((mask.y + wy - info.pad_h) / info.scale);
            if (ox < 0 || ox >= img.w || oy < 0 || oy >= img.h) continue;
            min_x = std::min(min_x, ox);
            min_y = std::min(min_y, oy);
            max_x = std::max(max_x, ox);
            max_y = std::max(max_y, oy);
        }
    }
    if (max_x < min_x || max_y < min_y) return false;
    ox0 = min_x;
    oy0 = min_y;
    ow = max_x - min_x + 1;
    oh = max_y - min_y + 1;

    // Pass 2: fill the RGBA crop; alpha = 255 inside the mask, else 0.
    rgba.assign((size_t)ow * oh * 4, 0);
    for (int wy = 0; wy < mask.h; wy++) {
        for (int wx = 0; wx < mask.w; wx++) {
            if (!mask.bits[(size_t)wy * mask.w + wx]) continue;
            const int ox = (int)lroundf((mask.x + wx - info.pad_w) / info.scale);
            const int oy = (int)lroundf((mask.y + wy - info.pad_h) / info.scale);
            if (ox < ox0 || ox >= ox0 + ow || oy < oy0 || oy >= oy0 + oh) continue;
            uint8_t* dst = &rgba[((size_t)(oy - oy0) * ow + (ox - ox0)) * 4];
            const uint8_t* src = &img.rgb[((size_t)oy * img.w + ox) * 3];
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = 255;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-crop --model M-seg.gguf --source IMG [--dir crops] "
                        "[--conf F] [--threads N]\n");
        return 1;
    }
    const std::string dir = yolo_examples::arg_s(args, "dir", "crops");

    yolo_examples::Options opt;
    opt.conf = yolo_examples::arg_f(args, "conf", 0.25f);
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    yolo::Image img;
    std::vector<yolo::Detection> dets;
    std::vector<yolo::SegMask> masks;
    yolo::LetterboxInfo info;
    std::vector<std::string> names;
    if (!yolo_examples::run_boxes(model, source, opt, img, dets, masks, info, names)) return 1;

    // Ensure the output directory exists (mkdir -p semantics).
    {
        std::string cmd = "mkdir -p " + dir;
        if (system(cmd.c_str()) != 0) {
            fprintf(stderr, "failed to create output directory %s\n", dir.c_str());
            return 1;
        }
    }

    int saved = 0;
    for (size_t k = 0; k < dets.size(); k++) {
        if (k >= masks.size() || masks[k].w <= 0 || masks[k].h <= 0) continue;
        std::vector<uint8_t> rgba;
        int ow = 0, oh = 0, ox0 = 0, oy0 = 0;
        if (!crop_masked(img, masks[k], info, rgba, ow, oh, ox0, oy0)) continue;
        const std::string cname = dets[k].class_id < (int)names.size() ? names[dets[k].class_id] : "?";
        char path[512];
        snprintf(path, sizeof(path), "%s/k%03zu_%s_%.2f.png", dir.c_str(), k, cname.c_str(), dets[k].score);
        if (!stbi_write_png(path, ow, oh, 4, rgba.data(), ow * 4)) {
            fprintf(stderr, "failed to write %s\n", path);
            return 1;
        }
        printf("  cropped %s (score %.2f): %dx%d at (%d, %d) -> %s\n", cname.c_str(), dets[k].score, ow, oh, ox0,
               oy0, path);
        saved++;
    }
    printf("object cropping: %d/%zu instance%s saved to %s/\n", saved, dets.size(),
           dets.size() == 1 ? "" : "s", dir.c_str());
    return 0;
}
