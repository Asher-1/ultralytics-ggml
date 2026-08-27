// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>

#if defined(YOLO_USE_OPENMP)
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace yolo {

bool load_image(const std::string& path, Image& img) {
    int n = 0;
    uint8_t* data = stbi_load(path.c_str(), &img.w, &img.h, &n, 3);
    if (!data) {
        YOLO_LOG_ERROR("failed to load image %s: %s", path.c_str(), stbi_failure_reason());
        return false;
    }
    img.c = 3;
    img.rgb.assign(data, data + (size_t)img.w * img.h * 3);
    stbi_image_free(data);
    return true;
}

static std::vector<float> resize_bilinear_float(const float* src, int sw, int sh, int dw, int dh) {
    std::vector<float> dst((size_t)dw * dh);
    const float fx = (float)sw / dw;
    const float fy = (float)sh / dh;
    std::vector<int> x0(dw), x1(dw), y0(dh), y1(dh);
    std::vector<float> wx(dw), wy(dh);
    for (int x = 0; x < dw; x++) {
        const float sx = (x + 0.5f) * fx - 0.5f;
        const int ix = (int)std::floor(sx);
        x0[x] = std::clamp(ix, 0, sw - 1);
        x1[x] = std::clamp(ix + 1, 0, sw - 1);
        wx[x] = sx - ix;
    }
    for (int y = 0; y < dh; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int iy = (int)std::floor(sy);
        y0[y] = std::clamp(iy, 0, sh - 1);
        y1[y] = std::clamp(iy + 1, 0, sh - 1);
        wy[y] = sy - iy;
    }
#if defined(YOLO_USE_OPENMP)
    const int resize_threads = std::min(8, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(resize_threads) if (dh >= 64)
#endif
    for (int y = 0; y < dh; y++) {
        const int yc0 = y0[y], yc1 = y1[y];
        const float wyv = wy[y];
        for (int x = 0; x < dw; x++) {
            const float wxv = wx[x];
            const float v0 = src[(size_t)yc0 * sw + x0[x]] * (1.0f - wxv) +
                             src[(size_t)yc0 * sw + x1[x]] * wxv;
            const float v1 = src[(size_t)yc1 * sw + x0[x]] * (1.0f - wxv) +
                             src[(size_t)yc1 * sw + x1[x]] * wxv;
            dst[(size_t)y * dw + x] = v0 * (1.0f - wyv) + v1 * wyv;
        }
    }
    return dst;
}

// torchvision antialias=True bilinear resize (two-pass, horizontal then
// vertical, float intermediate, uint8 round at the end). Downsample uses the
// ATen area-pixel weights w(i) = 1 - |i - src_idx| * (1/scale) over the support
// [src_idx - scale, src_idx + scale]; upsample (scale <= 1) falls back to the
// plain bilinear w = 1 - |i - src_idx| over 2 taps. src_idx = (dst + 0.5) *
// scale - 0.5 in both modes, matching F.interpolate(align_corners=False).
static void tv_resize_linear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    std::vector<float> tmp((size_t)dw * sh * 3);
    const double scale_x = (double)sw / dw;
    const double inv_x = scale_x > 1.0 ? 1.0 / scale_x : 1.0;
    const double support_x = scale_x > 1.0 ? scale_x : 1.0;
    for (int yy = 0; yy < sh; yy++) {
        for (int xx = 0; xx < dw; xx++) {
            const double src_idx = scale_x * (xx + 0.5) - 0.5;
            const int i0 = std::max(0, (int)std::ceil(src_idx - support_x));
            const int i1 = std::min(sw - 1, (int)std::floor(src_idx + support_x));
            float ww = 0.0f;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            const uint8_t* row = src + (size_t)yy * sw * 3;
            for (int i = i0; i <= i1; i++) {
                const float w = (float)std::max(0.0, 1.0 - std::abs((i - src_idx) * inv_x));
                if (w <= 0.0f) continue;
                for (int c = 0; c < 3; c++) acc[c] += row[(size_t)i * 3 + c] * w;
                ww += w;
            }
            if (ww <= 0.0f) ww = 1.0f;
            float* out = &tmp[((size_t)yy * dw + xx) * 3];
            for (int c = 0; c < 3; c++) out[c] = acc[c] / ww;
        }
    }
    const double scale_y = (double)sh / dh;
    const double inv_y = scale_y > 1.0 ? 1.0 / scale_y : 1.0;
    const double support_y = scale_y > 1.0 ? scale_y : 1.0;
    for (int yy = 0; yy < dh; yy++) {
        const double src_idx = scale_y * (yy + 0.5) - 0.5;
        const int j0 = std::max(0, (int)std::ceil(src_idx - support_y));
        const int j1 = std::min(sh - 1, (int)std::floor(src_idx + support_y));
        for (int xx = 0; xx < dw; xx++) {
            float ww = 0.0f;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int j = j0; j <= j1; j++) {
                const float w = (float)std::max(0.0, 1.0 - std::abs((j - src_idx) * inv_y));
                if (w <= 0.0f) continue;
                const float* px = &tmp[((size_t)j * dw + xx) * 3];
                for (int c = 0; c < 3; c++) acc[c] += px[c] * w;
                ww += w;
            }
            if (ww <= 0.0f) ww = 1.0f;
            uint8_t* out = &dst[((size_t)yy * dw + xx) * 3];
            for (int c = 0; c < 3; c++) out[c] = (uint8_t)std::clamp((int)(acc[c] / ww + 0.5f), 0, 255);
        }
    }
}

void classify_preprocess(const Image& img, int size, std::vector<float>& out) {
    // The released yolo26-cls checkpoint bakes its own transforms: Resize(size,
    // BILINEAR, antialias=True) on the shortest edge, CenterCrop(size), then a
    // plain /255 (ImageNet mean/std are NOT applied).
    const int min_edge = std::min(img.w, img.h);
    // torchvision: new_long = int(size * long / short) — multiply before divide
    // (1080x810 -> int(224*1080/810) = 298, not int(1080*0.2765...) = 223).
    const int new_w = std::max(1, (int)(size * (double)img.w / min_edge));
    const int new_h = std::max(1, (int)(size * (double)img.h / min_edge));

    std::vector<uint8_t> resized((size_t)new_w * new_h * 3);
    tv_resize_linear(img.rgb.data(), img.w, img.h, resized.data(), new_w, new_h);

    const int left = (new_w - size) / 2, top = (new_h - size) / 2;  // floor, torchvision
    const size_t plane = (size_t)size * size;
    out.resize(3 * plane);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            const size_t src = ((size_t)(y + top) * new_w + x + left) * 3;
            for (int c = 0; c < 3; c++) out[(size_t)c * plane + (size_t)y * size + x] = resized[src + c] / 255.0f;
        }
    }
}

void letterbox_image(const Image& img, int imgsz, LetterboxInfo& info, std::vector<float>& out) {
    const float r = std::min((float)imgsz / img.w, (float)imgsz / img.h);
    // nearbyint = round-half-to-even, matching Python round().
    const int new_w = (int)std::nearbyint(img.w * r);
    const int new_h = (int)std::nearbyint(img.h * r);

    // Ultralytics LetterBox(auto=True, center=True): mod stride first, then split padding.
    int dw = (imgsz - new_w) % 32, dh = (imgsz - new_h) % 32;
    const float hw = dw / 2.0f, hh = dh / 2.0f;
    const int left = (int)std::nearbyint(hw - 0.1f), right = (int)std::nearbyint(hw + 0.1f);
    const int top = (int)std::nearbyint(hh - 0.1f), bottom = (int)std::nearbyint(hh + 0.1f);
    const int canvas_w = new_w + left + right;
    const int canvas_h = new_h + top + bottom;

    info = LetterboxInfo{r, left, top, new_w, new_h, canvas_w, canvas_h};

    const size_t plane = (size_t)canvas_w * canvas_h;
    out.resize(3 * plane);
    if (left || right || top || bottom) {
        constexpr float pad = 114.0f / 255.0f;
        for (int c = 0; c < 3; c++) {
            float* channel = out.data() + (size_t)c * plane;
            std::fill(channel, channel + (size_t)top * canvas_w, pad);
            std::fill(channel + (size_t)(top + new_h) * canvas_w, channel + plane, pad);
            for (int y = top; y < top + new_h; y++) {
                float* row = channel + (size_t)y * canvas_w;
                std::fill(row, row + left, pad);
                std::fill(row + left + new_w, row + canvas_w, pad);
            }
        }
    }

    const float fx = (float)img.w / new_w;
    const float fy = (float)img.h / new_h;
    std::vector<int> x0(new_w), x1(new_w);
    std::vector<float> wx(new_w);
    for (int x = 0; x < new_w; x++) {
        // OpenCV INTER_LINEAR sampling: sx = (x + 0.5) * scale - 0.5, pixel-
        // center aligned (verified bit-exact against cv2.resize on bus.jpg).
        const float sx = (x + 0.5f) * fx - 0.5f;
        const int ix0 = (int)std::floor(sx);
        x0[x] = std::clamp(ix0, 0, img.w - 1);
        x1[x] = std::clamp(ix0 + 1, 0, img.w - 1);
        wx[x] = sx - ix0;
    }

#if defined(YOLO_USE_OPENMP)
    const int resize_threads = std::min(8, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(resize_threads) if (new_h >= 64)
#endif
    for (int y = 0; y < new_h; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int iy0 = (int)std::floor(sy);
        const int yc0 = std::clamp(iy0, 0, img.h - 1);
        const int yc1 = std::clamp(iy0 + 1, 0, img.h - 1);
        const float wy = sy - iy0;
        for (int x = 0; x < new_w; x++) {
            const size_t p00 = ((size_t)yc0 * img.w + x0[x]) * 3;
            const size_t p01 = ((size_t)yc0 * img.w + x1[x]) * 3;
            const size_t p10 = ((size_t)yc1 * img.w + x0[x]) * 3;
            const size_t p11 = ((size_t)yc1 * img.w + x1[x]) * 3;
            const size_t dst = (size_t)(y + top) * canvas_w + x + left;
            for (int c = 0; c < 3; c++) {
                const float v0 = img.rgb[p00 + c] + (img.rgb[p01 + c] - img.rgb[p00 + c]) * wx[x];
                const float v1 = img.rgb[p10 + c] + (img.rgb[p11 + c] - img.rgb[p10 + c]) * wx[x];
                const uint8_t value = (uint8_t)(v0 + (v1 - v0) * wy + 0.5f);
                out[(size_t)c * plane + dst] = value / 255.0f;
            }
        }
    }
}

void unscale_boxes(std::vector<Detection>& dets, const LetterboxInfo& info) {
    for (auto& d : dets) {
        d.x1 = (d.x1 - info.pad_w) / info.scale;
        d.y1 = (d.y1 - info.pad_h) / info.scale;
        d.x2 = (d.x2 - info.pad_w) / info.scale;
        d.y2 = (d.y2 - info.pad_h) / info.scale;
    }
}

void unscale_pose(std::vector<PoseDetection>& poses, const LetterboxInfo& info) {
    for (auto& p : poses) {
        p.det.x1 = (p.det.x1 - info.pad_w) / info.scale;
        p.det.y1 = (p.det.y1 - info.pad_h) / info.scale;
        p.det.x2 = (p.det.x2 - info.pad_w) / info.scale;
        p.det.y2 = (p.det.y2 - info.pad_h) / info.scale;
        for (size_t k = 0; k + 1 < p.kpts.size(); k += 2) {
            p.kpts[k] = (p.kpts[k] - info.pad_w) / info.scale;
            p.kpts[k + 1] = (p.kpts[k + 1] - info.pad_h) / info.scale;
        }
    }
}

void unscale_obb(std::vector<OBBDetection>& obbs, const LetterboxInfo& info) {
    for (auto& o : obbs) {
        o.cx = (o.cx - info.pad_w) / info.scale;
        o.cy = (o.cy - info.pad_h) / info.scale;
        o.w /= info.scale;
        o.h /= info.scale;
    }
}

static uint8_t clamp8(int v) { return (uint8_t)std::clamp(v, 0, 255); }

static const uint8_t* glyph_rows(char ch) {
    static constexpr uint8_t glyphs[][7] = {
        {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},  // 0
        {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},  // 1
        {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},  // 2
        {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E},  // 3
        {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},  // 4
        {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E},  // 5
        {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E},  // 6
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},  // 7
        {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},  // 8
        {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E},  // 9
        {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  // A
        {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},  // B
        {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},  // C
        {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E},  // D
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},  // E
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},  // F
        {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},  // G
        {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  // H
        {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},  // I
        {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C},  // J
        {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},  // K
        {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},  // L
        {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},  // M
        {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},  // N
        {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},  // O
        {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},  // P
        {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},  // Q
        {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},  // R
        {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},  // S
        {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},  // T
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},  // U
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},  // V
        {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},  // W
        {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},  // X
        {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},  // Y
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},  // Z
    };
    static constexpr uint8_t blank[7] = {};
    static constexpr uint8_t dot[7] = {0, 0, 0, 0, 0, 0x0C, 0x0C};
    static constexpr uint8_t dash[7] = {0, 0, 0, 0x0E, 0, 0, 0};
    static constexpr uint8_t underscore[7] = {0, 0, 0, 0, 0, 0, 0x1F};
    static constexpr uint8_t question[7] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04};

    unsigned char c = (unsigned char)ch;
    if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    if (c >= '0' && c <= '9') return glyphs[c - '0'];
    if (c >= 'A' && c <= 'Z') return glyphs[10 + c - 'A'];
    if (c == '.') return dot;
    if (c == '-') return dash;
    if (c == '_') return underscore;
    if (c == ' ') return blank;
    return question;
}

bool draw_detections(const std::string& out_path, Image& img,
                     const std::vector<Detection>& dets, const std::vector<std::string>& names,
                     const std::vector<SegMask>* masks, const LetterboxInfo* info) {
    const int t = 2;
    for (size_t k = 0; k < dets.size(); k++) {
        const auto& d = dets[k];
        const int x1 = std::clamp((int)d.x1, 0, img.w - 1), y1 = std::clamp((int)d.y1, 0, img.h - 1);
        const int x2 = std::clamp((int)d.x2, 0, img.w - 1), y2 = std::clamp((int)d.y2, 0, img.h - 1);
        const uint8_t col[3] = {clamp8(d.class_id * 53 + 30), clamp8(220 - d.class_id * 37),
                                clamp8(d.class_id * 91 + 60)};
        if (masks && info && k < masks->size() && (*masks)[k].w > 0 && (*masks)[k].h > 0) {
            // Original pixel -> letterbox canvas -> mask window, nearest lookup.
            // This mirrors ultralytics scale_masks: masks live in canvas space,
            // original pixels map through scale + pad.
            const SegMask& m = (*masks)[k];
            for (int y = y1; y <= y2; y++)
                for (int x = x1; x <= x2; x++) {
                    const int u = std::clamp((int)((x + 0.5f) * info->scale + info->pad_w - m.x), 0, m.w - 1);
                    const int v = std::clamp((int)((y + 0.5f) * info->scale + info->pad_h - m.y), 0, m.h - 1);
                    if (!m.bits[(size_t)v * m.w + u]) continue;
                    for (int c = 0; c < 3; c++) {
                        uint8_t& px = img.rgb[(size_t)(y * img.w + x) * 3 + c];
                        px = (uint8_t)((px + col[c]) >> 1);
                    }
                }
        }
        for (int y = y1; y <= y2; y++)
            for (int x = x1; x <= x2; x++)
                if (x < x1 + t || x >= x2 - t + 1 || y < y1 + t || y >= y2 - t + 1)
                    for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        char label[128];
        const char* cname = d.class_id >= 0 && d.class_id < (int)names.size() ? names[d.class_id].c_str() : "?";
        snprintf(label, sizeof(label), "%s %.2f", cname, d.score);
        constexpr int scale = 2, glyph_h = 7 * scale, advance = 6 * scale, pad = 2;
        constexpr int label_h = 2 * pad + glyph_h;
        const int available = img.w - x1 - 2 * pad;
        const size_t visible_chars = std::min(std::strlen(label), (size_t)std::max(0, (available + scale) / advance));
        if (visible_chars == 0 || img.h < label_h) continue;
        const int label_w = 2 * pad + (int)visible_chars * advance - scale;
        const int ly = y1 >= label_h ? y1 - label_h : std::min(y1 + t, img.h - label_h);
        for (int y = ly; y < ly + label_h; y++)
            for (int x = x1; x < x1 + label_w; x++)
                for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        for (size_t i = 0; i < visible_chars; i++) {
            const uint8_t* glyph = glyph_rows(label[i]);
            const int ox = x1 + pad + (int)i * advance, oy = ly + pad;
            for (int gy = 0; gy < 7; gy++)
                for (int gx = 0; gx < 5; gx++)
                    if (glyph[gy] & (1 << (4 - gx)))
                        for (int sy = 0; sy < scale; sy++)
                            for (int sx = 0; sx < scale; sx++)
                                for (int c = 0; c < 3; c++)
                                    img.rgb[(size_t)((oy + gy * scale + sy) * img.w + ox + gx * scale + sx) * 3 + c] =
                                        255;
        }
    }
    return stbi_write_png(out_path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

// COCO-17 skeleton edges (1-indexed pairs from ultralytics pose.yaml, minus 1).
static const int kpt_edges[][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                                   {5, 6},  {5, 7},  {6, 8},  {7, 9},  {8, 10}, {1, 2},  {0, 1},
                                   {0, 2},  {1, 3},  {2, 4},  {3, 5},  {4, 6}};

static void draw_line(Image& img, float x0, float y0, float x1, float y1, const uint8_t col[3], int t = 2) {
    const int dx = std::abs((int)x1 - (int)x0), dy = std::abs((int)y1 - (int)y0);
    const int steps = std::max(dx, dy);
    if (steps == 0) {
        for (int sy = -t / 2; sy <= t / 2; sy++)
            for (int sx = -t / 2; sx <= t / 2; sx++) {
                const int px = (int)x0 + sx, py = (int)y0 + sy;
                if (px < 0 || py < 0 || px >= img.w || py >= img.h) continue;
                for (int c = 0; c < 3; c++) img.rgb[(size_t)(py * img.w + px) * 3 + c] = col[c];
            }
        return;
    }
    for (int i = 0; i <= steps; i++) {
        const float x = x0 + (x1 - x0) * i / steps;
        const float y = y0 + (y1 - y0) * i / steps;
        for (int sy = -t / 2; sy <= t / 2; sy++)
            for (int sx = -t / 2; sx <= t / 2; sx++) {
                const int px = (int)x + sx, py = (int)y + sy;
                if (px < 0 || py < 0 || px >= img.w || py >= img.h) continue;
                for (int c = 0; c < 3; c++) img.rgb[(size_t)(py * img.w + px) * 3 + c] = col[c];
            }
    }
}

bool draw_pose(const std::string& out_path, Image& img, const std::vector<PoseDetection>& poses,
               const std::vector<std::string>& names) {
    const int nkpt = 17;  // COCO-17 is the only shipped pose family
    for (const auto& p : poses) {
        const auto& d = p.det;
        const uint8_t col[3] = {clamp8(d.class_id * 53 + 30), clamp8(220 - d.class_id * 37),
                                clamp8(d.class_id * 91 + 60)};
        const int x1 = std::clamp((int)d.x1, 0, img.w - 1), y1 = std::clamp((int)d.y1, 0, img.h - 1);
        const int x2 = std::clamp((int)d.x2, 0, img.w - 1), y2 = std::clamp((int)d.y2, 0, img.h - 1);
        for (int y = y1; y <= y2; y++)
            for (int x = x1; x <= x2; x++)
                if (x < x1 + 2 || x >= x2 - 1 || y < y1 + 2 || y >= y2 - 1)
                    for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        if ((int)p.kpts.size() >= nkpt * 3) {
            for (const auto& e : kpt_edges) {
                const float v0 = p.kpts[e[0] * 3 + 2], v1 = p.kpts[e[1] * 3 + 2];
                if (v0 < 0.5f || v1 < 0.5f) continue;  // invisible endpoints
                draw_line(img, p.kpts[e[0] * 3], p.kpts[e[0] * 3 + 1], p.kpts[e[1] * 3], p.kpts[e[1] * 3 + 1],
                          col);
            }
            const uint8_t dot[3] = {255, 255, 255};
            for (int k = 0; k < nkpt; k++) {
                if (p.kpts[k * 3 + 2] < 0.5f) continue;
                draw_line(img, p.kpts[k * 3], p.kpts[k * 3 + 1], p.kpts[k * 3], p.kpts[k * 3 + 1], dot, 3);
            }
        }
        const char* cname = d.class_id >= 0 && d.class_id < (int)names.size() ? names[d.class_id].c_str() : "?";
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", cname, d.score);
        constexpr int scale = 2, glyph_h = 7 * scale, advance = 6 * scale, pad = 2;
        constexpr int label_h = 2 * pad + glyph_h;
        const size_t visible_chars = std::min(std::strlen(label),
                                               (size_t)std::max(0, (img.w - x1 - 2 * pad + scale) / advance));
        if (visible_chars == 0 || img.h < label_h) continue;
        const int label_w = 2 * pad + (int)visible_chars * advance - scale;
        const int ly = y1 >= label_h ? y1 - label_h : std::min(y1 + 2, img.h - label_h);
        for (int y = ly; y < ly + label_h; y++)
            for (int x = x1; x < x1 + label_w; x++)
                for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        for (size_t i = 0; i < visible_chars; i++) {
            const uint8_t* glyph = glyph_rows(label[i]);
            const int ox = x1 + pad + (int)i * advance, oy = ly + pad;
            for (int gy = 0; gy < 7; gy++)
                for (int gx = 0; gx < 5; gx++)
                    if (glyph[gy] & (1 << (4 - gx)))
                        for (int sy = 0; sy < scale; sy++)
                            for (int sx = 0; sx < scale; sx++)
                                for (int c = 0; c < 3; c++)
                                    img.rgb[(size_t)((oy + gy * scale + sy) * img.w + ox + gx * scale + sx) * 3 + c] =
                                        255;
        }
    }
    return stbi_write_png(out_path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

bool draw_obb(const std::string& out_path, Image& img, const std::vector<OBBDetection>& obbs,
              const std::vector<std::string>& names) {
    for (const auto& o : obbs) {
        const uint8_t col[3] = {clamp8(o.class_id * 53 + 30), clamp8(220 - o.class_id * 37),
                                clamp8(o.class_id * 91 + 60)};
        const float cos_a = std::cos(o.angle), sin_a = std::sin(o.angle);
        const float hw = o.w * 0.5f, hh = o.h * 0.5f;
        const float c[4][2] = {{o.cx + hw * cos_a - hh * sin_a, o.cy + hw * sin_a + hh * cos_a},
                               {o.cx - hw * cos_a - hh * sin_a, o.cy - hw * sin_a + hh * cos_a},
                               {o.cx - hw * cos_a + hh * sin_a, o.cy - hw * sin_a - hh * cos_a},
                               {o.cx + hw * cos_a + hh * sin_a, o.cy + hw * sin_a - hh * cos_a}};
        for (int i = 0; i < 4; i++) draw_line(img, c[i][0], c[i][1], c[(i + 1) % 4][0], c[(i + 1) % 4][1], col);
        draw_line(img, o.cx, o.cy, o.cx, o.cy, col, 4);  // center mark
        const char* cname = o.class_id >= 0 && o.class_id < (int)names.size() ? names[o.class_id].c_str() : "?";
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f %.0fdeg", cname, o.score, o.angle * 180.0f / 3.14159265f);
        constexpr int scale = 2, glyph_h = 7 * scale, advance = 6 * scale, pad = 2;
        constexpr int label_h = 2 * pad + glyph_h;
        const int x1 = std::clamp((int)o.cx - 30, 0, img.w - 1);
        const size_t visible_chars = std::min(std::strlen(label),
                                               (size_t)std::max(0, (img.w - x1 - 2 * pad + scale) / advance));
        if (visible_chars == 0 || img.h < label_h) continue;
        const int label_w = 2 * pad + (int)visible_chars * advance - scale;
        const int ly = std::max((int)o.cy - 30 - label_h, 0);
        for (int y = ly; y < ly + label_h; y++)
            for (int x = x1; x < x1 + label_w; x++)
                for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        for (size_t i = 0; i < visible_chars; i++) {
            const uint8_t* glyph = glyph_rows(label[i]);
            const int ox = x1 + pad + (int)i * advance, oy = ly + pad;
            for (int gy = 0; gy < 7; gy++)
                for (int gx = 0; gx < 5; gx++)
                    if (glyph[gy] & (1 << (4 - gx)))
                        for (int sy = 0; sy < scale; sy++)
                            for (int sx = 0; sx < scale; sx++)
                                for (int c = 0; c < 3; c++)
                                    img.rgb[(size_t)((oy + gy * scale + sy) * img.w + ox + gx * scale + sx) * 3 + c] =
                                        255;
        }
    }
    return stbi_write_png(out_path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

bool draw_semantic(const std::string& out_path, Image& img, const std::vector<uint8_t>& classes,
                   int grid_w, int grid_h, int nc) {
    if ((int)classes.size() != grid_w * grid_h || grid_w <= 0 || grid_h <= 0) return false;
    // Nearest upsample of the canvas/8 grid onto the original image, then 40%
    // alpha blend of the per-class color. Palette: Cityscapes 19-class colors
    // (first 19 rows), then a deterministic pseudo-color for extra classes.
    // The old arithmetic palette (cls*53+30, ...) overflowed clamp8 for
    // cls>=4, collapsing distinct classes (e.g. person and train) onto the
    // same color.
    static const uint8_t kPalette[][3] = {
        {128, 64, 128}, {244, 35, 232}, {70, 70, 70},   {102, 102, 156}, {190, 153, 153},
        {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35},  {152, 251, 152},
        {70, 130, 180}, {220, 20, 60},  {255, 0, 0},   {0, 0, 142},     {0, 0, 70},
        {0, 60, 100},   {0, 80, 100},   {0, 0, 230},   {119, 11, 32},
    };
    const size_t n_pal = sizeof(kPalette) / sizeof(kPalette[0]);
    const float fx = (float)grid_w / img.w, fy = (float)grid_h / img.h;
    for (int y = 0; y < img.h; y++) {
        const int gy = std::clamp((int)((y + 0.5f) * fy), 0, grid_h - 1);
        for (int x = 0; x < img.w; x++) {
            const int gx = std::clamp((int)((x + 0.5f) * fx), 0, grid_w - 1);
            const uint8_t cls = classes[(size_t)gy * grid_w + gx];
            if (cls >= nc) continue;
            uint8_t col[3];
            if (cls < n_pal) {
                col[0] = kPalette[cls][0];
                col[1] = kPalette[cls][1];
                col[2] = kPalette[cls][2];
            } else {
                col[0] = clamp8(cls * 53 + 30);
                col[1] = clamp8(220 - cls * 37);
                col[2] = clamp8(cls * 91 + 60);
            }
            uint8_t* px = &img.rgb[(size_t)(y * img.w + x) * 3];
            for (int c = 0; c < 3; c++) px[c] = (uint8_t)((px[c] * 3 + col[c]) / 4);
        }
    }
    return stbi_write_png(out_path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

std::vector<float> restore_depth(const std::vector<float>& depth, int depth_w, int depth_h,
                                 const LetterboxInfo& info, int image_w, int image_h) {
    if ((int)depth.size() != depth_w * depth_h || depth_w <= 0 || depth_h <= 0 || image_w <= 0 || image_h <= 0) {
        return {};
    }
    std::vector<float> canvas = resize_bilinear_float(depth.data(), depth_w, depth_h, info.imgsz_w, info.imgsz_h);
    std::vector<float> crop((size_t)info.new_w * info.new_h);
    for (int y = 0; y < info.new_h; y++) {
        memcpy(crop.data() + (size_t)y * info.new_w,
               canvas.data() + (size_t)(y + info.pad_h) * info.imgsz_w + info.pad_w,
               (size_t)info.new_w * sizeof(float));
    }
    return resize_bilinear_float(crop.data(), info.new_w, info.new_h, image_w, image_h);
}

bool write_depth_png(const std::string& out_path, const std::vector<float>& depth, int width, int height,
                     float max_depth) {
    if ((int)depth.size() != width * height || width <= 0 || height <= 0) return false;
    std::vector<float> valid;
    valid.reserve(depth.size());
    for (float v : depth)
        if (std::isfinite(v) && v > 0.0f) valid.push_back(v);
    if (valid.empty()) return false;
    const float min_depth = *std::min_element(valid.begin(), valid.end());
    if (!(max_depth > min_depth)) {
        const size_t p95 = std::min(valid.size() - 1, valid.size() * 95 / 100);
        std::nth_element(valid.begin(), valid.begin() + p95, valid.end());
        max_depth = valid[p95];
    }
    const float scale = 1.0f / std::max(max_depth - min_depth, 1e-6f);
    std::vector<uint8_t> rgb(depth.size() * 3);
    for (size_t i = 0; i < depth.size(); i++) {
        const float t = std::clamp((depth[i] - min_depth) * scale, 0.0f, 1.0f);
        const float r = std::clamp(1.5f - std::fabs(4.0f * t - 3.0f), 0.0f, 1.0f);
        const float g = std::clamp(1.5f - std::fabs(4.0f * t - 2.0f), 0.0f, 1.0f);
        const float b = std::clamp(1.5f - std::fabs(4.0f * t - 1.0f), 0.0f, 1.0f);
        rgb[i * 3 + 0] = (uint8_t)std::nearbyint(r * 255.0f);
        rgb[i * 3 + 1] = (uint8_t)std::nearbyint(g * 255.0f);
        rgb[i * 3 + 2] = (uint8_t)std::nearbyint(b * 255.0f);
    }
    return stbi_write_png(out_path.c_str(), width, height, 3, rgb.data(), width * 3) != 0;
}

}  // namespace yolo
