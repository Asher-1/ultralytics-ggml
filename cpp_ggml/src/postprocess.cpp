// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace yolo {

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Exclusive-boundary IoU, identical to torchvision.ops.nms / box_iou.
inline float box_iou(const Detection& a, const Detection& b) {
    const float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, xx2 - xx1), h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float u = area_a + area_b - inter;
    return u > 0.0f ? inter / u : 0.0f;
}

struct Cand {
    int a;
    float score;
    int cls;
};

}  // namespace

std::vector<Detection> postprocess(const std::vector<float>& raw, int no, int na, const ModelMeta& meta,
                                   const float* anchors, const float* strides, const PostprocConfig& cfg) {
    const int nc = meta.nc;
    const int rm = meta.reg_max;
    const int box_ch = 4 * rm;
    if (no < box_ch + nc) {
        YOLO_LOG_ERROR("output channels %d < 4*reg_max + nc (%d)", no, box_ch + nc);
        return {};
    }

    // Per-anchor best class, confidence filtered (shared by both heads). sigmoid is
    // monotonic, so argmax and the conf filter run on the raw logits — sigmoid itself
    // is applied only to the few surviving candidates. The class loop is outer so the
    // anchor walk streams raw[c] contiguously instead of hopping na*4B per class.
    const float logit_thr = -std::log(1.0f / cfg.conf_thres - 1.0f);
    const float* cls_base = raw.data() + (size_t)box_ch * na;
    std::vector<Cand> cands;
    cands.reserve(na / 8);
    if (nc > 1) {
        std::vector<float> best(na, -INFINITY);
        std::vector<int> bc(na, 0);
        for (int c = 0; c < nc; c++) {
            const float* row = cls_base + (size_t)c * na;
            for (int a = 0; a < na; a++) {
                if (row[a] > best[a]) {
                    best[a] = row[a];
                    bc[a] = c;
                }
            }
        }
        for (int a = 0; a < na; a++) {
            if (best[a] > logit_thr) cands.push_back({a, sigmoid(best[a]), bc[a]});
        }
    } else {
        for (int a = 0; a < na; a++) {
            if (cls_base[a] > logit_thr) cands.push_back({a, sigmoid(cls_base[a]), 0});
        }
    }

    // Cap the NMS input by confidence like ultralytics max_nms.
    if ((int)cands.size() > cfg.max_nms) {
        std::partial_sort(cands.begin(), cands.begin() + cfg.max_nms, cands.end(),
                          [](const Cand& x, const Cand& y) { return x.score > y.score; });
        cands.resize(cfg.max_nms);
    }

    // DFL softmax scratch (reg_max may be any size, e.g. 16 for v8, 1 for yolo26).
    std::vector<float> probs(rm > 0 ? rm : 1);

    std::vector<Detection> dets;
    dets.reserve(cands.size());
    for (const Cand& cd : cands) {
        const int a = cd.a;
        const float ax = anchors[2 * a], ay = anchors[2 * a + 1], st = strides[a];
        float d[4];
        if (rm > 1) {
            // torch DFL: x.view(B, 4, reg_max, A) — channel k = j*rm + t (edge j outer, bin t inner).
            for (int j = 0; j < 4; j++) {
                float m = -INFINITY;
                for (int t = 0; t < rm; t++) m = std::max(m, raw[(size_t)(j * rm + t) * na + a]);
                float sum = 0.0f, val = 0.0f;
                for (int t = 0; t < rm; t++) {
                    probs[t] = std::exp(raw[(size_t)(j * rm + t) * na + a] - m);
                    sum += probs[t];
                }
                for (int t = 0; t < rm; t++) val += probs[t] * t;
                d[j] = val / sum;
            }
        } else {
            for (int j = 0; j < 4; j++) d[j] = raw[(size_t)j * na + a];
        }

        if (meta.end2end) {
            // dist2bbox xyxy: anchor -/+ dist, then scaled by stride.
            dets.push_back({(ax - d[0]) * st, (ay - d[1]) * st, (ax + d[2]) * st, (ay + d[3]) * st, cd.score, cd.cls,
                            cd.a});
        } else {
            // dist2bbox xywh: center = (lt + rb) / 2, wh = lt + rb, scaled by stride.
            const float x1 = ax - d[0], y1 = ay - d[1], x2 = ax + d[2], y2 = ay + d[3];
            const float cx = (x1 + x2) * 0.5f * st, cy = (y1 + y2) * 0.5f * st;
            const float w = (x2 - x1) * st, h = (y2 - y1) * st;
            dets.push_back({cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, cd.score, cd.cls, cd.a});
        }
    }

    if (meta.end2end) {
        // Detect.postprocess topk (score descending) then the conf filter in the
        // non_max_suppression end2end branch keeps the order; cap at max_det.
        std::stable_sort(dets.begin(), dets.end(),
                         [](const Detection& x, const Detection& y) { return x.score > y.score; });
        if ((int)dets.size() > cfg.max_det) dets.resize(cfg.max_det);
        return dets;
    }

    // Greedy class-aware NMS, matching torchvision.ops.nms with the class offset
    // trick: only same-class boxes can suppress each other, IoU > threshold.
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int x, int y) { return dets[x].score > dets[y].score; });
    std::vector<int> keep;
    keep.reserve(std::min((size_t)cfg.max_det, dets.size()));
    for (int i : order) {
        bool suppressed = false;
        for (int k : keep) {
            if (dets[k].class_id == dets[i].class_id && box_iou(dets[k], dets[i]) > cfg.iou_thres) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(i);
        if ((int)keep.size() == cfg.max_det) break;
    }

    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int k : keep) out.push_back(dets[k]);
    return out;
}

std::vector<SegMask> compose_masks(const std::vector<Detection>& dets, const std::vector<float>& raw, int na,
                                   const ModelMeta& meta, const std::vector<float>& proto, int proto_w, int proto_h,
                                   int canvas_w, int canvas_h) {
    std::vector<SegMask> masks(dets.size());
    const int nm = meta.nm;
    if (nm <= 0 || (int)proto.size() != nm * proto_w * proto_h || canvas_w <= 0 || canvas_h <= 0) return masks;

    const int mc_base = 4 * meta.reg_max + meta.nc;  // mask-coefficient channel start
    for (size_t k = 0; k < dets.size(); k++) {
        const Detection& d = dets[k];
        if (d.anchor < 0 || d.anchor >= na) continue;

        // Canvas-space window (ints, clamped): the mask canvas extent. The crop
        // window on the proto grid follows ultralytics crop_mask semantics —
        // proto cell px survives iff px >= ceil(bx1) and px < ceil(bx2) with the
        // box mapped onto the proto grid — so edges land on the same cells.
        const int x1 = std::clamp((int)std::floor(d.x1), 0, canvas_w - 1);
        const int y1 = std::clamp((int)std::floor(d.y1), 0, canvas_h - 1);
        const int x2 = std::clamp((int)std::ceil(d.x2), x1 + 1, canvas_w);
        const int y2 = std::clamp((int)std::ceil(d.y2), y1 + 1, canvas_h);
        const int px1 = std::clamp((int)std::ceil(d.x1 * proto_w / (float)canvas_w), 0, proto_w);
        const int px2 = std::clamp((int)std::ceil(d.x2 * proto_w / (float)canvas_w), 0, proto_w);
        const int py1 = std::clamp((int)std::ceil(d.y1 * proto_h / (float)canvas_h), 0, proto_h);
        const int py2 = std::clamp((int)std::ceil(d.y2 * proto_h / (float)canvas_h), 0, proto_h);

        SegMask& m = masks[k];
        m.x = x1;
        m.y = y1;
        m.w = x2 - x1;
        m.h = y2 - y1;
        m.bits.assign((size_t)m.w * m.h, 0);
        if (px2 <= px1 || py2 <= py1) continue;  // box degenerate on the proto grid

        float mc[64];  // mask coefficients; nm is 32 for every shipped seg model
        if (nm > 64) {
            m.w = m.h = 0;
            continue;
        }
        for (int c = 0; c < nm; c++) mc[c] = raw[(size_t)(mc_base + c) * na + d.anchor];

        // Evaluate the nm-dot product once per proto cell, then splat the bit
        // over the cell's canvas pixel block (4x4 at 640/160). Every pixel of a
        // block reads the same cell, so this is bit-identical to the naive
        // per-pixel walk at 1/16th the arithmetic.
        for (int py = py1; py < py2; py++) {
            const int by0 = std::max((int)std::ceil((float)py * canvas_h / proto_h), y1);
            const int by1 = std::min((int)std::ceil((float)(py + 1) * canvas_h / proto_h), y2);
            for (int px = px1; px < px2; px++) {
                const float* pc = proto.data() + (size_t)py * proto_w + px;
                float v = 0.0f;
                for (int c = 0; c < nm; c++) v += mc[c] * pc[(size_t)c * proto_w * proto_h];
                if (sigmoid(v) <= 0.5f) continue;
                const int bx0 = std::max((int)std::ceil((float)px * canvas_w / proto_w), x1);
                const int bx1 = std::min((int)std::ceil((float)(px + 1) * canvas_w / proto_w), x2);
                for (int y = by0; y < by1; y++) {
                    memset(&m.bits[(size_t)(y - y1) * m.w + (bx0 - x1)], 1, (size_t)(bx1 - bx0));
                }
            }
        }
    }
    return masks;
}

}  // namespace yolo
