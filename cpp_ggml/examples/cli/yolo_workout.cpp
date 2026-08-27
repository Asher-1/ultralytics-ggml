// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo-workout — official "Workout Monitoring" solution scenario
// (https://docs.ultralytics.com/solutions/workout-monitoring) on the ggml C++
// stack. Runs a pose model (COCO-17 keypoints), computes the elbow and knee
// joint angles for every person, classifies the pose stage, and renders the
// skeleton overlay.
//
// Usage:
//   yolo-workout --model yolo26n-pose-f16.gguf --source IMG \
//       [--conf 0.25] [--out out.png] [--threads N]
//
// Angle convention: elbow = shoulder-elbow-wrist, knee = hip-knee-ankle;
// a joint is used only when visible (kpt visibility > 0.5 for ndim == 3).

#include "examples_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// COCO-17 keypoint indices used by the workout scenarios.
constexpr int kLeftShoulder = 5, kRightShoulder = 6;
constexpr int kLeftElbow = 7, kRightElbow = 8;
constexpr int kLeftWrist = 9, kRightWrist = 10;
constexpr int kLeftHip = 11, kRightHip = 12;
constexpr int kLeftKnee = 13, kRightKnee = 14;
constexpr int kLeftAnkle = 15, kRightAnkle = 16;

// Angle at vertex b of the a-b-c triangle, in degrees. Returns NaN when any
// keypoint is missing or (for ndim == 3) not visible.
float joint_angle(const yolo::PoseDetection& p, int ndim, int a, int b, int c) {
    if ((int)p.kpts.size() < (c + 1) * ndim) return NAN;
    if (ndim == 3) {
        for (int k : {a, b, c}) {
            if (p.kpts[k * ndim + 2] < 0.5f) return NAN;
        }
    }
    const float ax = p.kpts[a * ndim], ay = p.kpts[a * ndim + 1];
    const float bx = p.kpts[b * ndim], by = p.kpts[b * ndim + 1];
    const float cx = p.kpts[c * ndim], cy = p.kpts[c * ndim + 1];
    const float ux = ax - bx, uy = ay - by;
    const float vx = cx - bx, vy = cy - by;
    const float dot = ux * vx + uy * vy;
    const float len = std::sqrt(ux * ux + uy * uy) * std::sqrt(vx * vx + vy * vy);
    if (len <= 0.0f) return NAN;
    const float cos_a = std::clamp(dot / len, -1.0f, 1.0f);
    return std::acos(cos_a) * 180.0f / (float)M_PI;
}

// Stage label for a squat-style knee bend / pushup-style elbow bend.
const char* stage_label(float left, float right) {
    const float bend = std::min(left, right);
    if (bend < 90.0f) return "bent (deep)";
    if (bend < 150.0f) return "bent (partial)";
    return "straight";
}

}  // namespace

int main(int argc, char** argv) {
    const yolo_examples::Args args = yolo_examples::parse_args(argc, argv);
    const std::string model = yolo_examples::arg_s(args, "model");
    const std::string source = yolo_examples::arg_s(args, "source");
    if (model.empty() || source.empty()) {
        fprintf(stderr, "usage: yolo-workout --model M-pose.gguf --source IMG "
                        "[--conf F] [--out OUT.png] [--threads N]\n");
        return 1;
    }

    yolo_examples::Options opt;
    opt.conf = yolo_examples::arg_f(args, "conf", 0.25f);
    opt.threads = yolo_examples::arg_i(args, "threads", 0);

    yolo::Image img;
    std::vector<yolo::PoseDetection> poses;
    if (!yolo_examples::run_pose(model, source, opt, img, poses)) return 1;
    const int ndim = 3;  // yolo26-pose checkpoints ship kpt_shape [17, 3]

    printf("workout monitoring on %dx%d image: %zu person%s detected\n", img.w, img.h, poses.size(),
           poses.size() == 1 ? "" : "s");
    for (size_t i = 0; i < poses.size(); i++) {
        const auto& p = poses[i];
        const float lelbow = joint_angle(p, ndim, kLeftShoulder, kLeftElbow, kLeftWrist);
        const float relbow = joint_angle(p, ndim, kRightShoulder, kRightElbow, kRightWrist);
        const float lknee = joint_angle(p, ndim, kLeftHip, kLeftKnee, kLeftAnkle);
        const float rknee = joint_angle(p, ndim, kRightHip, kRightKnee, kRightAnkle);
        printf("  person %zu (score %.2f):\n", i + 1, p.det.score);
        if (!std::isnan(lelbow) || !std::isnan(relbow))
            printf("    elbows   left %5.1f deg  right %5.1f deg  -> %s\n",
                   std::isnan(lelbow) ? NAN : lelbow, std::isnan(relbow) ? NAN : relbow,
                   stage_label(std::isnan(lelbow) ? relbow : lelbow, std::isnan(relbow) ? lelbow : relbow));
        if (!std::isnan(lknee) || !std::isnan(rknee))
            printf("    knees    left %5.1f deg  right %5.1f deg  -> %s\n",
                   std::isnan(lknee) ? NAN : lknee, std::isnan(rknee) ? NAN : rknee,
                   stage_label(std::isnan(lknee) ? rknee : lknee, std::isnan(rknee) ? lknee : rknee));
        if (std::isnan(lelbow) && std::isnan(relbow) && std::isnan(lknee) && std::isnan(rknee))
            printf("    no visible joints for angle estimation\n");
    }

    const std::string out = yolo_examples::arg_s(args, "out");
    if (!out.empty() && !yolo::draw_pose(out, img, poses, {"person"})) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}
