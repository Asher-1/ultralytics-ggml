// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "common.hpp"
#include "postprocess.hpp"

#include <string>
#include <vector>

namespace yolo {

struct Image {
    int w = 0, h = 0, c = 3;
    std::vector<uint8_t> rgb;  // interleaved RGB8
};

// Load an image (jpg/png/bmp). Returns false on failure.
bool load_image(const std::string& path, Image& img);

// Ultralytics-equivalent LetterBox(auto=True, stride=32): resize keeping aspect
// then pad to a stride-multiple rectangle inside imgsz x imgsz. Bilinear
// resampling matches cv2.INTER_LINEAR bit-for-bit.
void letterbox_image(const Image& img, int imgsz, LetterboxInfo& info, std::vector<float>& out);

// Classification preprocessing (torchvision classify_transforms): resize the
// shortest edge to `size`, center-crop size x size, then normalize with the
// ImageNet mean/std used by every ultralytics classify checkpoint. Output is
// the same CHW F32 layout as letterbox_image (0-1 scale is NOT applied).
void classify_preprocess(const Image& img, int size, std::vector<float>& out);

// Map boxes from the letterboxed canvas back to original image pixels.
void unscale_boxes(std::vector<Detection>& dets, const LetterboxInfo& info);

// Map pose keypoints from the letterboxed canvas back to original pixels
// (same scale/pad transform as unscale_boxes; visibility dims untouched).
void unscale_pose(std::vector<PoseDetection>& poses, const LetterboxInfo& info);

// Map OBB centers/extents back to original image pixels (angle unchanged).
void unscale_obb(std::vector<OBBDetection>& obbs, const LetterboxInfo& info);

// Draw detections onto the image (in place) and write a PNG. For segment
// models pass the compose_masks output plus the letterbox info: each mask is
// alpha-blended (50%) over its box before the outline and label are drawn.
bool draw_detections(const std::string& out_path, Image& img,
                     const std::vector<Detection>& dets, const std::vector<std::string>& names,
                     const std::vector<SegMask>* masks = nullptr, const LetterboxInfo* info = nullptr);

// Draw pose results: COCO-17 skeleton lines between visible keypoints, keypoint
// dots, and the detection box/label (keypoints must be in original pixels).
bool draw_pose(const std::string& out_path, Image& img, const std::vector<PoseDetection>& poses,
               const std::vector<std::string>& names);

// Draw oriented boxes as rotated rectangles with a center mark and label
// (coordinates must be in original pixels).
bool draw_obb(const std::string& out_path, Image& img, const std::vector<OBBDetection>& obbs,
              const std::vector<std::string>& names);

// Upsample a canvas/8 argmax class map to the original image size (nearest)
// and alpha-blend it over the image before writing the PNG.
bool draw_semantic(const std::string& out_path, Image& img, const std::vector<uint8_t>& classes,
                   int grid_w, int grid_h, int nc);

// Restore a model-resolution depth map to the original image size, matching
// DepthPredictor's bilinear resize, letterbox crop, and final resize.
std::vector<float> restore_depth(const std::vector<float>& depth, int depth_w, int depth_h,
                                 const LetterboxInfo& info, int image_w, int image_h);

// Write a colorized depth preview. The float depth values remain in meters;
// the PNG is display-only and scaled to max_depth (or the 95th percentile).
bool write_depth_png(const std::string& out_path, const std::vector<float>& depth, int width, int height,
                     float max_depth = 0.0f);

}  // namespace yolo
