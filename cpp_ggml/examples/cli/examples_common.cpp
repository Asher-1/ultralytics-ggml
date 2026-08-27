// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
// Implementation of the shared scenario-example pipelines. Each pipeline
// mirrors the corresponding yolo-cli command path in src/cli.cpp, which is
// the parity-verified reference.

#include "examples_common.hpp"

namespace yolo_examples {

bool run_letterbox(const std::string& gguf, const yolo::Image& img, const Options& opt, SessionPtr& sess,
                   yolo::LetterboxInfo& info, std::vector<float>& input) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.imgsz <= 0) return false;
    yolo::letterbox_image(img, meta.imgsz, info, input);
    yolo::SessionOptions sopts;
    sopts.threads = opt.threads;
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sess.reset(yolo::create_session(gguf, sopts));
    if (!sess) return false;
    return yolo::session_run(sess.get(), input.data());
}

bool run_boxes(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
               std::vector<yolo::Detection>& dets, std::vector<yolo::SegMask>& masks, yolo::LetterboxInfo& info,
               std::vector<std::string>& names) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.task != "detect" && meta.task != "segment") {
        fprintf(stderr, "run_boxes requires a detect or segment model, got task=%s\n", meta.task.c_str());
        return false;
    }
    if (!yolo::load_image(source, img)) return false;
    SessionPtr sess(nullptr, yolo::free_session);
    std::vector<float> input;
    if (!run_letterbox(gguf, img, opt, sess, info, input)) return false;
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(sess.get(), raw, no, na)) return false;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = opt.conf;
    cfg.iou_thres = opt.iou;
    cfg.max_det = opt.max_det;
    dets = yolo::postprocess(raw, no, na, sess->model.meta, sess->anchors.data(), sess->anchor_strides.data(), cfg);
    masks.clear();
    if (meta.task == "segment") {
        std::vector<float> proto;
        int nm = 0, pw = 0, ph = 0;
        if (!yolo::session_read_proto(sess.get(), proto, nm, pw, ph)) return false;
        masks = yolo::compose_masks(dets, raw, na, sess->model.meta, proto, pw, ph, info.imgsz_w, info.imgsz_h);
    }
    yolo::unscale_boxes(dets, info);
    names = sess->model.meta.class_names;
    return true;
}

bool run_pose(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
              std::vector<yolo::PoseDetection>& poses) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.task != "pose") {
        fprintf(stderr, "run_pose requires a pose model, got task=%s\n", meta.task.c_str());
        return false;
    }
    if (!yolo::load_image(source, img)) return false;
    SessionPtr sess(nullptr, yolo::free_session);
    yolo::LetterboxInfo info;
    std::vector<float> input;
    if (!run_letterbox(gguf, img, opt, sess, info, input)) return false;
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(sess.get(), raw, no, na)) return false;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = opt.conf;
    cfg.iou_thres = opt.iou;
    cfg.max_det = opt.max_det;
    poses = yolo::postprocess_pose(raw, no, na, sess->model.meta, sess->anchors.data(), sess->anchor_strides.data(), cfg);
    yolo::unscale_pose(poses, info);
    return true;
}

bool run_obb(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
             std::vector<yolo::OBBDetection>& obbs) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.task != "obb") {
        fprintf(stderr, "run_obb requires an obb model, got task=%s\n", meta.task.c_str());
        return false;
    }
    if (!yolo::load_image(source, img)) return false;
    SessionPtr sess(nullptr, yolo::free_session);
    yolo::LetterboxInfo info;
    std::vector<float> input;
    if (!run_letterbox(gguf, img, opt, sess, info, input)) return false;
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(sess.get(), raw, no, na)) return false;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = opt.conf;
    cfg.iou_thres = opt.iou;
    cfg.max_det = opt.max_det;
    obbs = yolo::postprocess_obb(raw, no, na, sess->model.meta, sess->anchors.data(), sess->anchor_strides.data(), cfg);
    yolo::unscale_obb(obbs, info);
    return true;
}

bool run_semantic(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
                  std::vector<uint8_t>& classes, int& gw, int& gh, std::vector<std::string>& names) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.task != "semantic") {
        fprintf(stderr, "run_semantic requires a semantic model, got task=%s\n", meta.task.c_str());
        return false;
    }
    if (!yolo::load_image(source, img)) return false;
    SessionPtr sess(nullptr, yolo::free_session);
    yolo::LetterboxInfo info;
    std::vector<float> input;
    if (!run_letterbox(gguf, img, opt, sess, info, input)) return false;
    std::vector<float> logits;
    int nc = 0;
    if (!yolo::session_read_semantic(sess.get(), logits, nc, gw, gh)) return false;
    classes = yolo::semantic_argmax(logits, nc, gw, gh);
    names = sess->model.meta.class_names;
    return true;
}

bool run_classify(const std::string& gguf, const std::string& source, const Options& opt,
                  std::vector<float>& probs, std::vector<std::string>& names) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.task != "classify") {
        fprintf(stderr, "run_classify requires a classify model, got task=%s\n", meta.task.c_str());
        return false;
    }
    yolo::Image img;
    if (!yolo::load_image(source, img)) return false;
    // Classification uses resize + center-crop (checkpoint transforms), not letterbox.
    std::vector<float> input;
    yolo::classify_preprocess(img, meta.imgsz, input);
    yolo::SessionOptions sopts;
    sopts.threads = opt.threads;
    sopts.input_w = meta.imgsz;
    sopts.input_h = meta.imgsz;
    SessionPtr sess(yolo::create_session(gguf, sopts), yolo::free_session);
    if (!sess) return false;
    if (!yolo::session_run(sess.get(), input.data())) return false;
    std::vector<float> logits;
    if (!yolo::session_read_logits(sess.get(), logits)) return false;
    probs = yolo::classify_softmax(logits);
    names = sess->model.meta.class_names;
    return true;
}

bool run_depth(const std::string& gguf, const std::string& source, const Options& opt, yolo::Image& img,
               std::vector<float>& depth) {
    const yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    if (meta.task != "depth") {
        fprintf(stderr, "run_depth requires a depth model, got task=%s\n", meta.task.c_str());
        return false;
    }
    if (!yolo::load_image(source, img)) return false;
    SessionPtr sess(nullptr, yolo::free_session);
    yolo::LetterboxInfo info;
    std::vector<float> input;
    if (!run_letterbox(gguf, img, opt, sess, info, input)) return false;
    std::vector<float> raw;
    int depth_w = 0, depth_h = 0;
    if (!yolo::session_read_depth(sess.get(), raw, depth_w, depth_h)) return false;
    depth = yolo::restore_depth(raw, depth_w, depth_h, info, img.w, img.h);
    return !depth.empty();
}

}  // namespace yolo_examples
