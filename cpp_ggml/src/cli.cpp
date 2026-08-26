// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "backend.hpp"
#include "common.hpp"
#include "image_io.hpp"
#include "postprocess.hpp"
#include "yolo_graph.hpp"

#if defined(YOLO_GGML_CLIP) && YOLO_GGML_CLIP
#include "clip_graph.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using SessionPtr = std::unique_ptr<yolo::Session, decltype(&yolo::free_session)>;

void usage() {
    fprintf(stderr,
            "usage:\n"
            "  yolo-cli info   --model M.gguf\n"
            "  yolo-cli detect --model M.gguf --source IMG [--out OUT.png] [--conf 0.25] [--iou 0.7]\n"
            "                 [--max-det 300] [--threads N] [--input-f32 IN.bin] [--dump-raw OUT.bin]\n"
            "                 [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "                 [--classes \"person,car\" --clip-model clip-ViT-B-32.gguf] (YOLO-World)\n"
            "                 (segment models run here too: boxes + instance masks, --out blends them)\n"
            "  yolo-cli pose   --model M.gguf --source IMG [--out OUT.png] [--conf 0.25] [--iou 0.7]\n"
            "                 [--max-det 300] [--threads N] [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "  yolo-cli obb    --model M.gguf --source IMG [--out OUT.png] [--conf 0.25] [--iou 0.7]\n"
            "                 [--max-det 300] [--threads N] [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "  yolo-cli semantic --model M.gguf --source IMG [--out OUT.png] [--threads N] [--profile ops|gaps]\n"
            "  yolo-cli classify --model M.gguf --source IMG [--topk 5] [--threads N] [--profile ops|gaps]\n"
            "  yolo-cli depth  --model M.gguf --source IMG [--out OUT.png] [--raw OUT.bin] [--max-depth M]\n"
            "                 [--threads N] [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "  yolo-cli bench  --model M.gguf --source IMG [--warmup 20] [--iters 100] [--threads N]\n"
            "                 [--profile ops|gaps] [--classes \"person,car\" --clip-model clip.gguf]\n"
            "                 [--text-embed vocabulary.ytxt] (YOLO-World)\n"
            "  --profile ops:   per-op wall-time table on exit (adds per-node sync; GPU builds)\n"
            "  --profile gaps:  per-stage (upload/compute/readback) traces on stderr\n"
            "\n"
            "raw binary formats (little endian, for pytorch parity tests):\n"
            "  --input-f32 / --dump-input: 8b magic \"YINP0001\", 3x i32 (C,H,W), then f32 CHW pixels\n"
            "  --dump-raw:                  8b magic \"YRAW0001\", 2x i32 (no,na), then f32 [no,na]\n"
            "  depth --raw:                 8b magic \"YDEP0001\", 2x i32 (H,W), then f32 meters\n");
}

using Args = std::unordered_map<std::string, std::string>;

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || k.size() <= 2) {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return {};
        }
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            args[k] = argv[++i];
        } else {
            args[k] = "1";  // boolean flag
        }
    }
    return args;
}

std::string arg_s(const Args& a, const char* k, const std::string& def = "") {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

double arg_f(const Args& a, const char* k, double def) {
    auto it = a.find(k);
    return it == a.end() ? def : atof(it->second.c_str());
}

int arg_i(const Args& a, const char* k, int def) {
    auto it = a.find(k);
    return it == a.end() ? def : atoi(it->second.c_str());
}

std::vector<std::string> parse_class_list(const std::string& text) {
    std::vector<std::string> classes;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t comma = text.find(',', pos);
        std::string name = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        while (!name.empty() && name.front() == ' ') name.erase(name.begin());
        while (!name.empty() && name.back() == ' ') name.pop_back();
        if (!name.empty()) classes.push_back(std::move(name));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return classes;
}

bool read_ytxt_shape(const std::string& path, int& nc) {
    FILE* f = fopen(path.c_str(), "rb");
    char magic[8];
    int32_t dims[2] = {};
    const bool ok = f && fread(magic, 1, sizeof(magic), f) == sizeof(magic) && !memcmp(magic, "YTXT0001", 8) &&
                    fread(dims, sizeof(int32_t), 2, f) == 2 && dims[0] > 0 && dims[1] == clip::EMBED_DIM;
    if (f) fclose(f);
    if (ok) nc = dims[0];
    return ok;
}

std::string json_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (unsigned char c : text) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    char encoded[7];
                    snprintf(encoded, sizeof(encoded), "\\u%04x", c);
                    escaped += encoded;
                } else {
                    escaped += (char)c;
                }
        }
    }
    return escaped;
}

// ---- raw f32 dump helpers (tensor-level parity with pytorch) ----------------

bool dump_f32(const char* path, const char* magic, const std::vector<int32_t>& dims, const float* data,
              size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(magic, 1, 8, f);
    fwrite(dims.data(), sizeof(int32_t), dims.size(), f);
    const bool ok = fwrite(data, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

bool read_f32(const char* path, const char* magic, std::vector<int32_t>& dims, std::vector<float>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char m[8];
    if (fread(m, 1, 8, f) != 8 || memcmp(m, magic, 8) || dims.empty() ||
        fread(dims.data(), sizeof(int32_t), dims.size(), f) != dims.size()) {
        fclose(f);
        return false;
    }
    size_t n = 1;
    for (int32_t v : dims) {
        if (v <= 0 || n > std::numeric_limits<size_t>::max() / (size_t)v) {
            fclose(f);
            return false;
        }
        n *= (size_t)v;
    }
    if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
        fclose(f);
        return false;
    }
    const long data_pos = ftell(f);
    if (data_pos < 0 || fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    const long file_end = ftell(f);
    if (file_end < data_pos || (uint64_t)n > (uint64_t)(file_end - data_pos) / sizeof(float) ||
        fseek(f, data_pos, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    out.resize(n);
    const bool ok = fread(out.data(), sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

// ---- info --------------------------------------------------------------------

int cmd_info(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    if (model_path.empty()) {
        fprintf(stderr, "--model is required\n");
        return 1;
    }
    auto model = yolo::load_gguf(model_path);
    if (!model) return 1;
    const auto& m = model->meta;

    printf("name       : %s\n", m.name.c_str());
    printf("task       : %s\n", m.task.c_str());
    printf("dtype      : %s\n", m.dtype.c_str());
    printf("imgsz      : %d\n", m.imgsz);
    printf("nc         : %d\n", m.nc);
    printf("layers     : %d (strides:", m.nl);
    for (float s : m.strides) printf(" %g", s);
    printf(")\n");
    printf("reg_max    : %d\n", m.reg_max);
    if (m.task == "segment") printf("nm         : %d\n", m.nm);
    if (m.task == "pose") printf("nk         : %d (ndim %d)\n", m.nk, m.kpt_ndim);
    if (m.task == "obb") printf("ne         : %d\n", m.ne);
    printf("end2end    : %s\n", m.end2end ? "true" : "false");
    printf("max_det    : %d\n", m.max_det);
    printf("ops        : %zu\n", model->ops.size());
    printf("tensors    : %zu\n", model->tensors.size());

    std::map<std::string, int> hist;
    for (const auto& op : model->ops) hist[op.type]++;
    printf("op types   :");
    for (const auto& kv : hist) printf(" %s=%d", kv.first.c_str(), kv.second);
    printf("\n");

    printf("classes    : %d [", (int)m.class_names.size());
    for (size_t i = 0; i < m.class_names.size() && i < 5; i++) printf("%s,", m.class_names[i].c_str());
    if (m.class_names.size() > 5) printf("...");
    printf("]\n");
    return 0;
}

// ---- detect ------------------------------------------------------------------

int cmd_detect(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    const std::string in_f32 = arg_s(args, "input-f32");
    if (model_path.empty() || (source.empty() && in_f32.empty())) {
        fprintf(stderr, "--model and (--source | --input-f32) are required\n");
        return 1;
    }

    // Preprocess first: the letterboxed canvas decides the graph input shape.
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "detect" && meta.task != "segment") {
        fprintf(stderr, "detect requires a detect or segment model, got task=%s\n", meta.task.c_str());
        return 1;
    }

    yolo::Image img;
    yolo::LetterboxInfo info{};
    std::vector<float> input;
    int canvas_w = meta.imgsz, canvas_h = meta.imgsz;
    if (!in_f32.empty()) {
        std::vector<int32_t> in_dims = {3, canvas_h, canvas_w};  // updated from file header
        if (!read_f32(in_f32.c_str(), "YINP0001", in_dims, input)) {
            fprintf(stderr, "failed to read --input-f32 %s\n", in_f32.c_str());
            return 1;
        }
        canvas_h = in_dims[1];
        canvas_w = in_dims[2];
        if (in_dims[0] != 3) {
            fprintf(stderr, "--input-f32 must contain three channels\n");
            return 1;
        }
        info = yolo::LetterboxInfo{1.0f, 0, 0, canvas_w, canvas_h, canvas_w, canvas_h};
    } else {
        if (!yolo::load_image(source, img)) return 1;
        yolo::letterbox_image(img, meta.imgsz, info, input);
        canvas_w = info.imgsz_w;
        canvas_h = info.imgsz_h;
    }

    const std::string dump_ops = arg_s(args, "dump-ops");
    // YOLO-World: --classes "person,car" sets the open-vocabulary class list.
    const std::vector<std::string> world_classes = parse_class_list(arg_s(args, "classes"));
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = canvas_w;
    sopts.input_h = canvas_h;
    sopts.keep_all_ops = !dump_ops.empty();
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    // YOLO-World class count: --classes wins; otherwise peek the --text-embed
    // blob header (dump_f32 layout: YTXT0001 magic + dims, no ndim field) so
    // the graph is built with the right nc instead of the COCO (80) default.
    int te_nc = 0;
    const std::string te_path = arg_s(args, "text-embed");
    if (!te_path.empty()) {
        if (!read_ytxt_shape(te_path, te_nc)) {
            fprintf(stderr, "--text-embed must be a [nc, 512] YTXT0001 f32 blob\n");
            return 1;
        }
    }
    if (!world_classes.empty() && te_nc && (int)world_classes.size() != te_nc) {
        fprintf(stderr, "--classes (%zu classes) and --text-embed nc (%d) disagree\n",
                world_classes.size(), te_nc);
        return 1;
    }
    sopts.world_nc = (int)world_classes.size() ? (int)world_classes.size() : te_nc;
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    const std::string dump_in = arg_s(args, "dump-input");
    if (!dump_in.empty() &&
        !dump_f32(dump_in.c_str(), "YINP0001", {3, canvas_h, canvas_w}, input.data(), input.size())) {
        fprintf(stderr, "failed to write --dump-input %s\n", dump_in.c_str());
        return 1;
    }

    if (s->text_input) {
        // YOLO-World text embedding input. Prefer a precomputed [nc, 512]
        // row-major f32 blob (--text-embed), else encode --classes through the
        // CLIP text encoder (--clip-model, default models/gguf/clip-ViT-B-32.gguf).
        std::vector<float> text_embed((size_t)s->world_nc * clip::EMBED_DIM, 0.0f);
        const std::string te_file = arg_s(args, "text-embed");
        if (!te_file.empty()) {
            std::vector<int32_t> dims = {s->world_nc, clip::EMBED_DIM};
            if (!read_f32(te_file.c_str(), "YTXT0001", dims, text_embed) ||
                dims[0] != s->world_nc || dims[1] != clip::EMBED_DIM) {
                fprintf(stderr, "--text-embed must be [nc=%d, 512] YTXT0001 f32 blob\n", s->world_nc);
                return 1;
            }
        } else {
            if (!s->model.meta.text_model.empty()) {
                fprintf(stderr,
                        "YOLOE model (%s) requires a post-reprta MobileCLIP YTXT file; "
                        "generate it with scripts/encode_mobileclip_text.py --detector <yoloe.pt>\n",
                        s->model.meta.text_model.c_str());
                return 1;
            }
#if defined(YOLO_GGML_CLIP) && YOLO_GGML_CLIP
            if (world_classes.empty()) {
                fprintf(stderr, "world model: pass --classes \"a,b,c\" or --text-embed file\n");
                return 1;
            }
            std::string clip_model = arg_s(args, "clip-model", "models/gguf/clip-ViT-B-32.gguf");
            // Keep CLIP's graph allocator separate from the YOLO allocator. The
            // two sessions may use the same physical device, but a gallocr or
            // scheduler is not a process-wide shared workspace.
            clip::ClipSession* cs = clip::clip_create_session(clip_model);
            if (!cs) return 1;
            for (int i = 0; i < s->world_nc; i++) {
                if (!clip::clip_encode_string(cs, world_classes[i].c_str(),
                                              text_embed.data() + (size_t)i * clip::EMBED_DIM)) {
                    fprintf(stderr, "failed to encode class '%s'\n", world_classes[i].c_str());
                    clip::clip_free_session(cs);
                    return 1;
                }
            }
            clip::clip_free_session(cs);
#else
            fprintf(stderr, "world model requires --text-embed file (built without CLIP)\n");
            return 1;
#endif
        }
        if (!yolo::session_set_text(s, text_embed.data())) return 1;
    }

    if (!yolo::session_run(s, input.data())) return 1;

    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;

    const std::string dump_raw = arg_s(args, "dump-raw");
    if (!dump_raw.empty() && !dump_f32(dump_raw.c_str(), "YRAW0001", {no, na}, raw.data(), raw.size())) {
        fprintf(stderr, "failed to write --dump-raw %s\n", dump_raw.c_str());
        return 1;
    }

    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }

    yolo::PostprocConfig cfg;
    cfg.conf_thres = (float)arg_f(args, "conf", 0.25);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", s->model.meta.max_det);
    if (!(cfg.conf_thres > 0.0f && cfg.conf_thres < 1.0f) || !(cfg.iou_thres >= 0.0f && cfg.iou_thres <= 1.0f) ||
        cfg.max_det <= 0) {
        fprintf(stderr, "--conf must be in (0,1), --iou in [0,1], and --max-det positive\n");
        return 1;
    }
    std::vector<yolo::Detection> dets =
        yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    std::vector<yolo::SegMask> masks;
    if (meta.task == "segment") {
        std::vector<float> proto;
        int nm = 0, pw = 0, ph = 0;
        if (!yolo::session_read_proto(s, proto, nm, pw, ph)) return 1;
        masks = yolo::compose_masks(dets, raw, na, s->model.meta, proto, pw, ph, canvas_w, canvas_h);
    }
    yolo::unscale_boxes(dets, info);

    const auto& names = world_classes.empty() ? s->model.meta.class_names : world_classes;
    printf("%d detection%s (%s, %s, %dx%d, backend=%s)\n", (int)dets.size(), dets.size() == 1 ? "" : "s",
           s->model.meta.name.c_str(), s->model.meta.dtype.c_str(), canvas_w, canvas_h,
           yolo::backend_name(s->backend));
    for (size_t i = 0; i < dets.size(); i++) {
        const auto& d = dets[i];
        const char* cname = d.class_id < (int)names.size() ? names[d.class_id].c_str() : "?";
        printf("  %-12s %.2f  [%.1f, %.1f, %.1f, %.1f]", cname, d.score, d.x1, d.y1, d.x2, d.y2);
        if (i < masks.size() && masks[i].w > 0) {
            size_t bits = 0;
            for (uint8_t b : masks[i].bits) bits += b;
            printf("  mask=%zu", bits);
        }
        printf("\n");
    }

    const std::string out = arg_s(args, "out");
    if (!out.empty()) {
        if (in_f32.empty()) {
            yolo::draw_detections(out, img, dets, names, meta.task == "segment" ? &masks : nullptr, &info);
        } else {
            fprintf(stderr, "note: --out skipped with --input-f32 (no source image)\n");
        }
    }

    return 0;
}

// ---- pose -------------------------------------------------------------------

int cmd_pose(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "pose") {
        fprintf(stderr, "pose requires a pose model, got task=%s\n", meta.task.c_str());
        return 1;
    }
    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    if (!yolo::session_run(s, input.data())) return 1;
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = (float)arg_f(args, "conf", 0.25);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", s->model.meta.max_det);
    std::vector<yolo::PoseDetection> poses =
        yolo::postprocess_pose(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    yolo::unscale_pose(poses, info);
    const auto& names = s->model.meta.class_names;
    printf("%zu pose result%s (%s, %s, %dx%d, backend=%s)\n", poses.size(), poses.size() == 1 ? "" : "s",
           s->model.meta.name.c_str(), s->model.meta.dtype.c_str(), info.imgsz_w, info.imgsz_h,
           yolo::backend_name(s->backend));
    for (const auto& p : poses) {
        const char* cname = p.det.class_id < (int)names.size() ? names[p.det.class_id].c_str() : "?";
        printf("  %-12s %.2f  box [%.1f, %.1f, %.1f, %.1f]  kpts %zu", cname, p.det.score, p.det.x1, p.det.y1,
               p.det.x2, p.det.y2, p.kpts.size());
        const int nkpt = meta.kpt_ndim ? (int)p.kpts.size() / meta.kpt_ndim : 0;
        for (int k = 0; k < nkpt && k < 5; k++)
            printf("  k%d=(%.1f,%.1f)", k, p.kpts[k * meta.kpt_ndim], p.kpts[k * meta.kpt_ndim + 1]);
        if (nkpt > 5) printf(" ...");
        printf("\n");
    }
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !yolo::draw_pose(out, img, poses, names)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}

// ---- obb --------------------------------------------------------------------

int cmd_obb(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "obb") {
        fprintf(stderr, "obb requires an obb model, got task=%s\n", meta.task.c_str());
        return 1;
    }
    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    if (!yolo::session_run(s, input.data())) return 1;
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = (float)arg_f(args, "conf", 0.25);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", s->model.meta.max_det);
    std::vector<yolo::OBBDetection> obbs =
        yolo::postprocess_obb(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    yolo::unscale_obb(obbs, info);
    const auto& names = s->model.meta.class_names;
    printf("%zu obb result%s (%s, %s, %dx%d, backend=%s)\n", obbs.size(), obbs.size() == 1 ? "" : "s",
           s->model.meta.name.c_str(), s->model.meta.dtype.c_str(), info.imgsz_w, info.imgsz_h,
           yolo::backend_name(s->backend));
    for (const auto& o : obbs) {
        const char* cname = o.class_id < (int)names.size() ? names[o.class_id].c_str() : "?";
        printf("  %-12s %.2f  rbox cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.1fdeg\n", cname, o.score, o.cx, o.cy, o.w,
               o.h, o.angle * 180.0f / 3.14159265f);
    }
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !yolo::draw_obb(out, img, obbs, names)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}

// ---- semantic ---------------------------------------------------------------

int cmd_semantic(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "semantic") {
        fprintf(stderr, "semantic requires a semantic model, got task=%s\n", meta.task.c_str());
        return 1;
    }
    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    if (!yolo::session_run(s, input.data())) return 1;
    std::vector<float> logits;
    int nc = 0, gw = 0, gh = 0;
    if (!yolo::session_read_semantic(s, logits, nc, gw, gh)) return 1;
    std::vector<uint8_t> classes = yolo::semantic_argmax(logits, nc, gw, gh);
    const auto& names = s->model.meta.class_names;
    printf("semantic %dx%d grid, %d classes, top classes:", gw, gh, nc);
    std::map<uint8_t, size_t> hist;
    for (uint8_t c : classes) hist[c]++;
    std::vector<std::pair<size_t, uint8_t>> ranked;
    for (const auto& [c, n] : hist) ranked.push_back({n, c});
    std::partial_sort(ranked.begin(), ranked.begin() + std::min<size_t>(5, ranked.size()), ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < std::min<size_t>(5, ranked.size()); i++) {
        const char* cname = ranked[i].second < names.size() ? names[ranked[i].second].c_str() : "?";
        printf(" %s=%zu", cname, ranked[i].first);
    }
    printf("\n");
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !yolo::draw_semantic(out, img, classes, gw, gh, nc)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}

// ---- classify ---------------------------------------------------------------

int cmd_classify(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "classify") {
        fprintf(stderr, "classify requires a classify model, got task=%s\n", meta.task.c_str());
        return 1;
    }
    yolo::Image img;
    if (!yolo::load_image(source, img)) return 1;
    // Classification uses resize+center-crop+ImageNet-normalize, not letterbox.
    std::vector<float> input;
    yolo::classify_preprocess(img, meta.imgsz, input);
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = meta.imgsz;
    sopts.input_h = meta.imgsz;
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    const std::string dump_in = arg_s(args, "dump-input");
    if (!dump_in.empty() &&
        !dump_f32(dump_in.c_str(), "YINP0001", {3, meta.imgsz, meta.imgsz}, input.data(), input.size())) {
        fprintf(stderr, "failed to write --dump-input %s\n", dump_in.c_str());
        return 1;
    }
    if (!yolo::session_run(s, input.data())) return 1;
    std::vector<float> logits;
    if (!yolo::session_read_logits(s, logits)) return 1;
    std::vector<float> probs = yolo::classify_softmax(logits);
    const int topk = std::clamp(arg_i(args, "topk", 5), 1, (int)probs.size());
    std::vector<size_t> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [&](size_t a, size_t b) { return probs[a] > probs[b]; });
    const auto& names = s->model.meta.class_names;
    printf("classify top-%d (%s, %s, backend=%s)\n", topk, s->model.meta.name.c_str(),
           s->model.meta.dtype.c_str(), yolo::backend_name(s->backend));
    for (int i = 0; i < topk; i++) {
        const size_t c = idx[i];
        const char* cname = c < names.size() ? names[c].c_str() : "?";
        printf("  %2d. %-24s %.4f\n", i + 1, cname, probs[c]);
    }
    return 0;
}

// ---- depth -------------------------------------------------------------------

int cmd_depth(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }

    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "depth") {
        fprintf(stderr, "depth requires a depth model, got task=%s\n", meta.task.c_str());
        return 1;
    }

    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);
    const std::string dump_ops = arg_s(args, "dump-ops");
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.keep_all_ops = !dump_ops.empty();
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    if (!yolo::session_run(s, input.data())) {
        return 1;
    }
    std::vector<float> raw;
    int depth_w = 0, depth_h = 0;
    if (!yolo::session_read_depth(s, raw, depth_w, depth_h)) {
        return 1;
    }
    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }
    std::vector<float> depth = yolo::restore_depth(raw, depth_w, depth_h, info, img.w, img.h);
    if (depth.empty()) return 1;

    const std::string dump_in = arg_s(args, "dump-input");
    if (!dump_in.empty() &&
        !dump_f32(dump_in.c_str(), "YINP0001", {3, info.imgsz_h, info.imgsz_w}, input.data(), input.size())) {
        fprintf(stderr, "failed to write --dump-input %s\n", dump_in.c_str());
        return 1;
    }
    const std::string raw_path = arg_s(args, "raw");
    if (!raw_path.empty() &&
        !dump_f32(raw_path.c_str(), "YDEP0001", {img.h, img.w}, depth.data(), depth.size())) {
        fprintf(stderr, "failed to write --raw %s\n", raw_path.c_str());
        return 1;
    }
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !yolo::write_depth_png(out, depth, img.w, img.h, (float)arg_f(args, "max-depth", 0.0))) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }

    const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
    const double mean = std::accumulate(depth.begin(), depth.end(), 0.0) / depth.size();
    printf("depth %dx%d meters (min=%.3f mean=%.3f max=%.3f, model=%s, dtype=%s)\n", img.w, img.h, *lo, mean,
           *hi, meta.name.c_str(), meta.dtype.c_str());
    return 0;
}

// ---- bench -------------------------------------------------------------------

struct Stats {
    std::vector<double> ms;
    double mean = 0, min = 0, p50 = 0, p90 = 0, max = 0;
    void finish() {
        std::sort(ms.begin(), ms.end());
        const int n = (int)ms.size();
        if (!n) return;
        double sum = 0;
        for (double v : ms) sum += v;
        mean = sum / n;
        min = ms.front();
        p50 = ms[n / 2];
        p90 = ms[std::min(n - 1, (int)(n * 0.9))];
        max = ms.back();
    }
};

int cmd_bench(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }
    const int warmup = std::max(1, arg_i(args, "warmup", 20));
    const int iters = std::max(1, arg_i(args, "iters", 100));

    // Preprocess first: the letterboxed canvas decides the graph input shape.
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "detect" && meta.task != "depth" && meta.task != "segment" && meta.task != "pose" &&
        meta.task != "obb" && meta.task != "semantic" && meta.task != "classify") {
        fprintf(stderr, "bench supports all yolo tasks, got task=%s\n", meta.task.c_str());
        return 1;
    }
    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    // Classify uses the checkpoint-baked resize+center-crop (no letterbox).
    if (meta.task == "classify") {
        yolo::classify_preprocess(img, meta.imgsz, input);
    } else {
        yolo::letterbox_image(img, meta.imgsz, info, input);
    }

    const std::string profile_mode = arg_s(args, "profile");
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = meta.task == "classify" ? meta.imgsz : info.imgsz_w;
    sopts.input_h = meta.task == "classify" ? meta.imgsz : info.imgsz_h;
    sopts.profile_ops = profile_mode == "ops";
    sopts.profile_gaps = profile_mode == "gaps";
    const std::string world_classes_arg = arg_s(args, "classes");
    const std::string world_text_embed = arg_s(args, "text-embed");
    const std::vector<std::string> wc = parse_class_list(world_classes_arg);
    int text_nc = 0;
    if (meta.has_text_input && !world_text_embed.empty() && !read_ytxt_shape(world_text_embed, text_nc)) {
        fprintf(stderr, "--text-embed must be a [nc, 512] YTXT0001 f32 blob\n");
        return 1;
    }
    if (meta.has_text_input && !wc.empty() && text_nc && (int)wc.size() != text_nc) {
        fprintf(stderr, "--classes (%zu classes) and --text-embed nc (%d) disagree\n", wc.size(), text_nc);
        return 1;
    }
    sopts.world_nc = meta.has_text_input ? (wc.empty() ? text_nc : (int)wc.size()) : 0;
    if (meta.has_text_input && sopts.world_nc <= 0) {
        fprintf(stderr, "world bench: pass --classes or --text-embed\n");
        return 1;
    }
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    std::string world_text_source;
    // Text setup is outside the timed per-frame loop: the vocabulary is set
    // once per session and remains constant for every measured frame.
    if (s->text_input) {
        std::vector<float> text_embed((size_t)s->world_nc * clip::EMBED_DIM, 0.0f);
        if (!world_text_embed.empty()) {
            std::vector<int32_t> dims = {s->world_nc, clip::EMBED_DIM};
            if (!read_f32(world_text_embed.c_str(), "YTXT0001", dims, text_embed) ||
                dims[0] != s->world_nc || dims[1] != clip::EMBED_DIM) {
                fprintf(stderr, "--text-embed must be [nc=%d, 512] YTXT0001 f32 blob\n", s->world_nc);
                return 1;
            }
            world_text_source = "ytxt";
        } else {
            if (!s->model.meta.text_model.empty()) {
                fprintf(stderr,
                        "YOLOE bench requires a post-reprta MobileCLIP YTXT file; "
                        "generate it with scripts/encode_mobileclip_text.py --detector <yoloe.pt>\n");
                return 1;
            }
#if defined(YOLO_GGML_CLIP) && YOLO_GGML_CLIP
        std::string clip_model = arg_s(args, "clip-model", "models/gguf/clip-ViT-B-32.gguf");
        clip::ClipSession* cs = clip::clip_create_session(clip_model);
        if (!cs) return 1;
        for (int i = 0; i < s->world_nc; i++) {
            if (!clip::clip_encode_string(cs, wc[i].c_str(),
                                          text_embed.data() + (size_t)i * clip::EMBED_DIM)) {
                clip::clip_free_session(cs);
                return 1;
            }
        }
        clip::clip_free_session(cs);
        world_text_source = "clip";
#else
        (void)wc;
        fprintf(stderr, "world bench requires YOLO_GGML_CLIP\n");
        return 1;
#endif
        }
        if (!yolo::session_set_text(s, text_embed.data())) return 1;
    }

    std::vector<float> raw;
    int no = 0, na = 0;
    yolo::PostprocConfig cfg;
    cfg.max_det = s->model.meta.max_det;
    int depth_w = 0, depth_h = 0;
    std::vector<float> depth;
    std::vector<float> proto;
    int nm = 0, pw = 0, ph = 0;
    int nc2 = 0, gw = 0, gh = 0;
    std::vector<float> probs;
    std::vector<uint8_t> classes;
    Stats preprocess, graph, post, e2e;
    // --profile gaps splits graph_ms on stderr: session_run (input upload +
    // graph record/submit host time) vs session_read_output (GPU fence wait +
    // output download + host cast). Used to attribute the wall-vs-GPU-busy gap.
    Stats gap_run, gap_read;

    const bool box_task = meta.task == "detect" || meta.task == "segment" || meta.task == "pose" || meta.task == "obb";
    auto preprocess_once = [&]() {
        if (meta.task == "classify") {
            yolo::classify_preprocess(img, meta.imgsz, input);
        } else {
            yolo::letterbox_image(img, meta.imgsz, info, input);
        }
    };
    for (int i = 0; i < warmup; i++) {
        preprocess_once();
        if (!yolo::session_run(s, input.data())) return 1;
        if (box_task) {
            if (!yolo::session_read_output(s, raw, no, na)) return 1;
        } else if (meta.task == "depth") {
            if (!yolo::session_read_depth(s, raw, depth_w, depth_h)) return 1;
        } else if (meta.task == "semantic") {
            if (!yolo::session_read_semantic(s, raw, nc2, gw, gh)) return 1;
        } else if (!yolo::session_read_logits(s, raw)) {
            return 1;
        }
    }
    for (int i = 0; i < iters; i++) {
        yolo::Clock ce;
        yolo::Clock c0;
        preprocess_once();
        preprocess.ms.push_back(c0.ms_since());
        yolo::Clock c1;
        if (!yolo::session_run(s, input.data())) return 1;
        gap_run.ms.push_back(c1.ms_since());
        yolo::Clock c1b;
        if (box_task) {
            if (!yolo::session_read_output(s, raw, no, na)) return 1;
        } else if (meta.task == "depth") {
            if (!yolo::session_read_depth(s, raw, depth_w, depth_h)) return 1;
        } else if (meta.task == "semantic") {
            if (!yolo::session_read_semantic(s, raw, nc2, gw, gh)) return 1;
        } else if (!yolo::session_read_logits(s, raw)) {
            return 1;
        }
        gap_read.ms.push_back(c1b.ms_since());
        graph.ms.push_back(c1.ms_since());
        yolo::Clock c2;
        std::vector<yolo::Detection> dets;
        if (meta.task == "detect" || meta.task == "segment") {
            dets = yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
            if (meta.task == "segment") {
                // proto readback + mask composition belong to postprocess, mirroring
                // the pytorch predict path (results include the masks).
                if (!yolo::session_read_proto(s, proto, nm, pw, ph)) return 1;
                yolo::compose_masks(dets, raw, na, s->model.meta, proto, pw, ph, s->input_w, s->input_h);
            }
            yolo::unscale_boxes(dets, info);
        } else if (meta.task == "pose") {
            std::vector<yolo::PoseDetection> poses =
                yolo::postprocess_pose(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
            yolo::unscale_pose(poses, info);
            dets.resize(poses.size());
            for (size_t k = 0; k < poses.size(); k++) dets[k] = poses[k].det;
        } else if (meta.task == "obb") {
            std::vector<yolo::OBBDetection> obbs =
                yolo::postprocess_obb(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
            yolo::unscale_obb(obbs, info);
            dets.resize(obbs.size());
            for (size_t k = 0; k < obbs.size(); k++)
                dets[k] = yolo::Detection{obbs[k].cx - obbs[k].w / 2, obbs[k].cy - obbs[k].h / 2,
                                           obbs[k].cx + obbs[k].w / 2, obbs[k].cy + obbs[k].h / 2, obbs[k].score,
                                           obbs[k].class_id};
        } else if (meta.task == "depth") {
            depth = yolo::restore_depth(raw, depth_w, depth_h, info, img.w, img.h);
        } else if (meta.task == "semantic") {
            classes = yolo::semantic_argmax(raw, nc2, gw, gh);
        } else {
            probs = yolo::classify_softmax(raw);
        }
        post.ms.push_back(c2.ms_since());
        e2e.ms.push_back(ce.ms_since());
        if (i == iters - 1) {
            if (box_task) {
                YOLO_LOG_INFO("sanity: %d detections, top score %.3f", (int)dets.size(),
                              dets.empty() ? 0.0f : dets.front().score);
            } else if (meta.task == "depth") {
                const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
                YOLO_LOG_INFO("sanity: depth range %.3f..%.3f meters", *lo, *hi);
            } else if (meta.task == "semantic") {
                YOLO_LOG_INFO("sanity: %zu class-map pixels", classes.size());
            } else {
                YOLO_LOG_INFO("sanity: top prob %.3f",
                              probs.empty() ? 0.0f : *std::max_element(probs.begin(), probs.end()));
            }
        }
    }
    preprocess.finish();
    graph.finish();
    post.finish();
    e2e.finish();
    if (sopts.profile_gaps) {
        gap_run.finish();
        gap_read.finish();
        fprintf(stderr,
                "[gap-prof] run=%.3fms (upload + graph host record)  read=%.3fms (fence wait + download + cast)\n",
                gap_run.mean, gap_read.mean);
    }

    printf("{\"backend\":\"%s\",\"model\":\"%s\",\"task\":\"%s\",\"dtype\":\"%s\",\"imgsz\":[%d,%d],\"threads\":%d,"
           "\"warmup\":%d,\"iters\":%d,"
           "\"preprocess_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f},"
           "\"graph_ms\":{\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,\"max\":%.3f},"
           "\"post_ms\":{\"mean\":%.3f,\"p50\":%.3f},"
           "\"e2e_ms\":{\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,\"max\":%.3f}",
           yolo::backend_name(s->backend), s->model.meta.name.c_str(), s->model.meta.task.c_str(),
           s->model.meta.dtype.c_str(), s->input_w, s->input_h, s->backend.n_threads, warmup, iters, preprocess.mean,
           preprocess.p50, preprocess.p90, graph.mean, graph.min, graph.p50, graph.p90, graph.max, post.mean, post.p50,
           e2e.mean, e2e.min, e2e.p50, e2e.p90, e2e.max);
    if (s->text_input) {
        const std::string escaped_classes = json_escape(world_classes_arg);
        printf(",\"world\":{\"classes\":\"%s\",\"class_count\":%d,\"text_source\":\"%s\"}",
               escaped_classes.c_str(), sopts.world_nc, world_text_source.c_str());
    }
    printf("}\n");

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string cmd = argv[1];
    const Args args = parse_args(argc, argv);
    if (args.empty() && argc > 2) return 1;
    if (cmd == "info") return cmd_info(args);
    if (cmd == "detect") return cmd_detect(args);
    if (cmd == "pose") return cmd_pose(args);
    if (cmd == "obb") return cmd_obb(args);
    if (cmd == "semantic") return cmd_semantic(args);
    if (cmd == "classify") return cmd_classify(args);
    if (cmd == "depth") return cmd_depth(args);
    if (cmd == "bench") return cmd_bench(args);
    usage();
    return 1;
}
