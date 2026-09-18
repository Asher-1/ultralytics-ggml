// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "backend.hpp"
#include "common.hpp"
#include "image_io.hpp"
#include "postprocess.hpp"
#include "tracker.hpp"
#include "yolo_graph.hpp"

#if defined(YOLO_GGML_CLIP) && YOLO_GGML_CLIP
#include "clip_graph.hpp"
#include "mobileclip_graph.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <sys/stat.h>
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
            "                 [--classes \"person,car\" --clip-model clip-ViT-B-32-f16.gguf] (YOLO-World;\n"
            "                 YOLOE takes the same --classes with --text-model mobileclip2_b-f16.gguf;\n"
            "                 without --classes the vocabulary stored in the GGUF is used;\n"
            "                 an empty trailing field is the background class row\n"
            "                 [--vp-boxes \"x1,y1,x2,y2,...\"] (YOLOE visual prompts: one example\n"
            "                 box per target in original-image pixels; needs a YOLOE GGUF with\n"
            "                 yolo.savpe=1 — see scripts/convert_yoloe_savpe_gguf.py;\n"
            "                 detections are labeled object0..objectN-1)\n"
            "                 [--dets-json OUT.json] writes the detections verbatim for tools\n"
            "                 (segment models run here too: boxes + instance masks, --out blends them)\n"
            "  yolo-cli pose   --model M.gguf --source IMG [--out OUT.png] [--conf 0.25] [--iou 0.7] [--dump-ops DIR]\n"
            "                 [--max-det 300] [--threads N] [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "  yolo-cli obb    --model M.gguf --source IMG [--out OUT.png] [--conf 0.25] [--iou 0.7] [--dump-ops DIR]\n"
            "                 [--max-det 300] [--threads N] [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "  yolo-cli track  --model M.gguf --source DIR|IMG[,IMG,...] [--tracker tracktrack]\n"
            "                 [--tracks-json OUT.jsonl] [--out PREFIX] [--conf 0.1] [--iou 0.7]\n"
            "                 [--max-det 300] [--threads N] [--tracker-config C.yaml] [--no-gmc]\n"
            "                 [--frames DIR|LIST.txt] (frames for GMC in --dets-jsonl replay mode)\n"
            "                 [--dets-jsonl IN.jsonl] (replay recorded detections instead of\n"
            "                 running a model; --model/--source then optional)\n"
            "                 (multi-object tracking over a frame sequence; tracker is one of\n"
            "                 bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack, default\n"
            "                 tracktrack, or an official ultralytics/cfg/trackers/*.yaml file;\n"
            "                 --tracker-config overrides individual keys from a YAML file;\n"
            "                 detect/segment/pose/obb models are supported; --out writes\n"
            "                 PREFIX_%%05d.png with #id labels)\n"
            "  yolo-cli semantic --model M.gguf --source IMG [--out OUT.png] [--threads N] [--dump-ops DIR] [--profile ops|gaps]\n"
            "                 [--raw OUT.bin] (YSEM0001: orig-size class map for A/B comparisons)\n"
            "  yolo-cli classify --model M.gguf --source IMG [--topk 5] [--threads N] [--dump-ops DIR] [--profile ops|gaps]\n"
            "                 [--raw OUT.bin] (YCLS0001: full softmax probs for A/B comparisons)\n"
            "  yolo-cli depth  --model M.gguf --source IMG [--out OUT.png] [--raw OUT.bin] [--max-depth M]\n"
            "                 [--threads N] [--dump-input OUT.bin] [--profile ops|gaps]\n"
            "  yolo-cli bench  --model M.gguf --source IMG [--warmup 20] [--iters 100] [--threads N]\n"
            "                 [--profile ops|gaps] [--classes \"person,car\" --clip-model clip.gguf]\n"
            "                 [--text-embed vocabulary.ytxt] (YOLO-World/YOLOE)\n"
            "  --profile ops:   per-op wall-time table on exit (adds per-node sync; GPU builds)\n"
            "  --profile gaps:  per-stage (upload/compute/readback) traces on stderr\n"
            "  pose/obb --dets-json: final detections as JSON for the A/B comparisons\n"
            "\n"
            "raw binary formats (little endian, for pytorch parity tests):\n"
            "  --input-f32 / --dump-input: 8b magic \"YINP0001\", 3x i32 (C,H,W), then f32 CHW pixels\n"
            "  --dump-ops:                  per-op dumps DIR/opNNN_<op>.bin (\"YLYR0001\" + 4x i32 dims +\n"
            "                               f32 data, F32 tensors only) for optrace regression diffs\n"
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

// Comma-separated open-vocabulary class list. Spaces around a name are padding, but an
// empty field is a real class row named "" -- the background prompt the YOLO-World docs
// add with set_classes(["person","bus",""]) -- so it must survive the split. Only a
// blank or absent argument means "no list given", i.e. use the stored vocabulary.
std::vector<std::string> parse_class_list(const std::string& text) {
    if (text.find_first_not_of(' ') == std::string::npos) return {};
    std::vector<std::string> classes;
    size_t pos = 0;
    while (true) {
        const size_t comma = text.find(',', pos);
        std::string name = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t begin = name.find_first_not_of(' '), end = name.find_last_not_of(' ');
        classes.push_back(begin == std::string::npos ? std::string() : name.substr(begin, end - begin + 1));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return classes;
}

bool read_ytxt_shape(const std::string& path, int& nc) {
    FILE* f = fopen(path.c_str(), "rb");
    char magic[8];
    int32_t dims[2] = {};
    const bool ok = f && fread(magic, 1, sizeof(magic), f) == sizeof(magic) && !memcmp(magic, "YTXT0002", 8) &&
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

// Full-canvas row-major run-length encoding of a segment mask (defined below);
// used by the A/B JSON outputs.
std::string mask_rle(int canvas_w, int canvas_h, const yolo::SegMask& m);

// Resolve the original-image size used for unscale/clip. Image mode uses the
// loaded image; --input-f32 runs may pass --img-size W,H to report boxes in
// original-image coordinates (with the Python-pipeline boundary clip) instead
// of canvas coordinates. Returns false on malformed input.
bool resolve_orig_size(const Args& args, const yolo::Image& img, bool has_input_f32, int meta_imgsz,
                       yolo::LetterboxInfo& info, int& ow, int& oh) {
    ow = img.w;
    oh = img.h;
    const std::string img_size = arg_s(args, "img-size");
    if (img_size.empty()) return true;
    const size_t comma = img_size.find(',');
    if (comma == std::string::npos) {
        fprintf(stderr, "--img-size expects W,H\n");
        return false;
    }
    const int w = atoi(img_size.substr(0, comma).c_str());
    const int h = atoi(img_size.c_str() + comma + 1);
    if (w <= 0 || h <= 0) {
        fprintf(stderr, "--img-size expects positive W,H\n");
        return false;
    }
    if (!has_input_f32) {
        fprintf(stderr, "note: --img-size only applies with --input-f32; using the loaded image size\n");
        return true;
    }
    info = yolo::letterbox_geometry(w, h, meta_imgsz);
    ow = w;
    oh = h;
    return true;
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

    if (m.has_text_input)
        printf("vocab      : %d class embeddings stored in the GGUF\n", (int)model->vocab_txt.size() / 512);
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
    // YOLOE visual prompts: --vp-boxes "x1,y1,x2,y2,..." (flat floats, one box
    // per target, original-image pixels). Non-empty switches the session to
    // the savpe mask input; --classes/--text-embed are then ignored (official
    // semantics: visual prompts take precedence over the class list).
    std::vector<float> vp_boxes;
    const std::string vp_spec = arg_s(args, "vp-boxes");
    if (!vp_spec.empty()) {
        size_t pos = 0;
        while (pos <= vp_spec.size()) {
            const size_t comma = vp_spec.find(',', pos);
            const std::string tok =
                vp_spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!tok.empty()) vp_boxes.push_back(strtof(tok.c_str(), nullptr));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (vp_boxes.empty() || vp_boxes.size() % 4 != 0) {
            fprintf(stderr, "--vp-boxes must be one or more x1,y1,x2,y2 quadruples\n");
            return 1;
        }
        if (!world_classes.empty() || !arg_s(args, "text-embed").empty()) {
            fprintf(stderr, "note: --vp-boxes takes precedence over --classes/--text-embed\n");
        }
    }
    const bool visual = !vp_boxes.empty();
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = canvas_w;
    sopts.input_h = canvas_h;
    sopts.keep_all_ops = !dump_ops.empty();
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    sopts.visual_count = visual ? (int)(vp_boxes.size() / 4) : 0;
    sopts.visual_boxes = vp_boxes;
    // YOLO-World class count: --classes wins; otherwise peek the --text-embed
    // blob header (dump_f32 layout: YTXT0002 magic + dims, no ndim field) so
    // the graph is built with the right nc instead of the COCO (80) default.
    int te_nc = 0;
    const std::string te_path = arg_s(args, "text-embed");
    if (!te_path.empty()) {
        if (!read_ytxt_shape(te_path, te_nc)) {
            fprintf(stderr, "--text-embed must be a [nc, 512] YTXT0002 f32 blob\n");
            return 1;
        }
    }
    if (!world_classes.empty() && te_nc && (int)world_classes.size() != te_nc) {
        fprintf(stderr, "--classes (%zu classes) and --text-embed nc (%d) disagree\n",
                world_classes.size(), te_nc);
        return 1;
    }
    // Neither knob means "use the vocabulary the checkpoint shipped with", which
    // the session loads itself; only an explicit list has to be encoded here.
    const bool supply_text = !visual && (!world_classes.empty() || te_nc > 0);
    sopts.world_nc = supply_text ? ((int)world_classes.size() ? (int)world_classes.size() : te_nc) : 0;
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    // Visual prompts: rasterize the example boxes onto the letterboxed P3
    // grid; must be re-done for every new canvas before session_run.
    if (visual && !yolo::session_prepare_visual_masks(s, info)) {
        fprintf(stderr, "failed to rasterize the visual prompt boxes\n");
        return 1;
    }

    const std::string dump_in = arg_s(args, "dump-input");
    if (!dump_in.empty() &&
        !dump_f32(dump_in.c_str(), "YINP0001", {3, canvas_h, canvas_w}, input.data(), input.size())) {
        fprintf(stderr, "failed to write --dump-input %s\n", dump_in.c_str());
        return 1;
    }

    if (supply_text) {
        // Text-conditioned head input. Prefer a precomputed [nc, 512] row-major
        // f32 blob (--text-embed), else encode --classes through the matching text
        // tower: MobileCLIP (--text-model) for YOLOE, CLIP (--clip-model) for World.
        std::vector<float> text_embed((size_t)s->world_nc * clip::EMBED_DIM, 0.0f);
        const std::string te_file = arg_s(args, "text-embed");
        if (!te_file.empty()) {
            std::vector<int32_t> dims = {s->world_nc, clip::EMBED_DIM};
            if (!read_f32(te_file.c_str(), "YTXT0002", dims, text_embed) ||
                dims[0] != s->world_nc || dims[1] != clip::EMBED_DIM) {
                fprintf(stderr, "--text-embed must be [nc=%d, 512] YTXT0002 f32 blob\n", s->world_nc);
                return 1;
            }
        } else {
#if defined(YOLO_GGML_CLIP) && YOLO_GGML_CLIP
            if (world_classes.empty()) {
                fprintf(stderr, "--classes is empty but --text-embed has %d rows\n", te_nc);
                return 1;
            }
            // Keep the text encoder's graph allocator separate from the YOLO
            // allocator. The two sessions may use the same physical device, but a
            // gallocr or scheduler is not a process-wide shared workspace.
            if (!s->model.meta.text_model.empty()) {
                // YOLOE: encode --classes with its MobileCLIP tower. The detector
                // graph applies the checkpoint's reprta itself (v4), so the raw
                // normalised MobileCLIP feature is exactly the graph text input.
                std::string tm = arg_s(args, "text-model", "models/gguf/mobileclip2_b-f16.gguf");
                mobileclip::MobileclipSession* ms = mobileclip::mobileclip_create_session(tm);
                if (!ms) return 1;
                for (int i = 0; i < s->world_nc; i++) {
                    if (!mobileclip::mobileclip_encode_string(ms, world_classes[i].c_str(),
                                                              text_embed.data() + (size_t)i * clip::EMBED_DIM)) {
                        fprintf(stderr, "failed to encode class '%s'\n", world_classes[i].c_str());
                        mobileclip::mobileclip_free_session(ms);
                        return 1;
                    }
                }
                mobileclip::mobileclip_free_session(ms);
            } else {
                std::string clip_model = arg_s(args, "clip-model", "models/gguf/clip-ViT-B-32-f16.gguf");
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
            }
#else
            fprintf(stderr, "world model requires --text-embed file (built without CLIP)\n");
            return 1;
#endif
        }
        if (!yolo::session_set_text(s, text_embed.data())) return 1;
    }

    if (!yolo::session_run(s, input.data())) return 1;

    // Debug: dump the savpe vpe node ([512, Q] F32) plus its fpn/emb inputs
    // for offline comparison. On F16 models the fpn/emb graph nodes hold F16
    // activations — a naive nelements*sizeof(float) readback overruns the
    // tensor buffer, so convert on the host from the node's real dtype.
    if (const char* vdump = getenv("YOLO_SAVPE_DUMP"); vdump && s->savpe_out) {
        auto dump_node = [](const char* path, ggml_tensor* t) {
            const size_t n = (size_t)ggml_nelements(t);
            std::vector<float> f(n);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, f.data(), 0, n * sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> raw(n);
                ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < n; i++) f[i] = ggml_fp16_to_fp32(raw[i]);
            } else {
                fprintf(stderr, "[savpe-dump] %s: unsupported dtype %d, skipped\n", path, (int)t->type);
                return false;
            }
            FILE* fp = fopen(path, "wb");
            if (!fp) return false;
            fwrite(f.data(), sizeof(float), n, fp);
            fclose(fp);
            fprintf(stderr, "[savpe-dump] wrote %s (%zu floats, dims [%lld,%lld,%lld,%lld])\n", path, n,
                    (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
            return true;
        };
        dump_node(vdump, s->savpe_out);
        if (ggml_tensor* emb = yolo::yolo_debug_emb()) {
            char path[512];
            snprintf(path, sizeof(path), "%s_emb0.bin", vdump);
            dump_node(path, emb);
        }
        for (int l = 0; l < 3; l++) {
            if (!s->savpe_fpn_dbg[l]) continue;
            char path[512];
            snprintf(path, sizeof(path), "%s_fpn%d.bin", vdump, l);
            dump_node(path, s->savpe_fpn_dbg[l]);
        }
    }

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
    int orig_w = 0, orig_h = 0;
    if (!resolve_orig_size(args, img, !in_f32.empty(), meta.imgsz, info, orig_w, orig_h)) return 1;
    yolo::unscale_boxes(dets, info, orig_w, orig_h);

    // Visual-prompt results group examples: official semantics label them
    // object0..objectN-1 regardless of any class list.
    std::vector<std::string> vp_names;
    if (visual) {
        for (int i = 0; i < sopts.visual_count; i++) vp_names.push_back("object" + std::to_string(i));
    }
    const auto& names = visual ? vp_names
                               : (world_classes.empty() ? s->model.meta.class_names : world_classes);
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

    // Machine-readable detections for tooling (scripts/val_map.py). The table above is
    // not a contract: it rounds scores to 2 decimals, which would flatten a precision-
    // recall curve, and a class name may contain spaces.
    const std::string dets_json = arg_s(args, "dets-json");
    if (!dets_json.empty()) {
        FILE* f = fopen(dets_json.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "failed to write --dets-json %s\n", dets_json.c_str());
            return 1;
        }
        // The vocabulary is part of the result: an open-vocabulary run maps class ids
        // to dataset categories by name, so re-deriving the list here would be a second
        // source of truth for --classes semantics. "canvas" is the letterbox size the
        // masks live in; segment detections carry a full-canvas row-major RLE so the
        // A/B comparisons can diff final masks pixel-for-pixel.
        fprintf(f, "{\"canvas\":[%d,%d],\"vocabulary\":[", canvas_w, canvas_h);
        for (size_t i = 0; i < names.size(); i++)
            fprintf(f, "%s\"%s\"", i ? "," : "", json_escape(names[i]).c_str());
        fputs("],\"detections\":[", f);
        for (size_t i = 0; i < dets.size(); i++) {
            const auto& d = dets[i];
            fprintf(f, "%s{\"cls\":%d,\"conf\":%.6f,\"xyxy\":[%.3f,%.3f,%.3f,%.3f]", i ? "," : "", d.class_id,
                    d.score, d.x1, d.y1, d.x2, d.y2);
            if (meta.task == "segment" && i < masks.size() && masks[i].w > 0)
                fprintf(f, ",\"mask\":\"%s\"", mask_rle(canvas_w, canvas_h, masks[i]).c_str());
            fputs("}", f);
        }
        fputs("]}\n", f);
        fclose(f);
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
    const std::string in_f32 = arg_s(args, "input-f32");
    if (model_path.empty() || (source.empty() && in_f32.empty())) {
        fprintf(stderr, "--model and (--source | --input-f32) are required\n");
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
    std::vector<float> input;
    int canvas_w = meta.imgsz, canvas_h = meta.imgsz;
    if (!in_f32.empty()) {
        // Same engine-level A/B path as detect --input-f32: consume a dumped
        // letterbox tensor verbatim; detections stay in canvas coordinates.
        std::vector<int32_t> in_dims = {3, canvas_h, canvas_w};
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
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    const std::string dump_ops = arg_s(args, "dump-ops");
    sopts.keep_all_ops = !dump_ops.empty();
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    if (!yolo::session_run(s, input.data())) return 1;
    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = (float)arg_f(args, "conf", 0.25);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", s->model.meta.max_det);
    std::vector<yolo::PoseDetection> poses =
        yolo::postprocess_pose(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    int orig_w = 0, orig_h = 0;
    if (!resolve_orig_size(args, img, !in_f32.empty(), meta.imgsz, info, orig_w, orig_h)) return 1;
    yolo::unscale_pose(poses, info, orig_w, orig_h);
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
    // Machine-readable pose detections for the A/B comparisons: final boxes and
    // keypoints in original-image pixels, the same quantities Python predict
    // exposes on Results.
    const std::string pose_json = arg_s(args, "dets-json");
    if (!pose_json.empty()) {
        FILE* f = fopen(pose_json.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "failed to write --dets-json %s\n", pose_json.c_str());
            return 1;
        }
        fprintf(f, "{\"canvas\":[%d,%d],\"detections\":[", info.imgsz_w, info.imgsz_h);
        for (size_t i = 0; i < poses.size(); i++) {
            const auto& p = poses[i];
            fprintf(f, "%s{\"cls\":%d,\"conf\":%.6f,\"xyxy\":[%.3f,%.3f,%.3f,%.3f],\"kpts\":[", i ? "," : "",
                    p.det.class_id, p.det.score, p.det.x1, p.det.y1, p.det.x2, p.det.y2);
            for (size_t j = 0; j < p.kpts.size(); j++) fprintf(f, "%s%.3f", j ? "," : "", p.kpts[j]);
            fputs("]}", f);
        }
        fputs("]}\n", f);
        fclose(f);
    }
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !in_f32.empty()) {
        fprintf(stderr, "note: --out skipped with --input-f32 (no source image)\n");
    } else if (!out.empty() && !yolo::draw_pose(out, img, poses, names)) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }
    return 0;
}

// ---- obb --------------------------------------------------------------------

int cmd_obb(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    const std::string obb_in_f32 = arg_s(args, "input-f32");
    if (model_path.empty() || (source.empty() && obb_in_f32.empty())) {
        fprintf(stderr, "--model and (--source | --input-f32) are required\n");
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
    std::vector<float> input;
    int obb_canvas_w = meta.imgsz, obb_canvas_h = meta.imgsz;
    if (!obb_in_f32.empty()) {
        // Engine-level A/B path: consume a dumped letterbox tensor verbatim;
        // detections stay in canvas coordinates (see detect --input-f32).
        std::vector<int32_t> in_dims = {3, obb_canvas_h, obb_canvas_w};
        if (!read_f32(obb_in_f32.c_str(), "YINP0001", in_dims, input)) {
            fprintf(stderr, "failed to read --input-f32 %s\n", obb_in_f32.c_str());
            return 1;
        }
        obb_canvas_h = in_dims[1];
        obb_canvas_w = in_dims[2];
        if (in_dims[0] != 3) {
            fprintf(stderr, "--input-f32 must contain three channels\n");
            return 1;
        }
        info = yolo::LetterboxInfo{1.0f, 0, 0, obb_canvas_w, obb_canvas_h, obb_canvas_w, obb_canvas_h};
    } else {
        if (!yolo::load_image(source, img)) return 1;
        yolo::letterbox_image(img, meta.imgsz, info, input);
        obb_canvas_w = info.imgsz_w;
        obb_canvas_h = info.imgsz_h;
    }
    yolo::SessionOptions sopts;
    sopts.threads = arg_i(args, "threads", 0);
    sopts.input_w = info.imgsz_w;
    sopts.input_h = info.imgsz_h;
    sopts.profile_ops = arg_s(args, "profile") == "ops";
    sopts.profile_gaps = arg_s(args, "profile") == "gaps";
    const std::string dump_ops = arg_s(args, "dump-ops");
    sopts.keep_all_ops = !dump_ops.empty();
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    if (!yolo::session_run(s, input.data())) return 1;
    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }
    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;
    yolo::PostprocConfig cfg;
    cfg.conf_thres = (float)arg_f(args, "conf", 0.25);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", s->model.meta.max_det);
    std::vector<yolo::OBBDetection> obbs =
        yolo::postprocess_obb(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    int obb_ow = 0, obb_oh = 0;
    if (!resolve_orig_size(args, img, !obb_in_f32.empty(), meta.imgsz, info, obb_ow, obb_oh)) return 1;
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
    // Machine-readable OBB detections for the A/B comparisons: rotated boxes in
    // original-image pixels (cx, cy, w, h, angle radians).
    const std::string obb_json = arg_s(args, "dets-json");
    if (!obb_json.empty()) {
        FILE* f = fopen(obb_json.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "failed to write --dets-json %s\n", obb_json.c_str());
            return 1;
        }
        fprintf(f, "{\"canvas\":[%d,%d],\"detections\":[", info.imgsz_w, info.imgsz_h);
        for (size_t i = 0; i < obbs.size(); i++) {
            const auto& o = obbs[i];
            fprintf(f, "%s{\"cls\":%d,\"conf\":%.6f,\"cx\":%.3f,\"cy\":%.3f,\"w\":%.3f,\"h\":%.3f,\"angle\":%.6f}",
                    i ? "," : "", o.class_id, o.score, o.cx, o.cy, o.w, o.h, o.angle);
        }
        fputs("]}\n", f);
        fclose(f);
    }
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !obb_in_f32.empty()) {
        fprintf(stderr, "note: --out skipped with --input-f32 (no source image)\n");
    } else if (!out.empty() && !yolo::draw_obb(out, img, obbs, names)) {
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
    const std::string dump_ops = arg_s(args, "dump-ops");
    sopts.keep_all_ops = !dump_ops.empty();
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;
    if (!yolo::session_run(s, input.data())) return 1;
    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }
    std::vector<float> logits;
    int nc = 0, gw = 0, gh = 0;
    if (!yolo::session_read_semantic(s, logits, nc, gw, gh)) return 1;
    std::vector<uint8_t> classes = yolo::semantic_argmax(logits, nc, gw, gh);
    // Final-mode class map for the A/B comparisons, replicating the Python
    // semantic postprocess (predict.py postprocess + ops.scale_masks): bilinear
    // logits up to the letterbox canvas, crop the padding, bilinear down to the
    // original image, then per-pixel argmax (nearest-upsampled argmax disagree
    // with Python wherever interpolated logits cross, which measurably changes
    // both pixels and the present class set). One uint8 class id per pixel as
    // f32 (YSEM0001).
    const std::string sem_raw = arg_s(args, "raw");
    if (!sem_raw.empty()) {
        const int ow = img.w, oh = img.h, cwv = info.imgsz_w, chv = info.imgsz_h;
        const float sgx = (float)gw / cwv, sgy = (float)gh / chv;
        const float gain = std::min((float)chv / oh, (float)cwv / ow);
        const float pad_w = (cwv - std::round(ow * gain)) / 2.0f;
        const float pad_h = (chv - std::round(oh * gain)) / 2.0f;
        const int left = (int)std::nearbyint(pad_w - 0.1f), top = (int)std::nearbyint(pad_h - 0.1f);
        const int crop_w = cwv - left - (int)std::nearbyint(pad_w + 0.1f);
        const int crop_h = chv - top - (int)std::nearbyint(pad_h + 0.1f);
        const float sx2 = (float)crop_w / ow, sy2 = (float)crop_h / oh;
        auto grid_at = [&](int c, float cx, float cy) {
            const float sx = std::max(0.0f, (cx + 0.5f) * sgx - 0.5f);
            const float sy = std::max(0.0f, (cy + 0.5f) * sgy - 0.5f);
            const int ix = (int)sx, iy = (int)sy;
            const int ix1 = std::min(ix + 1, gw - 1), iy1 = std::min(iy + 1, gh - 1);
            const float fx = sx - ix, fy = sy - iy;
            const float* base = logits.data() + (size_t)c * gw * gh;
            return base[(size_t)iy * gw + ix] * (1 - fx) * (1 - fy) + base[(size_t)iy * gw + ix1] * fx * (1 - fy) +
                   base[(size_t)iy1 * gw + ix] * (1 - fx) * fy + base[(size_t)iy1 * gw + ix1] * fx * fy;
        };
        std::vector<uint8_t> full((size_t)ow * oh);
        for (int y = 0; y < oh; y++) {
            const float cy = std::max(0.0f, (y + 0.5f) * sy2 - 0.5f) + top;
            for (int x = 0; x < ow; x++) {
                const float cx = std::max(0.0f, (x + 0.5f) * sx2 - 0.5f) + left;
                int best = 0;
                float best_v = grid_at(0, cx, cy);
                for (int c = 1; c < nc; c++) {
                    const float v = grid_at(c, cx, cy);
                    if (v > best_v) {
                        best_v = v;
                        best = c;
                    }
                }
                full[(size_t)y * ow + x] = (uint8_t)best;
            }
        }
        std::vector<float> full_f(full.begin(), full.end());
        if (!dump_f32(sem_raw.c_str(), "YSEM0001", {img.h, img.w}, full_f.data(), full_f.size())) {
            fprintf(stderr, "failed to write --raw %s\n", sem_raw.c_str());
            return 1;
        }
    }
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
    const std::string dump_ops = arg_s(args, "dump-ops");
    sopts.keep_all_ops = !dump_ops.empty();
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
    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }
    std::vector<float> logits;
    if (!yolo::session_read_logits(s, logits)) return 1;
    std::vector<float> probs = yolo::classify_softmax(logits);
    // Final-mode softmax probabilities for the A/B comparisons (YCLS0001).
    const std::string cls_raw = arg_s(args, "raw");
    if (!cls_raw.empty() &&
        !dump_f32(cls_raw.c_str(), "YCLS0001", {(int)probs.size()}, probs.data(), probs.size())) {
        fprintf(stderr, "failed to write --raw %s\n", cls_raw.c_str());
        return 1;
    }
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

// ---- track -------------------------------------------------------------------

// Frame sources: a directory of images (sorted by name), a comma-separated file
// list, or a single image (a single-frame track). Tracking needs consecutive
// frames of one stream; video decoding is out of scope for the C++ runtime.
std::vector<std::string> list_frame_files(const std::string& source) {
    auto has_image_ext = [](const std::string& s) {
        const size_t dot = s.rfind('.');
        if (dot == std::string::npos) return false;
        std::string ext = s.substr(dot);
        for (char& c : ext) c = (char)std::tolower((unsigned char)c);
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tga";
    };
    std::vector<std::string> files;
    if (source.find(',') != std::string::npos) {
        size_t pos = 0;
        while (true) {
            const size_t comma = source.find(',', pos);
            std::string tok = source.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            const size_t b = tok.find_first_not_of(' '), e = tok.find_last_not_of(' ');
            if (b != std::string::npos) files.push_back(tok.substr(b, e - b + 1));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        return files;
    }
    struct stat st;
    if (stat(source.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        DIR* d = opendir(source.c_str());
        if (!d) return files;
        while (const dirent* ent = readdir(d)) {
            const std::string name = ent->d_name;
            if (name.front() == '.') continue;
            if (has_image_ext(name)) files.push_back(source + "/" + name);
        }
        closedir(d);
        std::sort(files.begin(), files.end());
        return files;
    }
    files.push_back(source);
    return files;
}

yolo::track::TrackDet det_to_track(const yolo::Detection& d, int idx) {
    yolo::track::TrackDet t;
    t.cx = (d.x1 + d.x2) / 2;
    t.cy = (d.y1 + d.y2) / 2;
    t.w = d.x2 - d.x1;
    t.h = d.y2 - d.y1;
    t.angle = -10.0f;
    t.score = d.score;
    t.class_id = d.class_id;
    t.idx = idx;
    return t;
}

yolo::track::TrackDet obb_to_track(const yolo::OBBDetection& o, int idx) {
    yolo::track::TrackDet t;
    t.cx = o.cx;
    t.cy = o.cy;
    t.w = o.w;
    t.h = o.h;
    t.angle = o.angle;
    t.score = o.score;
    t.class_id = o.class_id;
    t.idx = idx;
    return t;
}

// Full-canvas row-major run-length encoding of a binary mask (runs alternate
// 0/1 starting with the leading zero-run count) — the compact form used by the
// A/B comparisons to diff final masks pixel-for-pixel.
std::string mask_rle(int canvas_w, int canvas_h, const yolo::SegMask& m) {
    std::vector<uint8_t> full((size_t)canvas_w * canvas_h, 0);
    for (int y = 0; y < m.h; y++)
        for (int x = 0; x < m.w; x++)
            if (m.bits[(size_t)y * m.w + x]) {
                const int cy = m.y + y, cx = m.x + x;
                if (cy >= 0 && cy < canvas_h && cx >= 0 && cx < canvas_w) full[(size_t)cy * canvas_w + cx] = 1;
            }
    std::vector<uint64_t> runs;
    uint64_t run = 0;
    uint8_t val = 0;
    for (size_t i = 0; i < full.size(); i++) {
        if (full[i] == val) {
            run++;
        } else {
            runs.push_back(run);
            run = 1;
            val ^= 1;
        }
    }
    runs.push_back(run);
    std::string out;
    char buf[24];
    for (size_t i = 0; i < runs.size(); i++) {
        snprintf(buf, sizeof(buf), "%s%llu", i ? "," : "", (unsigned long long)runs[i]);
        out += buf;
    }
    return out;
}

void write_track_json_entry(FILE* f, const yolo::track::TrackDet& t, int track_id,
                            const std::vector<float>* kpts = nullptr) {
    fprintf(f, "{\"cx\":%.6f,\"cy\":%.6f,\"w\":%.6f,\"h\":%.6f", t.cx, t.cy, t.w, t.h);
    if (t.angled()) fprintf(f, ",\"angle\":%.6f", t.angle);
    if (track_id >= 0) fprintf(f, ",\"id\":%d", track_id);
    fprintf(f, ",\"score\":%.6f,\"cls\":%d,\"idx\":%d", t.score, t.class_id, t.idx);
    if (kpts && !kpts->empty()) {
        fputs(",\"kpts\":[", f);
        for (size_t j = 0; j < kpts->size(); j++) fprintf(f, "%s%.3f", j ? "," : "", (*kpts)[j]);
        fputs("]", f);
    }
    fputs("}", f);
}

// --dets-jsonl replay input: per-frame detections in the same schema
// --tracks-json writes (flat objects; angle/id optional). Driving the tracker
// with recorded or hand-written detections skips the model entirely — the
// tracking analog of --input-f32, and how the parity script replays frames.
struct ReplayFrame {
    std::vector<yolo::track::TrackDet> dets, dets_del;
};

float json_num(const std::string& obj, const char* key, float def) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t p = obj.find(pat);
    if (p == std::string::npos) return def;
    return strtof(obj.c_str() + p + pat.size(), nullptr);
}

int json_int(const std::string& obj, const char* key, int def) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t p = obj.find(pat);
    if (p == std::string::npos) return def;
    return atoi(obj.c_str() + p + pat.size());
}

std::vector<yolo::track::TrackDet> json_det_array(const std::string& line, const char* array_key) {
    std::vector<yolo::track::TrackDet> out;
    // Colon suffix disambiguates "detections": from "detections_del":.
    const size_t arr = line.find(std::string("\"") + array_key + "\":");
    if (arr == std::string::npos) return out;
    size_t pos = line.find('[', arr);
    if (pos == std::string::npos) return out;
    while (true) {
        const size_t obj_begin = line.find('{', pos);
        const size_t arr_end = line.find(']', pos);
        if (obj_begin == std::string::npos || (arr_end != std::string::npos && arr_end < obj_begin)) break;
        const size_t obj_end = line.find('}', obj_begin);
        if (obj_end == std::string::npos) break;
        const std::string obj = line.substr(obj_begin, obj_end - obj_begin + 1);
        yolo::track::TrackDet d;
        d.cx = json_num(obj, "cx", 0);
        d.cy = json_num(obj, "cy", 0);
        d.w = json_num(obj, "w", 0);
        d.h = json_num(obj, "h", 0);
        d.score = json_num(obj, "score", 0);
        d.class_id = json_int(obj, "cls", 0);
        d.idx = json_int(obj, "idx", 0);
        const size_t angle_key = obj.find("\"angle\":");
        d.angle = angle_key == std::string::npos ? -10.0f : json_num(obj, "angle", -10.0f);
        out.push_back(d);
        pos = obj_end + 1;
    }
    return out;
}

std::vector<ReplayFrame> read_replay_frames(const std::string& path) {
    std::vector<ReplayFrame> frames;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return frames;
    char buf[65536];
    size_t n;
    std::string content;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) content.append(buf, n);
    fclose(f);
    size_t pos = 0;
    while (pos < content.size()) {
        size_t eol = content.find('\n', pos);
        if (eol == std::string::npos) eol = content.size();
        const std::string line = content.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.find("detections") == std::string::npos) continue;
        ReplayFrame fr;
        fr.dets = json_det_array(line, "detections");
        fr.dets_del = json_det_array(line, "detections_del");
        frames.push_back(std::move(fr));
    }
    return frames;
}

int cmd_track(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    const std::string dets_in_path = arg_s(args, "dets-jsonl");
    if (model_path.empty() && dets_in_path.empty()) {
        fprintf(stderr, "--model (or --dets-jsonl) is required\n");
        return 1;
    }
    if (source.empty() && dets_in_path.empty()) {
        fprintf(stderr, "--source is required\n");
        return 1;
    }
    yolo::ModelMeta meta;
    std::vector<ReplayFrame> replay;
    if (!dets_in_path.empty()) {
        // Replay mode: drive the tracker with recorded detections, no model.
        replay = read_replay_frames(dets_in_path);
        if (replay.empty()) {
            fprintf(stderr, "no frames parsed from --dets-jsonl %s\n", dets_in_path.c_str());
            return 1;
        }
        meta.task = "detect";
        meta.imgsz = 640;
        meta.max_det = 300;
    } else {
        meta = yolo::read_gguf_meta(model_path);
        if (meta.imgsz <= 0) return 1;
        // Trackable tasks in canonical order, mirroring trackers/track.py's on_predict_start.
        if (meta.task != "detect" && meta.task != "segment" && meta.task != "pose" && meta.task != "obb") {
            fprintf(stderr, "task '%s' doesn't support mode=track, valid tasks are detect, segment, pose, obb\n",
                    meta.task.c_str());
            return 1;
        }
    }

    // Tracker config: bare type (official YAML defaults) or a YAML file, plus
    // optional per-key overrides via --tracker-config.
    yolo::track::TrackConfig tcfg;
    if (!yolo::track::resolve_tracker_config(arg_s(args, "tracker", "tracktrack"), tcfg)) return 1;
    const std::string cfg_override = arg_s(args, "tracker-config");
    if (!cfg_override.empty() && !yolo::track::load_tracker_config_yaml(cfg_override, tcfg)) return 1;
    auto tracker = yolo::track::create_tracker(tcfg);
    if (!tracker) return 1;

    const std::vector<std::string> frames = list_frame_files(source);
    // Optional frames for GMC while replaying detections (--dets-jsonl + --frames).
    std::vector<std::string> gmc_frames;
    const std::string gmc_source = arg_s(args, "frames");
    if (!gmc_source.empty()) {
        gmc_frames = list_frame_files(gmc_source);
        if (gmc_frames.empty()) {
            fprintf(stderr, "no frames found for --frames %s\n", gmc_source.c_str());
            return 1;
        }
        if (gmc_frames.size() != replay.size()) {
            fprintf(stderr, "--frames has %zu files but --dets-jsonl has %zu frames; counts must match\n",
                    gmc_frames.size(), replay.size());
            return 1;
        }
    }
    if (replay.empty() && frames.empty()) {
        fprintf(stderr, "no frames found for --source %s\n", source.c_str());
        return 1;
    }

    yolo::PostprocConfig cfg;
    // ByteTrack-family association needs low-confidence detections; 0.1 mirrors
    // Model.track()'s conf default.
    cfg.conf_thres = (float)arg_f(args, "conf", 0.1);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", meta.max_det);
    if (!(cfg.conf_thres > 0.0f && cfg.conf_thres < 1.0f) || !(cfg.iou_thres >= 0.0f && cfg.iou_thres <= 1.0f) ||
        cfg.max_det <= 0) {
        fprintf(stderr, "--conf must be in (0,1), --iou in [0,1], and --max-det positive\n");
        return 1;
    }
    // Loose-NMS recovery for TrackTrack (detect/obb only), mirroring
    // track_tracker.attach_raw_preds_hook + compute_dets_del.
    yolo::PostprocConfig loose = cfg;
    loose.iou_thres = 0.95f;
    const bool want_recovered = tcfg.tracker_type == "tracktrack" && (meta.task == "detect" || meta.task == "obb");

    const std::string tracks_json = arg_s(args, "tracks-json");
    const std::string out_prefix = arg_s(args, "out");
    FILE* jf = nullptr;
    if (!tracks_json.empty()) {
        jf = fopen(tracks_json.c_str(), "wb");
        if (!jf) {
            fprintf(stderr, "failed to write --tracks-json %s\n", tracks_json.c_str());
            return 1;
        }
    }

    SessionPtr session(nullptr, yolo::free_session);
    yolo::Session* s = nullptr;
    yolo::LetterboxInfo info{};
    int canvas_w = 0, canvas_h = 0;
    int rc = 0;

    const size_t n_frames = replay.empty() ? frames.size() : replay.size();
    for (size_t k = 0; k < n_frames; k++) {
        yolo::Image img;
        yolo::track::FrameInput fin;
        fin.frame = nullptr;
        // --no-gmc feeds a null frame: GMC is skipped exactly like Python's
        // update(results, img=None) path (used for deterministic parity replays).
        if (replay.empty()) {
            fin.frame = arg_s(args, "no-gmc").empty() ? &img : nullptr;
        } else if (!gmc_frames.empty()) {
            // Replay mode with optional GMC frames: detections come from the
            // JSONL, frames feed the GMC estimator.
            if (!yolo::load_image(gmc_frames[k], img)) {
                rc = 1;
                break;
            }
            fin.frame = &img;
        }
        auto fin_dets = [&](yolo::track::TrackDet d) { fin.dets.push_back(d); };
        auto fin_del = [&](yolo::track::TrackDet d) { fin.dets_del.push_back(d); };
        std::vector<yolo::Detection> dets;
        std::vector<yolo::SegMask> masks;
        std::vector<yolo::PoseDetection> poses;
        std::vector<yolo::OBBDetection> obbs;
        if (replay.empty()) {
            if (!yolo::load_image(frames[k], img)) {
                rc = 1;
                break;
            }
            std::vector<float> input;
            yolo::letterbox_image(img, meta.imgsz, info, input);
            if (!s) {
                canvas_w = info.imgsz_w;
                canvas_h = info.imgsz_h;
                yolo::SessionOptions sopts;
                sopts.threads = arg_i(args, "threads", 0);
                sopts.input_w = canvas_w;
                sopts.input_h = canvas_h;
                session.reset(yolo::create_session(model_path, sopts));
                s = session.get();
                if (!s) {
                    rc = 1;
                    break;
                }
            } else if (info.imgsz_w != canvas_w || info.imgsz_h != canvas_h) {
                fprintf(stderr, "frame %s letterboxes to %dx%d; tracking requires every frame to share the\n"
                                "first frame's canvas (%dx%d)\n",
                        frames[k].c_str(), info.imgsz_w, info.imgsz_h, canvas_w, canvas_h);
                rc = 1;
                break;
            }
            if (!yolo::session_run(s, input.data())) {
                rc = 1;
                break;
            }
            std::vector<float> raw;
            int no = 0, na = 0;
            if (!yolo::session_read_output(s, raw, no, na)) {
                rc = 1;
                break;
            }

            if (meta.task == "detect" || meta.task == "segment") {
                dets = yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
                if (meta.task == "segment") {
                    std::vector<float> proto;
                    int nm = 0, pw = 0, ph = 0;
                    if (!yolo::session_read_proto(s, proto, nm, pw, ph)) {
                        rc = 1;
                        break;
                    }
                    masks = yolo::compose_masks(dets, raw, na, s->model.meta, proto, pw, ph, canvas_w, canvas_h);
                }
                yolo::unscale_boxes(dets, info, img.w, img.h);
            } else if (meta.task == "pose") {
                poses =
                    yolo::postprocess_pose(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
                yolo::unscale_pose(poses, info, img.w, img.h);
            } else if (meta.task == "obb") {
                obbs = yolo::postprocess_obb(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
                yolo::unscale_obb(obbs, info);
            }

            if (meta.task == "obb") {
                for (size_t i = 0; i < obbs.size(); i++) fin_dets(obb_to_track(obbs[i], (int)i));
            } else if (meta.task == "pose") {
                for (size_t i = 0; i < poses.size(); i++) fin_dets(det_to_track(poses[i].det, (int)i));
            } else {
                for (size_t i = 0; i < dets.size(); i++) fin_dets(det_to_track(dets[i], (int)i));
            }
            if (want_recovered) {
                std::vector<yolo::Detection> loose_dets =
                    yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), loose);
                yolo::unscale_boxes(loose_dets, info);
                for (const auto& ld : loose_dets) {
                    float best = 0.0f;
                    for (const auto& td : dets) {
                        const float xx1 = std::max(ld.x1, td.x1), yy1 = std::max(ld.y1, td.y1);
                        const float xx2 = std::min(ld.x2, td.x2), yy2 = std::min(ld.y2, td.y2);
                        const float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
                        const float uni =
                            (ld.x2 - ld.x1) * (ld.y2 - ld.y1) + (td.x2 - td.x1) * (td.y2 - td.y1) - inter;
                        if (uni > 0) best = std::max(best, inter / uni);
                    }
                    if (best < 0.97f) fin_del(det_to_track(ld, -1));
                }
            }
        } else {
            for (const auto& d : replay[k].dets) fin_dets(d);
            for (const auto& d : replay[k].dets_del) fin_del(d);
        }

        const std::vector<yolo::track::TrackedBox> tracks = tracker->update(fin);
        static const std::vector<std::string> kNoNames;
        const auto& names = s ? s->model.meta.class_names : kNoNames;
        const char* image_label = replay.empty() ? frames[k].c_str() : "replay";
        printf("frame %zu %s: %zu track%s", k, image_label, tracks.size(), tracks.size() == 1 ? "" : "s");
        if (s)
            printf(" (%s, %s, %dx%d, backend=%s)", s->model.meta.name.c_str(), s->model.meta.dtype.c_str(), img.w,
                   img.h, yolo::backend_name(s->backend));
        printf("\n");
        for (const auto& t : tracks) {
            const char* cname = t.det.class_id < (int)names.size() ? names[t.det.class_id].c_str() : "?";
            if (t.det.angled())
                printf("  #%d %-12s %.2f  rbox cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.1fdeg\n", t.track_id, cname,
                       t.det.score, t.det.cx, t.det.cy, t.det.w, t.det.h, t.det.angle * 180.0f / 3.14159265f);
            else
                printf("  #%d %-12s %.2f  [%.1f, %.1f, %.1f, %.1f]\n", t.track_id, cname, t.det.score, t.det.cx - t.det.w / 2,
                       t.det.cy - t.det.h / 2, t.det.cx + t.det.w / 2, t.det.cy + t.det.h / 2);
        }

        if (jf) {
            fprintf(jf, "{\"frame\":%zu,\"image\":\"%s\",\"detections\":[", k, json_escape(image_label).c_str());
            for (size_t i = 0; i < fin.dets.size(); i++) {
                if (i) fputs(",", jf);
                write_track_json_entry(jf, fin.dets[i], -1);
            }
            fputs("]", jf);
            if (want_recovered) {
                fputs(",\"detections_del\":[", jf);
                for (size_t i = 0; i < fin.dets_del.size(); i++) {
                    if (i) fputs(",", jf);
                    write_track_json_entry(jf, fin.dets_del[i], -1);
                }
                fputs("]", jf);
            }
            fputs(",\"tracks\":[", jf);
            // ModelMeta.nk is the total keypoint float count (nkpts * ndim).
            const size_t kpt_len = meta.nk > 0 ? (size_t)meta.nk : 51;
            for (size_t i = 0; i < tracks.size(); i++) {
                if (i) fputs(",", jf);
                const std::vector<float>* kpts = nullptr;
                if (meta.task == "pose" && tracks[i].det.idx >= 0 && tracks[i].det.idx < (int)poses.size() &&
                    poses[tracks[i].det.idx].kpts.size() >= kpt_len)
                    kpts = &poses[tracks[i].det.idx].kpts;
                write_track_json_entry(jf, tracks[i].det, tracks[i].track_id, kpts);
            }
            fputs("]}\n", jf);
        }

        if (!out_prefix.empty()) {
            char path[1024];
            snprintf(path, sizeof(path), "%s_%05zu.png", out_prefix.c_str(), k);
            std::vector<std::string> labels;
            if (meta.task == "obb") {
                labels.resize(obbs.size());
                for (const auto& t : tracks) {
                    if (t.det.idx < 0 || t.det.idx >= (int)obbs.size()) continue;
                    const char* cname = t.det.class_id < (int)names.size() ? names[t.det.class_id].c_str() : "?";
                    char buf[128];
                    snprintf(buf, sizeof(buf), "%s #%d %.2f", cname, t.track_id, t.det.score);
                    labels[t.det.idx] = buf;
                }
                if (!yolo::draw_obb(path, img, obbs, names, &labels)) {
                    rc = 1;
                    break;
                }
            } else if (meta.task == "pose") {
                labels.resize(poses.size());
                for (const auto& t : tracks) {
                    if (t.det.idx < 0 || t.det.idx >= (int)poses.size()) continue;
                    const char* cname = t.det.class_id < (int)names.size() ? names[t.det.class_id].c_str() : "?";
                    char buf[128];
                    snprintf(buf, sizeof(buf), "%s #%d %.2f", cname, t.track_id, t.det.score);
                    labels[t.det.idx] = buf;
                }
                if (!yolo::draw_pose(path, img, poses, names, &labels)) {
                    rc = 1;
                    break;
                }
            } else {
                labels.resize(dets.size());
                for (const auto& t : tracks) {
                    if (t.det.idx < 0 || t.det.idx >= (int)dets.size()) continue;
                    const char* cname = t.det.class_id < (int)names.size() ? names[t.det.class_id].c_str() : "?";
                    char buf[128];
                    snprintf(buf, sizeof(buf), "%s #%d %.2f", cname, t.track_id, t.det.score);
                    labels[t.det.idx] = buf;
                }
                if (!yolo::draw_detections(path, img, dets, names, meta.task == "segment" ? &masks : nullptr,
                                           &info, &labels)) {
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (jf) fclose(jf);
    return rc;
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
        fprintf(stderr, "--text-embed must be a [nc, 512] YTXT0002 f32 blob\n");
        return 1;
    }
    if (meta.has_text_input && !wc.empty() && text_nc && (int)wc.size() != text_nc) {
        fprintf(stderr, "--classes (%zu classes) and --text-embed nc (%d) disagree\n", wc.size(), text_nc);
        return 1;
    }
    const bool supply_text = meta.has_text_input && (!wc.empty() || text_nc > 0);
    sopts.world_nc = supply_text ? (wc.empty() ? text_nc : (int)wc.size()) : 0;
    SessionPtr session(yolo::create_session(model_path, sopts), yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    std::string world_text_source;
    // Text setup is outside the timed per-frame loop: the vocabulary is set
    // once per session and remains constant for every measured frame.
    if (!supply_text) {
        world_text_source = "builtin";
    } else {
        std::vector<float> text_embed((size_t)s->world_nc * clip::EMBED_DIM, 0.0f);
        if (!world_text_embed.empty()) {
            std::vector<int32_t> dims = {s->world_nc, clip::EMBED_DIM};
            if (!read_f32(world_text_embed.c_str(), "YTXT0002", dims, text_embed) ||
                dims[0] != s->world_nc || dims[1] != clip::EMBED_DIM) {
                fprintf(stderr, "--text-embed must be [nc=%d, 512] YTXT0002 f32 blob\n", s->world_nc);
                return 1;
            }
            world_text_source = "ytxt";
        } else {
#if defined(YOLO_GGML_CLIP) && YOLO_GGML_CLIP
        if (!s->model.meta.text_model.empty()) {
            // YOLOE: MobileCLIP tower encodes --classes; reprta runs inside the
            // detector graph (v4), so the raw feature is the graph text input.
            std::string tm = arg_s(args, "text-model", "models/gguf/mobileclip2_b-f16.gguf");
            mobileclip::MobileclipSession* ms = mobileclip::mobileclip_create_session(tm);
            if (!ms) return 1;
            for (int i = 0; i < s->world_nc; i++) {
                if (!mobileclip::mobileclip_encode_string(ms, wc[i].c_str(),
                                                          text_embed.data() + (size_t)i * clip::EMBED_DIM)) {
                    mobileclip::mobileclip_free_session(ms);
                    return 1;
                }
            }
            mobileclip::mobileclip_free_session(ms);
            world_text_source = "mobileclip";
        } else {
            std::string clip_model = arg_s(args, "clip-model", "models/gguf/clip-ViT-B-32-f16.gguf");
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
        }
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
    if (cmd == "track") return cmd_track(args);
    if (cmd == "semantic") return cmd_semantic(args);
    if (cmd == "classify") return cmd_classify(args);
    if (cmd == "depth") return cmd_depth(args);
    if (cmd == "bench") return cmd_bench(args);
    usage();
    return 1;
}
