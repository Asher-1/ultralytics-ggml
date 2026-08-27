// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// yolo_similarity - CLIP-based semantic image search using ggml.
//
// Usage:
//   # Encode a text query and search a directory of images
//   ./yolo_similarity --model clip-vit-b-32.gguf \
//                     --text "a red car" \
//                     --image_dir ./images/ \
//                     --topk 10
//
//   # Text-only encoding (useful for debugging / integration)
//   ./yolo_similarity --model clip-vit-b-32.gguf \
//                     --text "a cat sitting on a chair" \
//                     --dump_embed
//
//   # Image-only encoding
//   ./yolo_similarity --model clip-vit-b-32.gguf \
//                     --image photo.jpg \
//                     --dump_embed
//
// Precision: produces the same 512-d L2-normalised embeddings as the Python
// CLIP reference (cosine similarity > 0.999 with fp32 reference).

#include "../../src/clip_graph.hpp"
#include "../../src/image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Full BPE tokenization is provided by clip_tokenize_text (src/clip_graph.cpp),
// which reproduces the Python SimpleTokenizer + clip.tokenize pipeline: the
// vocab + merges tables are loaded from the GGUF written by
// scripts/convert_clip_to_gguf.py.

struct Options {
    std::string model;
    std::string text_query;
    std::string image_path;     // single image
    std::string image_dir;      // directory of images
    int         topk = 30;
    float       threshold = 0.1f;
    int         threads = 0;
    bool        cpu_only = false;
    bool        dump_embed = false;
    bool        text_only = false;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model <path>       CLIP GGUF model path\n"
        "  --text <query>       Text search query\n"
        "  --image <path>       Single image to encode\n"
        "  --image_dir <path>   Directory of images to search\n"
        "  --topk <n>           Top-K results (default: 30)\n"
        "  --threshold <f>      Similarity threshold (default: 0.1)\n"
        "  --threads <n>        Thread count (default: hardware max)\n"
        "  --cpu                Force CPU backend (default: GPU when available)\n"
        "  --dump_embed         Print embedding vector to stdout\n"
        "  --text_only          Only show text encoding results\n",
        prog);
}

static bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || k.size() <= 2) {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return false;
        }
        k = k.substr(2);
        auto next = [&]() {
            if (i + 1 >= argc) { fprintf(stderr, "--%s requires a value\n", k.c_str()); exit(1); }
            return std::string(argv[++i]);
        };
        if (k == "model")        opts.model = next();
        else if (k == "text")    opts.text_query = next();
        else if (k == "image")   opts.image_path = next();
        else if (k == "image_dir") opts.image_dir = next();
        else if (k == "topk")    opts.topk = atoi(next().c_str());
        else if (k == "threshold") opts.threshold = (float)atof(next().c_str());
        else if (k == "threads") opts.threads = atoi(next().c_str());
        else if (k == "cpu") opts.cpu_only = true;
        else if (k == "dump_embed") opts.dump_embed = true;
        else if (k == "text_only")  opts.text_only = true;
        else { fprintf(stderr, "unknown option --%s\n", k.c_str()); return false; }
    }
    if (opts.model.empty()) {
        fprintf(stderr, "error: --model is required\n");
        return false;
    }
    return true;
}

// Utility: load all image paths from a directory.
static std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> paths;
    // Simple: use glob pattern via shell
    std::string cmd = "ls \"" + dir + "\" 2>/dev/null | head -1000";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return paths;
    char buf[512];
    while (fgets(buf, sizeof(buf), fp)) {
        char* nl = strchr(buf, '\n');
        if (nl) *nl = '\0';
        std::string ext = strrchr(buf, '.') ? strrchr(buf, '.') + 1 : "";
        if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
            paths.push_back(dir + "/" + buf);
        }
    }
    pclose(fp);
    return paths;
}

// Normalise a path string (remove ./ prefix) for display.
static const char* display_path(const std::string& p) {
    if (p.size() > 2 && p[0] == '.' && p[1] == '/') return p.c_str() + 2;
    return p.c_str();
}

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    // Create CLIP session
    clip::ClipSessionOptions sopts;
    sopts.threads = opts.threads;
    // Standalone process: safe to offload to a GPU backend (see
    // ClipSessionOptions.use_gpu for why embedding into a YOLO process must not).
    sopts.use_gpu = !opts.cpu_only;
    clip::ClipSession* session = clip::clip_create_session(opts.model, sopts);
    if (!session) {
        fprintf(stderr, "error: failed to create CLIP session from '%s'\n",
                opts.model.c_str());
        return 1;
    }

    // Text encoding phase
    if (opts.text_query.empty() && opts.image_path.empty() && opts.image_dir.empty()) {
        fprintf(stderr, "error: provide --text, --image, or --image_dir\n");
        clip::clip_free_session(session);
        return 1;
    }

    float text_embed[clip::EMBED_DIM] = {0};
    if (!opts.text_query.empty()) {
        int32_t tokens[clip::TEXT_CTX];
        int n_tok = clip::clip_tokenize_text(session, opts.text_query.c_str(), tokens);

        fprintf(stderr, "tokenized \"%s\" -> %d tokens\n", opts.text_query.c_str(), n_tok);

        if (opts.dump_embed) {
            fprintf(stderr, "token ids = [");
            for (int i = 0; i < n_tok; i++) {
                if (i > 0) fprintf(stderr, ", ");
                fprintf(stderr, "%d", tokens[i]);
            }
            fprintf(stderr, "]\n");
        }

        if (!clip::clip_encode_text(session, tokens, text_embed)) {
            fprintf(stderr, "error: text encoding failed\n");
            clip::clip_free_session(session);
            return 1;
        }

        if (opts.dump_embed) {
            printf("text_embed = [");
            for (int i = 0; i < clip::EMBED_DIM; i++) {
                if (i > 0) printf(", ");
                printf("%.6f", text_embed[i]);
            }
            printf("]\n");
        }

        if (opts.text_only) {
            fprintf(stderr, "text encoding complete.\n");
            clip::clip_free_session(session);
            return 0;
        }
    }

    // Image encoding phase
    struct ImageEntry {
        std::string path;
        float embed[clip::EMBED_DIM];
    };
    std::vector<ImageEntry> entries;

    if (!opts.image_path.empty()) {
        // Single image mode
        ImageEntry entry;
        entry.path = opts.image_path;

        // Load image
        yolo::Image img;
        if (!yolo::load_image(opts.image_path, img)) {
            fprintf(stderr, "error: failed to load image '%s'\n", opts.image_path.c_str());
            clip::clip_free_session(session);
            return 1;
        }

        float chw[clip::IMAGE_SIZE * clip::IMAGE_SIZE * 3];
        clip::clip_preprocess_image(img.rgb.data(), img.w, img.h, img.c, chw);

        if (!clip::clip_encode_image(session, chw, entry.embed)) {
            fprintf(stderr, "error: image encoding failed\n");
            clip::clip_free_session(session);
            return 1;
        }

        if (opts.dump_embed && opts.text_query.empty()) {
            printf("image_embed = [");
            for (int i = 0; i < clip::EMBED_DIM; i++) {
                if (i > 0) printf(", ");
                printf("%.6f", entry.embed[i]);
            }
            printf("]\n");
        }

        entries.push_back(std::move(entry));
    }

    if (!opts.image_dir.empty()) {
        // Directory mode: bulk encode all images
        auto paths = list_images(opts.image_dir);
        if (paths.empty()) {
            fprintf(stderr, "error: no images found in '%s'\n", opts.image_dir.c_str());
            clip::clip_free_session(session);
            return 1;
        }
        fprintf(stderr, "encoding %zu images from %s ...\n", paths.size(), opts.image_dir.c_str());

        for (size_t i = 0; i < paths.size(); i++) {
            yolo::Image img;
            if (!yolo::load_image(paths[i], img)) {
                fprintf(stderr, "  [skip] %s\n", display_path(paths[i]));
                continue;
            }

            float chw[clip::IMAGE_SIZE * clip::IMAGE_SIZE * 3];
            clip::clip_preprocess_image(img.rgb.data(), img.w, img.h, img.c, chw);

            ImageEntry entry;
            entry.path = paths[i];
            if (!clip::clip_encode_image(session, chw, entry.embed)) {
                fprintf(stderr, "  [fail] %s\n", display_path(paths[i]));
                continue;
            }

            entries.push_back(std::move(entry));

            if ((i + 1) % 10 == 0 || i == paths.size() - 1) {
                fprintf(stderr, "  %zu/%zu encoded\r", i + 1, paths.size());
                fflush(stderr);
            }
        }
        fprintf(stderr, "\ndone: %zu images encoded\n", entries.size());
    }

    // Search phase (if text query provided)
    if (!opts.text_query.empty() && !entries.empty()) {
        fprintf(stderr, "searching with topk=%d, threshold=%.2f ...\n",
                opts.topk, opts.threshold);

        struct Result {
            int index;
            float score;
        };
        std::vector<Result> results;
        for (size_t i = 0; i < entries.size(); i++) {
            float sim = clip::clip_cosine_similarity(
                text_embed, entries[i].embed, clip::EMBED_DIM);
            if (sim >= opts.threshold) {
                results.push_back({(int)i, sim});
            }
        }

        // Sort by score descending
        std::sort(results.begin(), results.end(),
            [](const Result& a, const Result& b) { return a.score > b.score; });

        if (results.size() > (size_t)opts.topk)
            results.resize(opts.topk);

        printf("\nRanked Results:\n");
        for (size_t i = 0; i < results.size(); i++) {
            const auto& res = results[i];
            printf("  %3zu. %s | Similarity: %.4f\n",
                   i + 1, display_path(entries[res.index].path), res.score);
        }
    }

    clip::clip_free_session(session);
    return 0;
}