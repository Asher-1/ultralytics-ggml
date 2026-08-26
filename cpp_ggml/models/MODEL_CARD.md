# YOLO ggml model card

This directory is the single model store for the C++ integration. PyTorch checkpoints are conversion inputs; GGUF
files are runtime artifacts. Both are ignored by Git and can be regenerated.

```text
models/
├── MODEL_CARD.md
├── pytorch/                    # canonical .pt conversion inputs
│   ├── yolov8{n,s,m,l,x}.pt
│   ├── yolov8{n,s,m,l,x}-seg.pt
│   ├── yolov8{s,m,l,x}-world.pt
│   ├── yoloe-{v8,11}{s,m,l}-seg.pt
│   ├── yoloe-26{n,s,m,l,x}-seg.pt
│   ├── yolo26{n,s,m,l,x}.pt
│   ├── yolo26{n,s,m,l,x}-seg.pt
│   ├── yolo26{n,s,m,l,x}-depth.pt
│   ├── yolo26{n,s,m,l,x}-pose.pt
│   ├── yolo26{n,s,m,l,x}-obb.pt
│   ├── yolo26{n,s,m,l,x}-sem.pt
│   └── yolo26{n,s,m,l,x}-cls.pt
└── gguf/                       # generated runtime models
    ├── <detect-model>-{f32,f16,q8_0}.gguf
    ├── <detect-model>-seg-{f32,f16,q8_0}.gguf
    ├── <detect-model>-world-{f32,f16,q8_0}.gguf
    ├── yoloe-{v8,11}{s,m,l}-seg-{f32,f16,q8_0}.gguf
    ├── yoloe-26{n,s,m,l,x}-seg-{f32,f16,q8_0}.gguf
    ├── yolo26{n,s,m,l,x}-depth-{f32,f16,q8_0}.gguf
    ├── yolo26{n,s,m,l,x}-pose-{f32,f16,q8_0}.gguf
    ├── yolo26{n,s,m,l,x}-obb-{f32,f16,q8_0}.gguf
    ├── yolo26{n,s,m,l,x}-sem-{f32,f16,q8_0}.gguf
    ├── yolo26{n,s,m,l,x}-cls-{f32,f16,q8_0}.gguf
    └── clip-ViT-B-32.gguf
```

Do not put checkpoints in `cpp_ggml/` or the repository root. The converter resolves model aliases against
`models/pytorch/` and writes to `models/gguf/` by default.

## Layout migration

Older checkouts may contain the 11 source checkpoints directly under `cpp_ggml/` (or a duplicate `yolo26n.pt` in the
repository root). Those paths are retired. Move any locally retained files once, then remove the old copies:

```bash
mkdir -p cpp_ggml/models/pytorch
for name in yolov8n yolov8s yolov8m yolov8l yolov8x yolo26n yolo26s yolo26m yolo26l yolo26x yolo26n-depth; do
    test -f "cpp_ggml/$name.pt" && mv "cpp_ggml/$name.pt" "cpp_ggml/models/pytorch/$name.pt"
done
```

All conversion, benchmark, parity, and rendering scripts resolve this canonical directory; no script should reference
`cpp_ggml/<model>.pt` or a root-level checkpoint.

## Supported models

| Model         | Task           | Default input | Recommended use                                      |
| ------------- | -------------- | ------------: | ---------------------------------------------------- |
| YOLOv8n       | detect         |           640 | Lowest detection latency and memory use              |
| YOLOv8s       | detect         |           640 | Small edge deployments needing more capacity than n  |
| YOLOv8m       | detect         |           640 | Balanced accuracy and compute                        |
| YOLOv8l       | detect         |           640 | Accuracy-oriented GPU deployment                     |
| YOLOv8x       | detect         |           640 | Highest-capacity YOLOv8 integration target           |
| YOLO26n       | detect         |           640 | Lowest-latency end-to-end YOLO26 detector            |
| YOLO26s       | detect         |           640 | Compact end-to-end detector                          |
| YOLO26m       | detect         |           640 | Balanced end-to-end detector                         |
| YOLO26l       | detect         |           640 | Accuracy-oriented end-to-end detector                |
| YOLO26x       | detect         |           640 | Highest-capacity YOLO26 detection target             |
| YOLOv8s-world .. YOLOv8x-world | detect (open-vocabulary) |          640 | Open-vocabulary detection with CLIP text embeddings (--classes, --text-embed) |
| YOLOE-v8/11 s..l and YOLOE-26 n..x, `-seg` | open-vocabulary instance segment | 640 | MobileCLIP YTXT prepared for the exact YOLOE checkpoint |
| YOLOv8n-seg .. YOLOv8x-seg | instance segment |   640 | YOLOv8 boxes + on-device instance masks        |
| YOLO26n-seg .. YOLO26x-seg   | instance segment |   640 | YOLO26 boxes + on-device instance masks         |
| YOLO26n-depth .. YOLO26x-depth | absolute depth |        768 | Monocular metric-depth preview and spatial reasoning |
| YOLO26n-pose .. YOLO26x-pose    | keypoints      |          640 | COCO-17 person pose (RLE head), boxes + 17 keypoints |
| YOLO26n-obb .. YOLO26x-obb      | oriented boxes |          640 | DOTA-15 rotated boxes (raw angle, no sigmoid)       |
| YOLO26n-sem .. YOLO26x-sem      | semantic seg   |          640 | Cityscapes-19 dense per-pixel class map             |
| YOLO26n-cls .. YOLO26x-cls      | classification |          224 | ImageNet-1000 logits (checkpoint-baked transforms)  |
| CLIP ViT-B/32       | text + image encoder | 224x224 / 77 tokens | 512-d L2-normalised embeddings for semantic similarity search |

Detection models use COCO's 80 classes. YOLO26 detection checkpoints use the end-to-end head exported by the local
Ultralytics checkout. YOLO-World detection models are open-vocabulary: they accept a class list at runtime
(--classes) or a precomputed text embedding blob (--text-embed). The CLIP model (clip-ViT-B-32.gguf) is
used to encode class text when --classes is provided without --text-embed; it can also be used independently
for image/text similarity via the similarity subcommand (--model clip-ViT-B-32.gguf --source img.jpg).
Segmentation models additionally emit 32 mask prototypes at one-quarter resolution and compose
instance masks on device. Depth models produce one floating-point distance in meters per source pixel. Pose models
emit one box plus 17 COCO keypoints (x, y, visibility) per person; OBB models emit rotated boxes in the DOTA-15 class
set; semantic models emit an argmax class map on the Cityscapes-19 class set; classify models emit ImageNet-1000
softmax probabilities. The five YOLO26 scales (n/s/m/l/x) share one graph per task; scale changes tensor shapes, not
the public CLI or GGUF contract.

YOLOE models use MobileCLIP plus the detector checkpoint's `reprta` block before
the `[nc, 512]` text tensor is consumed. Generate this detector-specific YTXT
with `scripts/encode_mobileclip_text.py --detector models/pytorch/yoloe-26n-seg.pt`;
the runtime rejects `--classes` without `--text-embed` for YOLOE so it cannot
mistake a ViT-B/32 CLIP vector for a MobileCLIP vector.

### YOLOv8 detector family

- **YOLOv8n** is the default for latency-sensitive applications and constrained GPUs.
- **YOLOv8s** trades a small latency increase for more capacity while remaining suitable for edge deployment.
- **YOLOv8m** is the balanced choice when throughput and detection quality have similar weight.
- **YOLOv8l** targets accuracy-oriented GPU services where a larger memory and latency budget is available.
- **YOLOv8x** is the highest-capacity YOLOv8 checkpoint in this integration and the runtime/memory stress case.

All five use the same graph and postprocessing owners; model scale changes tensor shapes, not the public CLI or GGUF
contract.

### YOLOv8-World open-vocabulary family

**YOLOv8{s,m,l,x}-world** extend YOLOv8 detection with a CLIP text-embedding branch, enabling open-vocabulary
detection at runtime. Use --classes "person,bus,car" to set the class list; the CLI loads the CLIP text encoder
(clip-ViT-B-32.gguf) and encodes each class name into a 512-d L2-normalised embedding. The text tensors are fed
through C2fAttn and ImagePoolingAttn layers that attend image features to the class labels, then a ContrastiveHead
produces per-anchor class scores on the (L2-normalised) similarity between text and image embeddings.

For batch inference or to reuse precomputed embeddings, pass --text-embed file.bin with a [nc, 512] row-major F32
blob (YTXT0001 format). When both --classes and --text-embed are provided, the class counts must agree.

Example:

```bash
# Encode classes via CLIP text encoder at startup
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car" --conf 0.25

# Precomputed text embeddings (avoids CLIP model load)
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --text-embed path/to/embeddings.ytxt --classes "person,bus,car" --conf 0.25
```

YOLO-World checkpoints convert to f32, f16 and q8_0 (12 GGUF files for the s/m/l/x family). Its benchmark protocol
must identify the vocabulary and text embedding. The fixed-YTXT CUDA F32 raw-head gate passes
(score p99 `6.676e-5`, score max `2.956e-4`); see [the World report](../docs/world.md)
rather than inferring dataset accuracy from the qualitative grid.

### YOLOv8-World benchmark parity

![World Parity Bus & Zidane](../benchmarks/world_parity_bus_zidane.png)

### CLIP text+image encoder

**CLIP ViT-B/32** (clip-ViT-B-32.gguf) provides 512-d L2-normalised text and image embeddings. The text encoder
accepts up to 77 BPE-tokenised class/phrase inputs; the image encoder processes 224×224 RGB images through a
12-layer ViT. The model is used internally by YOLO-World for on-the-fly text encoding, or standalone for semantic
similarity tasks.

Integration parity:
- Text encoding cosine similarity against PyTorch: **1.0000000** ("a photo of a bus", ref in clip-ViT-B-32.ref.npz)
- Image encoding cosine similarity against PyTorch: **0.99994** (bus.jpg, same bilinear preprocessing as C++)

Use with the similarity subcommand:

```bash
yolo-cli similarity --model models/gguf/clip-ViT-B-32.gguf --source image.jpg --text "a bus on the street"
```

![CLIP Validation](../benchmarks/clip_validation.png)

### YOLOv8-World runtime notes

- YOLO-World requires the world_nc (number of open-vocabulary classes) to be set at session creation. CLIP encoding
  happens before graph build, not during inference, so changing the class list requires session recreation.
- The --text-embed path bypasses the CLIP encoder entirely and feeds precomputed [nc, 512] F32 embeddings directly
  into the ContrastiveHead. This is useful for server-side pre-encoding of a fixed class vocabulary.

### YOLO26 detector family

- **YOLO26n** is the lowest-latency end-to-end model and the strongest throughput choice in the measured matrix.
- **YOLO26s** is the compact accuracy/latency step above n.
- **YOLO26m** is the general balanced deployment model.
- **YOLO26l** uses a larger capacity budget for accuracy-oriented inference.
- **YOLO26x** is the largest YOLO26 integration target and the narrowest measured CUDA speedup case.

The exported end-to-end head is part of the model contract. Do not substitute YOLOv8 head decoding or assume that raw
tensor layouts are interchangeable across the two families.

### Segmentation families

**YOLOv8-seg and YOLO26-seg (n/s/m/l/x)** run through the same `detect` CLI command: the graph appends the mask
coefficient channels to the detection head and keeps the 32-prototype map as a second output. Postprocessing composes
per-detection instance masks on device after one proto readback; `--out` blends them onto the source image. Segmentation
parity against PyTorch was verified on the documented image (sub-pixel boxes, mask area within 2% per mask).

### YOLO26 absolute depth family

**YOLO26{n,s,m,l,x}-depth** predict a dense metric-depth map rather than COCO detections. They have a 768 default
input and a task-specific restoration step that removes letterbox padding and resizes values to source resolution. Use
the `depth` CLI command; the `detect` command intentionally rejects these checkpoints. YOLO26n-depth is the
lowest-latency depth model; the larger scales trade latency for denser, higher-quality depth maps.

### YOLO26 pose family

**YOLO26{n,s,m,l,x}-pose** predict person boxes plus 17 COCO keypoints using the RLE-based Pose26 head. Keypoint
decoding follows Pose26.kpts_decode (export path): `(raw + grid) * stride` with a sigmoid visibility. Use the `pose`
CLI command. Output text lists each box with its first five keypoints; `--out` renders the COCO-17 skeleton.

### YOLO26 OBB family

**YOLO26{n,s,m,l,x}-obb** predict oriented boxes in the DOTA-15 class set. The OBB26 head emits the angle in raw
radians (no sigmoid), decoded with dist2rbox into (cx, cy, w, h, angle). Use the `obb` CLI command; `--out` renders
rotated rectangles.

### YOLO26 semantic segmentation family

**YOLO26{n,s,m,l,x}-sem** predict a dense per-pixel class map over the 19 Cityscapes classes at one-eighth
resolution, argmax-reduced on device. Use the `semantic` CLI command; `--out` blends the class map over the source
image.

### YOLO26 classification family

**YOLO26{n,s,m,l,x}-cls** predict ImageNet-1000 class probabilities from a 224 input. The released checkpoints bake
their own transforms (Resize with antialias, center crop, plain /255 normalization), which the CLI reproduces exactly;
no letterbox or ImageNet mean/std is applied. Use the `classify` CLI command.

## Formats

| Format | Precision                                                           | Intended use                                 |
| ------ | ------------------------------------------------------------------- | -------------------------------------------- |
| F32    | Float32 weights and activations                                     | Reference parity and debugging               |
| F16    | Float16 matrix/conv weights; backend-dependent activations          | Default GPU deployment format                |
| Q8_0   | Block-quantized eligible conv weights; other tensors remain F16/F32 | Smaller files; validate accuracy per dataset |

F16 and Q8_0 are not expected to be bit-identical to PyTorch F32. Integration parity means equivalent task output
within declared tolerances, not identical intermediate activations. Dataset accuracy must be validated on the target
dataset before production deployment.

## Runtime support

| Backend | F32 | F16 | Q8_0 | Status                                                               |
| ------- | --- | --- | ---- | -------------------------------------------------------------------- |
| CPU     | yes | yes | yes  | Closed-set 405-key matrix measured; World is vocabulary-dependent |
| CUDA    | yes | yes | yes  | Closed-set matrix measured; fixed-YTXT World CUDA F32 raw parity passes |
| Vulkan  | yes | yes | yes  | Closed-set matrix measured; World requires its own declared-vocabulary report |

The [benchmark and parity report](../benchmarks/README.md) covers the 45 closed-set checkpoints
(10 detect, 10 segment, 5 depth, 5 pose, 5 obb, 5 semantic, 5 classify) x F32/F16/Q8_0 x CPU/CUDA/Vulkan. World
uses a separate vocabulary-bound protocol. The checkpoint-baked preprocessing is reproduced per
task (letterbox for box tasks and depth, antialiased resize + center crop for classify) so the ggml numbers are
directly comparable to the PyTorch CUDA reference in the same report. Every task model ships in all three
precisions; validate dataset accuracy on the target dataset before production deployment.

## Performance snapshot (checked-in charts)

Measured on the documented machine (RTX 3060 + CPU, bus.jpg; see `benchmarks/README.md` for the exact protocol):

| Chart | What it shows |
| ----- | ------------- |
| [speed_by_model.png](../benchmarks/speed_by_model.png) | Detect-family end-to-end latency: YOLOv8 + YOLO26, all scales, PyTorch vs ggml CUDA/Vulkan/CPU |
| [latency_by_backend.png](../benchmarks/latency_by_backend.png) | F16 detection latency grouped by backend |
| [latency_matrix.png](../benchmarks/latency_matrix.png) | Full heat matrix: all 45 checkpoints x 3 backends x 3 precisions |
| [task_latency.png](../benchmarks/task_latency.png) | Seven closed-set task families (detect/segment/depth/pose/obb/semantic/classify) vs PyTorch |
| [depth_latency.png](../benchmarks/depth_latency.png) | YOLO26 depth family by backend and precision (768 input) |
| [seg_latency.png](../benchmarks/seg_latency.png) | Segmentation families by scale vs PyTorch |
| [speedup_table.md](../benchmarks/speedup_table.md) | Per-model speedup over the PyTorch CUDA reference |
| [world_parity_bus_zidane.png](../benchmarks/world_parity_bus_zidane.png) | YOLOv8s-World open-vocabulary detection, PyTorch vs ggml CPU |
| [clip_validation.png](../benchmarks/clip_validation.png) | CLIP ViT-B/32 C++ vs PyTorch embedding cosine + per-dim overlay |
| [clip_architecture.png](../benchmarks/clip_architecture.png) | CLIP text/image encoder architecture in pure GGML ops |
| [model_family_overview.png](../benchmarks/model_family_overview.png) | Visual inference results across all model families |

World timing is not included in this table until all rows name the exact vocabulary and text-embedding source. Its
CPU attention path remains slower than GPU paths; treat existing legacy numbers as diagnostic only.

## Rebuild

From `cpp_ggml/`:

```bash
# Missing supported release checkpoints are downloaded into models/pytorch/.
bash scripts/convert_all.sh

# Or convert one model. Both commands resolve the canonical directory.
python3 scripts/convert_yolo_to_gguf.py --model yolo26n --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26n-depth --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26s-pose --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26l-obb --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26m-sem --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26n-cls --dtype f16
```

The depth source checkpoint is not a runtime model. Generate all three C++ artifacts explicitly when only
`models/pytorch/yolo26n-depth.pt` exists:

```bash
for dtype in f32 f16 q8_0; do
    python3 scripts/convert_yolo_to_gguf.py --model yolo26n-depth --dtype "$dtype"
done

for model in models/gguf/yolo26n-depth-{f32,f16,q8_0}.gguf; do
    build-cpu/bin/yolo-cli info --model "$model"
done
```

The converter writes task metadata, the depth calibration pair, and the depth-specific operation tail into each GGUF.
The loader rejects a file whose declared task and final operation disagree, and the `depth` command restores the dense
metric map to source resolution before writing `YDEP0001` float metres or the display-only PNG.

## Absolute-depth notes

YOLO26n-depth is monocular absolute-depth estimation. It is useful for approximate scene layout, obstacle-distance
priors, measurement assistance, and robotics perception when a calibrated depth sensor is unavailable. The raw
`YDEP0001` output is the machine-readable result in meters; the PNG is only a colorized visualization.

Depth from one RGB image remains sensitive to camera intrinsics, domain shift, reflective or transparent surfaces,
low texture, occlusion boundaries, and objects outside the training distribution. Do not use it as the sole input for
safety-critical distance decisions. Preserve aspect ratio and use the default 768 input for comparisons with the
released checkpoint.
