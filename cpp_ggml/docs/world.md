# YOLO-World: open-vocabulary detection in ggml

YOLO-World is the open-vocabulary extension of the YOLOv8 detector family. Unlike
the closed-set YOLOv8/YOLO26 checkpoints (fixed 80 COCO classes), a World model
accepts **any class vocabulary at runtime**: the class list is encoded into CLIP
text embeddings, and a contrastive head scores every anchor against those
embeddings. This document covers the architecture, the C++/GGML integration,
the application workflow, and how to combine YOLO-World with camera calibration
(标定) pipelines.

> Why is there no `yolo26-world`? YOLO-World is defined only for the YOLOv8
> architecture (`ultralytics/cfg/models/v8/yolov8-world.yaml`). The YOLO26
> family's open-vocabulary counterpart is **YOLOE-26**
> (`ultralytics/cfg/models/26/yoloe-26.yaml`), which uses a different text
> encoder (MobileCLIP) and an end-to-end DFL head. It is exported as a
> text-conditioned GGUF graph whose text input is the **pre-reprta** MobileCLIP
> feature (the checkpoint's `reprta` block rides inside the graph); pass
> `--classes` for native plaintext encoding or a `YTXT0002` embedding blob.
> Upstream publishes World checkpoints only as
> `yolov8{s,m,l,x}-world.pt`.

## 1. Architecture

World detection = YOLOv8 detection + three text-conditioned modules:

```
backbone (Conv/C2f/SPPF) ──► P3/P4/P5 feature maps
                                 │
head:  C2fAttn (MaxSigmoidAttnBlock)     ← text embeddings [512, nc] gate image features
       ImagePoolingAttn                   ← adaptive-max-pooled image tokens attend into text
       WorldDetect (ContrastiveHead)      ← L2-normalized dot product × logit_scale + bias
```

| Module                          | Role                                                                                 | GGML ops used                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `C2fAttn` (MaxSigmoidAttnBlock) | per-head gate `proj_conv(x) * sigmoid(max_n(embed·guide_n)/√hc + bias)`              | `mul_mat`, `scale_bias`, `sigmoid`, `mul`, static max-tree (`relu`/`sub`/`add`) |
| `ImagePoolingAttn`              | image tokens (adaptive max pool → k×k grid) attend into text embeddings, residual    | `pool_2d` (max), `norm`, `mul_mat`, `soft_max`, `concat`                        |
| `WorldDetect` (ContrastiveHead) | L2-normalized cosine score per anchor × `logit_scale.exp()` + `bias`, DFL box decode | `sum_rows`, `sqrt`, `div`, `mul_mat`, `scale_bias`                              |

Text is a **graph input**, not baked weights: the GGUF carries `yolo.world` (nc)
metadata and the session exposes a `text_input` tensor
([512, nc] F32). The C++ graph builder lowers the three modules in
`src/yolo_graph.cpp` (`max_sigmoid_attn`, `image_pooling_attn`, `world_detect`).

A World GGUF also carries the vocabulary its checkpoint shipped with
(`yolo.vocab_txt`, the flattened [nc, 512] f32 `txt_feats`, 160 KB for the 80
COCO classes). That is what the head sees when the caller supplies no class
list, mirroring `YOLO("yolov8s-world.pt")(...)` in Python.

## 2. Models and formats

| Item                | Location                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| PyTorch checkpoints | `models/pytorch/yolov8{s,m,l,x}-world.pt`                                                              |
| GGUF runtime models | `models/gguf/yolov8{s,m,l,x}-world-{f32,f16,q8_0}.gguf` (12 files)                                     |
| CLIP text encoder   | `models/gguf/clip-ViT-B-32-{f32,f16,q8_0}.gguf` (+ `clip-ViT-B-32-f16.ref.npz` verification reference) |

YOLOE v8, 11, and 26 use the same `[nc, 512]` text-input shape, while
YOLOE-26 adds an end-to-end DFL head. All six repository YAMLs (detect and
segment for each generation) convert to a text-conditioned GGUF; segment models
also carry mask coefficients and a prototype output through `world_segment`.
YOLO-World YTXT blobs are direct CLIP ViT-B/32 outputs. A YOLOE YTXT0002 blob
is the raw L2-normalised MobileCLIP output: the checkpoint's `reprta` block is
embedded in the GGUF graph (v4), so one blob serves every YOLOE checkpoint that
shares the text tower. Both families also accept `--classes` alone: the runtime
encodes plaintext through the matching native GGUF text tower (CLIP for World,
MobileCLIP for YOLOE), so no offline Python preparation is required.

World execution status on the measured `yolov8s-world` asset is:

| Backend | Status                                                                                                     | Measured YOLOv8s-World F32 graph (480x640, fixed YTXT)               |
| ------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| CPU     | Near-parity raw output recorded; dataset accuracy not established                                          | 188.468 ms graph / 189.044 ms e2e, 30 iterations, fixed 4-class YTXT |
| CUDA    | Strict raw-head tolerance gate passed; F16 deployment graph also runs without a mixed-dtype matmul failure | 7.144 ms graph / 7.936 ms e2e, 50 iterations, fixed 4-class YTXT     |
| Vulkan  | Near-parity raw output recorded; dataset accuracy not established                                          | 21.932 ms graph / 22.759 ms e2e, 50 iterations, fixed 4-class YTXT   |

The separately measured `yoloe-26n-seg` F32 run uses the same four display
classes but a fixed 4-class MobileCLIP YTXT: CUDA 4.886 ms graph / 12.388
ms end-to-end, Vulkan 7.903 / 18.078 ms, and CPU 106.711 / 108.174 ms. These
rows are recorded in `benchmarks/yoloe.jsonl`; they exclude offline MobileCLIP
preparation and do not assert dataset-level accuracy.

Plaintext `--classes` adds only a one-time text-tower setup (tower GGUF load,
graph build, class-name encoding); per-frame latency is identical to the YTXT
path once the session exists. Measured wall-clock overhead on this dev machine
(`yoloe-26n-seg-q8_0`, 4 classes): ~10 s on both CPU and CUDA, versus ~0 ms
for a precomputed `YTXT0002` blob — encode once, serve every frame.

## 3. CLI usage

```bash
# 1. Default vocabulary: whatever the checkpoint shipped with (80 COCO
#    classes). No text encoder is loaded and no embedding is passed.
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg

# 2. On-the-fly encoding: CLIP text encoder turns class names into embeddings
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car" --conf 0.25 \
    --clip-model models/gguf/clip-ViT-B-32-f16.gguf

# 3. Precomputed embeddings: skip CLIP load, reuse a fixed vocabulary
#    (YTXT0002 blob: magic(8) + dims(2×i32: nc, 512) + nc×512×f32 row-major)
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --text-embed path/to/embeddings.ytxt \
    --classes "person,bus,car" --conf 0.25

# 4. Per-frame latency benchmark with a fixed vocabulary. Text setup occurs
#    once before the timed loop and is recorded as text_source="ytxt".
yolo-cli bench --model models/gguf/yolov8s-world-f32.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car,truck" --text-embed path/to/vocabulary.ytxt
```

Notes:

- `world_nc` (the class count) is fixed at **session creation**: the graph is
  built with `nc` max-trees, attention heads and score rows. Changing the
  vocabulary requires a new session.
- `--classes` and `--text-embed` counts must agree when both are given;
  `--classes` alone encodes through the model's text tower (CLIP for World,
  native MobileCLIP for YOLOE).
- `bench` writes a `world` JSON object containing the supplied classes, class
  count and `clip` or `ytxt` text source. CLIP encoding and session creation
  are one-time setup, not per-frame latency. With neither knob the object
  reports `text_source="builtin"` and `class_count:0` (the stored vocabulary).
- Class display names come from `--classes`. Without it the GGUF's 80-class
  table is used: with the built-in vocabulary those are the scored classes,
  with a `--text-embed` blob they are labels by row index only.
- A text-conditioned model needs its vocabulary from somewhere: the stored
  `yolo.vocab_txt`, or an explicit `--classes`/`--text-embed`. A YOLOE GGUF
  stores none (its checkpoints ship no `txt_feats`), so one of the two knobs
  is required; the checkpoint's `reprta` weights ride inside the graph. Prompt-free
  YOLOE is a
  different mechanism: its vocabulary is baked into the head weights, the GGUF
  sets no `yolo.world` flag, and it needs neither.

### YOLOE vocabulary generation

```bash
# Plaintext end to end: the runtime encodes --classes with the native
# MobileCLIP GGUF tower (mobileclip2_b-f16.gguf by default), reprta runs inside the graph.
yolo-cli detect --model models/gguf/yoloe-26n-seg-f32.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car,truck"

# Or precompute a YTXT0002 blob once (checkpoint-agnostic raw MobileCLIP output):
python scripts/encode_mobileclip_text.py \
    --classes "person,bus,car,truck" --output yoloe-vocabulary.ytxt
yolo-cli detect --model models/gguf/yoloe-26n-seg-f32.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car,truck" --text-embed yoloe-vocabulary.ytxt
```

The YTXT0002 blob holds pre-reprta features, so it is reusable across every
YOLOE checkpoint that shares the MobileCLIP text tower (but not for
YOLO-World, which needs CLIP features). Native ggml MobileCLIP encoding ships
in `src/mobileclip_graph.cpp`; offline YTXT preparation is optional and
excluded from per-frame latency measurements.

## 4. Application guide

### 4.1 Server-side fixed vocabulary (recommended)

Encode the class list **once** on a server (Python or the C++ `yolo-similarity`
tool), write a `YTXT0002` blob, and ship it with the model. Edge devices then
never load CLIP:

```python
# scripts/gen_clip_ref.py shows the exact PyTorch encoding; for the CLI path:
yolo-cli similarity --model models/gguf/clip-ViT-B-32-f16.gguf \
    --text "person,bus,car" --dump_embed
```

### 4.2 Dynamic vocabulary

For a changing scene vocabulary (e.g. "forklift" at dock A, "pallet" at dock
B), keep one World session per fixed class list, or re-create the session when
the list changes. Session creation is ~0.04 s of graph build plus CLIP text
encoding (~10 ms), so per-switch recreation is cheap on GPU. CLIP and YOLO
sessions use separate graph allocators even when they select the same physical
CUDA device; GGML backends can coexist, but a gallocr/scheduler is not a
process-wide workspace.

### 4.3 Combining with depth / metric tasks

World gives boxes + class scores; it does not give distance. For spatial
applications pair it with the YOLO26 depth family
(`yolo-cli depth --model models/gguf/yolo26n-depth-f16.gguf`) or with a
calibrated sensor (see §5).

## 5. Integration with camera calibration (标定)

Calibration provides the camera intrinsics/extrinsics that turn detection
outputs into world-space quantities. YOLO-World integrates with a calibration
pipeline in three ways:

### 5.1 Calibration-constrained class vocabulary

After extrinsic calibration you know which scene regions matter (e.g. the lane
corridor in front of the vehicle, or a fixed dock area). Use that to:

- **Restrict the vocabulary** to scene-relevant classes: fewer classes reduce
  the contrastive head width (`nc` rows) and the max-tree depth, cutting graph
  build time and post-processing work.
- **Apply ROI gating** on decoded boxes using the calibrated frustum: drop
  detections whose projected position falls outside the calibrated region
  before NMS, so stray open-vocabulary hits (e.g. a "person" far outside the
  ROI) never reach downstream logic.

### 5.2 Metric estimation via calibrated geometry

With intrinsics `K` and the ground plane (extrinsics) known:

```text
for each detection box (person / vehicle):
    foot-point pixel (cx, y_bottom) → ray in camera frame → intersect ground plane
    → 3D position and distance in meters
```

The depth-from-single-view error grows with distance; cross-check World boxes
against the YOLO26 depth map (or a LiDAR/TOF sensor fused during calibration)
for the final distance used by planning. The `YDEP0001` depth output is raw
meters at source resolution, ready to sample at the box foot-point.

### 5.3 Calibration workflow with a fixed pre-encoded vocabulary

A typical integration loop:

1. **Offline**: run intrinsic + extrinsic calibration; define the scene class
   list (e.g. `person,car,truck,bus`).
2. **Offline**: encode the class list once with CLIP → `vocab.bin`; convert
   `yolov8s-world.pt` → `yolov8s-world-f16.gguf`.
3. **On target**: load GGUF + `vocab.bin` (`--text-embed`), run detection at
   the calibrated input size, apply the calibrated ROI/frustum gate, and
   convert foot-points to world coordinates with the calibration extrinsics.
4. **Re-calibration**: only step 1 changes; the model and vocabulary blob stay
   the same unless the scene classes change.

## 6. Verification and accuracy

The parity reference is the PyTorch GPU `predict()` detection output, not a
bit-exact raw tensor: PyTorch GPU itself diverges from PyTorch CPU by up to
`0.197` on the yoloe-26n-seg raw head (cuDNN algorithm selection), so any
raw-tensor gate stricter than PyTorch's own cross-device variance would reject
PyTorch itself. The acceptance gate is therefore defined at detection level:
identical `YINP0001` preprocessing and `YTXT0002` embedding, then the same NMS
detection set (count, classes, anchor assignment) with confidence and box
coordinates within noise.

CUDA F32 conv runs the TF32 tensor-core kernel, mirroring PyTorch's Ampere
default (10-bit mantissa products, F32 accumulate); F16/Q8 paths are
unchanged. Measured on the documented bus input:

- `yolov8s-world` (640x480, world.ytxt): 5 detections match PyTorch GPU,
  confidence diff `<= 0.001`, box coordinates `<= 1 px` (letterbox scale
  0.5926); sigmoid score-channel diff `max=0.016, p99=4.2e-5`.
- `yoloe-26n-seg` (bus and zidane): identical detection sets, confidence diff
  `<= 0.002`, coordinates `<= 0.2 px`; sigmoid score diff `max=0.0074,
p99=9e-6`.

Raw-head logit maxima (`4.48` world, `8.94` yoloe) sit on deeply negative
anchors whose sigmoid is `0.0000` on both sides, invisible downstream of the
score threshold. CPU remains near-parity (`score mean=0.006502,
p99=0.031689` on world) rather than a strict numerical claim. Do not infer
dataset mAP from the visual parity image or from a matching detection count.

The stored default vocabulary is verified the same way, with the PyTorch
reference falling back to its own `txt_feats` (no YTXT argument) and the engine
receiving no `--classes`:

```bash
tmpdir=$(mktemp -d)
python3 scripts/parity_reference.py prep models/pytorch/yolov8s-world.pt \
    ../ultralytics/assets/bus.jpg "$tmpdir/input.bin"
python3 scripts/parity_reference.py raw models/pytorch/yolov8s-world.pt \
    "$tmpdir/input.bin" "$tmpdir/torch-raw.bin"
./build-cpu/bin/yolo-cli detect --model models/gguf/yolov8s-world-f32.gguf \
    --input-f32 "$tmpdir/input.bin" --dump-raw "$tmpdir/ggml-raw.bin"
python3 scripts/parity_reference.py diff "$tmpdir/torch-raw.bin" "$tmpdir/ggml-raw.bin"
```

On the 80-class default vocabulary the raw head (144x6300) records
`max=5.19e-4, mean=1.19e-5` for the F32 GGUF on CPU, the same tolerance class as
the 4-class fixed-YTXT gate above. Flows that carry the text embedding in F16
are looser in absolute terms: `max=0.376` for the F16 GGUF on CPU, `max=0.614`
and `max=1.364` for the F32 GGUF on the Vulkan and CUDA builds. The 76 extra
negative class channels contribute pre-sigmoid logits around -20, where an
F16-class relative error is a large absolute difference; after the sigmoid it is
invisible - yolov8s-world returns the same 7 detections on bus.jpg on CPU, CUDA
and Vulkan, and yolov8x-world the same 9 in F32 and F16, each within 0.01 of the
PyTorch score.

CLIP ViT-B/32 text and image validation is recorded in
`benchmarks/clip_validation.png`; it applies to YOLO-World only, not MobileCLIP
used by YOLOE.

- Dataset-level mAP must be validated per vocabulary on the target data —
  open-vocabulary recall depends on how well the class names match the
  training text distribution (use the full phrase, e.g. "a person walking"
  performs differently from "person").

## 7. Limitations

- CPU latency is dominated by the two attention modules (no SIMD path yet);
  use CUDA/Vulkan for real-time applications.
- `world_nc` is fixed per session; vocabularies are not hot-swappable inside
  one session.
- YOLO-World checkpoints are COCO-pretrained; extreme open-set categories may
  need fine-tuning (see `ultralytics/models/yolo/world/train_world.py`).
- YOLOE-26's text encoder is MobileCLIP, not the ViT-B/32 CLIP asset used by
  YOLO-World. Its `YTXT0002` blob is raw MobileCLIP output, valid for every
  YOLOE checkpoint; native ggml MobileCLIP encoding (`src/mobileclip_graph.cpp`)
  also makes `--classes` alone work end to end.
