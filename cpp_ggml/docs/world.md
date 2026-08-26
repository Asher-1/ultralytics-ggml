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
> text-conditioned GGUF graph and accepts a **post-reprta** MobileCLIP
> `YTXT0001` embedding blob. Upstream publishes World checkpoints only as
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

| Module | Role | GGML ops used |
| ------ | ---- | ------------- |
| `C2fAttn` (MaxSigmoidAttnBlock) | per-head gate `proj_conv(x) * sigmoid(max_n(embed·guide_n)/√hc + bias)` | `mul_mat`, `scale_bias`, `sigmoid`, `mul`, static max-tree (`relu`/`sub`/`add`) |
| `ImagePoolingAttn` | image tokens (adaptive max pool → k×k grid) attend into text embeddings, residual | `pool_2d` (max), `norm`, `mul_mat`, `soft_max`, `concat` |
| `WorldDetect` (ContrastiveHead) | L2-normalized cosine score per anchor × `logit_scale.exp()` + `bias`, DFL box decode | `sum_rows`, `sqrt`, `div`, `mul_mat`, `scale_bias` |

Text is a **graph input**, not baked weights: the GGUF carries `yolo.world` (nc)
metadata and the session exposes a `text_input` tensor
([512, nc] F32). The C++ graph builder lowers the three modules in
`src/yolo_graph.cpp` (`max_sigmoid_attn`, `image_pooling_attn`, `world_detect`).

## 2. Models and formats

| Item | Location |
| ---- | -------- |
| PyTorch checkpoints | `models/pytorch/yolov8{s,m,l,x}-world.pt` |
| GGUF runtime models | `models/gguf/yolov8{s,m,l,x}-world-{f32,f16,q8_0}.gguf` (12 files) |
| CLIP text encoder | `models/gguf/clip-ViT-B-32-{f32,f16,q8_0}.gguf` (+ `clip-ViT-B-32.ref.npz` verification reference) |

YOLOE v8, 11, and 26 use the same `[nc, 512]` text-input shape, while
YOLOE-26 adds an end-to-end DFL head. All six repository YAMLs (detect and
segment for each generation) convert to a text-conditioned GGUF; segment models
also carry mask coefficients and a prototype output through `world_segment`.
YOLO-World YTXT blobs are direct CLIP ViT-B/32 outputs. A YOLOE YTXT instead
must be generated from its exact detector checkpoint: MobileCLIP output passes
through that checkpoint's `reprta` block and L2 normalization before the GGML
graph consumes it. The current runtime intentionally rejects `--classes` alone
for a YOLOE GGUF, so a ViT-B/32 vector cannot silently corrupt scores.

World execution status on the measured `yolov8s-world` asset is:

| Backend | Status | Measured YOLOv8s-World F32 graph (480x640, fixed YTXT) |
| ------- | ------ | -------------------------------------------------------- |
| CPU     | Near-parity raw output recorded; dataset accuracy not established | 261.662 ms graph / 262.443 ms e2e, 10 iterations, fixed 4-class YTXT |
| CUDA    | Strict raw-head tolerance gate passed; F16 deployment graph also runs without a mixed-dtype matmul failure | 15.797 ms graph / 16.894 ms e2e, 30 iterations, fixed 4-class YTXT |
| Vulkan  | Near-parity raw output recorded; dataset accuracy not established | 29.639 ms graph / 30.499 ms e2e, 30 iterations, fixed 4-class YTXT |

The separately measured `yoloe-26n-seg` F32 run uses the same four display
classes but a detector-specific post-reprta YTXT: CUDA 12.922 ms graph / 14.882
ms end-to-end, Vulkan 5.041 / 7.195 ms, and CPU 110.643 / 112.396 ms. These
rows are recorded in `benchmarks/yoloe.jsonl`; they exclude offline MobileCLIP
preparation and do not assert dataset-level accuracy.

## 3. CLI usage

```bash
# 1. On-the-fly encoding: CLIP text encoder turns class names into embeddings
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car" --conf 0.25 \
    --clip-model models/gguf/clip-ViT-B-32.gguf

# 2. Precomputed embeddings: skip CLIP load, reuse a fixed vocabulary
#    (YTXT0001 blob: magic(8) + dims(2×i32: nc, 512) + nc×512×f32 row-major)
yolo-cli detect --model models/gguf/yolov8s-world-f16.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --text-embed path/to/embeddings.ytxt \
    --classes "person,bus,car" --conf 0.25

# 3. Per-frame latency benchmark with a fixed vocabulary. Text setup occurs
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
  `--classes` alone takes the CLIP-encoding path.
- `bench` writes a `world` JSON object containing the supplied classes, class
  count and `clip` or `ytxt` text source. CLIP encoding and session creation
  are one-time setup, not per-frame latency.
- Class display names come from `--classes`; without it the GGUF's 80-class
  table is used for labels only (scores are still computed for your nc rows).

### YOLOE vocabulary generation

```bash
python scripts/encode_mobileclip_text.py \
    --detector models/pytorch/yoloe-26n-seg.pt \
    --classes "person,bus,car,truck" --output yoloe-26n-vocabulary.ytxt
yolo-cli detect --model models/gguf/yoloe-26n-seg-f32.gguf \
    --source ../ultralytics/assets/bus.jpg \
    --classes "person,bus,car,truck" --text-embed yoloe-26n-vocabulary.ytxt
```

The YTXT is detector-specific: do not reuse it for another YOLOE checkpoint or
for a YOLO-World model. Native ggml MobileCLIP encoding is not yet shipped, so
YTXT creation is an offline PyTorch preparation step and is excluded from
per-frame latency measurements.

## 4. Application guide

### 4.1 Server-side fixed vocabulary (recommended)

Encode the class list **once** on a server (Python or the C++ `yolo-similarity`
tool), write a `YTXT0001` blob, and ship it with the model. Edge devices then
never load CLIP:

```python
# scripts/gen_clip_ref.py shows the exact PyTorch encoding; for the CLI path:
yolo-cli similarity --model models/gguf/clip-ViT-B-32.gguf \
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

The raw-head gate uses an identical `YINP0001` image tensor and an identical
PyTorch-produced `YTXT0001` embedding. Its CUDA F32 acceptance thresholds are
absolute head maximum `<= 1e-3` and score-channel p99 `<= 1e-4`. On the
documented YOLOv8s-World F32 input, CUDA records all-head
`mean=1.675e-5, p99=1.221e-4, max=6.456e-4`, score-channel
`mean=1.284e-5, p99=6.676e-5, max=2.956e-4`, and correlation `1.0` at the
reported precision, so the CUDA F32 gate passes. CPU remains near-parity
(`score mean=0.006502, p99=0.031689`) rather than a strict numerical claim.
The fix keeps F32 `im2col`, F32 depthwise weights, and pedantic cuBLAS F32
math; F16/Q8 tensor-core paths remain unchanged. Do not infer dataset mAP from
the visual parity image or from a matching detection count.

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
  YOLO-World. Its post-reprta YTXT must come from the matching YOLOE checkpoint;
  native ggml MobileCLIP encoding remains an explicit unimplemented boundary.
