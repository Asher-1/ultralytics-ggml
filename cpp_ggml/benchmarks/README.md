# ggml inference benchmark and parity report

This directory is the reproducible evidence store for the C++ integration. It compares the same source image and model
input geometry across PyTorch CUDA, ggml CUDA, ggml Vulkan, and ggml CPU for the seven closed-set task families
(detect, segment, depth, pose, obb, semantic, classify). Open-vocabulary World is separate because every timing is
tied to a class vocabulary and text-embedding path.

## Artifacts

| File                  | Content                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| [bench.jsonl](bench.jsonl)      | Append-only closed-set GGML rows resolving to 405 keys (45 checkpoints x 3 precisions x 3 backends) |
| [pytorch.jsonl](pytorch.jsonl)  | 45 PyTorch CUDA reference rows, one per checkpoint (full dump in [Raw measurements](#raw-measurements)) |
| [world.jsonl](world.jsonl)      | 12 fixed-YTXT F32 World rows: s/m/l/x x CUDA/Vulkan/CPU, with vocabulary metadata |
| [yoloe.jsonl](yoloe.jsonl)      | 3 fixed post-reprta-YTXT F32 YOLOE-26n-seg rows: CUDA/Vulkan/CPU |
| [speedup_table.md](speedup_table.md) | Complete model/precision/backend latency and speedup matrix for all seven task families |
| PNG charts             | Embedded in [Visualized results](#visualized-results) below                                         |
| [old_snapshot_20260825/](old_snapshot_20260825) | Prior snapshot (11-model detection/depth + 10-model segmentation evidence) kept for before/after comparison |

Plot generation validates the full 405-key matrix before writing anything, keeps the latest entry for each
model/backend/precision key, and ignores legacy rows without a structured `e2e_ms` field. All charts, tables, and
figures in this report are rendered from the checked-in measurements in the sections below.

## Visualized results

### End-to-end latency by model family (detect)

![End-to-end latency by model family](speed_by_model.png)

Grouped end-to-end latency on `bus.jpg` for YOLOv8 and YOLO26 detectors across PyTorch CUDA, ggml CUDA, ggml Vulkan, and
ggml CPU (8 threads), plotted on a log scale with the mean value on each bar.

### Detection backend comparison (GGML F16)

![Detection backend latency comparison](latency_by_backend.png)

Full detection-model backend latency comparison in GGML F16.

### Precision comparison

![F32/F16/Q8_0 precision comparison](speed_by_dtype.png)

F32, F16, and Q8_0 end-to-end latency for the n/s deployment models.

### Depth family latency

![YOLO26 depth family latency](depth_latency.png)

All five YOLO26 depth scales by backend and precision at the checkpoint default 768 input size.

### Complete latency matrix

![Complete GGML latency matrix](latency_matrix.png)

All 45 checkpoints across all three GGML backends and all three precisions on a log-scale heatmap. Darker cells are
faster: the CUDA and Vulkan columns track each other closely across the board, with the best precision per model
frequently beating PyTorch; the CPU rows form the bright tail.

### Segmentation latency

![Segmentation end-to-end latency by model family](seg_latency.png)

End-to-end segmentation latency for YOLOv8-seg and YOLO26-seg (n/s/m/l/x) on `bus.jpg` across PyTorch CUDA, ggml CUDA,
and ggml Vulkan (best precision per model, F16 shown).

### Closed-set task families (detect / segment / depth / pose / obb / semantic / classify)

![All task family latency](task_latency.png)

End-to-end latency for every task family (n/s/m/l/x where applicable), comparing PyTorch CUDA, ggml CUDA, and ggml
Vulkan. detect/segment cover both YOLOv8 and YOLO26; depth/pose/obb/semantic/classify cover YOLO26.
Pose runs on `bus.jpg` at 640, OBB and semantic at 640, and classify at the checkpoint 224 input.

### YOLO-World open-vocabulary parity

![YOLO-World parity on bus/zidane](world_parity_bus_zidane.png)

YOLOv8s-World detection with a `person,bus,car` vocabulary at conf 0.25. This
is a qualitative rendering, not a dataset-accuracy claim. The separate fixed
YTXT raw-head protocol records CUDA F32 strict parity: score p99 `6.676e-5`,
score max `2.956e-4`, and all-head max `6.456e-4`. Reproduce a World timing
only with its class vocabulary and either a declared CLIP model or a declared
YTXT embedding input.

### YOLO-World fixed-vocabulary latency

![YOLO-World F32 latency](world_latency.png)

This separate chart covers yolov8s/m/l/x-World on CUDA, Vulkan and CPU using
the same F32 `person,bus,car,truck` YTXT at 480x640. The 30-iteration GPU and
shorter CPU samples report repeated-frame cost after one-time text setup; it
is not a PyTorch speedup chart because no matching PyTorch World protocol is
checked in.

### YOLOE fixed-vocabulary latency

![YOLOE-26n-seg F32 latency](yoloe_latency.png)

YOLOE-26n-seg uses a MobileCLIP embedding transformed by the exact checkpoint's
`reprta` block before it enters GGML. This chart uses the fixed
`person,bus,car,truck` post-reprta YTXT at 480x640 and is not comparable to
the ViT-B/32-encoded YOLO-World chart. MobileCLIP preparation is offline
PyTorch work and excluded from repeated-frame timing.

### CLIP ViT-B/32 embedding validation

![CLIP ViT-B/32 validation](clip_validation.png)

Full C++ CLIP (BPE tokenizer + text encoder + image encoder) versus PyTorch: text embedding cosine = **1.0000000**
("a photo of a bus"), image embedding cosine = **0.99994** (bus.jpg, identical bilinear preprocessing). The right
panel overlays the first 200 text-embedding dimensions; the two implementations are visually indistinguishable.
The reference data (`clip-ViT-B-32.ref.npz`) is regenerated by `scripts/gen_clip_ref.py`.

### CLIP ViT-B/32 architecture

![CLIP ViT-B/32 architecture](clip_architecture.png)

Text and image encoder architecture lowered to pure GGML graph ops (`get_rows`, `conv_2d`, `mul_mat`, `soft_max`,
`gelu_quick`), with the exact tensor shapes used by the C++ graph builder.

### Model family inference overview

![Model family overview](model_family_overview.png)

Visual inference results for every model family on `bus.jpg`/`zidane.jpg`, with `boats.jpg` used for OBB because the
COCO-trained oriented-box checkpoint has no meaningful rotated targets in the people-and-bus scene: detect, world
(open-vocabulary), depth, pose, obb, semantic segmentation, instance segmentation and classify.

### Detection parity

![Detection parity on bus.jpg](parity_grid_bus.png)

![Detection parity on zidane.jpg](parity_grid_zidane.png)

Rendered detection output at conf 0.25 and imgsz 640 for PyTorch CUDA F32, ggml CUDA F16, ggml Vulkan F16, and ggml CPU
8T F16. These grids come from the parity verification pass; the four engines produce visually identical boxes and
classes on both images.

### Depth parity

![Absolute-depth parity on bus.jpg](depth_parity_bus.png)

Restored per-pixel absolute depth in meters for PyTorch CUDA F32 versus ggml CUDA, ggml Vulkan, and ggml CPU F16. This
grid comes from the parity verification pass; the four panels are visually indistinguishable.

## Current status

| Integration goal                               | Evidence-based status                                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| YOLOv8 n/s/m/l/x detection                     | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26 n/s/m/l/x detection                     | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLOv8/YOLO26 n/s/m/l/x segmentation          | 10 models converted (F32/F16/Q8_0) and exercised end to end on all backends with on-device masks        |
| YOLO26 n/s/m/l/x absolute depth                | All 5 scales converted (F32/F16/Q8_0), CPU/CUDA/Vulkan execution, source-size restoration verified       |
| YOLO26 n/s/m/l/x pose (COCO-17)                | All 5 scales converted (F32/F16/Q8_0), CPU/CUDA/Vulkan execution, RLE-head decode verified               |
| YOLO26 n/s/m/l/x obb (DOTA-15)                 | All 5 scales converted (F32/F16/Q8_0), CPU/CUDA/Vulkan execution, dist2rbox decode verified              |
| YOLO26 n/s/m/l/x semantic (Cityscapes-19)      | All 5 scales converted (F32/F16/Q8_0), CPU/CUDA/Vulkan execution, argmax map verified                    |
| YOLO26 n/s/m/l/x classify (ImageNet-1000)      | All 5 scales converted (F32/F16/Q8_0), CPU/CUDA/Vulkan execution, checkpoint-baked preprocessing reproduced |
| Full q8_0/f16/f32 x cuda/cpu/vulkan matrix     | All 405 **closed-set** keys measured end to end; World is not part of this matrix |
| YOLOv8 s/m/l/x World (open-vocabulary)          | 12 fixed-YTXT F32 backend/model latency rows and chart; CUDA raw-head gate passes (`score p99=6.676e-5`, `max=2.956e-4`) |
| CLIP ViT-B/32 text+image encoder                | GGUF export script (`convert_clip_to_gguf.py`) and C++ inference engine (`clip_graph.hpp/cpp`) complete; text cos=1.0000000, image cos=0.99994 vs PyTorch |
| CPU Q8_0→F16 host dequant                       | Implemented in `yolo_graph.cpp` matching the existing Vulkan path; eliminates dynamic quantisation from Q8_0 im2col pipeline on CPU |
| CUDA close to or faster than PyTorch CUDA      | Met per family at best precision; see [Measured result](#measured-result-on-this-machine)               |
| Vulkan close to PyTorch CUDA                   | Met per family at best precision; see [Measured result](#measured-result-on-this-machine)               |
| Focused numerical parity with official PyTorch | Closed-set family checks are recorded below; World CUDA F32 passes its fixed-YTXT raw-head gate, while CPU/Vulkan remain near-parity only |
| Dataset-level accuracy for every GGML format   | Not established; focused output parity is not a substitute for COCO, NYU Depth V2, Cityscapes, DOTA, ImageNet validation |
| Runtime stability                              | 500-iteration, restart, alternate-resolution, and malformed-model focused checks pass (prior snapshot); GPU sweeps re-run 50 iterations per key |
| Stable production embedding API                | Session creation takes an explicit `SessionOptions` struct; a versioned installed public C API is still outstanding |

Performance and numerical equivalence are measurements, not properties that can be guaranteed by documentation. F32
is the parity reference. F16 and Q8_0 deliberately change weight representation and must use declared task-level
tolerances. A production accuracy claim requires dataset-level validation on every format/backend combination.

## Measured result on this machine

The evidence store contains 405 unique backend/model/precision keys: all 45 checkpoints (10 detect, 10 segment,
5 depth, 5 pose, 5 obb, 5 semantic, 5 classify) on CPU, CUDA, and Vulkan in F32, F16, and Q8_0, plus one PyTorch CUDA
F32 reference per checkpoint. The release performance target is F16, the default GPU deployment format. Values above
`x1.00` are faster than PyTorch.

This snapshot re-ran the full matrix — CUDA, Vulkan, and CPU (20 warmups, 50 timed iterations, 3 s cooldown per entry;
CPU m/l/x trimmed to 3 warmups / 10 iterations) — against the current ggml integration patch, including the
Q8_0 pose-head epilogue alignment fix (K not a multiple of 8 now stores scalar rows; the previous 16B vector store
crashed `yolo26n-pose-q8_0` on CUDA with `misaligned address`). All three backends were re-measured this round;
nothing is carried over. The PyTorch references were re-measured on an idle machine so CPU-bound pre/postprocessing
is not distorted by concurrent sweeps.

Best per-family speedups (e2e mean, bus.jpg; classify at 224, depth at 768):

| Family   | Model (best CUDA) | PyTorch CUDA | Best GGML CUDA | Speedup | Best GGML Vulkan | Speedup |
| -------- | ----------------- | -----------: | -------------: | ------: | ---------------: | ------: |
| detect   | yolo26n           |      10.09 ms |   4.18 ms (Q8_0) | x2.42 |   4.54 ms (Q8_0) | x2.22 |
| segment  | yolo26n-seg       |       9.84 ms |   6.21 ms (Q8_0) | x1.58 |   6.23 ms (Q8_0) | x1.58 |
| depth    | yolo26x-depth     |      41.18 ms |  27.07 ms (Q8_0) | x1.52 |  27.05 ms (Q8_0) | x1.52 |
| pose     | yolo26n-pose      |       9.75 ms |   7.81 ms (Q8_0) | x1.25 |   4.72 ms (Q8_0) | x2.07 |
| obb      | yolo26n-obb       |       9.27 ms |   3.86 ms (F16) | x2.40 |   4.45 ms (F16) | x2.08 |
| semantic | yolo26n-sem       |       6.45 ms |   2.95 ms (Q8_0) | x2.19 |   3.27 ms (Q8_0) | x1.97 |
| classify | yolo26x-cls       |      13.45 ms |  12.57 ms (Q8_0) | x1.07 |  13.23 ms (Q8_0) | x1.02 |

YOLO-World has no PyTorch CUDA row in `pytorch.jsonl`: a valid reference must declare the prompt template, class
vocabulary, text encoder or YTXT blob, source image, and input geometry. The dedicated `world.jsonl` contains the
fixed-YTXT F32 GPU/CPU comparison and is excluded from closed-set speedup claims.

GPU graph-time stability: 60/60 detect keys, 58/60 segment keys, and 30/30 keys for each of depth, pose, obb,
semantic, and classify stay under 30 ms of graph time (the two segment exceptions are the yolov8x-seg F32/F16 keys).

The plot generator validates this matrix before writing any artifact. Missing backend/model/precision keys or missing
preprocess, graph, postprocess, or end-to-end latency statistics stop generation with a concrete error rather than
silently rendering an incomplete chart.

### Full matrix

The complete 405-key matrix with CPU 8T and every precision is reproduced in [speedup_table.md](speedup_table.md)
after validating completeness. All values are end-to-end mean latencies in milliseconds; `x1.00` means equal to
PyTorch CUDA.

### Parity notes (focused, single-image)

- **detect / segment**: PyTorch-CUDA F32 vs ggml CUDA/Vulkan F16 agree on boxes, classes, and composed masks on
  `bus.jpg` and `zidane.jpg` (freshly rendered grids above).
- **depth**: restored meters agree within F16 rounding on `bus.jpg` (min 1.734 m both engines; mean within 0.008 m).
- **pose**: decoded keypoints match PyTorch exactly at the reported resolution (e.g. `k0.x = 142.3` on the reference
  image).
- **obb**: rotated ship detections on `boats.jpg` are rendered in the family overview; `bus.jpg` remains a valid
  zero-detection parity case because the COCO-trained OBB head finds no rotated target there. Decode math is verified
  against `dist2rbox`.
- **semantic**: Cityscapes-19 class map on `bus.jpg` matches PyTorch argmax; `person` labels are semantic pixels in
  the scene rather than an object-detector false positive, so a bus image can correctly contain both bus and person.
- **classify**: top-1 softmax on `bus.jpg` reproduces PyTorch within 0.004 (minibus 0.5128 vs 0.5159), using the
  checkpoint-baked transforms (antialiased resize + center crop + /255 only).

### Raw measurements

`bench.jsonl` is append-only and resolves to 405 closed-set backend/model/precision keys with warmup/iteration counts
and preprocess/graph/postprocess/e2e latency statistics. `pytorch.jsonl` holds the 45 reference rows. `world.jsonl`
contains exactly 12 fixed-YTXT F32 backend/model keys; its `world` object records the vocabulary and text source. The
plot script de-duplicates closed-set rows by (backend, model, dtype) and rejects incomplete World protocol data. The
prior 11-model + 10-model evidence is preserved under [old_snapshot_20260825/](old_snapshot_20260825/).

## Reproduce

```bash
# GGML sweeps (one build dir per backend; see the top-level README for configure flags)
bash scripts/bench_all.sh build-cuda cuda
bash scripts/bench_all.sh build-vulkan vulkan
bash scripts/bench_all.sh build-cpu cpu

# World F32, fixed vocabulary and text embeddings; writes world.jsonl.
YOLO_BENCH_DTYPES=f32 \
YOLO_BENCH_WORLD_CLASSES='person,bus,car,truck' \
YOLO_BENCH_WORLD_TEXT_EMBED=path/to/vocabulary.ytxt \
    bash scripts/bench_all.sh build-cuda cuda yolov8s-world yolov8m-world yolov8l-world yolov8x-world

# PyTorch references (run with the machine otherwise idle)
python3 scripts/bench_pytorch.py > benchmarks/pytorch.jsonl

# Validate and render
python3 scripts/plot_benchmarks.py
```

## CPU Q8_0→F16 host dequant optimization

Starting from the checked-in evidence, a CPU Q8_0→F16 host dequantisation pass was added to `yolo_graph.cpp`. When
running on the CPU-only backend (no CUDA or Vulkan), Q8_0 weights are expanded to F16 once at session-creation time,
so the entire conv path uses `im2col(F16) + mul_mat(F16, F16)` instead of `im2col(F32) + mul_mat(Q8_0, F32)dynamic-quant`.

| Effect | Before | After (estimated) | Mechanism |
|--------|--------|-------------------|-----------|
| Q8_0 weight load | 1.06 B/element (Q8_0) | 2 B/element (F16) | Dequant on host, one-time cost |
| im2col output | 4 B/element (F32) | 2 B/element (F16) | Bandwidth halved |
| mul_mat kwant | F32 dynamic → Q8_0 | none (F16×F16) | Extra quantization pass removed |
| EPFLOPs | Q8_0×F32 mixed | F16×F16 native | On ARM/AVX-512-FP16 CPUs compute-bound win |

This optimization matches the existing Vulkan Q8_0→F16 path (see `yolo_graph.cpp` line 323-341) and is gated by the
same `gb.q8_direct` tensor compliance check. For x86 CPUs without native F16 arithmetic the memory-bandwidth saving
from the halved im2col is the primary benefit.

![CPU Q8_0 latency before optimization](cpu_q8_optimization.png)
**Figure**: CPU (8T) detection latency by precision (before optimization). After the host dequant patch, Q8_0 latencies
are expected to converge toward the F16 row for each model since both paths share the same `im2col(F16) + mul_mat(F16,F16)`
compute pipeline — the only difference is the one-time dequant cost at session creation.
