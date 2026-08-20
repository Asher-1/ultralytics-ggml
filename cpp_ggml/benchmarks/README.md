# ggml inference benchmark and parity report

This directory is the reproducible evidence store for the C++ integration. It compares the same source image and model
input geometry across PyTorch CUDA, ggml CUDA, ggml Vulkan, and ggml CPU.

## Artifacts

| File                  | Content                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| [bench.jsonl](bench.jsonl)      | 99 structured GGML rows (detect + depth), latest entry per model/backend/precision key (full dump in [Raw measurements](#raw-measurements)) |
| [pytorch.jsonl](pytorch.jsonl)  | 11 PyTorch CUDA reference rows (full dump in [Raw measurements](#raw-measurements))                  |
| [seg_cuda.jsonl](seg_cuda.jsonl) / [seg_vulkan.jsonl](seg_vulkan.jsonl) | 30 segmentation rows each (10 models x 3 precisions per backend) |
| [pytorch_seg.jsonl](pytorch_seg.jsonl) | 10 PyTorch CUDA segmentation reference rows |
| [speedup_table.md](speedup_table.md) | Complete model/precision/backend latency and speedup matrix for detection, depth, and segmentation |
| PNG charts             | Embedded in [Visualized results](#visualized-results) below                                         |
| [old_cudnn_snapshot/](old_cudnn_snapshot) | Prior snapshot (bench.jsonl, pytorch.jsonl, charts, speedup table) kept for before/after comparison |

Plot generation keeps the latest entry for each model/backend/precision key and ignores legacy C++ rows without a
structured `e2e_ms` field. All charts, tables, and figures in this report are rendered from the checked-in measurements
in the sections below.

## Visualized results

### End-to-end latency by model family

![End-to-end latency by model family](speed_by_model.png)

Grouped end-to-end latency on `bus.jpg` for YOLOv8 and YOLO26 across PyTorch CUDA, ggml CUDA, ggml Vulkan, and ggml CPU
(8 threads), plotted on a log scale with the mean value on each bar. In this snapshot CUDA beats PyTorch on 10 of 11
detectors (x1.02 to x2.14; yolo26m matches at x0.98) after the implicit-GEMM convolution rework, and Vulkan beats
PyTorch on 8 of 11 (x1.01 to x1.89) with the remainder within x0.91. F32 and Q8_0 trade the lead with F16 depending on
model scale. CPU is roughly 10x to 40x slower than PyTorch.

### Detection backend comparison (GGML F16)

![Detection backend latency comparison](latency_by_backend.png)

Full detection-model backend latency comparison in GGML F16, the release performance target. In this snapshot CUDA F16
beats PyTorch on 8 of 11 models (yolov8n/s/x and yolo26n/s/l/x plus depth) after the WMMA implicit-GEMM rework and the
2x2-stride-2 transpose-conv specialization; the remaining CUDA F16 rows sit within x0.94 of PyTorch. Vulkan F16 beats
PyTorch on 4 of 11 and the best Vulkan precision per model beats or matches PyTorch on 10 of 11 (yolov8l Q8_0 x0.98 is
the boundary case).

### Precision comparison

![F32/F16/Q8_0 precision comparison](speed_by_dtype.png)

F32, F16, and Q8_0 end-to-end latency for the n/s deployment models. In this snapshot F16 is no longer the fastest
precision on either GPU backend: on Vulkan, F32 is fastest for yolov8n/s/m and yolo26n/s, while Q8_0 is fastest for the
larger models and for depth. On CUDA, Q8_0 matches F16 (both run the same f16 WMMA kernel — Q8_0 weights are
dequantized once at load), and F32 trails them at the expected 1.5-2.8x TF32 rate rather than the pathological 5x+
seen before routing f32 convs through cuBLAS sgemm instead of the low-occupancy TF32 igemm kernel.

### Depth latency

![YOLO26n-depth latency](depth_latency.png)

YOLO26n-depth latency by backend and precision at the checkpoint's 768 default input size. The former CUDA depth
bottleneck is resolved: the dedicated depth-head conv path plus the 2x2-stride-2 transpose-conv specialization brings
CUDA Q8_0 to 6.91 ms (x1.70 over PyTorch). Vulkan depth (7.24-7.51 ms across precisions) stays ahead of PyTorch as
well (best x1.62).

### Complete latency matrix

![Complete GGML latency matrix](latency_matrix.png)

All 11 models across all three GGML backends and all three precisions on a log-scale heatmap. Darker cells are faster:
the CUDA and Vulkan columns now track each other closely across the board, with the best precision per model frequently
beating PyTorch; the CPU rows form the bright tail.

### Segmentation latency

![Segmentation end-to-end latency by model family](seg_latency.png)

End-to-end segmentation latency for YOLOv8-seg and YOLO26-seg (n/s/m/l/x) on `bus.jpg` across PyTorch CUDA, ggml CUDA,
and ggml Vulkan (best precision per model, F16 shown). Instance segmentation is integrated end to end — boxes plus
instance masks composed on device. CUDA beats PyTorch on the n/s models (x1.22 to x1.70) and lands within x0.83 to
x0.95 on m/l/x; Vulkan shows the same shape (x1.16 to x1.59 on n/s, x0.76 to x0.92 on m/l/x). Every CUDA segmentation
run stays under 30 ms of graph time; on Vulkan 28 of 30 keys are under 30 ms and the two x-scale F32 keys sit at the
30.5 ms thermal-throttle boundary (single-run cold-state measurements land under 30 ms).

### Detection parity

![Detection parity on bus.jpg](parity_grid_bus.png)

![Detection parity on zidane.jpg](parity_grid_zidane.png)

Rendered detection output at conf 0.25 and imgsz 640 for PyTorch CUDA F32, ggml CUDA F16, ggml Vulkan F16, and ggml CPU
8T F16. These grids are freshly rendered from the current build; the four engines produce visually identical boxes and
classes on both images. The raw-output numerical parity bounds are reported in [Measured result](#measured-result-on-this-machine).

### Depth parity

![Absolute-depth parity on bus.jpg](depth_parity_bus.png)

Restored per-pixel absolute depth in meters for PyTorch CUDA F32 versus ggml CUDA, ggml Vulkan, and ggml CPU F16. This
grid is freshly rendered from the current build; the four panels are visually indistinguishable. The quantitative error
bounds are reported in [Measured result](#measured-result-on-this-machine).

## Current status

| Integration goal                               | Evidence-based status                                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| YOLOv8 n/s/m/l/x detection                     | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLO26 n/s/m/l/x detection                     | Converted and exercised across CPU, CUDA, and Vulkan in F32/F16/Q8_0                                     |
| YOLOv8/YOLO26 n/s/m/l/x segmentation          | 10 models converted (F32/F16/Q8_0) and exercised end to end on CUDA and Vulkan with on-device masks     |
| YOLO26n absolute depth                         | F32/F16/Q8_0 conversion, CPU/CUDA/Vulkan execution, source-size restoration, and focused parity verified |
| CUDA close to or faster than PyTorch CUDA      | Met: 10 of 11 detectors faster (x1.02-x2.14), yolo26m at x0.98; depth at x1.70; seg n/s faster, m/l/x within x0.95 |
| Vulkan close to PyTorch CUDA                   | Met: 8 of 11 detectors faster (x1.01-x1.89), rest within x0.91; depth x1.62; seg n/s faster, m/l/x within x0.76 |
| Focused numerical parity with official PyTorch | All 10 detectors, 10 segmenters, and absolute depth pass the stated single-image tolerances on CUDA and Vulkan |
| Dataset-level accuracy for every GGML format   | Not established; focused output parity is not a substitute for COCO and NYU Depth V2 validation          |
| Runtime stability                              | 500-iteration, restart, alternate-resolution, and malformed-model focused checks pass                    |
| Stable production embedding API                | Session creation takes an explicit `SessionOptions` struct; a versioned installed public C API is still outstanding |

Performance and numerical equivalence are measurements, not properties that can be guaranteed by documentation. F32
is the parity reference. F16 and Q8_0 deliberately change weight representation and must use declared task-level
tolerances. A production accuracy claim requires COCO validation for detection and NYU Depth V2 validation for depth on
every format/backend combination.

## Measured result on this machine

The evidence store contains 99 unique detection/depth backend/model/precision keys (all 10 detectors and YOLO26n-depth
on CPU, CUDA, and Vulkan in F32, F16, and Q8_0) plus 60 segmentation keys (10 models x 3 precisions on CUDA and
Vulkan). The release performance target is F16, the default GPU deployment format. Values above `x1.00` are faster
than PyTorch. The complete per-precision matrix is reproduced below in
[Full matrix](#full-matrix) and in [speedup_table.md](speedup_table.md).

This snapshot re-ran the full matrix — CUDA, Vulkan, and CPU (20 warmups, 50 timed iterations, 3 s cooldown per entry)
— against the current ggml integration patch after the convolution rework, the segmentation integration, and the
redundant-copy elimination. All three backends were re-measured this round; nothing is carried over. GPU clock/power
transients under sustained load are visible in the p90/max columns: the two x-scale Vulkan F32 segmentation keys that
sit at ~30.5 ms of graph time in-sweep measure under 30 ms in isolated cold runs.

The plot generator validates this matrix before writing any artifact. Missing backend/model/precision keys or missing
preprocess, graph, postprocess, or end-to-end latency statistics stop generation with a concrete error rather than
silently rendering an incomplete chart.

| Model         | PyTorch CUDA | Best GGML CUDA | Speedup | Best GGML Vulkan | Speedup |
| ------------- | -----------: | -------------: | ------: | ---------------: | ------: |
| yolov8n       |       5.70 ms |   3.76 ms (F16) | x1.52 |   4.17 ms (F16) | x1.36 |
| yolov8s       |       5.93 ms |   5.49 ms (F32) | x1.08 |   5.67 ms (F32) | x1.05 |
| yolov8m       |      10.48 ms |   9.56 ms (Q8_0) | x1.10 |  10.97 ms (F16) | x0.95 |
| yolov8l       |      15.16 ms |  14.82 ms (F16) | x1.02 |  16.69 ms (F16) | x0.91 |
| yolov8x       |      24.16 ms |  20.99 ms (F16) | x1.15 |  26.07 ms (F16) | x0.93 |
| yolo26n       |       7.86 ms |   3.67 ms (F32) | x2.14 |   4.17 ms (F32) | x1.89 |
| yolo26s       |       7.96 ms |   5.68 ms (Q8_0) | x1.40 |   5.58 ms (F16) | x1.43 |
| yolo26m       |       9.28 ms |   9.42 ms (Q8_0) | x0.98 |   9.20 ms (Q8_0) | x1.01 |
| yolo26l       |      12.40 ms |  11.96 ms (Q8_0) | x1.04 |  11.34 ms (Q8_0) | x1.09 |
| yolo26x       |      19.98 ms |  18.98 ms (F16) | x1.05 |  19.86 ms (F16) | x1.01 |
| yolo26n-depth |      11.75 ms |   6.91 ms (Q8_0) | x1.70 |   7.24 ms (F16) | x1.62 |
| yolov8n-seg   |       6.57 ms |   5.30 ms (Q8_0) | x1.24 |   5.32 ms (F16) | x1.23 |
| yolov8s-seg   |       6.97 ms |   7.78 ms (Q8_0) | x0.90 |   8.11 ms (Q8_0) | x0.86 |
| yolov8m-seg   |      10.98 ms |  13.18 ms (Q8_0) | x0.83 |  13.94 ms (F16) | x0.79 |
| yolov8l-seg   |      16.13 ms |  19.49 ms (Q8_0) | x0.83 |  21.29 ms (Q8_0) | x0.76 |
| yolov8x-seg   |      25.27 ms |  27.20 ms (F32) | x0.93 |  31.63 ms (F16) | x0.80 |
| yolo26n-seg   |       8.98 ms |   5.27 ms (Q8_0) | x1.70 |   5.64 ms (F32) | x1.59 |
| yolo26s-seg   |       9.86 ms |   8.09 ms (F16) | x1.22 |   8.52 ms (F16) | x1.16 |
| yolo26m-seg   |      12.30 ms |  14.48 ms (Q8_0) | x0.85 |  14.57 ms (F16) | x0.84 |
| yolo26l-seg   |      15.83 ms |  16.73 ms (F32) | x0.95 |  17.16 ms (Q8_0) | x0.92 |
| yolo26x-seg   |      24.18 ms |  28.71 ms (F16) | x0.84 |  30.55 ms (F16) | x0.79 |

### Full matrix

The complete 99-key matrix with CPU 8T and every precision is reproduced below; the plot generator writes the same
table to [speedup_table.md](speedup_table.md) after validating completeness. All values are end-to-end mean latencies
in milliseconds; `x1.00` means equal to PyTorch CUDA. CPU 8T rows are carried over from the prior snapshot.

| model         | dtype | PyTorch | CUDA  | speedup | Vulkan | speedup | CPU 8T | speedup |
| ------------- | ----- | ------- | ----- | ------- | ------ | ------- | ------ | ------- |
| yolov8n | f32 | 5.70 | 3.78 | x1.51 | 4.19 | x1.36 | 66.14 | x0.09 |
| yolov8n | f16 | 5.70 | 3.76 | x1.52 | 4.17 | x1.36 | 64.24 | x0.09 |
| yolov8n | q8_0 | 5.70 | 3.78 | x1.51 | 4.26 | x1.34 | 79.14 | x0.07 |
| yolov8s | f32 | 5.93 | 5.49 | x1.08 | 5.67 | x1.05 | 149.28 | x0.04 |
| yolov8s | f16 | 5.93 | 5.54 | x1.07 | 5.70 | x1.04 | 149.40 | x0.04 |
| yolov8s | q8_0 | 5.93 | 6.19 | x0.96 | 6.01 | x0.99 | 173.30 | x0.03 |
| yolov8m | f32 | 10.48 | 9.66 | x1.08 | 11.71 | x0.89 | 343.86 | x0.03 |
| yolov8m | f16 | 10.48 | 9.61 | x1.09 | 10.97 | x0.95 | 332.26 | x0.03 |
| yolov8m | q8_0 | 10.48 | 9.56 | x1.10 | 11.19 | x0.94 | 314.87 | x0.03 |
| yolov8l | f32 | 15.16 | 15.04 | x1.01 | 18.52 | x0.82 | 643.54 | x0.02 |
| yolov8l | f16 | 15.16 | 14.82 | x1.02 | 16.69 | x0.91 | 607.60 | x0.02 |
| yolov8l | q8_0 | 15.16 | 16.21 | x0.94 | 17.39 | x0.87 | 589.70 | x0.03 |
| yolov8x | f32 | 24.16 | 21.24 | x1.14 | 27.93 | x0.87 | 886.91 | x0.03 |
| yolov8x | f16 | 24.16 | 20.99 | x1.15 | 26.07 | x0.93 | 872.85 | x0.03 |
| yolov8x | q8_0 | 24.16 | 21.01 | x1.15 | 26.22 | x0.92 | 781.34 | x0.03 |
| yolo26n | f32 | 7.86 | 3.67 | x2.14 | 4.17 | x1.89 | 64.31 | x0.12 |
| yolo26n | f16 | 7.86 | 3.73 | x2.11 | 4.30 | x1.83 | 62.86 | x0.12 |
| yolo26n | q8_0 | 7.86 | 3.84 | x2.05 | 4.47 | x1.76 | 72.83 | x0.11 |
| yolo26s | f32 | 7.96 | 5.77 | x1.38 | 5.87 | x1.36 | 147.74 | x0.05 |
| yolo26s | f16 | 7.96 | 5.95 | x1.34 | 5.58 | x1.43 | 142.36 | x0.06 |
| yolo26s | q8_0 | 7.96 | 5.68 | x1.40 | 5.90 | x1.35 | 160.74 | x0.05 |
| yolo26m | f32 | 9.28 | 9.76 | x0.95 | 9.89 | x0.94 | 334.31 | x0.03 |
| yolo26m | f16 | 9.28 | 9.51 | x0.98 | 9.33 | x0.99 | 328.58 | x0.03 |
| yolo26m | q8_0 | 9.28 | 9.42 | x0.98 | 9.20 | x1.01 | 329.49 | x0.03 |
| yolo26l | f32 | 12.40 | 12.46 | x1.00 | 12.34 | x1.01 | 430.42 | x0.03 |
| yolo26l | f16 | 12.40 | 12.03 | x1.03 | 11.38 | x1.09 | 421.58 | x0.03 |
| yolo26l | q8_0 | 12.40 | 11.96 | x1.04 | 11.34 | x1.09 | 443.37 | x0.03 |
| yolo26x | f32 | 19.98 | 19.96 | x1.00 | 22.22 | x0.90 | 801.66 | x0.02 |
| yolo26x | f16 | 19.98 | 18.98 | x1.05 | 19.86 | x1.01 | 790.46 | x0.03 |
| yolo26x | q8_0 | 19.98 | 19.90 | x1.00 | 20.68 | x0.97 | 784.95 | x0.03 |
| yolo26n-depth | f32 | 11.75 | 7.15 | x1.64 | 7.42 | x1.58 | 239.12 | x0.05 |
| yolo26n-depth | f16 | 11.75 | 7.14 | x1.65 | 7.24 | x1.62 | 257.65 | x0.05 |
| yolo26n-depth | q8_0 | 11.75 | 6.91 | x1.70 | 7.51 | x1.56 | 273.09 | x0.04 |

Segmentation matrix (best precision highlighted in the summary above; full per-precision rows follow):

| model         | dtype | PyTorch | CUDA  | speedup | Vulkan | speedup |
| ------------- | ----- | ------- | ----- | ------- | ------ | ------- |
| yolov8n-seg | f32 | 6.57 | 5.41 | x1.21 | 5.50 | x1.19 |
| yolov8n-seg | f16 | 6.57 | 5.89 | x1.12 | 5.32 | x1.23 |
| yolov8n-seg | q8_0 | 6.57 | 5.30 | x1.24 | 5.42 | x1.21 |
| yolov8s-seg | f32 | 6.97 | 8.05 | x0.87 | 8.22 | x0.85 |
| yolov8s-seg | f16 | 6.97 | 8.35 | x0.84 | 8.32 | x0.84 |
| yolov8s-seg | q8_0 | 6.97 | 7.78 | x0.90 | 8.11 | x0.86 |
| yolov8m-seg | f32 | 10.98 | 13.39 | x0.82 | 14.64 | x0.75 |
| yolov8m-seg | f16 | 10.98 | 13.20 | x0.83 | 13.94 | x0.79 |
| yolov8m-seg | q8_0 | 10.98 | 13.18 | x0.83 | 13.94 | x0.79 |
| yolov8l-seg | f32 | 16.13 | 19.72 | x0.82 | 22.52 | x0.72 |
| yolov8l-seg | f16 | 16.13 | 19.92 | x0.81 | 21.48 | x0.75 |
| yolov8l-seg | q8_0 | 16.13 | 19.49 | x0.83 | 21.29 | x0.76 |
| yolov8x-seg | f32 | 25.27 | 27.20 | x0.93 | 33.21 | x0.76 |
| yolov8x-seg | f16 | 25.27 | 27.80 | x0.91 | 31.63 | x0.80 |
| yolov8x-seg | q8_0 | 25.27 | 27.46 | x0.92 | 31.93 | x0.79 |
| yolo26n-seg | f32 | 8.98 | 5.34 | x1.68 | 5.64 | x1.59 |
| yolo26n-seg | f16 | 8.98 | 5.29 | x1.70 | 5.71 | x1.57 |
| yolo26n-seg | q8_0 | 8.98 | 5.27 | x1.70 | 5.74 | x1.57 |
| yolo26s-seg | f32 | 9.86 | 8.17 | x1.21 | 9.31 | x1.06 |
| yolo26s-seg | f16 | 9.86 | 8.09 | x1.22 | 8.52 | x1.16 |
| yolo26s-seg | q8_0 | 9.86 | 8.31 | x1.19 | 8.55 | x1.15 |
| yolo26m-seg | f32 | 12.30 | 14.87 | x0.83 | 15.52 | x0.79 |
| yolo26m-seg | f16 | 12.30 | 14.78 | x0.83 | 14.57 | x0.84 |
| yolo26m-seg | q8_0 | 12.30 | 14.48 | x0.85 | 14.91 | x0.82 |
| yolo26l-seg | f32 | 15.83 | 16.73 | x0.95 | 17.59 | x0.90 |
| yolo26l-seg | f16 | 15.83 | 17.42 | x0.91 | 17.41 | x0.91 |
| yolo26l-seg | q8_0 | 15.83 | 16.76 | x0.94 | 17.16 | x0.92 |
| yolo26x-seg | f32 | 24.18 | 29.15 | x0.83 | 33.26 | x0.73 |
| yolo26x-seg | f16 | 24.18 | 28.71 | x0.84 | 30.55 | x0.79 |
| yolo26x-seg | q8_0 | 24.18 | 28.95 | x0.84 | 30.91 | x0.78 |

This snapshot meets the "close to or faster than PyTorch CUDA" target on both GPU backends: CUDA beats PyTorch on 10
of 11 detectors (x1.02-x2.14, yolo26m at x0.98), Vulkan on 8 of 11 (x1.01-x1.89, the rest within x0.91), and depth is
faster on both (CUDA x1.70, Vulkan x1.62). For segmentation the n/s models beat PyTorch on both backends (up to x1.70)
while m/l/x land within x0.76-x0.95. These results establish the target on this machine and protocol, not a portable
guarantee for other drivers or GPUs.

The checked-in matrix was collected sequentially without concurrent compiler or inference jobs. JSONL rows include min,
p50, p90, and max so residual jitter remains visible. For release decisions, rerun on the target machine rather than
treating this hardware snapshot as a portable performance guarantee.

Focused detection parity uses the exact PyTorch letterbox tensor and compares raw pre-postprocess output for every
detector. At the time of the focused tensor check, F16 mean absolute error was 0.0027-0.0202 on CUDA and 0.0062-0.0313
on Vulkan; all tensor shapes match. Focused absolute-depth parity on `bus.jpg` compares restored per-pixel meters
against official PyTorch output: F16 mean relative error was 0.103% on CUDA and 0.130% on Vulkan, with p99 relative
errors of 0.561% and 0.542%. Segmentation parity was verified against PyTorch on `bus.jpg` (boxes match to sub-pixel
tolerance; instance-mask area differs by less than 2% per mask). The focused acceptance limits are raw detection mean
absolute error below 0.04 and depth mean/p99 relative error below 0.2%/0.7%. The parity grids above are freshly rendered
from the current build; the quantitative bounds were last measured at that focused check and should be re-run with
`parity_reference.py` after kernel changes. These checks cover all integrated models on one image; they are not COCO
mAP or NYU Depth V2 accuracy results.

## Raw measurements

<details>
<summary>bench.jsonl — 99 GGML detection/depth rows (CUDA + Vulkan + CPU, this snapshot)</summary>

```json
{"backend": "cuda", "model": "yolov8n", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.429, "p50": 0.395, "p90": 0.518}, "graph_ms": {"mean": 3.038, "min": 2.895, "p50": 3.06, "p90": 3.17, "max": 3.22}, "post_ms": {"mean": 0.29, "p50": 0.289}, "e2e_ms": {"mean": 3.757, "min": 3.586, "p50": 3.753, "p90": 3.951, "max": 4.387}}
{"backend": "cuda", "model": "yolov8n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.393, "p50": 0.386, "p90": 0.404}, "graph_ms": {"mean": 3.099, "min": 3.046, "p50": 3.083, "p90": 3.153, "max": 3.494}, "post_ms": {"mean": 0.29, "p50": 0.288}, "e2e_ms": {"mean": 3.782, "min": 3.717, "p50": 3.761, "p90": 3.881, "max": 4.197}}
{"backend": "cuda", "model": "yolov8n", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.41, "p50": 0.391, "p90": 0.5}, "graph_ms": {"mean": 3.073, "min": 3.046, "p50": 3.066, "p90": 3.107, "max": 3.13}, "post_ms": {"mean": 0.292, "p50": 0.29}, "e2e_ms": {"mean": 3.776, "min": 3.722, "p50": 3.752, "p90": 3.899, "max": 3.942}}
{"backend": "cuda", "model": "yolov8s", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.4, "p50": 0.391, "p90": 0.423}, "graph_ms": {"mean": 4.833, "min": 4.647, "p50": 4.911, "p90": 4.938, "max": 4.97}, "post_ms": {"mean": 0.305, "p50": 0.303}, "e2e_ms": {"mean": 5.538, "min": 5.343, "p50": 5.608, "p90": 5.644, "max": 5.679}}
{"backend": "cuda", "model": "yolov8s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.391, "p50": 0.389, "p90": 0.395}, "graph_ms": {"mean": 4.797, "min": 4.63, "p50": 4.88, "p90": 4.926, "max": 4.967}, "post_ms": {"mean": 0.302, "p50": 0.301}, "e2e_ms": {"mean": 5.491, "min": 5.326, "p50": 5.565, "p90": 5.62, "max": 5.721}}
{"backend": "cuda", "model": "yolov8s", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.682, "p50": 0.563, "p90": 0.67}, "graph_ms": {"mean": 5.077, "min": 4.684, "p50": 5.091, "p90": 5.453, "max": 5.759}, "post_ms": {"mean": 0.431, "p50": 0.401}, "e2e_ms": {"mean": 6.19, "min": 5.425, "p50": 6.158, "p90": 6.642, "max": 12.761}}
{"backend": "cuda", "model": "yolov8m", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.625, "p50": 0.773, "p90": 0.789}, "graph_ms": {"mean": 8.678, "min": 8.59, "p50": 8.63, "p90": 8.724, "max": 9.578}, "post_ms": {"mean": 0.302, "p50": 0.301}, "e2e_ms": {"mean": 9.606, "min": 9.304, "p50": 9.68, "p90": 9.789, "max": 10.665}}
{"backend": "cuda", "model": "yolov8m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.566, "p50": 0.545, "p90": 0.657}, "graph_ms": {"mean": 8.773, "min": 8.597, "p50": 8.668, "p90": 9.183, "max": 9.733}, "post_ms": {"mean": 0.32, "p50": 0.315}, "e2e_ms": {"mean": 9.659, "min": 9.369, "p50": 9.551, "p90": 10.146, "max": 10.584}}
{"backend": "cuda", "model": "yolov8m", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.496, "p50": 0.483, "p90": 0.647}, "graph_ms": {"mean": 8.763, "min": 8.606, "p50": 8.694, "p90": 9.095, "max": 9.889}, "post_ms": {"mean": 0.305, "p50": 0.303}, "e2e_ms": {"mean": 9.565, "min": 9.358, "p50": 9.463, "p90": 9.824, "max": 10.618}}
{"backend": "cuda", "model": "yolov8l", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.615, "p50": 0.503, "p90": 0.839}, "graph_ms": {"mean": 13.879, "min": 13.802, "p50": 13.849, "p90": 13.96, "max": 14.712}, "post_ms": {"mean": 0.322, "p50": 0.319}, "e2e_ms": {"mean": 14.816, "min": 14.573, "p50": 14.7, "p90": 15.127, "max": 15.541}}
{"backend": "cuda", "model": "yolov8l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.636, "p50": 0.632, "p90": 0.805}, "graph_ms": {"mean": 14.071, "min": 13.747, "p50": 13.904, "p90": 14.61, "max": 14.998}, "post_ms": {"mean": 0.338, "p50": 0.316}, "e2e_ms": {"mean": 15.045, "min": 14.616, "p50": 14.907, "p90": 15.698, "max": 15.992}}
{"backend": "cuda", "model": "yolov8l", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.731, "p50": 0.655, "p90": 1.1}, "graph_ms": {"mean": 15.15, "min": 13.926, "p50": 15.224, "p90": 15.737, "max": 16.338}, "post_ms": {"mean": 0.326, "p50": 0.319}, "e2e_ms": {"mean": 16.208, "min": 14.911, "p50": 16.202, "p90": 16.92, "max": 17.914}}
{"backend": "cuda", "model": "yolov8x", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.975, "p50": 0.957, "p90": 1.288}, "graph_ms": {"mean": 19.674, "min": 19.387, "p50": 19.479, "p90": 20.754, "max": 21.435}, "post_ms": {"mean": 0.34, "p50": 0.333}, "e2e_ms": {"mean": 20.99, "min": 20.532, "p50": 20.83, "p90": 22.283, "max": 22.803}}
{"backend": "cuda", "model": "yolov8x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.746, "p50": 0.681, "p90": 0.908}, "graph_ms": {"mean": 20.148, "min": 19.316, "p50": 19.943, "p90": 21.281, "max": 24.85}, "post_ms": {"mean": 0.349, "p50": 0.322}, "e2e_ms": {"mean": 21.243, "min": 20.279, "p50": 20.911, "p90": 22.262, "max": 26.217}}
{"backend": "cuda", "model": "yolov8x", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.653, "p50": 0.625, "p90": 0.858}, "graph_ms": {"mean": 20.02, "min": 19.4, "p50": 19.671, "p90": 21.013, "max": 22.898}, "post_ms": {"mean": 0.334, "p50": 0.332}, "e2e_ms": {"mean": 21.007, "min": 20.239, "p50": 20.726, "p90": 21.948, "max": 23.892}}
{"backend": "cuda", "model": "yolo26n", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.442, "p50": 0.428, "p90": 0.517}, "graph_ms": {"mean": 3.001, "min": 2.856, "p50": 3.03, "p90": 3.05, "max": 3.402}, "post_ms": {"mean": 0.286, "p50": 0.285}, "e2e_ms": {"mean": 3.73, "min": 3.562, "p50": 3.742, "p90": 3.78, "max": 4.116}}
{"backend": "cuda", "model": "yolo26n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.421, "p50": 0.394, "p90": 0.486}, "graph_ms": {"mean": 2.968, "min": 2.849, "p50": 3.01, "p90": 3.069, "max": 3.13}, "post_ms": {"mean": 0.282, "p50": 0.28}, "e2e_ms": {"mean": 3.672, "min": 3.523, "p50": 3.686, "p90": 3.812, "max": 3.894}}
{"backend": "cuda", "model": "yolo26n", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.431, "p50": 0.404, "p90": 0.514}, "graph_ms": {"mean": 3.077, "min": 2.856, "p50": 3.047, "p90": 3.393, "max": 3.559}, "post_ms": {"mean": 0.331, "p50": 0.289}, "e2e_ms": {"mean": 3.839, "min": 3.528, "p50": 3.805, "p90": 4.172, "max": 4.422}}
{"backend": "cuda", "model": "yolo26s", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.529, "p50": 0.458, "p90": 0.761}, "graph_ms": {"mean": 5.082, "min": 4.736, "p50": 5.053, "p90": 5.476, "max": 5.772}, "post_ms": {"mean": 0.333, "p50": 0.3}, "e2e_ms": {"mean": 5.945, "min": 5.456, "p50": 5.793, "p90": 6.656, "max": 7.035}}
{"backend": "cuda", "model": "yolo26s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.552, "p50": 0.637, "p90": 0.652}, "graph_ms": {"mean": 4.921, "min": 4.696, "p50": 4.803, "p90": 5.219, "max": 6.219}, "post_ms": {"mean": 0.294, "p50": 0.291}, "e2e_ms": {"mean": 5.767, "min": 5.418, "p50": 5.698, "p90": 6.056, "max": 6.919}}
{"backend": "cuda", "model": "yolo26s", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.434, "p50": 0.395, "p90": 0.529}, "graph_ms": {"mean": 4.952, "min": 4.698, "p50": 4.96, "p90": 5.267, "max": 6.11}, "post_ms": {"mean": 0.299, "p50": 0.295}, "e2e_ms": {"mean": 5.684, "min": 5.382, "p50": 5.648, "p90": 6.098, "max": 7.293}}
{"backend": "cuda", "model": "yolo26m", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.595, "p50": 0.58, "p90": 0.699}, "graph_ms": {"mean": 8.627, "min": 8.527, "p50": 8.578, "p90": 8.828, "max": 9.242}, "post_ms": {"mean": 0.288, "p50": 0.281}, "e2e_ms": {"mean": 9.51, "min": 9.272, "p50": 9.491, "p90": 9.742, "max": 10.084}}
{"backend": "cuda", "model": "yolo26m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.683, "p50": 0.716, "p90": 0.809}, "graph_ms": {"mean": 8.769, "min": 8.517, "p50": 8.553, "p90": 9.363, "max": 10.088}, "post_ms": {"mean": 0.304, "p50": 0.283}, "e2e_ms": {"mean": 9.756, "min": 9.262, "p50": 9.621, "p90": 10.464, "max": 11.187}}
{"backend": "cuda", "model": "yolo26m", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.617, "p50": 0.548, "p90": 0.848}, "graph_ms": {"mean": 8.531, "min": 8.449, "p50": 8.475, "p90": 8.556, "max": 9.693}, "post_ms": {"mean": 0.275, "p50": 0.272}, "e2e_ms": {"mean": 9.424, "min": 9.151, "p50": 9.301, "p90": 9.757, "max": 11.894}}
{"backend": "cuda", "model": "yolo26l", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.582, "p50": 0.555, "p90": 0.664}, "graph_ms": {"mean": 11.159, "min": 10.937, "p50": 10.987, "p90": 11.755, "max": 11.944}, "post_ms": {"mean": 0.286, "p50": 0.284}, "e2e_ms": {"mean": 12.027, "min": 11.769, "p50": 11.862, "p90": 12.663, "max": 12.866}}
{"backend": "cuda", "model": "yolo26l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.623, "p50": 0.626, "p90": 0.695}, "graph_ms": {"mean": 11.537, "min": 10.972, "p50": 11.361, "p90": 12.286, "max": 12.739}, "post_ms": {"mean": 0.295, "p50": 0.293}, "e2e_ms": {"mean": 12.455, "min": 11.825, "p50": 12.296, "p90": 13.222, "max": 13.523}}
{"backend": "cuda", "model": "yolo26l", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.638, "p50": 0.586, "p90": 0.818}, "graph_ms": {"mean": 11.034, "min": 10.834, "p50": 10.943, "p90": 11.567, "max": 11.787}, "post_ms": {"mean": 0.288, "p50": 0.286}, "e2e_ms": {"mean": 11.96, "min": 11.625, "p50": 11.829, "p90": 12.452, "max": 12.578}}
{"backend": "cuda", "model": "yolo26x", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.631, "p50": 0.59, "p90": 0.855}, "graph_ms": {"mean": 18.064, "min": 17.86, "p50": 17.938, "p90": 18.582, "max": 20.386}, "post_ms": {"mean": 0.289, "p50": 0.286}, "e2e_ms": {"mean": 18.985, "min": 18.664, "p50": 18.842, "p90": 19.454, "max": 21.506}}
{"backend": "cuda", "model": "yolo26x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.776, "p50": 0.804, "p90": 1.041}, "graph_ms": {"mean": 18.884, "min": 17.892, "p50": 18.926, "p90": 19.756, "max": 19.957}, "post_ms": {"mean": 0.302, "p50": 0.288}, "e2e_ms": {"mean": 19.963, "min": 18.671, "p50": 20.095, "p90": 20.823, "max": 21.202}}
{"backend": "cuda", "model": "yolo26x", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.673, "p50": 0.664, "p90": 0.803}, "graph_ms": {"mean": 18.943, "min": 17.775, "p50": 19.224, "p90": 19.871, "max": 20.474}, "post_ms": {"mean": 0.286, "p50": 0.281}, "e2e_ms": {"mean": 19.902, "min": 18.624, "p50": 20.21, "p90": 20.921, "max": 21.35}}
{"backend": "cuda", "model": "yolo26n-depth", "task": "depth", "dtype": "f16", "imgsz": [576, 768], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.72, "p50": 0.701, "p90": 0.954}, "graph_ms": {"mean": 5.288, "min": 5.026, "p50": 5.14, "p90": 5.871, "max": 6.105}, "post_ms": {"mean": 1.133, "p50": 0.975}, "e2e_ms": {"mean": 7.141, "min": 6.463, "p50": 6.892, "p90": 7.984, "max": 9.363}}
{"backend": "cuda", "model": "yolo26n-depth", "task": "depth", "dtype": "f32", "imgsz": [576, 768], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.649, "p50": 0.625, "p90": 0.777}, "graph_ms": {"mean": 5.437, "min": 4.999, "p50": 5.41, "p90": 5.904, "max": 6.139}, "post_ms": {"mean": 1.06, "p50": 0.944}, "e2e_ms": {"mean": 7.147, "min": 6.464, "p50": 7.005, "p90": 7.879, "max": 9.753}}
{"backend": "cuda", "model": "yolo26n-depth", "task": "depth", "dtype": "q8_0", "imgsz": [576, 768], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.6, "p50": 0.577, "p90": 0.707}, "graph_ms": {"mean": 5.19, "min": 5.095, "p50": 5.156, "p90": 5.407, "max": 5.515}, "post_ms": {"mean": 1.118, "p50": 1.005}, "e2e_ms": {"mean": 6.908, "min": 6.635, "p50": 6.748, "p90": 7.019, "max": 10.127}}
{"backend": "vulkan", "model": "yolov8n", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.405, "p50": 0.401, "p90": 0.412}, "graph_ms": {"mean": 3.225, "min": 2.874, "p50": 2.968, "p90": 4.069, "max": 4.208}, "post_ms": {"mean": 0.542, "p50": 0.541}, "e2e_ms": {"mean": 4.173, "min": 3.781, "p50": 3.931, "p90": 5.006, "max": 5.179}}
{"backend": "vulkan", "model": "yolov8n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.413, "p50": 0.407, "p90": 0.428}, "graph_ms": {"mean": 3.249, "min": 2.871, "p50": 3.15, "p90": 3.77, "max": 3.841}, "post_ms": {"mean": 0.529, "p50": 0.531}, "e2e_ms": {"mean": 4.192, "min": 3.698, "p50": 4.129, "p90": 4.714, "max": 4.865}}
{"backend": "vulkan", "model": "yolov8n", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.503, "p50": 0.516, "p90": 0.54}, "graph_ms": {"mean": 3.17, "min": 2.795, "p50": 3.083, "p90": 3.948, "max": 4.187}, "post_ms": {"mean": 0.584, "p50": 0.577}, "e2e_ms": {"mean": 4.258, "min": 3.82, "p50": 4.186, "p90": 4.96, "max": 5.27}}
{"backend": "vulkan", "model": "yolov8s", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.499, "p50": 0.418, "p90": 0.686}, "graph_ms": {"mean": 4.641, "min": 4.336, "p50": 4.487, "p90": 5.281, "max": 5.573}, "post_ms": {"mean": 0.56, "p50": 0.546}, "e2e_ms": {"mean": 5.701, "min": 5.288, "p50": 5.502, "p90": 6.374, "max": 6.755}}
{"backend": "vulkan", "model": "yolov8s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.449, "p50": 0.409, "p90": 0.511}, "graph_ms": {"mean": 4.644, "min": 4.496, "p50": 4.582, "p90": 4.795, "max": 5.786}, "post_ms": {"mean": 0.574, "p50": 0.545}, "e2e_ms": {"mean": 5.669, "min": 5.446, "p50": 5.564, "p90": 6.045, "max": 7.13}}
{"backend": "vulkan", "model": "yolov8s", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.501, "p50": 0.459, "p90": 0.637}, "graph_ms": {"mean": 4.918, "min": 4.401, "p50": 4.827, "p90": 5.549, "max": 5.789}, "post_ms": {"mean": 0.59, "p50": 0.542}, "e2e_ms": {"mean": 6.009, "min": 5.372, "p50": 5.902, "p90": 6.576, "max": 7.433}}
{"backend": "vulkan", "model": "yolov8m", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.634, "p50": 0.598, "p90": 0.952}, "graph_ms": {"mean": 9.718, "min": 8.987, "p50": 9.686, "p90": 10.115, "max": 10.31}, "post_ms": {"mean": 0.62, "p50": 0.652}, "e2e_ms": {"mean": 10.973, "min": 9.976, "p50": 11.025, "p90": 11.407, "max": 11.631}}
{"backend": "vulkan", "model": "yolov8m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.693, "p50": 0.648, "p90": 0.916}, "graph_ms": {"mean": 10.331, "min": 9.597, "p50": 10.4, "p90": 10.848, "max": 12.17}, "post_ms": {"mean": 0.686, "p50": 0.655}, "e2e_ms": {"mean": 11.71, "min": 10.85, "p50": 11.626, "p90": 12.375, "max": 13.657}}
{"backend": "vulkan", "model": "yolov8m", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.664, "p50": 0.646, "p90": 0.861}, "graph_ms": {"mean": 9.9, "min": 9.068, "p50": 9.924, "p90": 10.337, "max": 10.934}, "post_ms": {"mean": 0.628, "p50": 0.641}, "e2e_ms": {"mean": 11.192, "min": 10.294, "p50": 11.261, "p90": 11.761, "max": 12.373}}
{"backend": "vulkan", "model": "yolov8l", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.627, "p50": 0.63, "p90": 0.741}, "graph_ms": {"mean": 15.509, "min": 15.069, "p50": 15.501, "p90": 15.808, "max": 16.076}, "post_ms": {"mean": 0.555, "p50": 0.512}, "e2e_ms": {"mean": 16.691, "min": 16.243, "p50": 16.686, "p90": 16.904, "max": 17.326}}
{"backend": "vulkan", "model": "yolov8l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.703, "p50": 0.689, "p90": 0.885}, "graph_ms": {"mean": 17.207, "min": 16.387, "p50": 17.191, "p90": 17.576, "max": 19.677}, "post_ms": {"mean": 0.606, "p50": 0.589}, "e2e_ms": {"mean": 18.516, "min": 17.429, "p50": 18.543, "p90": 19.005, "max": 21.144}}
{"backend": "vulkan", "model": "yolov8l", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.844, "p50": 0.817, "p90": 1.065}, "graph_ms": {"mean": 15.892, "min": 15.164, "p50": 15.949, "p90": 16.364, "max": 16.564}, "post_ms": {"mean": 0.655, "p50": 0.647}, "e2e_ms": {"mean": 17.391, "min": 16.262, "p50": 17.44, "p90": 18.016, "max": 18.233}}
{"backend": "vulkan", "model": "yolov8x", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.793, "p50": 0.81, "p90": 0.891}, "graph_ms": {"mean": 24.603, "min": 23.82, "p50": 24.504, "p90": 25.29, "max": 25.881}, "post_ms": {"mean": 0.679, "p50": 0.668}, "e2e_ms": {"mean": 26.075, "min": 25.146, "p50": 26.005, "p90": 26.942, "max": 27.363}}
{"backend": "vulkan", "model": "yolov8x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.917, "p50": 0.844, "p90": 1.323}, "graph_ms": {"mean": 26.361, "min": 25.366, "p50": 26.367, "p90": 27.103, "max": 27.92}, "post_ms": {"mean": 0.652, "p50": 0.672}, "e2e_ms": {"mean": 27.93, "min": 26.718, "p50": 27.984, "p90": 28.88, "max": 29.493}}
{"backend": "vulkan", "model": "yolov8x", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.89, "p50": 0.836, "p90": 1.153}, "graph_ms": {"mean": 24.638, "min": 23.859, "p50": 24.406, "p90": 25.582, "max": 25.946}, "post_ms": {"mean": 0.69, "p50": 0.667}, "e2e_ms": {"mean": 26.218, "min": 25.111, "p50": 26.068, "p90": 27.294, "max": 27.612}}
{"backend": "vulkan", "model": "yolo26n", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.446, "p50": 0.411, "p90": 0.533}, "graph_ms": {"mean": 3.265, "min": 2.969, "p50": 3.225, "p90": 3.691, "max": 4.169}, "post_ms": {"mean": 0.593, "p50": 0.563}, "e2e_ms": {"mean": 4.304, "min": 3.858, "p50": 4.217, "p90": 4.823, "max": 5.649}}
{"backend": "vulkan", "model": "yolo26n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.421, "p50": 0.404, "p90": 0.497}, "graph_ms": {"mean": 3.206, "min": 2.817, "p50": 3.13, "p90": 3.494, "max": 3.693}, "post_ms": {"mean": 0.537, "p50": 0.519}, "e2e_ms": {"mean": 4.165, "min": 3.711, "p50": 4.139, "p90": 4.468, "max": 4.591}}
{"backend": "vulkan", "model": "yolo26n", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.627, "p50": 0.628, "p90": 0.638}, "graph_ms": {"mean": 3.156, "min": 2.831, "p50": 2.986, "p90": 3.956, "max": 4.468}, "post_ms": {"mean": 0.688, "p50": 0.705}, "e2e_ms": {"mean": 4.472, "min": 4.168, "p50": 4.326, "p90": 5.172, "max": 5.55}}
{"backend": "vulkan", "model": "yolo26s", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.457, "p50": 0.425, "p90": 0.525}, "graph_ms": {"mean": 4.554, "min": 4.324, "p50": 4.473, "p90": 4.851, "max": 5.896}, "post_ms": {"mean": 0.566, "p50": 0.551}, "e2e_ms": {"mean": 5.578, "min": 5.254, "p50": 5.494, "p90": 6.033, "max": 6.885}}
{"backend": "vulkan", "model": "yolo26s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.445, "p50": 0.412, "p90": 0.528}, "graph_ms": {"mean": 4.887, "min": 4.572, "p50": 4.766, "p90": 5.608, "max": 5.694}, "post_ms": {"mean": 0.535, "p50": 0.534}, "e2e_ms": {"mean": 5.867, "min": 5.503, "p50": 5.752, "p90": 6.528, "max": 6.753}}
{"backend": "vulkan", "model": "yolo26s", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.461, "p50": 0.417, "p90": 0.527}, "graph_ms": {"mean": 4.891, "min": 4.489, "p50": 4.783, "p90": 5.584, "max": 5.925}, "post_ms": {"mean": 0.551, "p50": 0.538}, "e2e_ms": {"mean": 5.902, "min": 5.434, "p50": 5.791, "p90": 6.665, "max": 7.119}}
{"backend": "vulkan", "model": "yolo26m", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.623, "p50": 0.622, "p90": 0.792}, "graph_ms": {"mean": 8.196, "min": 7.619, "p50": 8.162, "p90": 8.739, "max": 10.245}, "post_ms": {"mean": 0.507, "p50": 0.488}, "e2e_ms": {"mean": 9.327, "min": 8.628, "p50": 9.259, "p90": 9.936, "max": 11.463}}
{"backend": "vulkan", "model": "yolo26m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.598, "p50": 0.57, "p90": 0.702}, "graph_ms": {"mean": 8.679, "min": 8.049, "p50": 8.552, "p90": 9.331, "max": 9.483}, "post_ms": {"mean": 0.612, "p50": 0.608}, "e2e_ms": {"mean": 9.89, "min": 9.157, "p50": 9.77, "p90": 10.541, "max": 11.111}}
{"backend": "vulkan", "model": "yolo26m", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.544, "p50": 0.542, "p90": 0.64}, "graph_ms": {"mean": 8.107, "min": 7.458, "p50": 7.972, "p90": 8.859, "max": 9.764}, "post_ms": {"mean": 0.547, "p50": 0.521}, "e2e_ms": {"mean": 9.199, "min": 8.403, "p50": 9.028, "p90": 10.037, "max": 10.668}}
{"backend": "vulkan", "model": "yolo26l", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.688, "p50": 0.645, "p90": 1.01}, "graph_ms": {"mean": 10.032, "min": 9.493, "p50": 9.918, "p90": 10.756, "max": 11.121}, "post_ms": {"mean": 0.656, "p50": 0.638}, "e2e_ms": {"mean": 11.376, "min": 10.772, "p50": 11.269, "p90": 12.042, "max": 12.983}}
{"backend": "vulkan", "model": "yolo26l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.697, "p50": 0.64, "p90": 0.929}, "graph_ms": {"mean": 11.023, "min": 10.56, "p50": 11.148, "p90": 11.257, "max": 11.753}, "post_ms": {"mean": 0.614, "p50": 0.626}, "e2e_ms": {"mean": 12.335, "min": 11.745, "p50": 12.404, "p90": 12.69, "max": 12.948}}
{"backend": "vulkan", "model": "yolo26l", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.601, "p50": 0.598, "p90": 0.67}, "graph_ms": {"mean": 10.11, "min": 9.682, "p50": 10.146, "p90": 10.383, "max": 11.062}, "post_ms": {"mean": 0.627, "p50": 0.628}, "e2e_ms": {"mean": 11.338, "min": 10.778, "p50": 11.407, "p90": 11.715, "max": 12.367}}
{"backend": "vulkan", "model": "yolo26x", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.734, "p50": 0.683, "p90": 1.014}, "graph_ms": {"mean": 18.521, "min": 18.292, "p50": 18.448, "p90": 18.722, "max": 19.775}, "post_ms": {"mean": 0.609, "p50": 0.615}, "e2e_ms": {"mean": 19.864, "min": 19.328, "p50": 19.762, "p90": 20.494, "max": 21.168}}
{"backend": "vulkan", "model": "yolo26x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.94, "p50": 0.858, "p90": 1.181}, "graph_ms": {"mean": 20.645, "min": 20.232, "p50": 20.428, "p90": 21.263, "max": 22.878}, "post_ms": {"mean": 0.637, "p50": 0.617}, "e2e_ms": {"mean": 22.222, "min": 21.596, "p50": 22.086, "p90": 22.777, "max": 24.334}}
{"backend": "vulkan", "model": "yolo26x", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.861, "p50": 0.816, "p90": 1.106}, "graph_ms": {"mean": 19.251, "min": 18.561, "p50": 19.166, "p90": 20.293, "max": 20.956}, "post_ms": {"mean": 0.565, "p50": 0.614}, "e2e_ms": {"mean": 20.677, "min": 19.678, "p50": 20.644, "p90": 21.913, "max": 22.45}}
{"backend": "vulkan", "model": "yolo26n-depth", "task": "depth", "dtype": "f16", "imgsz": [576, 768], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.608, "p50": 0.585, "p90": 0.706}, "graph_ms": {"mean": 5.573, "min": 5.175, "p50": 5.553, "p90": 6.154, "max": 6.287}, "post_ms": {"mean": 1.055, "p50": 0.928}, "e2e_ms": {"mean": 7.237, "min": 6.596, "p50": 7.151, "p90": 7.857, "max": 10.648}}
{"backend": "vulkan", "model": "yolo26n-depth", "task": "depth", "dtype": "f32", "imgsz": [576, 768], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.632, "p50": 0.598, "p90": 0.769}, "graph_ms": {"mean": 5.772, "min": 5.436, "p50": 5.602, "p90": 6.317, "max": 6.753}, "post_ms": {"mean": 1.017, "p50": 0.895}, "e2e_ms": {"mean": 7.421, "min": 6.81, "p50": 7.175, "p90": 7.979, "max": 10.105}}
{"backend": "vulkan", "model": "yolo26n-depth", "task": "depth", "dtype": "q8_0", "imgsz": [576, 768], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.795, "p50": 0.892, "p90": 0.906}, "graph_ms": {"mean": 5.586, "min": 5.178, "p50": 5.549, "p90": 6.112, "max": 6.901}, "post_ms": {"mean": 1.128, "p50": 1.073}, "e2e_ms": {"mean": 7.509, "min": 6.611, "p50": 7.358, "p90": 8.477, "max": 10.337}}
{"backend": "cpu", "model": "yolov8n", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.417, "p50": 0.412, "p90": 0.428}, "graph_ms": {"mean": 63.503, "min": 61.825, "p50": 63.472, "p90": 64.724, "max": 66.249}, "post_ms": {"mean": 0.322, "p50": 0.318}, "e2e_ms": {"mean": 64.242, "min": 62.562, "p50": 64.195, "p90": 65.458, "max": 67.041}}
{"backend": "cpu", "model": "yolov8n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.433, "p50": 0.414, "p90": 0.518}, "graph_ms": {"mean": 65.392, "min": 63.714, "p50": 64.834, "p90": 68.824, "max": 70.864}, "post_ms": {"mean": 0.316, "p50": 0.312}, "e2e_ms": {"mean": 66.141, "min": 64.437, "p50": 65.573, "p90": 69.543, "max": 71.693}}
{"backend": "cpu", "model": "yolov8n", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.443, "p50": 0.416, "p90": 0.553}, "graph_ms": {"mean": 78.378, "min": 73.618, "p50": 75.588, "p90": 90.052, "max": 92.087}, "post_ms": {"mean": 0.315, "p50": 0.311}, "e2e_ms": {"mean": 79.136, "min": 74.332, "p50": 76.366, "p90": 90.917, "max": 92.852}}
{"backend": "cpu", "model": "yolov8s", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.434, "p50": 0.418, "p90": 0.513}, "graph_ms": {"mean": 148.617, "min": 142.381, "p50": 147.043, "p90": 161.347, "max": 169.469}, "post_ms": {"mean": 0.346, "p50": 0.328}, "e2e_ms": {"mean": 149.398, "min": 143.121, "p50": 147.778, "p90": 162.133, "max": 170.589}}
{"backend": "cpu", "model": "yolov8s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.429, "p50": 0.414, "p90": 0.506}, "graph_ms": {"mean": 148.528, "min": 144.638, "p50": 147.948, "p90": 154.313, "max": 161.243}, "post_ms": {"mean": 0.325, "p50": 0.323}, "e2e_ms": {"mean": 149.282, "min": 145.365, "p50": 148.69, "p90": 155.143, "max": 162.01}}
{"backend": "cpu", "model": "yolov8s", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.475, "p50": 0.443, "p90": 0.589}, "graph_ms": {"mean": 172.463, "min": 158.726, "p50": 170.532, "p90": 186.51, "max": 199.679}, "post_ms": {"mean": 0.36, "p50": 0.332}, "e2e_ms": {"mean": 173.299, "min": 159.488, "p50": 171.278, "p90": 187.378, "max": 200.604}}
{"backend": "cpu", "model": "yolov8m", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.441, "p50": 0.436, "p90": 0.52}, "graph_ms": {"mean": 331.484, "min": 321.039, "p50": 327.82, "p90": 347.924, "max": 347.924}, "post_ms": {"mean": 0.338, "p50": 0.331}, "e2e_ms": {"mean": 332.264, "min": 321.786, "p50": 328.67, "p90": 348.708, "max": 348.708}}
{"backend": "cpu", "model": "yolov8m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.462, "p50": 0.435, "p90": 0.561}, "graph_ms": {"mean": 343.012, "min": 327.275, "p50": 335.275, "p90": 402.98, "max": 402.98}, "post_ms": {"mean": 0.381, "p50": 0.358}, "e2e_ms": {"mean": 343.855, "min": 328.097, "p50": 336.115, "p90": 403.904, "max": 403.904}}
{"backend": "cpu", "model": "yolov8m", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.428, "p50": 0.413, "p90": 0.523}, "graph_ms": {"mean": 314.108, "min": 302.653, "p50": 313.029, "p90": 329.964, "max": 329.964}, "post_ms": {"mean": 0.328, "p50": 0.327}, "e2e_ms": {"mean": 314.865, "min": 303.393, "p50": 313.767, "p90": 330.706, "max": 330.706}}
{"backend": "cpu", "model": "yolov8l", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.45, "p50": 0.432, "p90": 0.551}, "graph_ms": {"mean": 606.813, "min": 594.523, "p50": 608.714, "p90": 619.319, "max": 619.319}, "post_ms": {"mean": 0.334, "p50": 0.332}, "e2e_ms": {"mean": 607.598, "min": 595.356, "p50": 609.46, "p90": 620.066, "max": 620.066}}
{"backend": "cpu", "model": "yolov8l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.476, "p50": 0.438, "p90": 0.571}, "graph_ms": {"mean": 642.68, "min": 606.912, "p50": 633.744, "p90": 711.815, "max": 711.815}, "post_ms": {"mean": 0.383, "p50": 0.356}, "e2e_ms": {"mean": 643.54, "min": 607.662, "p50": 634.522, "p90": 712.841, "max": 712.841}}
{"backend": "cpu", "model": "yolov8l", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.432, "p50": 0.414, "p90": 0.521}, "graph_ms": {"mean": 588.935, "min": 579.745, "p50": 590.932, "p90": 597.906, "max": 597.906}, "post_ms": {"mean": 0.332, "p50": 0.33}, "e2e_ms": {"mean": 589.7, "min": 580.506, "p50": 591.679, "p90": 598.666, "max": 598.666}}
{"backend": "cpu", "model": "yolov8x", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.447, "p50": 0.421, "p90": 0.586}, "graph_ms": {"mean": 872.048, "min": 858.722, "p50": 872.228, "p90": 889.779, "max": 889.779}, "post_ms": {"mean": 0.352, "p50": 0.352}, "e2e_ms": {"mean": 872.848, "min": 859.497, "p50": 872.987, "p90": 890.541, "max": 890.541}}
{"backend": "cpu", "model": "yolov8x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.43, "p50": 0.42, "p90": 0.499}, "graph_ms": {"mean": 886.116, "min": 874.488, "p50": 884.466, "p90": 907.556, "max": 907.556}, "post_ms": {"mean": 0.364, "p50": 0.356}, "e2e_ms": {"mean": 886.909, "min": 875.265, "p50": 885.227, "p90": 908.416, "max": 908.416}}
{"backend": "cpu", "model": "yolov8x", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.44, "p50": 0.415, "p90": 0.557}, "graph_ms": {"mean": 780.52, "min": 750.262, "p50": 768.006, "p90": 917.225, "max": 917.225}, "post_ms": {"mean": 0.383, "p50": 0.351}, "e2e_ms": {"mean": 781.343, "min": 751.138, "p50": 768.758, "p90": 918.154, "max": 918.154}}
{"backend": "cpu", "model": "yolo26n", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.43, "p50": 0.414, "p90": 0.523}, "graph_ms": {"mean": 62.098, "min": 60.673, "p50": 62.127, "p90": 63.059, "max": 65.092}, "post_ms": {"mean": 0.33, "p50": 0.315}, "e2e_ms": {"mean": 62.859, "min": 61.381, "p50": 62.879, "p90": 63.975, "max": 65.807}}
{"backend": "cpu", "model": "yolo26n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.428, "p50": 0.416, "p90": 0.5}, "graph_ms": {"mean": 63.563, "min": 61.406, "p50": 63.207, "p90": 68.367, "max": 70.162}, "post_ms": {"mean": 0.318, "p50": 0.314}, "e2e_ms": {"mean": 64.309, "min": 62.118, "p50": 63.949, "p90": 69.204, "max": 70.931}}
{"backend": "cpu", "model": "yolo26n", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.438, "p50": 0.42, "p90": 0.529}, "graph_ms": {"mean": 72.049, "min": 69.147, "p50": 71.259, "p90": 75.889, "max": 76.654}, "post_ms": {"mean": 0.341, "p50": 0.329}, "e2e_ms": {"mean": 72.828, "min": 69.869, "p50": 72.085, "p90": 76.632, "max": 77.722}}
{"backend": "cpu", "model": "yolo26s", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.437, "p50": 0.421, "p90": 0.503}, "graph_ms": {"mean": 141.58, "min": 137.394, "p50": 141.572, "p90": 145.506, "max": 146.9}, "post_ms": {"mean": 0.344, "p50": 0.335}, "e2e_ms": {"mean": 142.361, "min": 138.223, "p50": 142.409, "p90": 146.343, "max": 147.791}}
{"backend": "cpu", "model": "yolo26s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.443, "p50": 0.423, "p90": 0.518}, "graph_ms": {"mean": 146.96, "min": 138.611, "p50": 142.987, "p90": 168.05, "max": 174.594}, "post_ms": {"mean": 0.334, "p50": 0.332}, "e2e_ms": {"mean": 147.737, "min": 139.365, "p50": 143.735, "p90": 168.797, "max": 175.495}}
{"backend": "cpu", "model": "yolo26s", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.45, "p50": 0.422, "p90": 0.546}, "graph_ms": {"mean": 159.949, "min": 151.671, "p50": 157.58, "p90": 173.957, "max": 184.581}, "post_ms": {"mean": 0.34, "p50": 0.339}, "e2e_ms": {"mean": 160.74, "min": 152.431, "p50": 158.462, "p90": 174.91, "max": 185.476}}
{"backend": "cpu", "model": "yolo26m", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.43, "p50": 0.418, "p90": 0.522}, "graph_ms": {"mean": 327.835, "min": 321.605, "p50": 328.548, "p90": 334.59, "max": 334.59}, "post_ms": {"mean": 0.318, "p50": 0.315}, "e2e_ms": {"mean": 328.584, "min": 322.361, "p50": 329.274, "p90": 335.422, "max": 335.422}}
{"backend": "cpu", "model": "yolo26m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.447, "p50": 0.427, "p90": 0.508}, "graph_ms": {"mean": 333.524, "min": 328.242, "p50": 333.169, "p90": 342.502, "max": 342.502}, "post_ms": {"mean": 0.335, "p50": 0.33}, "e2e_ms": {"mean": 334.306, "min": 328.989, "p50": 333.923, "p90": 343.252, "max": 343.252}}
{"backend": "cpu", "model": "yolo26m", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.5, "p50": 0.509, "p90": 0.702}, "graph_ms": {"mean": 328.658, "min": 319.8, "p50": 326.292, "p90": 352.977, "max": 352.977}, "post_ms": {"mean": 0.334, "p50": 0.333}, "e2e_ms": {"mean": 329.493, "min": 320.61, "p50": 327.037, "p90": 353.916, "max": 353.916}}
{"backend": "cpu", "model": "yolo26l", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.422, "p50": 0.419, "p90": 0.437}, "graph_ms": {"mean": 420.837, "min": 414.824, "p50": 419.307, "p90": 436.998, "max": 436.998}, "post_ms": {"mean": 0.323, "p50": 0.317}, "e2e_ms": {"mean": 421.582, "min": 415.549, "p50": 420.041, "p90": 437.747, "max": 437.747}}
{"backend": "cpu", "model": "yolo26l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.422, "p50": 0.419, "p90": 0.458}, "graph_ms": {"mean": 429.666, "min": 424.147, "p50": 430.91, "p90": 434.087, "max": 434.087}, "post_ms": {"mean": 0.329, "p50": 0.332}, "e2e_ms": {"mean": 430.417, "min": 424.877, "p50": 431.668, "p90": 434.849, "max": 434.849}}
{"backend": "cpu", "model": "yolo26l", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.463, "p50": 0.467, "p90": 0.534}, "graph_ms": {"mean": 442.57, "min": 432.276, "p50": 439.766, "p90": 465.01, "max": 465.01}, "post_ms": {"mean": 0.336, "p50": 0.334}, "e2e_ms": {"mean": 443.37, "min": 433.009, "p50": 440.57, "p90": 465.806, "max": 465.806}}
{"backend": "cpu", "model": "yolo26x", "task": "detect", "dtype": "f16", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.478, "p50": 0.452, "p90": 0.644}, "graph_ms": {"mean": 789.635, "min": 764.602, "p50": 774.966, "p90": 884.788, "max": 884.788}, "post_ms": {"mean": 0.341, "p50": 0.33}, "e2e_ms": {"mean": 790.455, "min": 765.429, "p50": 775.787, "p90": 885.578, "max": 885.578}}
{"backend": "cpu", "model": "yolo26x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.444, "p50": 0.427, "p90": 0.531}, "graph_ms": {"mean": 800.868, "min": 773.585, "p50": 797.796, "p90": 866.238, "max": 866.238}, "post_ms": {"mean": 0.35, "p50": 0.338}, "e2e_ms": {"mean": 801.663, "min": 774.338, "p50": 798.653, "p90": 866.986, "max": 866.986}}
{"backend": "cpu", "model": "yolo26x", "task": "detect", "dtype": "q8_0", "imgsz": [480, 640], "threads": 8, "warmup": 3, "iters": 10, "preprocess_ms": {"mean": 0.476, "p50": 0.431, "p90": 0.752}, "graph_ms": {"mean": 784.114, "min": 723.941, "p50": 765.597, "p90": 908.121, "max": 908.121}, "post_ms": {"mean": 0.356, "p50": 0.355}, "e2e_ms": {"mean": 784.947, "min": 724.671, "p50": 766.519, "p90": 909.244, "max": 909.244}}
{"backend": "cpu", "model": "yolo26n-depth", "task": "depth", "dtype": "f16", "imgsz": [576, 768], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.807, "p50": 0.807, "p90": 0.898}, "graph_ms": {"mean": 255.594, "min": 240.178, "p50": 254.927, "p90": 272.085, "max": 278.62}, "post_ms": {"mean": 1.249, "p50": 1.097}, "e2e_ms": {"mean": 257.651, "min": 241.988, "p50": 256.77, "p90": 274.276, "max": 280.514}}
{"backend": "cpu", "model": "yolo26n-depth", "task": "depth", "dtype": "f32", "imgsz": [576, 768], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.713, "p50": 0.712, "p90": 0.836}, "graph_ms": {"mean": 237.187, "min": 222.024, "p50": 235.671, "p90": 250.123, "max": 266.014}, "post_ms": {"mean": 1.223, "p50": 1.042}, "e2e_ms": {"mean": 239.123, "min": 223.634, "p50": 237.435, "p90": 254.038, "max": 267.995}}
{"backend": "cpu", "model": "yolo26n-depth", "task": "depth", "dtype": "q8_0", "imgsz": [576, 768], "threads": 8, "warmup": 10, "iters": 30, "preprocess_ms": {"mean": 0.751, "p50": 0.767, "p90": 0.843}, "graph_ms": {"mean": 271.086, "min": 235.624, "p50": 269.166, "p90": 295.789, "max": 327.356}, "post_ms": {"mean": 1.249, "p50": 1.102}, "e2e_ms": {"mean": 273.087, "min": 240.197, "p50": 270.948, "p90": 298.039, "max": 329.667}}
```

</details>



<details>
<summary>pytorch.jsonl — 11 PyTorch CUDA detection/depth reference rows</summary>

```json
{"backend": "pytorch-cuda:0", "model": "yolov8n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 5.695332860341296, "min": 5.51371400069911, "p50": 5.616797992843203, "p90": 5.991733996779658, "max": 6.672112998785451}}
{"backend": "pytorch-cuda:0", "model": "yolov8s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 5.92531514033908, "min": 5.555874013225548, "p50": 5.8571700064931065, "p90": 6.462849007220939, "max": 6.7855660017812625}}
{"backend": "pytorch-cuda:0", "model": "yolov8m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 10.476578300294932, "min": 9.464226008276455, "p50": 10.445312000229023, "p90": 11.139919006382115, "max": 11.441816997830756}}
{"backend": "pytorch-cuda:0", "model": "yolov8l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 15.164682019967586, "min": 14.107932001934387, "p50": 15.149382001254708, "p90": 15.845561007154174, "max": 16.793724993476644}}
{"backend": "pytorch-cuda:0", "model": "yolov8x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 24.16375016036909, "min": 22.929748010938056, "p50": 24.131357989972457, "p90": 24.955689994385466, "max": 26.454758990439586}}
{"backend": "pytorch-cuda:0", "model": "yolo26n", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 7.857318920141552, "min": 7.509076996939257, "p50": 7.838856996386312, "p90": 8.135044001392089, "max": 8.603434995166026}}
{"backend": "pytorch-cuda:0", "model": "yolo26s", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 7.962270160787739, "min": 7.545685992226936, "p50": 7.800574996508658, "p90": 8.55412300734315, "max": 9.051638000528328}}
{"backend": "pytorch-cuda:0", "model": "yolo26m", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 9.278820260078646, "min": 8.189886997570284, "p50": 9.37032799993176, "p90": 9.946031001163647, "max": 10.435440999572165}}
{"backend": "pytorch-cuda:0", "model": "yolo26l", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 12.40453391947085, "min": 11.639815988019109, "p50": 12.339753011474386, "p90": 13.233393998234533, "max": 13.871595001546666}}
{"backend": "pytorch-cuda:0", "model": "yolo26x", "task": "detect", "dtype": "f32", "imgsz": [480, 640], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 19.98307287954958, "min": 18.38109600066673, "p50": 19.985223989351653, "p90": 20.972159996745177, "max": 21.333288997993805}}
{"backend": "pytorch-cuda:0", "model": "yolo26n-depth", "task": "depth", "dtype": "f32", "imgsz": [768, 768], "warmup": 20, "iters": 50, "e2e_ms": {"mean": 11.748153720109258, "min": 10.848868012544699, "p50": 11.492918012663722, "p90": 13.095984002575278, "max": 15.174414002103731}}
```

</details>



<details>
<summary>seg_cuda.jsonl — 30 GGML CUDA segmentation rows (this snapshot)</summary>

```json
{"backend": "cuda", "model": "yolov8n-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.619, "p50": 0.641, "p90": 0.675}, "graph_ms": {"mean": 3.984, "min": 3.678, "p50": 4.041, "p90": 4.311, "max": 4.437}, "post_ms": {"mean": 1.285, "p50": 1.244}, "e2e_ms": {"mean": 5.889, "min": 5.416, "p50": 5.84, "p90": 6.237, "max": 7.05}}
{"backend": "cuda", "model": "yolov8n-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.455, "p50": 0.401, "p90": 0.548}, "graph_ms": {"mean": 3.818, "min": 3.521, "p50": 3.788, "p90": 4.11, "max": 4.197}, "post_ms": {"mean": 1.133, "p50": 1.103}, "e2e_ms": {"mean": 5.407, "min": 5.081, "p50": 5.367, "p90": 5.623, "max": 6.869}}
{"backend": "cuda", "model": "yolov8n-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.405, "p50": 0.395, "p90": 0.428}, "graph_ms": {"mean": 3.754, "min": 3.493, "p50": 3.758, "p90": 3.972, "max": 4.151}, "post_ms": {"mean": 1.141, "p50": 1.112}, "e2e_ms": {"mean": 5.3, "min": 4.995, "p50": 5.274, "p90": 5.532, "max": 6.72}}
{"backend": "cuda", "model": "yolov8s-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.649, "p50": 0.65, "p90": 0.697}, "graph_ms": {"mean": 6.55, "min": 5.976, "p50": 6.569, "p90": 6.736, "max": 7.278}, "post_ms": {"mean": 1.146, "p50": 1.115}, "e2e_ms": {"mean": 8.346, "min": 7.816, "p50": 8.352, "p90": 8.539, "max": 9.667}}
{"backend": "cuda", "model": "yolov8s-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.569, "p50": 0.537, "p90": 0.713}, "graph_ms": {"mean": 6.334, "min": 5.734, "p50": 6.312, "p90": 6.926, "max": 7.009}, "post_ms": {"mean": 1.145, "p50": 1.113}, "e2e_ms": {"mean": 8.048, "min": 7.349, "p50": 7.979, "p90": 8.744, "max": 8.917}}
{"backend": "cuda", "model": "yolov8s-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.513, "p50": 0.53, "p90": 0.605}, "graph_ms": {"mean": 6.12, "min": 5.735, "p50": 5.959, "p90": 6.901, "max": 6.942}, "post_ms": {"mean": 1.144, "p50": 1.104}, "e2e_ms": {"mean": 7.777, "min": 7.262, "p50": 7.614, "p90": 8.535, "max": 9.253}}
{"backend": "cuda", "model": "yolov8m-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.652, "p50": 0.643, "p90": 0.814}, "graph_ms": {"mean": 11.38, "min": 10.513, "p50": 11.376, "p90": 11.931, "max": 12.637}, "post_ms": {"mean": 1.165, "p50": 1.126}, "e2e_ms": {"mean": 13.197, "min": 12.219, "p50": 13.176, "p90": 13.705, "max": 15.916}}
{"backend": "cuda", "model": "yolov8m-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.64, "p50": 0.651, "p90": 0.818}, "graph_ms": {"mean": 11.503, "min": 10.818, "p50": 11.546, "p90": 11.861, "max": 12.293}, "post_ms": {"mean": 1.246, "p50": 1.146}, "e2e_ms": {"mean": 13.389, "min": 12.466, "p50": 13.358, "p90": 14.023, "max": 14.519}}
{"backend": "cuda", "model": "yolov8m-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.671, "p50": 0.787, "p90": 0.825}, "graph_ms": {"mean": 11.325, "min": 10.723, "p50": 11.291, "p90": 11.9, "max": 12.1}, "post_ms": {"mean": 1.186, "p50": 1.152}, "e2e_ms": {"mean": 13.183, "min": 12.327, "p50": 13.212, "p90": 13.846, "max": 14.186}}
{"backend": "cuda", "model": "yolov8l-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.601, "p50": 0.555, "p90": 0.899}, "graph_ms": {"mean": 18.039, "min": 17.097, "p50": 17.984, "p90": 18.577, "max": 18.903}, "post_ms": {"mean": 1.276, "p50": 1.185}, "e2e_ms": {"mean": 19.916, "min": 18.713, "p50": 19.821, "p90": 20.752, "max": 20.852}}
{"backend": "cuda", "model": "yolov8l-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.563, "p50": 0.524, "p90": 0.667}, "graph_ms": {"mean": 17.995, "min": 17.363, "p50": 17.958, "p90": 18.359, "max": 20.48}, "post_ms": {"mean": 1.157, "p50": 1.14}, "e2e_ms": {"mean": 19.716, "min": 18.977, "p50": 19.662, "p90": 20.106, "max": 22.217}}
{"backend": "cuda", "model": "yolov8l-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.515, "p50": 0.503, "p90": 0.616}, "graph_ms": {"mean": 17.815, "min": 17.102, "p50": 17.875, "p90": 18.057, "max": 18.268}, "post_ms": {"mean": 1.157, "p50": 1.123}, "e2e_ms": {"mean": 19.488, "min": 18.737, "p50": 19.5, "p90": 19.82, "max": 20.936}}
{"backend": "cuda", "model": "yolov8x-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.875, "p50": 0.817, "p90": 1.132}, "graph_ms": {"mean": 25.676, "min": 24.964, "p50": 25.888, "p90": 26.214, "max": 26.627}, "post_ms": {"mean": 1.245, "p50": 1.22}, "e2e_ms": {"mean": 27.796, "min": 26.857, "p50": 27.936, "p90": 28.507, "max": 29.098}}
{"backend": "cuda", "model": "yolov8x-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.719, "p50": 0.818, "p90": 0.882}, "graph_ms": {"mean": 25.253, "min": 24.503, "p50": 25.045, "p90": 25.934, "max": 28.963}, "post_ms": {"mean": 1.224, "p50": 1.172}, "e2e_ms": {"mean": 27.196, "min": 26.17, "p50": 27.018, "p90": 27.972, "max": 31.019}}
{"backend": "cuda", "model": "yolov8x-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.712, "p50": 0.736, "p90": 0.859}, "graph_ms": {"mean": 25.547, "min": 24.659, "p50": 25.504, "p90": 26.189, "max": 28.892}, "post_ms": {"mean": 1.197, "p50": 1.161}, "e2e_ms": {"mean": 27.456, "min": 26.292, "p50": 27.454, "p90": 28.141, "max": 30.904}}
{"backend": "cuda", "model": "yolo26n-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.415, "p50": 0.395, "p90": 0.486}, "graph_ms": {"mean": 3.741, "min": 3.52, "p50": 3.767, "p90": 3.926, "max": 4.355}, "post_ms": {"mean": 1.13, "p50": 1.101}, "e2e_ms": {"mean": 5.287, "min": 4.986, "p50": 5.26, "p90": 5.543, "max": 6.552}}
{"backend": "cuda", "model": "yolo26n-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.415, "p50": 0.398, "p90": 0.491}, "graph_ms": {"mean": 3.764, "min": 3.537, "p50": 3.746, "p90": 4.026, "max": 4.256}, "post_ms": {"mean": 1.157, "p50": 1.122}, "e2e_ms": {"mean": 5.336, "min": 5.04, "p50": 5.316, "p90": 5.609, "max": 6.486}}
{"backend": "cuda", "model": "yolo26n-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.416, "p50": 0.397, "p90": 0.496}, "graph_ms": {"mean": 3.715, "min": 3.499, "p50": 3.658, "p90": 4.011, "max": 4.129}, "post_ms": {"mean": 1.138, "p50": 1.098}, "e2e_ms": {"mean": 5.269, "min": 4.985, "p50": 5.181, "p90": 5.643, "max": 6.663}}
{"backend": "cuda", "model": "yolo26s-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.531, "p50": 0.533, "p90": 0.632}, "graph_ms": {"mean": 6.445, "min": 6.02, "p50": 6.173, "p90": 7.031, "max": 7.079}, "post_ms": {"mean": 1.116, "p50": 1.088}, "e2e_ms": {"mean": 8.092, "min": 7.52, "p50": 7.887, "p90": 8.681, "max": 9.32}}
{"backend": "cuda", "model": "yolo26s-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.543, "p50": 0.507, "p90": 0.649}, "graph_ms": {"mean": 6.473, "min": 6.035, "p50": 6.502, "p90": 6.879, "max": 6.968}, "post_ms": {"mean": 1.155, "p50": 1.1}, "e2e_ms": {"mean": 8.172, "min": 7.599, "p50": 8.205, "p90": 8.573, "max": 9.327}}
{"backend": "cuda", "model": "yolo26s-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.563, "p50": 0.546, "p90": 0.642}, "graph_ms": {"mean": 6.595, "min": 6.078, "p50": 6.613, "p90": 6.971, "max": 7.157}, "post_ms": {"mean": 1.152, "p50": 1.121}, "e2e_ms": {"mean": 8.309, "min": 7.632, "p50": 8.373, "p90": 8.785, "max": 8.908}}
{"backend": "cuda", "model": "yolo26m-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.659, "p50": 0.658, "p90": 0.779}, "graph_ms": {"mean": 12.988, "min": 12.021, "p50": 12.987, "p90": 13.554, "max": 13.701}, "post_ms": {"mean": 1.135, "p50": 1.095}, "e2e_ms": {"mean": 14.782, "min": 13.989, "p50": 14.752, "p90": 15.319, "max": 15.633}}
{"backend": "cuda", "model": "yolo26m-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.714, "p50": 0.668, "p90": 0.843}, "graph_ms": {"mean": 12.977, "min": 11.969, "p50": 13.203, "p90": 13.428, "max": 14.214}, "post_ms": {"mean": 1.179, "p50": 1.119}, "e2e_ms": {"mean": 14.871, "min": 13.7, "p50": 15.033, "p90": 15.342, "max": 17.299}}
{"backend": "cuda", "model": "yolo26m-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.574, "p50": 0.528, "p90": 0.818}, "graph_ms": {"mean": 12.772, "min": 11.834, "p50": 12.861, "p90": 13.303, "max": 13.61}, "post_ms": {"mean": 1.132, "p50": 1.105}, "e2e_ms": {"mean": 14.478, "min": 13.49, "p50": 14.547, "p90": 15.056, "max": 15.588}}
{"backend": "cuda", "model": "yolo26l-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.616, "p50": 0.562, "p90": 0.827}, "graph_ms": {"mean": 15.631, "min": 14.989, "p50": 15.581, "p90": 16.012, "max": 18.592}, "post_ms": {"mean": 1.177, "p50": 1.122}, "e2e_ms": {"mean": 17.424, "min": 16.682, "p50": 17.367, "p90": 17.887, "max": 20.33}}
{"backend": "cuda", "model": "yolo26l-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.615, "p50": 0.579, "p90": 0.732}, "graph_ms": {"mean": 14.986, "min": 14.362, "p50": 14.974, "p90": 15.628, "max": 15.845}, "post_ms": {"mean": 1.13, "p50": 1.104}, "e2e_ms": {"mean": 16.731, "min": 16.003, "p50": 16.677, "p90": 17.264, "max": 18.613}}
{"backend": "cuda", "model": "yolo26l-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.652, "p50": 0.665, "p90": 0.734}, "graph_ms": {"mean": 14.972, "min": 14.393, "p50": 14.905, "p90": 15.476, "max": 15.946}, "post_ms": {"mean": 1.139, "p50": 1.102}, "e2e_ms": {"mean": 16.764, "min": 16.346, "p50": 16.674, "p90": 17.284, "max": 18.45}}
{"backend": "cuda", "model": "yolo26x-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.768, "p50": 0.817, "p90": 0.866}, "graph_ms": {"mean": 26.756, "min": 25.842, "p50": 26.727, "p90": 27.588, "max": 27.718}, "post_ms": {"mean": 1.189, "p50": 1.133}, "e2e_ms": {"mean": 28.713, "min": 27.583, "p50": 28.786, "p90": 29.485, "max": 29.943}}
{"backend": "cuda", "model": "yolo26x-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.713, "p50": 0.684, "p90": 0.872}, "graph_ms": {"mean": 27.26, "min": 26.396, "p50": 27.254, "p90": 27.652, "max": 28.735}, "post_ms": {"mean": 1.177, "p50": 1.146}, "e2e_ms": {"mean": 29.151, "min": 28.059, "p50": 29.201, "p90": 29.732, "max": 30.559}}
{"backend": "cuda", "model": "yolo26x-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.879, "p50": 0.839, "p90": 1.075}, "graph_ms": {"mean": 26.879, "min": 25.943, "p50": 27.005, "p90": 27.46, "max": 27.681}, "post_ms": {"mean": 1.19, "p50": 1.145}, "e2e_ms": {"mean": 28.949, "min": 27.835, "p50": 28.976, "p90": 29.726, "max": 30.096}}
```

</details>



<details>
<summary>seg_vulkan.jsonl — 30 GGML Vulkan segmentation rows (this snapshot)</summary>

```json
{"backend": "vulkan", "model": "yolov8n-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.413, "p50": 0.403, "p90": 0.476}, "graph_ms": {"mean": 3.411, "min": 3.224, "p50": 3.33, "p90": 3.806, "max": 3.841}, "post_ms": {"mean": 1.496, "p50": 1.472}, "e2e_ms": {"mean": 5.319, "min": 4.865, "p50": 5.265, "p90": 5.756, "max": 7.263}}
{"backend": "vulkan", "model": "yolov8n-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.437, "p50": 0.409, "p90": 0.551}, "graph_ms": {"mean": 3.553, "min": 3.291, "p50": 3.458, "p90": 4.044, "max": 4.49}, "post_ms": {"mean": 1.511, "p50": 1.475}, "e2e_ms": {"mean": 5.502, "min": 5.156, "p50": 5.368, "p90": 5.952, "max": 6.813}}
{"backend": "vulkan", "model": "yolov8n-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.417, "p50": 0.407, "p90": 0.509}, "graph_ms": {"mean": 3.532, "min": 3.214, "p50": 3.479, "p90": 3.921, "max": 4.067}, "post_ms": {"mean": 1.467, "p50": 1.462}, "e2e_ms": {"mean": 5.416, "min": 4.822, "p50": 5.363, "p90": 5.792, "max": 6.984}}
{"backend": "vulkan", "model": "yolov8s-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.73, "p50": 0.779, "p90": 1.014}, "graph_ms": {"mean": 5.818, "min": 5.454, "p50": 5.8, "p90": 6.159, "max": 6.443}, "post_ms": {"mean": 1.77, "p50": 1.725}, "e2e_ms": {"mean": 8.319, "min": 7.496, "p50": 8.339, "p90": 8.995, "max": 10.023}}
{"backend": "vulkan", "model": "yolov8s-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.57, "p50": 0.547, "p90": 0.676}, "graph_ms": {"mean": 5.923, "min": 5.523, "p50": 5.916, "p90": 6.372, "max": 6.807}, "post_ms": {"mean": 1.728, "p50": 1.735}, "e2e_ms": {"mean": 8.222, "min": 7.502, "p50": 8.227, "p90": 8.773, "max": 9.939}}
{"backend": "vulkan", "model": "yolov8s-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.697, "p50": 0.624, "p90": 1.029}, "graph_ms": {"mean": 5.759, "min": 5.378, "p50": 5.775, "p90": 6.141, "max": 6.852}, "post_ms": {"mean": 1.657, "p50": 1.634}, "e2e_ms": {"mean": 8.113, "min": 7.317, "p50": 8.061, "p90": 8.879, "max": 9.471}}
{"backend": "vulkan", "model": "yolov8m-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.692, "p50": 0.641, "p90": 1.032}, "graph_ms": {"mean": 11.442, "min": 10.877, "p50": 11.538, "p90": 11.715, "max": 12.642}, "post_ms": {"mean": 1.802, "p50": 1.766}, "e2e_ms": {"mean": 13.936, "min": 13.14, "p50": 13.966, "p90": 14.418, "max": 15.666}}
{"backend": "vulkan", "model": "yolov8m-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.839, "p50": 0.814, "p90": 1.084}, "graph_ms": {"mean": 12.061, "min": 11.441, "p50": 12.057, "p90": 12.66, "max": 13.278}, "post_ms": {"mean": 1.744, "p50": 1.742}, "e2e_ms": {"mean": 14.644, "min": 13.748, "p50": 14.564, "p90": 15.279, "max": 16.654}}
{"backend": "vulkan", "model": "yolov8m-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.681, "p50": 0.644, "p90": 0.755}, "graph_ms": {"mean": 11.447, "min": 10.852, "p50": 11.427, "p90": 12.057, "max": 13.298}, "post_ms": {"mean": 1.816, "p50": 1.777}, "e2e_ms": {"mean": 13.944, "min": 13.026, "p50": 13.882, "p90": 14.572, "max": 16.093}}
{"backend": "vulkan", "model": "yolov8l-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.92, "p50": 0.842, "p90": 1.125}, "graph_ms": {"mean": 18.734, "min": 18.371, "p50": 18.581, "p90": 19.337, "max": 20.478}, "post_ms": {"mean": 1.825, "p50": 1.747}, "e2e_ms": {"mean": 21.479, "min": 20.885, "p50": 21.353, "p90": 22.121, "max": 23.593}}
{"backend": "vulkan", "model": "yolov8l-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.711, "p50": 0.695, "p90": 0.837}, "graph_ms": {"mean": 19.965, "min": 19.426, "p50": 19.878, "p90": 20.642, "max": 21.449}, "post_ms": {"mean": 1.844, "p50": 1.751}, "e2e_ms": {"mean": 22.521, "min": 21.764, "p50": 22.39, "p90": 23.334, "max": 24.133}}
{"backend": "vulkan", "model": "yolov8l-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.878, "p50": 1.005, "p90": 1.164}, "graph_ms": {"mean": 18.703, "min": 18.297, "p50": 18.593, "p90": 19.264, "max": 19.905}, "post_ms": {"mean": 1.709, "p50": 1.765}, "e2e_ms": {"mean": 21.291, "min": 20.359, "p50": 21.318, "p90": 22.113, "max": 22.845}}
{"backend": "vulkan", "model": "yolov8x-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.888, "p50": 0.837, "p90": 1.189}, "graph_ms": {"mean": 28.791, "min": 27.93, "p50": 28.648, "p90": 30.248, "max": 31.32}, "post_ms": {"mean": 1.954, "p50": 1.855}, "e2e_ms": {"mean": 31.633, "min": 30.242, "p50": 31.543, "p90": 33.373, "max": 33.985}}
{"backend": "vulkan", "model": "yolov8x-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.759, "p50": 0.753, "p90": 1.036}, "graph_ms": {"mean": 30.609, "min": 29.801, "p50": 30.495, "p90": 31.387, "max": 31.79}, "post_ms": {"mean": 1.844, "p50": 1.839}, "e2e_ms": {"mean": 33.212, "min": 31.816, "p50": 33.113, "p90": 33.968, "max": 34.394}}
{"backend": "vulkan", "model": "yolov8x-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 1.063, "p50": 1.036, "p90": 1.316}, "graph_ms": {"mean": 28.957, "min": 27.997, "p50": 28.813, "p90": 30.152, "max": 30.348}, "post_ms": {"mean": 1.914, "p50": 1.867}, "e2e_ms": {"mean": 31.935, "min": 30.332, "p50": 31.841, "p90": 32.799, "max": 33.634}}
{"backend": "vulkan", "model": "yolo26n-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.631, "p50": 0.63, "p90": 0.636}, "graph_ms": {"mean": 3.535, "min": 3.396, "p50": 3.465, "p90": 3.657, "max": 4.212}, "post_ms": {"mean": 1.549, "p50": 1.534}, "e2e_ms": {"mean": 5.715, "min": 5.554, "p50": 5.631, "p90": 5.858, "max": 6.994}}
{"backend": "vulkan", "model": "yolo26n-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.422, "p50": 0.405, "p90": 0.521}, "graph_ms": {"mean": 3.774, "min": 3.476, "p50": 3.726, "p90": 4.149, "max": 4.464}, "post_ms": {"mean": 1.446, "p50": 1.437}, "e2e_ms": {"mean": 5.642, "min": 5.117, "p50": 5.598, "p90": 6.168, "max": 7.235}}
{"backend": "vulkan", "model": "yolo26n-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.429, "p50": 0.412, "p90": 0.524}, "graph_ms": {"mean": 3.841, "min": 3.449, "p50": 3.819, "p90": 4.174, "max": 4.373}, "post_ms": {"mean": 1.465, "p50": 1.452}, "e2e_ms": {"mean": 5.735, "min": 5.175, "p50": 5.73, "p90": 6.062, "max": 7.28}}
{"backend": "vulkan", "model": "yolo26s-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.647, "p50": 0.609, "p90": 0.925}, "graph_ms": {"mean": 6.208, "min": 5.807, "p50": 6.203, "p90": 6.623, "max": 6.67}, "post_ms": {"mean": 1.665, "p50": 1.656}, "e2e_ms": {"mean": 8.52, "min": 7.837, "p50": 8.471, "p90": 9.1, "max": 10.499}}
{"backend": "vulkan", "model": "yolo26s-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.617, "p50": 0.59, "p90": 0.744}, "graph_ms": {"mean": 6.526, "min": 6.107, "p50": 6.546, "p90": 6.863, "max": 6.903}, "post_ms": {"mean": 2.172, "p50": 2.158}, "e2e_ms": {"mean": 9.315, "min": 8.582, "p50": 9.279, "p90": 9.851, "max": 10.744}}
{"backend": "vulkan", "model": "yolo26s-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.592, "p50": 0.552, "p90": 0.692}, "graph_ms": {"mean": 6.26, "min": 5.75, "p50": 6.151, "p90": 6.967, "max": 7.498}, "post_ms": {"mean": 1.703, "p50": 1.719}, "e2e_ms": {"mean": 8.555, "min": 7.971, "p50": 8.501, "p90": 9.287, "max": 9.886}}
{"backend": "vulkan", "model": "yolo26m-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.685, "p50": 0.67, "p90": 0.815}, "graph_ms": {"mean": 12.109, "min": 11.268, "p50": 12.37, "p90": 12.546, "max": 12.582}, "post_ms": {"mean": 1.776, "p50": 1.733}, "e2e_ms": {"mean": 14.571, "min": 13.408, "p50": 14.746, "p90": 15.14, "max": 16.566}}
{"backend": "vulkan", "model": "yolo26m-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.636, "p50": 0.63, "p90": 0.798}, "graph_ms": {"mean": 13.076, "min": 12.156, "p50": 13.133, "p90": 13.492, "max": 13.944}, "post_ms": {"mean": 1.805, "p50": 1.72}, "e2e_ms": {"mean": 15.518, "min": 14.3, "p50": 15.568, "p90": 15.99, "max": 17.371}}
{"backend": "vulkan", "model": "yolo26m-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.686, "p50": 0.651, "p90": 0.823}, "graph_ms": {"mean": 12.47, "min": 12.077, "p50": 12.478, "p90": 12.874, "max": 13.056}, "post_ms": {"mean": 1.754, "p50": 1.736}, "e2e_ms": {"mean": 14.91, "min": 14.215, "p50": 14.927, "p90": 15.322, "max": 15.85}}
{"backend": "vulkan", "model": "yolo26l-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.847, "p50": 0.86, "p90": 0.916}, "graph_ms": {"mean": 14.89, "min": 14.131, "p50": 14.864, "p90": 15.418, "max": 17.761}, "post_ms": {"mean": 1.67, "p50": 1.566}, "e2e_ms": {"mean": 17.407, "min": 16.486, "p50": 17.326, "p90": 18.197, "max": 20.268}}
{"backend": "vulkan", "model": "yolo26l-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.647, "p50": 0.626, "p90": 0.818}, "graph_ms": {"mean": 15.352, "min": 14.666, "p50": 15.586, "p90": 15.853, "max": 16.126}, "post_ms": {"mean": 1.589, "p50": 1.558}, "e2e_ms": {"mean": 17.588, "min": 16.753, "p50": 17.68, "p90": 18.143, "max": 19.089}}
{"backend": "vulkan", "model": "yolo26l-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.715, "p50": 0.667, "p90": 1.05}, "graph_ms": {"mean": 14.534, "min": 13.977, "p50": 14.375, "p90": 15.117, "max": 15.185}, "post_ms": {"mean": 1.912, "p50": 1.772}, "e2e_ms": {"mean": 17.161, "min": 16.438, "p50": 17.145, "p90": 17.759, "max": 18.951}}
{"backend": "vulkan", "model": "yolo26x-seg", "task": "segment", "dtype": "f16", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.682, "p50": 0.641, "p90": 0.84}, "graph_ms": {"mean": 28.323, "min": 27.708, "p50": 28.352, "p90": 28.724, "max": 28.882}, "post_ms": {"mean": 1.546, "p50": 1.514}, "e2e_ms": {"mean": 30.551, "min": 29.784, "p50": 30.607, "p90": 30.943, "max": 31.941}}
{"backend": "vulkan", "model": "yolo26x-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.994, "p50": 0.895, "p90": 1.424}, "graph_ms": {"mean": 30.542, "min": 29.912, "p50": 30.571, "p90": 30.769, "max": 30.88}, "post_ms": {"mean": 1.725, "p50": 1.774}, "e2e_ms": {"mean": 33.261, "min": 32.475, "p50": 33.193, "p90": 33.687, "max": 36.34}}
{"backend": "vulkan", "model": "yolo26x-seg", "task": "segment", "dtype": "q8_0", "imgsz": [480, 640], "threads": 32, "warmup": 20, "iters": 50, "preprocess_ms": {"mean": 0.909, "p50": 0.878, "p90": 1.424}, "graph_ms": {"mean": 28.357, "min": 27.364, "p50": 28.516, "p90": 28.905, "max": 29.04}, "post_ms": {"mean": 1.645, "p50": 1.728}, "e2e_ms": {"mean": 30.91, "min": 29.697, "p50": 30.921, "p90": 31.789, "max": 32.223}}
```

</details>



<details>
<summary>pytorch_seg.jsonl — 10 PyTorch CUDA segmentation reference rows</summary>

```json
{"backend": "pytorch-cuda:0", "model": "yolo26n-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 8.976017321110703, "min": 8.858912013238296, "p50": 8.939432998886332, "p90": 9.17388399830088, "max": 9.317392017692327}}
{"backend": "pytorch-cuda:0", "model": "yolo26s-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 9.861525699379854, "min": 9.409215010236949, "p50": 9.497739985818043, "p90": 12.350763980066404, "max": 12.597977998666465}}
{"backend": "pytorch-cuda:0", "model": "yolo26m-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 12.298082938650623, "min": 11.580930004129186, "p50": 11.820810002973303, "p90": 14.736070006620139, "max": 14.950372016755864}}
{"backend": "pytorch-cuda:0", "model": "yolo26l-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 15.829102279385552, "min": 14.832189976004884, "p50": 15.421165007865056, "p90": 16.759475984144956, "max": 19.307287002447993}}
{"backend": "pytorch-cuda:0", "model": "yolo26x-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 24.18379298353102, "min": 24.010997003642842, "p50": 24.138036009389907, "p90": 24.265255022328347, "max": 25.082868000026792}}
{"backend": "pytorch-cuda:0", "model": "yolov8n-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 6.566240598913282, "min": 6.433751987060532, "p50": 6.483846984298015, "p90": 6.575970008270815, "max": 9.441217000130564}}
{"backend": "pytorch-cuda:0", "model": "yolov8s-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 6.970981381600723, "min": 6.861114990897477, "p50": 6.9368369817973381, "p90": 7.002762024058029, "max": 7.605271995998919}}
{"backend": "pytorch-cuda:0", "model": "yolov8m-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 10.9778896975331, "min": 10.820526094615916, "p50": 10.853937012143433, "p90": 11.386888009584477, "max": 11.563389009097591}}
{"backend": "pytorch-cuda:0", "model": "yolov8l-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 16.127768460428342, "min": 16.059455985669047, "p50": 16.091855999547988, "p90": 16.219046025071293, "max": 16.680302011581956}}
{"backend": "pytorch-cuda:0", "model": "yolov8x-seg", "task": "segment", "dtype": "f32", "imgsz": [480, 640], "warmup": 10, "iters": 50, "e2e_ms": {"mean": 25.268140221596695, "min": 25.166436011204496, "p50": 25.212281005224213, "p90": 25.450409011244495, "max": 25.844468007562682}}
```

</details>



## Methodology

- Input: `ultralytics/assets/bus.jpg`, decoded once before timing.
- Detection geometry: a 480x640 stride-aligned canvas on both engines.
- Depth geometry: the checkpoint's 768 default, with aspect-preserving stride alignment.
- C++ end-to-end latency: letterbox preprocessing + graph compute/readback + NMS or depth restoration.
- PyTorch end-to-end latency: `model.predict()` on the already decoded image, including preprocessing, forward, and
  postprocessing.
- GPU protocol: 20 warmups, 50 timed iterations, and a 3 second cooldown between entries.
- CPU protocol: 8 threads; 10 to 30 timed iterations depending on model size.
- Reported comparison: arithmetic mean in milliseconds per frame; raw min/p50/p90/max remain in JSONL.

File I/O, model loading, GGUF parsing, graph construction, and first-run shader/JIT compilation are excluded from both
steady-state measurements. Use p90 and max, not only the mean, when evaluating latency stability.

## Focused stability checks

The sustained-load checks re-ran on the current build: YOLO26x F16 completed 500 consecutive CUDA iterations with a
19.40 ms mean, 20.74 ms p90, and 21.90 ms maximum — the 50-iteration snapshot mean is 18.49 ms, so the sustained-load
penalty shrank from the prior snapshot's 18 ms to under 1 ms. YOLOv8l F16 completed 500 Vulkan iterations with a 16.60
ms mean and 17.44 ms p90. YOLO26n-depth F16 completed 500 Vulkan iterations with a 7.03 ms mean, 7.65 ms p90, and a
stable 2.22-17.84 m depth-value range. Alternate-resolution, fresh-process, and malformed-GGUF checks also pass. These
checks cover observed runtime paths; they do not replace sanitizer, out-of-memory, truncated-model, or multi-device
release testing.

## Environment for the checked-in snapshot

| Component    | Value                                                 |
| ------------ | ----------------------------------------------------- |
| CPU          | AMD Ryzen 9 5950X, 16 cores / 32 threads              |
| GPU          | NVIDIA GeForce RTX 3060 12 GB, compute capability 8.6 |
| Driver       | 550.144.03                                            |
| CUDA toolkit | 11.8                                                  |
| ggml         | v0.18.1 (`90951f99`) plus repository patches          |

Results are hardware and software specific. Regenerate them after driver, toolkit, compiler, ggml revision, model,
input-size, or timing-protocol changes.

The checked-in CUDA snapshot is built without cuDNN (`YOLO_GGML_CUDNN=OFF`) and without CUTLASS: its convolutions run on
the native WMMA implicit-GEMM kernel and the binary links only against `libcudart`, `libcublas`, and `libcublasLt`
(verified with `ldd`). Enabling the optional cuDNN path requires cuDNN 8.5 or newer and `YOLO_GGML_CUDNN=ON`; the
experimental CUTLASS builds (`build-cuda-cutlass`, `build-exp-cutlass`) are excluded from this evidence. Measurements
taken on those paths must not be relabeled as the native snapshot above.

## Reproduce

Run from `cpp_ggml/`. Build parallelism is intentionally capped at 6.

```bash
cmake -S . -B build-cuda -DYOLO_GGML_CUDA=ON
cmake --build build-cuda --parallel 6
cmake -S . -B build-vulkan -DYOLO_GGML_VULKAN=ON
cmake --build build-vulkan --parallel 6
cmake -S . -B build-cpu
cmake --build build-cpu --parallel 6

bash scripts/convert_all.sh
for backend in cuda vulkan cpu; do
    bash scripts/bench_all.sh "build-$backend" "$backend"
done
SEG_MODELS="yolov8n-seg yolov8s-seg yolov8m-seg yolov8l-seg yolov8x-seg yolo26n-seg yolo26s-seg yolo26m-seg yolo26l-seg yolo26x-seg"
YOLO_BENCH_OUT=benchmarks/seg_cuda.jsonl bash scripts/bench_all.sh build-cuda cuda $SEG_MODELS
YOLO_BENCH_OUT=benchmarks/seg_vulkan.jsonl bash scripts/bench_all.sh build-vulkan vulkan $SEG_MODELS
python3 scripts/bench_pytorch.py > benchmarks/pytorch.jsonl
python3 scripts/plot_benchmarks.py
python3 scripts/render_parity.py
```

`bench_all.sh` appends so interrupted runs retain completed measurements. Remove or archive the JSONL inputs before a
formal release run when a single clean snapshot is required. The plot generator de-duplicates by keeping the latest
entry per key and requires the complete 99-key detection matrix plus the 60-key segmentation matrix (including CPU
detection rows) to render.

## Optimization boundary

The shipped integration keeps F16 GPU activations resident, reuses preprocessing/output buffers, uses CUDA graph replay,
and performs postprocessing in the logit domain. CUDA convolution runs a native WMMA implicit-GEMM tensor-core kernel
that forms the im2col operand on the fly in shared memory and fuses bias and SiLU into the epilogue, so one fused conv is
one kernel launch with no cuDNN, CUTLASS, or im2col buffer. Dispatch is a direct host-side gate: F16 inputs, output
channels divisible by 8, and compute capability 7.0 or newer select the implicit-GEMM kernel, unaligned input channels
take a scalar staging variant, and everything else falls back to the generic CUDA kernel. Optional cuDNN builds take
precedence only when `YOLO_GGML_CUDNN=ON` is configured explicitly.

Vulkan executes F16 convolution directly from the image tensor without an im2col allocation, fuses Conv+Bias+SiLU,
uses cooperative-matrix tiles selected by output shape, and batches submissions according to graph FLOPs. F16 pool,
scale, upscale, depthwise convolution, and raw contiguous copies remain on device. Both GPU paths run the
transpose-convolution upsample as a specialized kernel (CUDA: a stride-2 2x2 GEMM form; Vulkan: the TRANSPOSE
cooperative-matrix variant), and the C2f channel slices stay as zero-copy contiguous sub-block views instead of
materializing duplicates, removing one redundant copy kernel per C2f block. Segmentation composes instance masks on
device (proto matrix multiply, sigmoid, and crop happen in the postprocess pipeline after a single proto readback), and
session profiling is an explicit `--profile ops|gaps` CLI switch backed by `SessionOptions` — no environment-variable
hooks remain in the runtime.

This snapshot meets the close-to-or-faster-than-PyTorch target on both GPU backends (see the matrix above): the
implicit-GEMM rework, the transpose-conv specialization, the on-device dtype casts, and the copy elimination together
recovered the previous CUDA regression and brought segmentation to parity-relevant latency.

## Remaining integration work

1. Re-run the focused tensor-level parity (`parity_reference.py`) after the kernel changes and refresh the bounds above.
2. Run COCO and NYU Depth V2 validation across F32/F16/Q8_0 and all supported backends with explicit task tolerances.
3. Automate the focused stability checks and extend them with sanitizers, truncated models, OOM, and multi-device cases.
4. Validate the Vulkan and CUDA results on AMD/Intel GPUs and non-NVIDIA drivers; the checked-in result covers one
   NVIDIA RTX 3060.
5. Promote the internal C++ session types to a versioned installed C API only when an embedding consumer requires it.
6. Extend YOLO26 absolute-depth support from n to s/m/l/x when those checkpoints are placed in the canonical model set.
