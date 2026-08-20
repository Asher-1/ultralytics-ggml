# ggml vs PyTorch speedup (e2e mean, bus.jpg)

## Coverage

| Evidence | Expected | Present | Status |
|---|---:|---:|---|
| GGML backend/model/precision keys | 99 | 99 | complete |
| PyTorch CUDA model references | 11 | 11 | complete |
| GGML per-row latency fields | preprocess mean/p50/p90; graph mean/min/p50/p90/max; post mean/p50; e2e mean/min/p50/p90/max | all required fields | complete |
| PyTorch per-row latency fields | e2e mean/min/p50/p90/max | all required fields | complete |

The latency matrix is complete. Accuracy evidence is separate: focused F16 parity covers every model on the documented image, while dataset-level COCO and NYU Depth V2 validation is not yet established.

## Latency and speedup

| model | dtype | PyTorch | CUDA | speedup | Vulkan | speedup | CPU 8T | speedup |
|---|---|---|---|---|---|---|---|---|
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

## Segmentation latency and speedup

| model | dtype | PyTorch | CUDA | speedup | Vulkan | speedup |
|---|---|---|---|---|---|---|
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
