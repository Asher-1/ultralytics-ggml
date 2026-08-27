# End-to-end solution scenarios (C++)

Each program in this directory pairs one YOLO26 task family with an official
[Ultralytics Solutions](https://docs.ultralytics.com/solutions/) scenario and
runs it end to end on the ggml C++ stack (CPU / CUDA / Vulkan, any GGUF
precision — the model file selects all of that). They share the inference
plumbing in `examples_common.{hpp,cpp}` and only implement the scenario logic.

| Program | Task model | Official scenario | Output |
| --- | --- | --- | --- |
| `yolo-region-count` | detect | [Region Counting](https://docs.ultralytics.com/solutions/region-counting) | per-ROI object counts + annotated PNG |
| `yolo-crop` | segment | [Object Cropping](https://docs.ultralytics.com/solutions/object-cropping) | transparent-background instance PNGs |
| `yolo-workout` | pose | [Workout Monitoring](https://docs.ultralytics.com/solutions/workout-monitoring) | joint angles + pose stage + skeleton PNG |
| `yolo-obb` | obb | oriented detection (DOTA-style) | rotated boxes + CSV export + PNG |
| `yolo-city` | semantic | Cityscapes semantic segmentation | per-class area histogram + class map PNG |
| `yolo-topk` | classify | image classification | top-k classes with probability bars |
| `yolo-distance` | depth | [Distance Calculation](https://docs.ultralytics.com/solutions/distance-calculation) | metric distances (center / point / 3x3 grid) + depth PNG |

## Build

```bash
# from cpp_ggml/ (see the top-level README for backend flags)
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release          # or -DYOLO_GGML_CUDA=ON / -DYOLO_GGML_VULKAN=ON
cmake --build build-cpu -j --target yolo-region-count yolo-crop yolo-workout yolo-obb yolo-city yolo-topk yolo-distance
```

## Usage

All programs take `--model` (a GGUF file from `models/gguf/`) and `--source`
(any jpg/png/bmp), plus optional `--conf` and `--threads`.

```bash
# Region counting: two half-image ROIs on bus.jpg
./build-cpu/bin/yolo-region-count --model models/gguf/yolo26n-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --roi 0,0,405,540 --roi 405,0,810,540

# Object cropping: every instance as a transparent PNG under crops/
./build-cpu/bin/yolo-crop --model models/gguf/yolo26n-seg-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --dir crops

# Workout monitoring: elbow/knee angles for every person
./build-cpu/bin/yolo-workout --model models/gguf/yolo26n-pose-f16.gguf \
    --source ../../ultralytics/assets/zidane.jpg --out pose.png

# Oriented boxes with CSV export
./build-cpu/bin/yolo-obb --model models/gguf/yolo26n-obb-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --csv obb.csv --out obb.png

# Cityscapes-style semantic segmentation
./build-cpu/bin/yolo-city --model models/gguf/yolo26n-sem-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --out city.png

# Top-k classification
./build-cpu/bin/yolo-topk --model models/gguf/yolo26n-cls-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --topk 5

# Distance calculation (center point + 3x3 grid means, meters)
./build-cpu/bin/yolo-distance --model models/gguf/yolo26n-depth-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --px 405 --py 300 --out depth.png
```

Any task model can be swapped for any other scale or precision: the same
binary reads `yolo26s-pose-q8_0.gguf`, `yolo26l-depth-f32.gguf`, and so on.
GPU backends are selected by the build directory, not by the program.

## World and YOLOE with CLIP

The `yolo-cli detect` command runs CLIP text encoding before YOLO inference.
The sessions may select the same physical CUDA device, but keep independent
graph allocators and schedulers; an allocator is graph-owned, not a process
singleton. In the embedded World path CLIP stays on CPU so a Vulkan backend is
never initialized or destroyed while the YOLO session owns the GPU:

```bash
./build-cpu/bin/yolo-cli detect \
    --model models/gguf/yolov8s-world-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg \
    --classes "bus,person,car" \
    --clip-model models/gguf/clip-ViT-B-32.gguf \
    --out world.png
```

YOLOE-26 uses the same `[512, nc]` runtime shape after MobileCLIP text
encoding, its detector-specific `reprta` transform, and L2 normalization.
Convert a `yoloe-26*.pt` checkpoint with
`scripts/convert_yolo_to_gguf.py`, encode the vocabulary, then provide the
`YTXT0001` blob via `--text-embed`:

```bash
python scripts/encode_mobileclip_text.py --detector models/pytorch/yoloe-26n-seg.pt \
    --classes "bus,person,car" --output classes.ytxt
./build-cpu/bin/yolo-cli detect --model models/gguf/yoloe-26n-seg-f16.gguf \
    --source ../../ultralytics/assets/bus.jpg --text-embed classes.ytxt --out yoloe.png
```

YOLOE deliberately requires `--text-embed`; `--classes` alone is only valid
for YOLO-World's ViT-B/32 CLIP encoder. A YTXT belongs to the exact YOLOE
checkpoint that generated it.
