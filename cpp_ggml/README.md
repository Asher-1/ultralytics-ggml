# YOLO inference with ggml

`cpp_ggml` is a standalone C++17 inference runtime for YOLOv8 and YOLO26 detection plus YOLO26n absolute depth. It
loads metadata-driven GGUF graphs without Python or PyTorch at runtime. CPU, CUDA, and Vulkan are validated here;
Metal and HIP have build wiring but remain experimental until they pass the same benchmark and parity matrix.
Conversion supports F32, F16, and Q8_0 weights.

This document is the operational guide from a clean clone to reproduced benchmark charts. See the
[model card](models/MODEL_CARD.md) for every supported checkpoint and the canonical model layout, and the
[benchmark report](benchmarks/README.md) for the checked-in measurements, parity evidence, hardware environment, and
remaining validation boundary.

## Repository layout

```text
cpp_ggml/
├── CMakeLists.txt
├── src/                       # loader, graph builder, backends, preprocessing, postprocessing, CLI
├── examples/cli/              # yolo-cli target
├── models/
│   ├── MODEL_CARD.md
│   ├── pytorch/               # canonical .pt conversion inputs (ignored)
│   └── gguf/                  # generated GGUF runtime models (ignored)
├── benchmarks/                # report, raw JSONL, charts, and parity images
├── scripts/                   # conversion, benchmark, parity, and plot tools
└── third_party/
    ├── ggml/                  # pinned v0.18.1 submodule
    ├── cudnn-frontend/        # pinned cuDNN Graph API headers
    ├── cutlass/               # pinned v3.5.1 implicit-GEMM kernels
    └── ggml-patches/          # one idempotent integration patch against pinned ggml
```

All commands below start at the repository root unless a section explicitly runs `cd cpp_ggml`.

## 1. Prerequisites

Every build needs Git, a C++17 compiler, CMake 3.14 or newer, and a build tool supported by CMake. Model conversion and
report generation additionally need Python 3.8 or newer and the Python dependencies declared by this repository.

| Target         | Additional requirement                                                                        |
| -------------- | --------------------------------------------------------------------------------------------- |
| CPU            | No accelerator SDK; OpenMP is used when CMake finds it                                        |
| Optimized CUDA | NVIDIA driver, CUDA toolkit, cuDNN 8.5 or newer, and a GPU supported by the installed toolkit |
| Generic CUDA   | NVIDIA driver and CUDA toolkit; cuDNN and CUTLASS are disabled                                |
| Vulkan         | Vulkan loader, headers, shader compiler, and a Vulkan 1.2-capable driver                      |

Check the tools before configuring:

```bash
git --version
cmake --version
c++ --version
python3 --version

# CUDA only
nvcc --version
nvidia-smi

# Vulkan only; it must list the intended physical device.
vulkaninfo --summary
```

Do not use more than six concurrent compiler jobs. The CUDA and CUTLASS translation units can exhaust system memory at
higher parallelism, so every build command in this guide uses `--parallel 6`.

## 2. Clone and initialize

Clone with all pinned third-party revisions:

```bash
git clone --recurse-submodules https://github.com/Asher-1/ultralytics-ggml.git
cd ultralytics-ggml
git submodule status --recursive
```

SSH users can use `git@github.com:Asher-1/ultralytics-ggml.git` instead. If the repository was cloned without
`--recurse-submodules`, or a network interruption left an empty dependency directory, recover it with:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

Each `git submodule status` line should start with a space, not `-`. The first CMake configure applies the repository's
single integration patch to the worktree-local `cpp_ggml/third_party/ggml` checkout. That submodule then appears dirty
in `git status`; this is expected generated state, not a source change to commit.

Create a Python environment for conversion, PyTorch reference runs, and chart generation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The repository also supports `uv pip install -e .` in an already activated environment. Neither Python nor these
packages are needed after GGUF conversion when deploying only `yolo-cli` and its runtime libraries.

## 3. Build

Use a different build directory for every backend. CPU fallback is included in GPU builds, but CMake intentionally
rejects configurations that enable more than one GPU backend in the same directory.

### CPU

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_ggml/build-cpu --parallel 6
cpp_ggml/build-cpu/bin/yolo-cli
```

The final command prints CLI usage and exits nonzero because no subcommand was supplied. CPU builds use host-native
instructions by default. Add `-DGGML_NATIVE=OFF` when the resulting binary must run on older CPUs than the build host.

### Optimized CUDA

The performance path uses cuDNN Graph and CUTLASS by default:

```bash
export CUDNN_ROOT=/path/to/cudnn
cmake -S cpp_ggml -B cpp_ggml/build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DYOLO_GGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 6
```

`CUDNN_ROOT` may be omitted when `cudnn_version.h` is already on a system search path. If cuDNN libraries are in a
nonstandard directory, also expose that directory to the platform dynamic loader before running the binary. CMake
versions older than 3.24 query the attached NVIDIA GPU for its compute capability. On a headless builder or while
cross-compiling, set it explicitly, for example:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda \
    -DYOLO_GGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cpp_ggml/build-cuda --parallel 6
```

Use the capability of the deployment GPU rather than copying `86` unconditionally.

### Generic CUDA compatibility build

cuDNN Graph and CUTLASS are optional implementation choices, not requirements of GGML itself. Disable both when the
target only provides the CUDA toolkit or cuDNN is older than 8.5:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda-generic \
    -DCMAKE_BUILD_TYPE=Release \
    -DYOLO_GGML_CUDA=ON \
    -DYOLO_GGML_CUDNN=OFF \
    -DYOLO_GGML_CUTLASS=OFF
cmake --build cpp_ggml/build-cuda-generic --parallel 6
```

This path is functional but is not the configuration used for the checked-in all-model CUDA performance result.

### Vulkan

```bash
cmake -S cpp_ggml -B cpp_ggml/build-vulkan \
    -DCMAKE_BUILD_TYPE=Release \
    -DYOLO_GGML_VULKAN=ON
cmake --build cpp_ggml/build-vulkan --parallel 6
```

If the Vulkan SDK is outside the system search path, export its environment setup or pass
`-DCMAKE_PREFIX_PATH=/path/to/vulkan-sdk`. Verify the selected runtime device from `yolo-cli` output before recording a
benchmark; `scripts/bench_all.sh` rejects silent GPU-to-CPU fallback.

### Why optimized CUDA has two extra dependencies

No single convolution implementation wins for every YOLO tensor shape. The optimized backend measures supported plans
for the complete shape on first execution, caches the winner, and retains fallback coverage:

- `cudnn-frontend` is a header-only C++ graph and execution-plan builder over the cuDNN backend API. It enables fused
  NHWC Conv+Bias+SiLU plans; the process links the cuDNN runtime library.
- CUTLASS is a header-only CUDA template library. Its Ampere implicit-GEMM kernels compile into `ggml-cuda`, so there
  is no separate CUTLASS runtime library.
- CPU and Vulkan builds include or link neither dependency.

The checked-in 11/11 CUDA result was measured with cuDNN 9.1. Use the generic CUDA build on older installations rather
than downgrading the pinned frontend and treating the result as equivalent performance evidence.

## 4. Convert models

The only supported source checkpoint directory is `cpp_ggml/models/pytorch/`; generated runtime files go to
`cpp_ggml/models/gguf/`. Old model copies under `cpp_ggml/` or the repository root are not read. Supported missing
release checkpoints are downloaded into the canonical PyTorch directory by Ultralytics when conversion starts.

Convert all 10 detection models and YOLO26n-depth to F32, F16, and Q8_0, producing 33 GGUF files:

```bash
cd cpp_ggml
bash scripts/convert_all.sh
find models/gguf -maxdepth 1 -name '*.gguf' -printf '%f\n' | sort
cd ..
```

Convert only selected models or precisions:

```bash
# convert_all accepts a model subset but always emits all three precisions.
cd cpp_ggml
bash scripts/convert_all.sh yolov8n yolo26n yolo26n-depth

# The Python converter emits one requested precision.
python3 scripts/convert_yolo_to_gguf.py --model yolo26n --dtype f16
python3 scripts/convert_yolo_to_gguf.py --model yolo26n-depth --dtype f16

# An explicit checkpoint and output path are also accepted.
python3 scripts/convert_yolo_to_gguf.py \
    --model models/pytorch/yolov8n.pt \
    --dtype q8_0 \
    --output models/gguf/yolov8n-q8_0.gguf
cd ..
```

F32 is the numerical reference, F16 is the recommended GPU deployment format, and Q8_0 reduces model size but needs
dataset-level accuracy validation. Conversion fuses the local Ultralytics model and serializes a small operation
vocabulary plus task metadata into GGUF, so no Python code is needed during C++ inference.

Inspect a generated model before inference:

```bash
cpp_ggml/build-cpu/bin/yolo-cli info \
    --model cpp_ggml/models/gguf/yolo26n-f16.gguf
cpp_ggml/build-cpu/bin/yolo-cli info \
    --model cpp_ggml/models/gguf/yolo26n-depth-f16.gguf
```

## 5. Run inference

Replace `build-cuda` with `build-vulkan`, `build-cpu`, or `build-cuda-generic` to select a compiled runtime. The command
prints the backend actually used.

### Detection

```bash
mkdir -p cpp_ggml/results
cpp_ggml/build-cuda/bin/yolo-cli detect \
    --model cpp_ggml/models/gguf/yolo26n-f16.gguf \
    --source ultralytics/assets/bus.jpg \
    --out cpp_ggml/results/yolo26n-bus.png \
    --conf 0.25 \
    --iou 0.70 \
    --max-det 300
```

Use `--threads N` to control CPU threads. `--out` is optional; detections are always printed. Input images supported by
the bundled image loader include JPEG and PNG.

### YOLO26n absolute depth

```bash
cpp_ggml/build-cuda/bin/yolo-cli depth \
    --model cpp_ggml/models/gguf/yolo26n-depth-f16.gguf \
    --source ultralytics/assets/bus.jpg \
    --raw cpp_ggml/results/yolo26n-depth-bus.bin \
    --out cpp_ggml/results/yolo26n-depth-bus.png
```

The `--raw` file starts with the eight-byte `YDEP0001` magic, two little-endian int32 dimensions `(height, width)`,
then row-major float32 meters. The PNG applies a color map and is display-only. Monocular depth is useful as an
approximate spatial prior, but is sensitive to camera intrinsics, domain shift, reflective surfaces, low texture, and
occlusion boundaries. It must not be the sole sensor for safety-critical distance decisions.

### One-model latency

`bench` times preprocessing, graph execution and device readback, and task postprocessing separately, then writes one
JSON record on its final line:

```bash
cpp_ggml/build-cuda/bin/yolo-cli bench \
    --model cpp_ggml/models/gguf/yolo26n-f16.gguf \
    --source ultralytics/assets/bus.jpg \
    --warmup 20 \
    --iters 50
```

Model loading, graph construction, file I/O, and first-run shader or kernel compilation are outside the steady-state
timing. Use the reported p90 and maximum together with the mean when evaluating runtime stability.

## 6. Test correctness and stability

The GGML dependency's own test targets are disabled in this standalone build. Validation is owned by the runtime CLI
and parity scripts instead of an empty `ctest` invocation.

### Smoke test every compiled backend

After generating the F16 models, run one detection and one depth inference through each available build:

```bash
mkdir -p cpp_ggml/results
for build in build-cpu build-cuda build-vulkan; do
    cpp_ggml/$build/bin/yolo-cli detect \
        --model cpp_ggml/models/gguf/yolo26n-f16.gguf \
        --source ultralytics/assets/bus.jpg \
        --out cpp_ggml/results/yolo26n-$build.png
    cpp_ggml/$build/bin/yolo-cli depth \
        --model cpp_ggml/models/gguf/yolo26n-depth-f16.gguf \
        --source ultralytics/assets/bus.jpg \
        --raw cpp_ggml/results/yolo26n-depth-$build.bin
done
```

Remove unavailable build names from the loop. For CPU-only validation, pass `--threads 8`.

### Detection tensor parity

Use an identical PyTorch letterbox tensor and compare the raw pre-postprocess head. The F32 model is the clearest
debugging reference:

```bash
tmpdir=$(mktemp -d)
python3 cpp_ggml/scripts/parity_reference.py prep \
    cpp_ggml/models/pytorch/yolo26n.pt ultralytics/assets/bus.jpg "$tmpdir/input.bin"
python3 cpp_ggml/scripts/parity_reference.py raw \
    cpp_ggml/models/pytorch/yolo26n.pt "$tmpdir/input.bin" "$tmpdir/torch-raw.bin"
cpp_ggml/build-cuda/bin/yolo-cli detect \
    --model cpp_ggml/models/gguf/yolo26n-f32.gguf \
    --input-f32 "$tmpdir/input.bin" \
    --dump-raw "$tmpdir/ggml-raw.bin"
python3 cpp_ggml/scripts/parity_reference.py diff "$tmpdir/torch-raw.bin" "$tmpdir/ggml-raw.bin"
```

Repeat with the CUDA and Vulkan builds and with every model required by the release. A matching detection count alone
is not numerical parity; inspect the reported tensor shape, mean error, maximum error, and task-level output.

### Absolute-depth parity

Compare restored per-pixel meters, not colorized images:

```bash
tmpdir=$(mktemp -d)
python3 cpp_ggml/scripts/parity_reference.py depth \
    cpp_ggml/models/pytorch/yolo26n-depth.pt \
    ultralytics/assets/bus.jpg "$tmpdir/torch-depth.bin" cuda:0
cpp_ggml/build-cuda/bin/yolo-cli depth \
    --model cpp_ggml/models/gguf/yolo26n-depth-f32.gguf \
    --source ultralytics/assets/bus.jpg \
    --raw "$tmpdir/ggml-depth.bin"
python3 cpp_ggml/scripts/parity_reference.py ddiff "$tmpdir/torch-depth.bin" "$tmpdir/ggml-depth.bin"
```

The checked-in focused acceptance limits and measured errors are documented in
[benchmarks/README.md](benchmarks/README.md). Focused single-image parity does not replace COCO mAP or NYU Depth V2
validation.

### Stability run

Use a fresh process and a long steady-state run for release-sensitive models:

```bash
cpp_ggml/build-cuda/bin/yolo-cli bench \
    --model cpp_ggml/models/gguf/yolo26x-f16.gguf \
    --source ultralytics/assets/bus.jpg \
    --warmup 20 \
    --iters 500
```

Run the command at least twice to cover allocator and cached-plan reuse across process lifetimes. Also exercise the
largest model, depth, alternate image sizes, and the narrowest performance-margin model on deployment hardware.

## 7. Reproduce the complete benchmark comparison

The complete GGML matrix contains 99 unique keys:

```text
11 models x 3 backends (CUDA, Vulkan, CPU) x 3 precisions (F32, F16, Q8_0)
```

The PyTorch reference contains one CUDA row for each of the same 11 models. `bench_all.sh` appends records so an
interrupted sweep keeps completed work; `plot_benchmarks.py` de-duplicates by `(backend, model, dtype)`, keeping the
last row. For a formal comparison, collect into temporary files and replace the evidence only after every command
succeeds:

```bash
cd cpp_ggml

# Required once: three builds and all 33 GGUF files.
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 6
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DYOLO_GGML_CUDA=ON
cmake --build build-cuda --parallel 6
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DYOLO_GGML_VULKAN=ON
cmake --build build-vulkan --parallel 6
bash scripts/convert_all.sh

# Collect a clean GGML snapshot. GPU entries use 20 warmups, 50 iterations,
# and a 3 second cooldown. CPU uses 8 threads and size-dependent iteration counts.
: > benchmarks/bench.json.tmp
for backend in cuda vulkan cpu; do
    YOLO_BENCH_OUT=benchmarks/bench.json.tmp \
        bash scripts/bench_all.sh "build-$backend" "$backend"
done

# Collect the matching official PyTorch CUDA reference.
python3 scripts/bench_pytorch.py > benchmarks/pytorch.json.tmp

# Publish the snapshot atomically only after all four sweeps complete.
mv benchmarks/bench.json.tmp benchmarks/bench.jsonl
mv benchmarks/pytorch.json.tmp benchmarks/pytorch.jsonl

# Validate coverage and regenerate all latency charts and the comparison table.
python3 scripts/plot_benchmarks.py

# Run official PyTorch plus all three GGML builds and regenerate visual parity grids.
python3 scripts/render_parity.py
cd ..
```

The full run is intentionally sequential: concurrent compiler or inference jobs distort clocks, thermals, available
memory, and latency. Do not run it while another GPU workload is active. `plot_benchmarks.py` stops with a concrete
error if any of the 99 GGML keys, 11 PyTorch rows, or required latency statistics are missing.

To collect a quick subset without overwriting the checked-in full matrix:

```bash
cd cpp_ggml
YOLO_BENCH_OUT=/tmp/yolo-ggml-smoke.jsonl \
YOLO_BENCH_DTYPES='f16' \
    bash scripts/bench_all.sh build-cuda cuda yolov8n yolo26n yolo26n-depth
python3 scripts/bench_pytorch.py \
    --models yolov8n yolo26n yolo26n-depth \
    --warmup 5 \
    --iters 10 > /tmp/yolo-pytorch-smoke.jsonl
cd ..
```

The subset cannot be passed to `plot_benchmarks.py`, because that script deliberately requires complete release
coverage.

### Generated benchmark artifacts

| Artifact                                  | Meaning                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------- |
| `benchmarks/bench.jsonl`                  | Raw GGML preprocess, graph, postprocess, and end-to-end timing statistics |
| `benchmarks/pytorch.jsonl`                | Official PyTorch CUDA end-to-end reference                                |
| `benchmarks/speed_by_model.png`           | Model-family latency and scale comparison                                 |
| `benchmarks/latency_by_backend.png`       | F16 detection latency by backend                                          |
| `benchmarks/speed_by_dtype.png`           | Precision comparison for n/s deployment models                            |
| `benchmarks/depth_latency.png`            | YOLO26n-depth backend and precision latency                               |
| `benchmarks/latency_matrix.png`           | All 11 models, three GGML backends, and three precisions                  |
| `benchmarks/speedup_table.md`             | Complete latency and PyTorch speedup table                                |
| `benchmarks/parity_grid_{bus,zidane}.png` | PyTorch/CUDA/Vulkan/CPU detection visual comparison                       |
| `benchmarks/depth_parity_bus.png`         | PyTorch/CUDA/Vulkan/CPU restored metric-depth visual comparison           |

If build directories have nondefault names, set `YOLO_CUDA_BUILD`, `YOLO_VULKAN_BUILD`, and `YOLO_CPU_BUILD` before
running `render_parity.py`.

## 8. Interpret the checked-in result

On the measured RTX 3060, F16 GGML CUDA is faster than PyTorch CUDA on all 11 models (x1.04-x2.01), and Vulkan reaches
the declared x0.70 proximity floor on all 11 (x0.70-x1.32). These are measurements for the documented hardware,
software, input, and protocol, not portable guarantees. Re-run the complete matrix after changing the GPU, driver,
toolkit, cuDNN, compiler, GGML revision, model, input shape, or timing protocol.

Focused F16 numerical parity covers every integrated model. F32/Q8_0 and production accuracy claims still require
tolerance-based COCO and NYU Depth V2 validation. Performance and accuracy gates should therefore be evaluated from
fresh raw JSONL and dataset metrics on the actual deployment system, not inferred from a chart alone.

## 9. GGML patch replay contract

`third_party/ggml-patches/` contains exactly one patch. It is the complete diff from the pinned GGML revision, and
`scripts/apply_ggml_patches.sh` rejects zero or multiple patch files. The seven predecessor patches and the
consolidated patch were independently replayed from the same clean base with identical results:

| Evidence                      | Value                                                              |
| ----------------------------- | ------------------------------------------------------------------ |
| GGML base                     | `90951f99af1fbebef3fbdd58ff5b8715b0bb9c43`                         |
| old seven-patch result tree   | `7b0acb29df2ddb5372f9c8f8ee99564a00629eec`                         |
| new single-patch result tree  | `7b0acb29df2ddb5372f9c8f8ee99564a00629eec`                         |
| normalized diff SHA-256, both | `a153d601095b02012887d772308ae331c21a36650fbf7f8a2da1158066d15916` |

Matching Git trees prove equality of every resulting path, mode, and blob. The apply script is idempotent and
invalidates its worktree-local stamp when either the patch or a patched file changes. Do not edit generated build-tree
copies of GGML; update the source patch and replay it from the pinned clean submodule revision.

## 10. Troubleshooting

| Symptom                                            | Check or action                                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `ggml submodule not found`                         | Run `git submodule sync --recursive` and `git submodule update --init --recursive`                 |
| Patch cannot apply forward or in reverse           | Inspect `git -C cpp_ggml/third_party/ggml status`; use a clean worktree if it contains local edits |
| CMake reports cuDNN older than 8.5                 | Install a supported cuDNN or use the generic CUDA build                                            |
| CMake cannot find `cudnn_version.h`                | Set `CUDNN_ROOT` to the cuDNN installation prefix                                                  |
| CUDA compile is killed or the host runs out of RAM | Delete no files; rerun the build with `cmake --build <dir> --parallel 6` or fewer jobs             |
| CUDA compilation targets the wrong GPU             | Set `CMAKE_CUDA_ARCHITECTURES` to the deployment compute capability                                |
| Vulkan configure cannot find its SDK               | Load the SDK environment or set `CMAKE_PREFIX_PATH`; verify with `vulkaninfo --summary`            |
| `models/gguf/<name>.gguf` is missing               | Activate the Python environment and run the converter from the canonical model layout              |
| Benchmark reports an unexpected backend            | Do not accept the row; verify the driver/build directory because the script rejects CPU fallback   |
| Plot generation reports missing keys               | Finish all 99 GGML entries and all 11 PyTorch entries; a subset is intentionally insufficient      |

Do not clean a dirty GGML submodule blindly: it may contain an in-progress patch investigation. A new Git worktree is
the safest way to obtain an independent clean submodule checkout.

## Architecture and extension points

- `convert_yolo_to_gguf.py` owns PyTorch-module-to-operation lowering and GGUF metadata.
- `gguf_loader.cpp` owns format validation; graph format version 2 adds depth operations while retaining version 1
  detection models.
- `yolo_graph.cpp` owns task-independent graph construction and task-specific output reads.
- `backend.cpp` owns CPU plus one optional GPU scheduler and CPU fallback.
- The single GGML integration patch owns cuDNN Graph/CUTLASS shape selection and Vulkan direct convolution.
- `image_io.cpp` and `postprocess.cpp` own preprocessing and task output restoration.
- `cli.cpp` is a thin command adapter for `detect`, `depth`, `info`, and `bench`.

To add an architecture, extend converter lowering only for genuinely new modules and implement the matching generic op
in the graph builder. To add a task, keep output decoding and source-size restoration in the task owner rather than
forking model execution. The current C++ session types are internal; add an installed, versioned public API only when
an embedding consumer defines that contract.
