#!/usr/bin/env bash
#
# bench_all.sh — collect the full performance matrix into benchmarks/bench.jsonl
#
# Runs yolo-cli bench for every (model x dtype) on the requested backend build
# directory and appends one JSON line per run. Closed-set lines go to
# bench.jsonl; World lines go to world.jsonl because their vocabulary is part
# of the measurement contract. Re-runs overwrite nothing: the plot script
# de-duplicates by (backend, model, dtype) keeping the last closed-set entry.
# The default sweep covers the 45 closed-set checkpoints (detect, segment,
# depth, pose, obb, semantic, classify x n/s/m/l/x, plus yolov8 detect/seg).
# Open-vocabulary World must be requested explicitly with a fixed vocabulary.
# YOLOE takes a checkpoint-agnostic YTXT0002 blob (pre-reprta MobileCLIP
# output; reprta runs inside the v4 graph) or plaintext --classes through the
# native MobileCLIP tower, and is therefore also explicit.
#
# Usage: scripts/bench_all.sh BUILD_DIR [BACKEND_TAG] [models...]
#   scripts/bench_all.sh build-cuda cuda
#   scripts/bench_all.sh build-cpu cpu yolo26n yolov8n   # subset
#
# GPU runs get warmup 20 / iters 50 plus a 3 s cool-down between entries:
# back-to-back sweeps otherwise ride GPU clock/power transients and report
# inflated means (observed yolo26n-f16 CUDA 17 ms in-sweep vs 10 ms solo).

set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="${1:?build dir required, e.g. build-cuda}"
TAG="${2:-$BUILD}"
shift 2 || true
if (( $# )); then
    MODELS=("$@")
else
    MODELS=(
        # yolov8 detection + segmentation
        yolov8n yolov8s yolov8m yolov8l yolov8x
        yolov8n-seg yolov8s-seg yolov8m-seg yolov8l-seg yolov8x-seg
        # yolo26 detection
        yolo26n yolo26s yolo26m yolo26l yolo26x
        # yolo26 segmentation
        yolo26n-seg yolo26s-seg yolo26m-seg yolo26l-seg yolo26x-seg
        # yolo26 depth
        yolo26n-depth yolo26s-depth yolo26m-depth yolo26l-depth yolo26x-depth
        # yolo26 pose
        yolo26n-pose yolo26s-pose yolo26m-pose yolo26l-pose yolo26x-pose
        # yolo26 obb
        yolo26n-obb yolo26s-obb yolo26m-obb yolo26l-obb yolo26x-obb
        # yolo26 semantic
        yolo26n-sem yolo26s-sem yolo26m-sem yolo26l-sem yolo26x-sem
        # yolo26 classify
        yolo26n-cls yolo26s-cls yolo26m-cls yolo26l-cls yolo26x-cls
    )
fi
read -r -a DTYPES <<< "${YOLO_BENCH_DTYPES:-f16 f32 q8_0}"
SRC="../ultralytics/assets/bus.jpg"

THREADS=0
GPU_ARGS=()
COOLDOWN=0
if [[ "$TAG" == "cpu" ]]; then THREADS=8; else GPU_ARGS=(--warmup 20 --iters 50); COOLDOWN=3; fi

mkdir -p benchmarks
OUT="${YOLO_BENCH_OUT:-benchmarks/bench.jsonl}"
WORLD_OUT="${YOLO_BENCH_WORLD_OUT:-benchmarks/world.jsonl}"
YOLOE_OUT="${YOLO_BENCH_YOLOE_OUT:-benchmarks/yoloe.jsonl}"
YOLOE_PF_OUT="${YOLO_BENCH_YOLOE_PF_OUT:-benchmarks/yoloe_pf.jsonl}"
WORLD_COUNT=0
YOLOE_COUNT=0
YOLOE_PF_COUNT=0
echo "collecting ${#MODELS[@]} models x ${#DTYPES[@]} dtypes on $TAG -> $OUT"

for m in "${MODELS[@]}"; do
    for dt in "${DTYPES[@]}"; do
        f="models/gguf/$m-$dt.gguf"
        [[ -f "$f" ]] || { echo "[fail] $f missing" >&2; exit 1; }
        args=(--model "$f" --source "$SRC" "${GPU_ARGS[@]}")
        if [[ "$m" == *-world ]]; then
            : "${YOLO_BENCH_WORLD_CLASSES:?set a fixed comma-separated World vocabulary}"
            args+=(--classes "$YOLO_BENCH_WORLD_CLASSES")
            if [[ -n "${YOLO_BENCH_WORLD_CLIP_MODEL:-}" ]]; then
                args+=(--clip-model "$YOLO_BENCH_WORLD_CLIP_MODEL")
            fi
            if [[ -n "${YOLO_BENCH_WORLD_TEXT_EMBED:-}" ]]; then
                args+=(--text-embed "$YOLO_BENCH_WORLD_TEXT_EMBED")
            fi
        elif [[ "$m" == *-pf ]]; then
            # Prompt-free YOLOE bakes its vocabulary into the checkpoint, so the bench
            # takes no --classes/--text-embed and pays no text setup at all.
            :
        elif [[ "$m" == yoloe-* ]]; then
            : "${YOLO_BENCH_YOLOE_CLASSES:?set a fixed comma-separated YOLOE vocabulary}"
            : "${YOLO_BENCH_YOLOE_TEXT_EMBED_DIR:?set the directory holding per-model YTXT0002 files}"
            ytxt="$YOLO_BENCH_YOLOE_TEXT_EMBED_DIR/$m.ytxt"
            [[ -f "$ytxt" ]] || { echo "[fail] missing YOLOE YTXT: $ytxt" >&2; exit 1; }
            args+=(--classes "$YOLO_BENCH_YOLOE_CLASSES" --text-embed "$ytxt")
        fi
        [[ "$THREADS" -gt 0 ]] && args+=(--threads "$THREADS")
        # CPU m/l/x are slow; trim iterations so the sweep stays tractable.
        if [[ "$TAG" == "cpu" && ("$m" == *m || "$m" == *l || "$m" == *x) ]]; then
            args+=(--iters 10 --warmup 3)
        elif [[ "$TAG" == "cpu" ]]; then
            args+=(--iters 30 --warmup 10)
        fi
        line=$(./"$BUILD"/bin/yolo-cli bench "${args[@]}" | tail -1)
        [[ "$line" == "{"* ]] || { echo "[fail] $m-$dt" >&2; exit 1; }
        # Reject silent GPU-to-CPU fallback, then normalize the verified backend tag for plots.
        line=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); tag=sys.argv[2]; actual=d['backend'].lower(); \
assert tag == 'cpu' and actual == 'cpu' or tag != 'cpu' and tag in actual, f'expected {tag}, got {actual}'; \
d['backend']=tag; print(json.dumps(d))" "$line" "$TAG")
        if [[ "$m" == *-world ]]; then
            echo "$line" >> "$WORLD_OUT"
            ((WORLD_COUNT += 1))
        elif [[ "$m" == *-pf ]]; then
            echo "$line" >> "$YOLOE_PF_OUT"
            ((YOLOE_PF_COUNT += 1))
        elif [[ "$m" == yoloe-* ]]; then
            echo "$line" >> "$YOLOE_OUT"
            ((YOLOE_COUNT += 1))
        else
            echo "$line" >> "$OUT"
        fi
        ms=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['e2e_ms']['mean'])" "$line")
        echo "[ok] $m-$dt e2e_ms=$ms"
        [[ "$COOLDOWN" -gt 0 ]] && sleep "$COOLDOWN"
    done
done
echo "done: $(wc -l < "$OUT") closed-set entries in $OUT"
if (( WORLD_COUNT )); then
    echo "      $(wc -l < "$WORLD_OUT") World entries in $WORLD_OUT"
fi
if (( YOLOE_COUNT )); then
    echo "      $(wc -l < "$YOLOE_OUT") YOLOE entries in $YOLOE_OUT"
fi
if (( YOLOE_PF_COUNT )); then
    echo "      $(wc -l < "$YOLOE_PF_OUT") prompt-free YOLOE entries in $YOLOE_PF_OUT"
fi
