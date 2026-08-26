#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Convert cached closed-set checkpoints to GGUF. Set YOLO_CONVERT_YOLOE=1 to
# add every officially published text-conditioned YOLOE v8/11/26 segment
# checkpoint after placing its source .pt files in models/pytorch/. Any local
# YOLOE detect checkpoint can still be passed as an explicit argument.
# Usage: scripts/convert_all.sh [models...]
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# )); then
    MODELS=("$@")
else
    MODELS=(yolov8n yolov8s yolov8m yolov8l yolov8x yolo26n yolo26s yolo26m yolo26l yolo26x \
        yolov8n-seg yolov8s-seg yolov8m-seg yolov8l-seg yolov8x-seg \
        yolo26n-seg yolo26s-seg yolo26m-seg yolo26l-seg yolo26x-seg \
        yolo26n-depth yolo26s-depth yolo26m-depth yolo26l-depth yolo26x-depth \
        yolo26n-pose yolo26s-pose yolo26m-pose yolo26l-pose yolo26x-pose \
        yolo26n-obb yolo26s-obb yolo26m-obb yolo26l-obb yolo26x-obb \
        yolo26n-sem yolo26s-sem yolo26m-sem yolo26l-sem yolo26x-sem \
        yolo26n-cls yolo26s-cls yolo26m-cls yolo26l-cls yolo26x-cls)
    if [[ "${YOLO_CONVERT_YOLOE:-0}" == "1" ]]; then
        for family in yoloe-v8 yoloe-11; do
            for scale in s m l; do
                MODELS+=("${family}${scale}-seg")
            done
        done
        for scale in n s m l x; do
            MODELS+=("yoloe-26${scale}-seg")
        done
    fi
fi
DTYPES=(f32 f16 q8_0)

for m in "${MODELS[@]}"; do
    for dt in "${DTYPES[@]}"; do
        out="models/gguf/$m-$dt.gguf"
        if [[ -f "$out" ]]; then
            echo "[skip] $out exists"
            continue
        fi
        python3 scripts/convert_yolo_to_gguf.py --model "$m" --dtype "$dt" --output "$out"
    done
done
echo "done: $(ls models/gguf/*.gguf | wc -l) gguf files in models/gguf/"
