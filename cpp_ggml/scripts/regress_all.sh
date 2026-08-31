#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# One-shot regression gate: precision + speed for every model family and dtype
# on one backend build. Run after any engine change to catch behavioral or
# performance regressions before they land.
#
# Usage:
#   scripts/regress_all.sh BUILD_DIR BACKEND [--speed-only|--precision-only]
#                          [--dtypes "f16 f32 q8_0"] [MODELS...]
#
#   BUILD_DIR  build tree (build-cuda / build-vulkan / build-cpu)
#   BACKEND    tag bench_all.sh expects: cuda | vulkan | cpu
#   MODELS     optional subset (names without dtype, e.g. yolo26n yoloe-26s-seg);
#              default sweeps every GGUF in models/gguf (clip/mobileclip excluded)
#
# Precision layers by model family:
#   detect / segment / YOLOE / PF    --input-f32 + --dump-raw vs a torch
#                                    baseline (parity_reference.py prep+raw,
#                                    generated once and cached — torch is NOT
#                                    needed on later runs; falls back to a
#                                    self-recorded baseline when the .pt is
#                                    unavailable)
#   depth                            --raw vs the torch metric-depth baseline
#   pose / obb / semantic / classify --dump-ops optrace vs a cached first-run
#                                    baseline (self-regression: catches any
#                                    graph-structure or kernel-numerics drift)
#   YOLOE visual prompts             --dets-json vs a cached detection baseline
#                                    (bus.jpg + fixed boxes; exercises SAVPE)
#   world                            speed only (no tensor raw channel yet)
#
# Environment:
#   REGRESS_BASELINE_DIR   baseline cache (default benchmarks/parity_baseline)
#   REGRESS_REFRESH=1      re-record every baseline from the current build
#   REGRESS_SKIP_TORCH=1   never invoke torch (missing baseline => SKIP)
#   REGRESS_MAX_ABS_F32/F16/Q8, REGRESS_MAX_REL_F32/F16/Q8   PASS thresholds
#   YOLO_BENCH_*           forwarded to bench_all.sh (classes, ytxt dir, ...)
#
# Exit code 0 iff every measured case passes; speed entries append to
# benchmarks/regress/<backend>-speed.jsonl.

set -uo pipefail
cd "$(dirname "$0")/.." # cpp_ggml/

BUILD_DIR=$1; shift
TAG=$1; shift
MODE=speed+precision
DTYPES=${REGRESS_DTYPES:-f16 f32 q8_0}
MODELS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    --speed-only) MODE=speed; shift ;;
    --precision-only) MODE=precision; shift ;;
    --dtypes) DTYPES="$2"; shift 2 ;;
    *) MODELS+=("$1"); shift ;;
    esac
done

CLI="./$BUILD_DIR/bin/yolo-cli"
SRC=../ultralytics/assets/bus.jpg
BASE_DIR=${REGRESS_BASELINE_DIR:-benchmarks/parity_baseline}
REG_DIR=benchmarks/regress
YTXT_DIR=${YOLO_BENCH_YOLOE_TEXT_EMBED_DIR:-benchmarks/ytxt}
mkdir -p "$BASE_DIR" "$REG_DIR"

# Thresholds: |cpp - base| <= max_abs + max_rel * |base|, per element. The CUDA
# build runs every model on the F16 activation flow by design (see the
# unconditional input cast in yolo_graph.cpp), so f32 carries the same
# accumulated-rounding profile as f16 there and gets the same headroom.
max_abs() { case "$1" in f32) echo "${REGRESS_MAX_ABS_F32:-0.5}" ;; f16) echo "${REGRESS_MAX_ABS_F16:-0.5}" ;; *) echo "${REGRESS_MAX_ABS_Q8:-1.0}" ;; esac; }
max_rel() { case "$1" in f32) echo "${REGRESS_MAX_REL_F32:-0.05}" ;; f16) echo "${REGRESS_MAX_REL_F16:-0.05}" ;; *) echo "${REGRESS_MAX_REL_Q8:-0.10}" ;; esac; }

RESULTS=$(mktemp)
PASS=0; FAIL=0; SKIP=0

note() { printf '%s\n' "$*" | tee -a "$RESULTS"; }

# precision <model> <dtype> <gguf> — one model-dtype through its family channel.
precision_one() {
    local m=$1 dt=$2 f=$3
    local base="$BASE_DIR/$m-$dt"
    local ma mr sub ytxt pt
    ma=$(max_abs "$dt"); mr=$(max_rel "$dt")
    pt="models/pytorch/$m.pt"; [[ -f $pt ]] || pt="$m.pt" # ultralytics auto-downloads
    case "$m" in
    # world: speed-only channel for now — the torch raw reference needs a
    # set_classes flow (WorldModel.predict takes no tpe), unlike YOLOE's tpe
    # injection, and worldv2 folds the vocabulary into cv3 weights.
    *-world*) note "SKIP  $m-$dt  precision (world: speed-only channel)"; ((SKIP++)); return 0 ;;
    *-depth*) sub=depth ;;
    *-pose) sub=pose ;;
    *-obb) sub=obb ;;
    *-sem) sub=semantic ;;
    *-cls) sub=classify ;;
    *) sub=detect ;;
    esac
    mkdir -p "$base"

    if [[ $sub == detect ]]; then
        ytxt=""
        if [[ $m == yoloe-* && $m != *-pf ]]; then
            ytxt="$YTXT_DIR/$m.ytxt"
        fi
        if [[ ! -f $base/input.bin ]]; then
            python3 scripts/parity_reference.py prep "$pt" "$SRC" "$base/input.bin" >&2 || return 1
        fi
        if [[ ! -f $base/torch.bin && ! -f $base/self.bin ]]; then
            if [[ -n ${REGRESS_SKIP_TORCH:-} ]]; then
                note "SKIP  $m-$dt  precision (REGRESS_SKIP_TORCH, no baseline)"; ((SKIP++)); return 0
            fi
            if [[ -n $ytxt ]]; then
                python3 scripts/parity_reference.py raw "$pt" "$base/input.bin" "$base/torch.bin" "$ytxt" >&2
            else
                python3 scripts/parity_reference.py raw "$pt" "$base/input.bin" "$base/torch.bin" >&2
            fi
            if [[ ! -f $base/torch.bin ]]; then
                echo "[regress] torch baseline failed for $m — falling back to self baseline" >&2
            fi
        fi
        local ref="torch"
        [[ ! -f $base/torch.bin ]] && ref="self"
        local args=(detect --model "$f" --input-f32 "$base/input.bin" --dump-raw "$base/cur.bin")
        [[ -n $ytxt ]] && args+=(--classes "${YOLO_BENCH_YOLOE_CLASSES:-person,bus,car,truck}" --text-embed "$ytxt")
        if [[ $ref == self && ! -f $base/self.bin ]]; then
            # Record the self baseline from this run (assumed-good code state).
            local base_args=("${args[@]/$base\/cur.bin/$base/self.bin}")
            "$CLI" "${base_args[@]}" >/dev/null 2>&1 || { note "FAIL  $m-$dt  precision (self baseline run failed)"; ((FAIL++)); return 0; }
        fi
        "$CLI" "${args[@]}" >/dev/null 2>&1 || { note "FAIL  $m-$dt  precision (cli run failed)"; ((FAIL++)); return 0; }
        if python3 scripts/regress_compare.py raw "$base/$ref.bin" "$base/cur.bin" --max-abs "$ma" --max-rel "$mr" | sed "s/^/      [$m-$dt\/$ref] /" | tee /dev/stderr 2>/dev/null | grep -q PASS; then
            note "PASS  $m-$dt  precision ($ref)"; ((PASS++))
        else
            note "FAIL  $m-$dt  precision ($ref)"; ((FAIL++))
        fi

    elif [[ $sub == depth ]]; then
        if [[ ! -f $base/torch.bin && ! -f $base/self.bin ]]; then
            if [[ -n ${REGRESS_SKIP_TORCH:-} ]]; then
                note "SKIP  $m-$dt  precision (REGRESS_SKIP_TORCH, no baseline)"; ((SKIP++)); return 0
            fi
            python3 scripts/parity_reference.py depth "$pt" "$SRC" "$base/torch.bin" cuda:0 >&2 || true
        fi
        if [[ -f $base/torch.bin ]]; then
            "$CLI" depth --model "$f" --source "$SRC" --raw "$base/cur.bin" >/dev/null 2>&1 || { note "FAIL  $m-$dt  precision (cli run failed)"; ((FAIL++)); return 0; }
            if python3 scripts/regress_compare.py raw "$base/torch.bin" "$base/cur.bin" --max-abs "$ma" --max-rel "$mr" | sed "s/^/      [$m-$dt\/torch] /" | tee -a "$RESULTS" | tail -1 | grep -q PASS; then
                note "PASS  $m-$dt  precision (torch)"; ((PASS++))
            else
                note "FAIL  $m-$dt  precision (torch)"; ((FAIL++))
            fi
        else
            "$CLI" depth --model "$f" --source "$SRC" --raw "$base/self.bin" >/dev/null 2>&1 || { note "FAIL  $m-$dt  precision (self baseline run failed)"; ((FAIL++)); return 0; }
            "$CLI" depth --model "$f" --source "$SRC" --raw "$base/cur.bin" >/dev/null 2>&1
            if python3 scripts/regress_compare.py raw "$base/self.bin" "$base/cur.bin" --max-abs 0 --max-rel 0 | tail -1 | grep -q PASS; then
                note "PASS  $m-$dt  precision (self, bit-exact)"; ((PASS++))
            else
                note "FAIL  $m-$dt  precision (self, bit-exact)"; ((FAIL++))
            fi
        fi

    else # pose / obb / semantic / classify: optrace self-regression
        if [[ ! -d $base/opsbase || -n ${REGRESS_REFRESH:-} ]]; then
            rm -rf "$base/opsbase" "$base/opscur"
            "$CLI" "$sub" --model "$f" --source "$SRC" --dump-ops "$base/opsbase" >/dev/null 2>&1 || { note "FAIL  $m-$dt  precision (baseline run failed)"; ((FAIL++)); return 0; }
        fi
        rm -rf "$base/opscur"
        "$CLI" "$sub" --model "$f" --source "$SRC" --dump-ops "$base/opscur" >/dev/null 2>&1 || { note "FAIL  $m-$dt  precision (cli run failed)"; ((FAIL++)); return 0; }
        if python3 scripts/regress_compare.py ops "$base/opsbase" "$base/opscur" --max-abs "$ma" --max-rel "$mr" | sed "s/^/      [$m-$dt\/ops] /" | tee -a "$RESULTS" | tail -1 | grep -q PASS; then
            note "PASS  $m-$dt  precision (ops)"; ((PASS++))
        else
            note "FAIL  $m-$dt  precision (ops)"; ((FAIL++))
        fi
    fi
    return 0
}

# --- speed -------------------------------------------------------------------
if [[ $MODE != precision ]]; then
    echo "=== [1/2] speed sweep on $TAG (bench_all.sh) ==="
    export YOLO_BENCH_YOLOE_CLASSES="${YOLO_BENCH_YOLOE_CLASSES:-person,bus,car,truck}"
    export YOLO_BENCH_YOLOE_TEXT_EMBED_DIR="$YTXT_DIR"
    export YOLO_BENCH_WORLD_CLASSES="${YOLO_BENCH_WORLD_CLASSES:-person,bus,car}"
    export YOLO_BENCH_WORLD_TEXT_EMBED="${YOLO_BENCH_WORLD_TEXT_EMBED:-$YTXT_DIR/world.ytxt}"
    export YOLO_BENCH_OUT="$REG_DIR/$TAG-speed.jsonl"
    export YOLO_BENCH_WORLD_OUT="$REG_DIR/$TAG-speed-world.jsonl"
    export YOLO_BENCH_YOLOE_OUT="$REG_DIR/$TAG-speed-yoloe.jsonl"
    export YOLO_BENCH_YOLOE_PF_OUT="$REG_DIR/$TAG-speed-pf.jsonl"
    if bash scripts/bench_all.sh "$BUILD_DIR" "$TAG" ${MODELS[@]+"${MODELS[@]}"}; then
        note "PASS  speed sweep (entries appended to $YOLO_BENCH_OUT)"; ((PASS++))
    else
        note "FAIL  speed sweep (bench_all.sh failed — see its output above)"; ((FAIL++))
    fi
fi

# --- precision ---------------------------------------------------------------
if [[ $MODE != speed ]]; then
    echo "=== [2/2] precision gate on $TAG ==="
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        for f in models/gguf/*.gguf; do
            m=$(basename "$f" | sed 's/-f16$//; s/-f32$//; s/-q8_0$//')
            [[ $m == clip-* || $m == mobileclip* || $m == *.ref ]] && continue
            MODELS+=("$m")
        done
    fi
    for m in "${MODELS[@]}"; do
        for dt in $DTYPES; do
            f="models/gguf/$m-$dt.gguf"
            [[ -f $f ]] || { note "SKIP  $m-$dt  (missing $f)"; ((SKIP++)); continue; }
            precision_one "$m" "$dt" "$f"
        done
    done

    # YOLOE visual-prompt (SAVPE) case: self-recorded detection baseline.
    vp_f=""
    for cand in yoloe-26s-seg-f16 yoloe-26n-seg-f16 yoloe-26s-seg-f32; do
        [[ -f models/gguf/$cand.gguf ]] && { vp_f="models/gguf/$cand.gguf"; break; }
    done
    if [[ -n $vp_f ]]; then
        base="$BASE_DIR/yoloe-vp"
        mkdir -p "$base"
        VP_BOXES="60,420,240,900,10,230,800,740,660,380,800,580"
        if [[ ! -f $base/vp.json || -n ${REGRESS_REFRESH:-} ]]; then
            "$CLI" detect --model "$vp_f" --source "$SRC" --vp-boxes "$VP_BOXES" --conf 0.25 --dets-json "$base/vp.json" >/dev/null 2>&1
        fi
        "$CLI" detect --model "$vp_f" --source "$SRC" --vp-boxes "$VP_BOXES" --conf 0.25 --dets-json "$base/cur.json" >/dev/null 2>&1
        if python3 scripts/regress_compare.py dets "$base/vp.json" "$base/cur.json" | sed 's/^/      [savpe-vp] /' | tee -a "$RESULTS" | tail -1 | grep -q PASS; then
            note "PASS  yoloe-vp savpe precision"; ((PASS++))
        else
            note "FAIL  yoloe-vp savpe precision"; ((FAIL++))
        fi
    fi
fi

echo
echo "=== regress_all [$TAG] summary: $PASS passed, $FAIL failed, $SKIP skipped ==="
grep -E "^FAIL" "$RESULTS" || true
rm -f "$RESULTS"
[[ $FAIL -eq 0 ]]
