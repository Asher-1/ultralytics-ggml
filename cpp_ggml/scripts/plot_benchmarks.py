#!/usr/bin/env python3
"""Render the speed comparison charts for the ggml integration report.

Inputs:  benchmarks/bench.jsonl (yolo-cli bench lines, any backend/task)
         benchmarks/pytorch.jsonl (PyTorch reference lines, all tasks)
         benchmarks/world.jsonl (fixed-vocabulary YOLO-World lines)
         benchmarks/yoloe.jsonl (fixed-vocabulary YOLOE lines)
Output:  benchmarks/speed_by_model.png, latency_by_backend.png,
         speed_by_dtype.png, depth_latency.png, latency_matrix.png,
         seg_latency.png, task_latency.png, world_latency.png, yoloe_latency.png, and speedup_table.md

De-duplicates bench.jsonl by (backend, model, dtype), keeping the last run,
then rejects incomplete model/backend/precision or latency-statistic coverage.
The seven closed-set task families (detect, segment, depth, pose, obb,
semantic, classify) are validated and charted. Open-vocabulary World has a
vocabulary-dependent contract and is reported separately, never folded into
this closed-set matrix.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RES = Path(__file__).resolve().parent.parent / "benchmarks"
YOLOV8_DETECT = [f"yolov8{n}" for n in "nsmlx"]
YOLO26_DETECT = [f"yolo26{n}" for n in "nsmlx"]
DETECT_MODELS = YOLOV8_DETECT + YOLO26_DETECT
SEG_MODELS = [f"yolov8{n}-seg" for n in "nsmlx"] + [f"yolo26{n}-seg" for n in "nsmlx"]
DEPTH_MODELS = [f"yolo26{n}-depth" for n in "nsmlx"]
POSE_MODELS = [f"yolo26{n}-pose" for n in "nsmlx"]
OBB_MODELS = [f"yolo26{n}-obb" for n in "nsmlx"]
SEM_MODELS = [f"yolo26{n}-sem" for n in "nsmlx"]
CLS_MODELS = [f"yolo26{n}-cls" for n in "nsmlx"]
WORLD_MODELS = [f"yolov8{n}-world" for n in "smlx"]
YOLOE_MODELS = ["yoloe-26n-seg"]
FAMILIES = {
    "detect": DETECT_MODELS,
    "segment": SEG_MODELS,
    "depth": DEPTH_MODELS,
    "pose": POSE_MODELS,
    "obb": OBB_MODELS,
    "semantic": SEM_MODELS,
    "classify": CLS_MODELS,
}
MODELS = DETECT_MODELS + SEG_MODELS + DEPTH_MODELS + POSE_MODELS + OBB_MODELS + SEM_MODELS + CLS_MODELS
BACKENDS = ["cuda", "vulkan", "cpu"]
DTYPES = ["f32", "f16", "q8_0"]
WORLD_PROTOCOL = {"classes": "person,bus,car,truck", "class_count": 4, "text_source": "ytxt"}
SERIES = [  # (label, backend tag, color)
    ("PyTorch CUDA (f32)", "pytorch-cuda", "#d62728"),
    ("ggml CUDA", "cuda", "#1f77b4"),
    ("ggml Vulkan", "vulkan", "#2ca02c"),
    ("ggml CPU 8T", "cpu", "#7f7f7f"),
]
GPU_ONLY_SERIES = [series for series in SERIES if series[1] != "cpu"]


def present_series(bench):
    """Return only backend series represented in the current benchmark snapshot."""
    tags = {key[0] for key in bench}
    return [series for series in SERIES if series[1] == "pytorch-cuda" or series[1] in tags]


def plot_values(values):
    """Use NaN for missing samples so charts never imply a measured zero latency."""
    return [value if value is not None else math.nan for value in values]


def load(path, key):
    """Load the latest structured record for every requested key."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d.get("e2e_ms", {}).get("mean"):
            out[key(d)] = d
    return out


def require_metrics(record, sections):
    """Return missing nested latency fields for one benchmark record."""
    return [
        f"{section}.{field}"
        for section, fields in sections.items()
        for field in fields
        if field not in record.get(section, {})
    ]


def validate_coverage(bench_records, torch_records):
    """Fail before rendering when the checked-in evidence matrix is incomplete.

    Open-vocabulary World is deliberately absent from this matrix. Its rows
    need a declared vocabulary and are collected with an explicit invocation
    of bench_all.sh rather than being mixed with closed-set timings.
    """
    expected_ggml = {(backend, model, dtype) for backend in BACKENDS for model in MODELS for dtype in DTYPES}
    missing_ggml = sorted(expected_ggml - set(bench_records))
    missing_torch = sorted({(model,) for model in MODELS} - set(torch_records))
    ggml_sections = {
        "preprocess_ms": ("mean", "p50", "p90"),
        "graph_ms": ("mean", "min", "p50", "p90", "max"),
        "post_ms": ("mean", "p50"),
        "e2e_ms": ("mean", "min", "p50", "p90", "max"),
    }
    torch_sections = {"e2e_ms": ("mean", "min", "p50", "p90", "max")}
    incomplete = []
    for key in sorted(expected_ggml & set(bench_records)):
        missing = require_metrics(bench_records[key], ggml_sections)
        if missing:
            incomplete.append(f"ggml {key}: {', '.join(missing)}")
    for key in sorted({(model,) for model in MODELS} & set(torch_records)):
        missing = require_metrics(torch_records[key], torch_sections)
        if missing:
            incomplete.append(f"pytorch {key}: {', '.join(missing)}")
    problems = []
    if missing_ggml:
        problems.append(f"missing ggml keys ({len(missing_ggml)}): {missing_ggml}")
    if missing_torch:
        problems.append(f"missing PyTorch keys ({len(missing_torch)}): {missing_torch}")
    problems.extend(incomplete)
    if problems:
        raise RuntimeError("incomplete benchmark evidence:\n" + "\n".join(problems))


def validate_world_coverage(world_records):
    """Require comparable World rows rather than silently accepting legacy timings."""
    expected = {(backend, model) for backend in BACKENDS for model in WORLD_MODELS}
    missing = sorted(expected - set(world_records))
    unexpected = sorted(set(world_records) - expected)
    incomplete = []
    sections = {
        "preprocess_ms": ("mean", "p50", "p90"),
        "graph_ms": ("mean", "min", "p50", "p90", "max"),
        "post_ms": ("mean", "p50"),
        "e2e_ms": ("mean", "min", "p50", "p90", "max"),
    }
    for key in sorted(expected & set(world_records)):
        record = world_records[key]
        fields = require_metrics(record, sections)
        if fields:
            incomplete.append(f"world {key}: {', '.join(fields)}")
        if record.get("dtype") != "f32" or record.get("imgsz") != [480, 640]:
            incomplete.append(f"world {key}: expected f32 at 480x640")
        if record.get("world") != WORLD_PROTOCOL:
            incomplete.append(f"world {key}: fixed vocabulary/text protocol missing")
    if missing or unexpected or incomplete:
        problems = []
        if missing:
            problems.append(f"missing World keys ({len(missing)}): {missing}")
        if unexpected:
            problems.append(f"unexpected World keys ({len(unexpected)}): {unexpected}")
        problems.extend(incomplete)
        raise RuntimeError("incomplete World benchmark evidence:\n" + "\n".join(problems))


def validate_yoloe_coverage(yoloe_records):
    """Require the checkpoint-specific YTXT contract for every YOLOE backend row."""
    expected = {(backend, model) for backend in BACKENDS for model in YOLOE_MODELS}
    missing = sorted(expected - set(yoloe_records))
    unexpected = sorted(set(yoloe_records) - expected)
    problems = []
    for key in sorted(expected & set(yoloe_records)):
        record = yoloe_records[key]
        fields = require_metrics(record, {"e2e_ms": ("mean",), "graph_ms": ("mean",)})
        if fields:
            problems.append(f"YOLOE {key}: {', '.join(fields)}")
        if record.get("dtype") != "f32" or record.get("imgsz") != [480, 640]:
            problems.append(f"YOLOE {key}: expected f32 at 480x640")
        if record.get("world") != WORLD_PROTOCOL:
            problems.append(f"YOLOE {key}: fixed post-reprta YTXT protocol missing")
    if missing:
        problems.append(f"missing YOLOE keys ({len(missing)}): {missing}")
    if unexpected:
        problems.append(f"unexpected YOLOE keys ({len(unexpected)}): {unexpected}")
    if problems:
        raise RuntimeError("incomplete YOLOE benchmark evidence:\n" + "\n".join(problems))


def family_bars(ax, models, bench, torchref, series, title, label_fn, value_dtype="f16"):
    """Grouped bar chart: PyTorch reference plus one ggml value per series."""
    x = range(len(models))
    width = 0.8 / len(series)
    center = (len(series) - 1) / 2
    for i, (label, tag, color) in enumerate(series):
        vals = []
        for m in models:
            if tag == "pytorch-cuda":
                vals.append(torchref.get(m))
            else:
                if value_dtype:
                    dt = (
                        value_dtype
                        if (tag, m, value_dtype) in bench
                        else next((d for d in DTYPES if (tag, m, d) in bench), None)
                    )
                else:
                    dt = (
                        "f16"
                        if (tag, m, "f16") in bench
                        else next((d for d in ("f32", "q8_0") if (tag, m, d) in bench), None)
                    )
                vals.append(bench.get((tag, m, dt)))
        bars = ax.bar([xi + (i - center) * width for xi in x], plot_values(vals), width, label=label, color=color)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label_fn(m) for m in models])
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("ms / frame (log scale)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main():
    bench_records = load(RES / "bench.jsonl", lambda d: (d["backend"], d["model"], d["dtype"]))
    torch_records = load(RES / "pytorch.jsonl", lambda d: (d["model"],))
    world_records = load(RES / "world.jsonl", lambda d: (d["backend"], d["model"]))
    yoloe_records = load(RES / "yoloe.jsonl", lambda d: (d["backend"], d["model"]))
    validate_coverage(bench_records, torch_records)
    validate_world_coverage(world_records)
    validate_yoloe_coverage(yoloe_records)
    bench = {key: record["e2e_ms"]["mean"] for key, record in bench_records.items()}
    torchref = {model: record["e2e_ms"]["mean"] for (model,), record in torch_records.items()}
    series = present_series(bench)

    # ---- Chart 1: e2e by model scale, detect family ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, family in zip(axes, ["yolov8", "yolo26"]):
        models = [m for m in DETECT_MODELS if m.startswith(family)]
        family_bars(ax, models, bench, torchref, series, f"{family} — end-to-end latency on bus.jpg (lower is better)",
                    lambda m, fam=family: m.replace(fam, ""))
    fig.tight_layout()
    fig.savefig(RES / "speed_by_model.png", dpi=150)

    # ---- Chart 2: direct latency comparison across all detection models ----
    fig, ax = plt.subplots(figsize=(15, 6))
    x = range(len(DETECT_MODELS))
    width = 0.8 / len(series)
    center = (len(series) - 1) / 2
    for i, (label, tag, color) in enumerate(series):
        vals = []
        for model in DETECT_MODELS:
            if tag == "pytorch-cuda":
                vals.append(torchref.get(model))
            else:
                vals.append(bench.get((tag, model, "f16")))
        ax.bar([p + (i - center) * width for p in x], plot_values(vals), width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(DETECT_MODELS, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_title("End-to-end latency by backend (ggml F16, lower is better)")
    ax.set_ylabel("ms / frame (log scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "latency_by_backend.png", dpi=150)

    # ---- Chart 3: dtype sweep at n/s scale, CUDA + Vulkan ----
    small = ["yolov8n", "yolov8s", "yolo26n", "yolo26s"]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.13
    colors = {"f32": "#aec7e8", "f16": "#1f77b4", "q8_0": "#ff9896", "pytorch": "#d62728"}
    x = range(len(small))
    col_i = 0
    vals = [torchref.get(m) for m in small]
    bars = ax.bar(
        [xi - 3.5 * width for xi in x], plot_values(vals), width, label="PyTorch CUDA f32", color=colors["pytorch"]
    )
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7)
    for tag, taglabel in [("cuda", "CUDA"), ("vulkan", "Vulkan")]:
        for j, dt in enumerate(DTYPES):
            vals = [bench.get((tag, m, dt)) for m in small]
            pos = [xi + (col_i - 1.5) * width for xi in x]
            bars = ax.bar(
                pos,
                plot_values(vals),
                width,
                label=f"ggml {taglabel} {dt}",
                color=colors[dt],
                alpha=0.85 if tag == "cuda" else 0.5,
            )
            for b, v in zip(bars, vals):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=6)
            col_i += 1
    ax.set_xticks(list(x))
    ax.set_xticklabels(small)
    ax.set_title("dtype sweep — end-to-end latency (lower is better)")
    ax.set_ylabel("ms / frame")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "speed_by_dtype.png", dpi=150)

    # ---- Chart 4: depth family backend and dtype latency ----
    depth_rows = []
    for tag, color in (("cuda", "#1f77b4"), ("vulkan", "#2ca02c"), ("cpu", "#7f7f7f")):
        for model in DEPTH_MODELS:
            for dt in DTYPES:
                value = bench.get((tag, model, dt))
                if value is not None:
                    depth_rows.append((f"{model} {tag} {dt}", value, color))
    fig, ax = plt.subplots(figsize=(16, 5.5))
    labels, vals, bar_colors = zip(*depth_rows)
    bars = ax.bar(range(len(labels)), vals, color=bar_colors)
    for bar, value in zip(bars, vals):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("YOLO26 depth family end-to-end latency (768 input)")
    ax.set_ylabel("ms / frame")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "depth_latency.png", dpi=150)

    # ---- Chart 5: complete 45-model x 3-backend x 3-precision matrix ----
    columns = [(backend, dtype) for backend in BACKENDS for dtype in DTYPES]
    matrix = [[bench[(backend, model, dtype)] for backend, dtype in columns] for model in MODELS]
    values = [value for row in matrix for value in row]
    norm = LogNorm(vmin=min(values), vmax=max(values))
    fig, ax = plt.subplots(figsize=(16, 22))
    image = ax.imshow(matrix, cmap="viridis", norm=norm, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{backend}\n{dtype}" for backend, dtype in columns])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=7)
    ax.set_title("Complete GGML end-to-end latency matrix (ms, lower is better)")
    for row, values_by_column in enumerate(matrix):
        for column, value in enumerate(values_by_column):
            color = "white" if norm(value) < 0.58 else "black"
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=color, fontsize=6)
    fig.colorbar(image, ax=ax, label="ms / frame (log scale)")
    fig.tight_layout()
    fig.savefig(RES / "latency_matrix.png", dpi=150)

    # ---- Chart 6: segment e2e by model scale (cuda/vulkan vs PyTorch) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, family in zip(axes, ["yolov8", "yolo26"]):
        models = [m for m in SEG_MODELS if m.startswith(family)]
        family_bars(ax, models, bench, torchref, GPU_ONLY_SERIES,
                    f"{family}-seg — end-to-end latency on bus.jpg (lower is better)",
                    lambda m, fam=family: m.replace(fam + "-seg", ""))
    fig.tight_layout()
    fig.savefig(RES / "seg_latency.png", dpi=150)

    # ---- Chart 7: every closed-set task family ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    chart7 = [
        ("detect", DETECT_MODELS, lambda m: m.replace("yolov8", "v8").replace("yolo26", "26")),
        ("segment", SEG_MODELS, lambda m: m.replace("yolov8", "v8").replace("yolo26", "26").replace("-seg", "")),
        ("depth", DEPTH_MODELS, lambda m: m.replace("yolo26", "").replace("-depth", "")),
        ("pose", POSE_MODELS, lambda m: m.replace("yolo26", "").replace("-pose", "")),
        ("obb", OBB_MODELS, lambda m: m.replace("yolo26", "").replace("-obb", "")),
        ("semantic", SEM_MODELS, lambda m: m.replace("yolo26", "").replace("-sem", "")),
        ("classify", CLS_MODELS, lambda m: m.replace("yolo26", "").replace("-cls", "")),
    ]
    titles7 = {
        "detect": "detect (YOLOv8 + YOLO26)",
        "segment": "segment (YOLOv8 + YOLO26)",
        "depth": "depth (YOLO26)",
        "pose": "pose (YOLO26)",
        "obb": "obb (YOLO26)",
        "semantic": "semantic (YOLO26)",
        "classify": "classify (YOLO26)",
    }
    for ax, (fam, models, label_fn) in zip(axes.flat, chart7):
        family_bars(ax, models, bench, torchref, GPU_ONLY_SERIES, f"{titles7[fam]} — e2e latency (lower is better)", label_fn)
    axes[1, 3].axis("off")
    fig.suptitle("Closed-set task families — end-to-end latency by backend (bus.jpg, log scale)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RES / "task_latency.png", dpi=150)

    # ---- Chart 8: open-vocabulary World, fixed YTXT protocol ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    world_labels = [model.replace("yolov8", "v8") for model in WORLD_MODELS]
    width = 0.22
    for axis, metric, title in (
        (axes[0], "e2e_ms", "end-to-end latency"),
        (axes[1], "graph_ms", "graph latency"),
    ):
        for index, (backend, color) in enumerate((("cuda", "#1f77b4"), ("vulkan", "#2ca02c"), ("cpu", "#7f7f7f"))):
            values = [world_records[(backend, model)][metric]["mean"] for model in WORLD_MODELS]
            bars = axis.bar(
                [x + (index - 1) * width for x in range(len(WORLD_MODELS))], values, width, label=backend, color=color
            )
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.1f}", ha="center", fontsize=7)
        axis.set_xticks(range(len(WORLD_MODELS)))
        axis.set_xticklabels(world_labels)
        axis.set_yscale("log")
        axis.set_ylabel("ms / frame (log scale)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    fig.suptitle("YOLO-World F32 latency: fixed YTXT person,bus,car,truck at 480x640", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(RES / "world_latency.png", dpi=150)

    # ---- Chart 9: YOLOE uses detector-specific post-reprta MobileCLIP YTXT ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in (
        (axes[0], "e2e_ms", "end-to-end latency"),
        (axes[1], "graph_ms", "graph latency"),
    ):
        values = [yoloe_records[(backend, YOLOE_MODELS[0])][metric]["mean"] for backend in BACKENDS]
        bars = axis.bar(BACKENDS, values, color=["#1f77b4", "#2ca02c", "#7f7f7f"])
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value * 1.05, f"{value:.1f}", ha="center", fontsize=8)
        axis.set_yscale("log")
        axis.set_ylabel("ms / frame (log scale)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("YOLOE-26n-seg F32: fixed post-reprta YTXT person,bus,car,truck at 480x640", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(RES / "yoloe_latency.png", dpi=150)

    # ---- Speedup table (markdown) ----
    lines = [
        "# ggml vs PyTorch speedup (e2e mean, bus.jpg)",
        "",
        "## Coverage",
        "",
        "| Evidence | Expected | Present | Status |",
        "|---|---:|---:|---|",
        f"| GGML backend/model/precision keys | {len(BACKENDS) * len(MODELS) * len(DTYPES)} | "
        f"{len(BACKENDS) * len(MODELS) * len(DTYPES)} | complete |",
        f"| PyTorch CUDA model references | {len(MODELS)} | {len(MODELS)} | complete |",
        f"| World backend/model keys (fixed YTXT F32) | {len(BACKENDS) * len(WORLD_MODELS)} | "
        f"{len(world_records)} | complete |",
        f"| YOLOE backend/model keys (fixed post-reprta YTXT F32) | {len(BACKENDS) * len(YOLOE_MODELS)} | "
        f"{len(yoloe_records)} | complete |",
        "| GGML per-row latency fields | preprocess mean/p50/p90; graph mean/min/p50/p90/max; "
        "post mean/p50; e2e mean/min/p50/p90/max | all required fields | complete |",
        "| PyTorch per-row latency fields | e2e mean/min/p50/p90/max | all required fields | complete |",
        "",
        "The latency matrix is complete across all seven task families "
        "(detect, segment, depth, pose, obb, semantic, classify). Accuracy evidence is separate: "
        "focused parity covers every family on the documented image, while dataset-level validation "
        "(COCO, NYU Depth V2, Cityscapes, ImageNet, DOTA) is not yet established.",
        "",
    ]
    for fam, models in FAMILIES.items():
        lines += [f"## {fam} latency and speedup", "", "| model | dtype | PyTorch | CUDA | speedup | Vulkan | speedup | CPU 8T | speedup |", "|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            pt = torchref.get(m)
            for dt in DTYPES:
                row = [m, dt, f"{pt:.2f}" if pt else "-"]
                for tag in ("cuda", "vulkan", "cpu"):
                    v = bench.get((tag, m, dt))
                    row += [f"{v:.2f}" if v else "-", f"x{pt / v:.2f}" if (v and pt) else "-"]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    (RES / "speedup_table.md").write_text("\n".join(lines).rstrip() + "\n")
    n_keys = len(BACKENDS) * len(MODELS) * len(DTYPES)
    print(
        f"validated {n_keys} GGML + {len(torch_records)} PyTorch keys across 7 task families; "
        "wrote speed_by_model.png, latency_by_backend.png, speed_by_dtype.png, depth_latency.png, "
        "latency_matrix.png, seg_latency.png, task_latency.png, world_latency.png, yoloe_latency.png, speedup_table.md"
    )


if __name__ == "__main__":
    main()
