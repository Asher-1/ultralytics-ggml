#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression comparators for the cpp_ggml engine (used by regress_all.sh).

Every subcommand prints a metrics block, then exactly one line "PASS" or
"FAIL", and exits 0/1 — scripts/regress_all.sh aggregates those verdicts.

Subcommands:
  raw  BASE.bin CUR.bin [--max-abs F] [--max-rel F]
        Tensor-level diff of two YRAW0001/YDEP0001/YINP0001 dumps (the exact
        same tensors parity_reference.py produces). Per-element NumPy-allclose
        tolerance: diff <= max-abs + max-rel * |base|. FAIL when any element
        exceeds it.
  ops  BASE_DIR CUR_DIR [--max-abs F] [--max-rel F]
        Optrace regression: compares every op dump the two directories share
        with the same per-element tolerance. A differing file set means the
        graph structure changed — that is a FAIL too (re-record the baseline
        deliberately if the change is the point of the patch).
  dets BASE.json CUR.json [--conf-tol F] [--box-tol F]
        Detection-list regression on --dets-json output: same detection count
        and, pairing by class then position, per-box conf/xyxy deltas.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC_DIMS = {b"YINP0001": 3, b"YRAW0001": 2, b"YDEP0001": 2, b"YLYR0001": 4}


def read_bin(path):
    import numpy as np

    with open(path, "rb") as f:
        magic = f.read(8)
        nd = MAGIC_DIMS.get(magic)
        if nd is None:
            raise ValueError(f"{path}: unknown magic {magic!r}")
        dims = struct.unpack(f"<{nd}i", f.read(4 * nd))
    return magic, dims, np.fromfile(path, dtype=np.float32, offset=8 + 4 * nd)


def verdict(ok, detail):
    print(detail)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def diff_stats(a, b):
    """Per-element metrics vs the allclose tolerance max-abs + max-rel*|base|."""
    import numpy as np

    diff = np.abs(a - b)
    rel = diff / np.maximum(np.abs(a), 1e-3)
    idx = np.unravel_index(np.argmax(diff), diff.shape) if diff.size else (0,)
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "rel_p99": float(np.quantile(rel, 0.99)) if diff.size else 0.0,
        "worst_idx": [int(i) for i in idx],
        "worst_base": float(a[idx]) if diff.size else 0.0,
        "worst_cur": float(b[idx]) if diff.size else 0.0,
    }


def within_tolerance(a, b, max_abs, max_rel):
    """NumPy-allclose semantics on |a - b| <= max_abs + max_rel*|a|, per element."""
    import numpy as np

    return bool(np.all(np.abs(a - b) <= max_abs + max_rel * np.abs(a)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("raw", help="tensor diff of two binary dumps")
    pr.add_argument("base")
    pr.add_argument("cur")
    pr.add_argument("--max-abs", type=float, required=True)
    pr.add_argument("--max-rel", type=float, required=True)

    po = sub.add_parser("ops", help="optrace directory diff")
    po.add_argument("base")
    po.add_argument("cur")
    po.add_argument("--max-abs", type=float, required=True)
    po.add_argument("--max-rel", type=float, required=True)

    pd = sub.add_parser("dets", help="detection-list diff of two --dets-json files")
    pd.add_argument("base")
    pd.add_argument("cur")
    pd.add_argument("--conf-tol", type=float, default=0.02)
    pd.add_argument("--box-tol", type=float, default=3.0)

    a = p.parse_args()

    if a.cmd == "raw":
        _, d1, base = read_bin(a.base)
        _, d2, cur = read_bin(a.cur)
        if d1 != d2 or base.shape != cur.shape:
            return verdict(False, f"shape mismatch {d1} vs {d2}")
        base, cur = base.reshape(d1), cur.reshape(d2)
        s = diff_stats(base, cur)
        ok = within_tolerance(base, cur, a.max_abs, a.max_rel)
        detail = (
            f"shape={s['shape']} max_abs={s['max_abs']:.6f} rel_p99={s['rel_p99']:.6f} "
            f"(tolerance max_abs<={a.max_abs} + max_rel<={a.max_rel}*|base|) "
            f"worst@{s['worst_idx']} base={s['worst_base']:.4f} cur={s['worst_cur']:.4f}"
        )
        return verdict(ok, detail)

    if a.cmd == "ops":
        base_files = sorted(f.name for f in Path(a.base).glob("*.bin"))
        cur_files = sorted(f.name for f in Path(a.cur).glob("*.bin"))
        if base_files != cur_files:
            only_base = [f for f in base_files if f not in cur_files]
            only_cur = [f for f in cur_files if f not in base_files]
            return verdict(
                False,
                f"op set changed (graph structure drift): only-base={only_base[:4]} "
                f"only-cur={only_cur[:4]} — re-record the baseline if intended",
            )
        worst, worst_name, fails = None, "", []
        for name in base_files:
            _, d1, base = read_bin(str(Path(a.base) / name))
            _, d2, cur = read_bin(str(Path(a.cur) / name))
            if d1 != d2:
                fails.append(name)
                continue
            s = diff_stats(base.reshape(d1), cur.reshape(d2))
            if not within_tolerance(base.reshape(d1), cur.reshape(d2), a.max_abs, a.max_rel):
                fails.append(name)
            if worst is None or s["max_abs"] > worst["max_abs"]:
                worst, worst_name = s, name
        detail = (
            f"ops={len(base_files)} failed={len(fails)} worst={worst_name} "
            f"max_abs={worst['max_abs']:.6f} rel_p99={worst['rel_p99']:.6f} "
            f"(tolerance max_abs<={a.max_abs} + max_rel<={a.max_rel}*|base|)"
        )
        return verdict(not fails, detail)

    if a.cmd == "dets":
        base = json.loads(Path(a.base).read_text())
        cur = json.loads(Path(a.cur).read_text())
        bd = sorted(base["detections"], key=lambda d: (d["cls"], d["xyxy"][0], d["xyxy"][1]))
        cd = sorted(cur["detections"], key=lambda d: (d["cls"], d["xyxy"][0], d["xyxy"][1]))
        if len(bd) != len(cd):
            return verdict(
                False,
                f"detection count changed: base={len(bd)} cur={len(cd)} "
                f"vocab={base['vocabulary']} vs {cur['vocabulary']}",
            )
        worst_conf, worst_box = 0.0, 0.0
        for b, c in zip(bd, cd):
            if b["cls"] != c["cls"]:
                return verdict(False, f"class changed: {b} vs {c}")
            worst_conf = max(worst_conf, abs(b["conf"] - c["conf"]))
            worst_box = max(worst_box, max(abs(x - y) for x, y in zip(b["xyxy"], c["xyxy"])))
        ok = worst_conf <= a.conf_tol and worst_box <= a.box_tol
        detail = (
            f"detections={len(bd)} max_conf_delta={worst_conf:.6f} max_box_delta={worst_box:.3f}px "
            f"(limits conf<={a.conf_tol} box<={a.box_tol}px)"
        )
        return verdict(ok, detail)

    return 2


if __name__ == "__main__":
    sys.exit(main())
