"""Compare standard benchmark artifacts such as soma traces and LFP."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.result_artifacts import find_soma_trace_artifact, load_saved_result_artifact


def load_pickle(path: str | Path):
    """Load a pickle artifact from disk."""
    return load_saved_result_artifact(path)


def max_abs_diff(a, b) -> float | None:
    """Return the maximum absolute difference when array shapes match."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def compare_soma_vs(before_path: str | Path, after_path: str | Path) -> dict[str, object]:
    """Compare saved soma traces by cell label."""
    before = {cell: (np.asarray(t), np.asarray(v)) for cell, t, v in load_pickle(before_path)}
    after = {cell: (np.asarray(t), np.asarray(v)) for cell, t, v in load_pickle(after_path)}
    common = sorted(set(before) & set(after))

    max_t = 0.0
    max_v = 0.0
    mismatched_shapes = []
    for cell in common:
        bt, bv = before[cell]
        at, av = after[cell]
        dt = max_abs_diff(bt, at)
        dv = max_abs_diff(bv, av)
        if dt is None or dv is None:
            mismatched_shapes.append(cell)
            continue
        max_t = max(max_t, dt)
        max_v = max(max_v, dv)

    return {
        "before_cells": len(before),
        "after_cells": len(after),
        "common_cells": len(common),
        "only_before": len(set(before) - set(after)),
        "only_after": len(set(after) - set(before)),
        "max_abs_time_diff": max_t,
        "max_abs_voltage_diff": max_v,
        "mismatched_shapes": mismatched_shapes[:10],
    }


def compare_input_times(before_path: str | Path, after_path: str | Path) -> dict[str, object]:
    """Compare saved input times by target segment."""
    before = {seg: np.asarray(times) for seg, times in load_pickle(before_path)}
    after = {seg: np.asarray(times) for seg, times in load_pickle(after_path)}
    common = sorted(set(before) & set(after))

    max_diff = 0.0
    mismatched_shapes = []
    for seg in common:
        diff = max_abs_diff(before[seg], after[seg])
        if diff is None:
            mismatched_shapes.append(seg)
            continue
        max_diff = max(max_diff, diff)

    return {
        "before_segments": len(before),
        "after_segments": len(after),
        "common_segments": len(common),
        "only_before": len(set(before) - set(after)),
        "only_after": len(set(after) - set(before)),
        "max_abs_time_diff": max_diff,
        "mismatched_shapes": mismatched_shapes[:10],
    }


def compare_lfp(before_path: str | Path, after_path: str | Path) -> dict[str, object]:
    """Compare saved LFP traces."""
    bt, bv = load_pickle(before_path)
    at, av = load_pickle(after_path)
    return {
        "before_len": len(bt),
        "after_len": len(at),
        "max_abs_time_diff": max_abs_diff(bt, at),
        "max_abs_value_diff": max_abs_diff(bv, av),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("before_dir")
    parser.add_argument("after_dir")
    args = parser.parse_args()

    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)

    report = {}
    before_soma = find_soma_trace_artifact(before_dir)
    after_soma = find_soma_trace_artifact(after_dir)
    if before_soma is not None and after_soma is not None:
        report["soma_vs"] = compare_soma_vs(before_soma, after_soma)
    for filename, fn in [("input_times.pkl", compare_input_times), ("lfp.pkl", compare_lfp)]:
        before_path = before_dir / filename
        after_path = after_dir / filename
        if before_path.exists() and after_path.exists():
            report[filename] = fn(before_path, after_path)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
