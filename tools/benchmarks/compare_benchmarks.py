"""Compare two benchmark summaries and report timing/file-hash deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str | Path) -> dict:
    """Load a benchmark summary JSON file."""
    with open(path) as f:
        return json.load(f)


def pct_change(before: float, after: float) -> float | None:
    """Return percent change from ``before`` to ``after``."""
    if before == 0:
        return None
    return ((after - before) / before) * 100.0


def speedup(before: float, after: float) -> float | None:
    """Return multiplicative speedup ``before / after``."""
    if after == 0:
        return None
    return before / after


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before = load_summary(args.before)
    after = load_summary(args.after)

    phases = ["build_max_rank", "run_max_rank", "save_max_rank", "total_max_rank"]
    report = {
        "before": before["label"],
        "after": after["label"],
        "paramset_before": before["paramset"],
        "paramset_after": after["paramset"],
        "nranks_before": before["nranks"],
        "nranks_after": after["nranks"],
        "timing": {},
        "file_hashes_match": {},
    }

    for phase in phases:
        b = before["timing_seconds"][phase]
        a = after["timing_seconds"][phase]
        report["timing"][phase] = {
            "before_seconds": b,
            "after_seconds": a,
            "speedup": speedup(b, a),
            "percent_change": pct_change(b, a),
        }

    all_files = set(before.get("files", {}).keys()) | set(after.get("files", {}).keys())
    for filename in sorted(all_files):
        before_info = before.get("files", {}).get(filename)
        after_info = after.get("files", {}).get(filename)
        before_hash = None if before_info is None else before_info.get("canonical_sha256", before_info.get("sha256"))
        after_hash = None if after_info is None else after_info.get("canonical_sha256", after_info.get("sha256"))
        report["file_hashes_match"][filename] = (
            before_info is not None
            and after_info is not None
            and before_hash == after_hash
        )

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
