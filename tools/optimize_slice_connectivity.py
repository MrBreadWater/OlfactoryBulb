#!/usr/bin/env python3
"""Offline connectivity optimization against exported slice JSON geometry.

Examples:
    python tools/optimize_slice_connectivity.py reference \
        --slice DorsalColumnSlice \
        --synapse-set GCs__MCs

    python tools/optimize_slice_connectivity.py epli \
        --slice olfactorybulb/slices/DorsalColumnSliceEPLI_opt_scan \
        --source-group EPLIs \
        --target-group MCs \
        --max-distances 5 10 15 20 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.slice_connectivity_optimizer import (
    grid_search_against_reference,
    grid_search_epli_candidates,
    load_slice_geometry,
    observed_metrics_for_synapse_set,
    pretty_top_results,
    suggest_section_patterns,
)


def _comma_or_space_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in str(value).split(",") if part.strip())
    return items


def _resolve_patterns(
    explicit_patterns: list[str] | None,
    *,
    group_patterns: list[str],
    preferred_patterns: list[str] | None = None,
) -> list[str]:
    explicit = _comma_or_space_list(explicit_patterns)
    if explicit:
        return explicit
    if preferred_patterns:
        ordered = [pattern for pattern in preferred_patterns if pattern in group_patterns]
        extras = [pattern for pattern in group_patterns if pattern not in ordered]
        return ordered + extras
    return group_patterns


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference", help="Recover canonical connectivity rules from an exported slice.")
    reference.add_argument("--slice", default="DorsalColumnSlice")
    reference.add_argument("--synapse-set", default="GCs__MCs")
    reference.add_argument("--source-patterns", nargs="*")
    reference.add_argument("--target-patterns", nargs="*")
    reference.add_argument("--max-distances", type=float, nargs="+", default=[4, 5, 6, 8])
    reference.add_argument("--use-radii", choices=["both", "true", "false"], default="both")
    reference.add_argument("--max-syns-per-pts", type=int, nargs="+", default=[1, 2, 3])
    reference.add_argument("--top-n", type=int, default=10)
    reference.add_argument("--json-out")

    epli = subparsers.add_parser("epli", help="Search candidate EPLI connectivity rules on an exported slice.")
    epli.add_argument("--slice", required=True)
    epli.add_argument("--source-group", default="EPLIs")
    epli.add_argument("--target-group", default="MCs")
    epli.add_argument("--reference-synapse-set", default="GCs__MCs")
    epli.add_argument("--source-patterns", nargs="*")
    epli.add_argument("--target-patterns", nargs="*")
    epli.add_argument("--max-distances", type=float, nargs="+", default=[5, 10, 15, 20, 30, 40])
    epli.add_argument("--use-radii", choices=["both", "true", "false"], default="both")
    epli.add_argument("--max-syns-per-pts", type=int, nargs="+", default=[1, 2, 3])
    epli.add_argument("--top-n", type=int, default=10)
    epli.add_argument("--json-out")

    return parser


def _bool_search_values(mode: str) -> list[bool]:
    if mode == "true":
        return [True]
    if mode == "false":
        return [False]
    return [True, False]


def _reference_payload(args: argparse.Namespace) -> dict[str, Any]:
    groups = load_slice_geometry(args.slice)
    source_group, target_group = args.synapse_set.split("__", 1)
    source_patterns = _resolve_patterns(
        args.source_patterns,
        group_patterns=suggest_section_patterns(groups[source_group]),
        preferred_patterns=["*apic*", "*dend*", "*soma*", "*axon*"],
    )
    target_patterns = _resolve_patterns(
        args.target_patterns,
        group_patterns=suggest_section_patterns(groups[target_group]),
        preferred_patterns=["*dend*", "*apic*", "*soma*", "*axon*"],
    )
    results = grid_search_against_reference(
        args.slice,
        reference_synapse_set=args.synapse_set,
        source_patterns=source_patterns,
        target_patterns=target_patterns,
        max_distances_um=args.max_distances,
        use_radii=_bool_search_values(args.use_radii),
        max_syns_per_pts=args.max_syns_per_pts,
    )
    reference_metrics = observed_metrics_for_synapse_set(args.slice, args.synapse_set, groups=groups)
    return {
        "mode": "reference",
        "slice": args.slice,
        "synapse_set": args.synapse_set,
        "reference_metrics": reference_metrics.to_dict(),
        "source_patterns": source_patterns,
        "target_patterns": target_patterns,
        "max_distances_um": list(args.max_distances),
        "use_radii": _bool_search_values(args.use_radii),
        "max_syns_per_pts": list(args.max_syns_per_pts),
        "top_results": [result.to_dict() for result in results[: args.top_n]],
        "top_pretty": pretty_top_results(results, top_n=args.top_n),
    }


def _epli_payload(args: argparse.Namespace) -> dict[str, Any]:
    groups = load_slice_geometry(args.slice)
    source_patterns = _resolve_patterns(
        args.source_patterns,
        group_patterns=suggest_section_patterns(groups[args.source_group]),
        preferred_patterns=["*dend*", "*dend_branch*", "*dend_primary*", "*soma*", "*apic*"],
    )
    target_patterns = _resolve_patterns(
        args.target_patterns,
        group_patterns=suggest_section_patterns(groups[args.target_group]),
        preferred_patterns=["*dend*", "*soma*", "*apic*", "*axon*"],
    )
    results = grid_search_epli_candidates(
        args.slice,
        source_group=args.source_group,
        target_group=args.target_group,
        source_patterns=source_patterns,
        target_patterns=target_patterns,
        max_distances_um=args.max_distances,
        use_radii=_bool_search_values(args.use_radii),
        max_syns_per_pts=args.max_syns_per_pts,
        reference_synapse_set=args.reference_synapse_set,
    )
    return {
        "mode": "epli",
        "slice": args.slice,
        "source_group": args.source_group,
        "target_group": args.target_group,
        "reference_synapse_set": args.reference_synapse_set,
        "source_patterns": source_patterns,
        "target_patterns": target_patterns,
        "max_distances_um": list(args.max_distances),
        "use_radii": _bool_search_values(args.use_radii),
        "max_syns_per_pts": list(args.max_syns_per_pts),
        "top_results": [result.to_dict() for result in results[: args.top_n]],
        "top_pretty": pretty_top_results(results, top_n=args.top_n),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "reference":
        payload = _reference_payload(args)
    elif args.command == "epli":
        payload = _epli_payload(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command {args.command!r}")

    print(payload["top_pretty"])
    _write_json(getattr(args, "json_out", None), payload)


if __name__ == "__main__":
    main()
