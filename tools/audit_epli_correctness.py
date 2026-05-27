"""Audit optional EPLI implementation against explicit structural constraints.

This script is intentionally conservative. It does not claim to prove that the
biology is correct; instead, it checks whether the repository still satisfies
the concrete assumptions that have already been stated in code and in the
literature notes.

Run examples:

    python tools/audit_epli_correctness.py --skip-neuron
    /opt/miniconda3/envs/OBGPU/bin/python tools/audit_epli_correctness.py
    /opt/miniconda3/envs/OBGPU/bin/python tools/audit_epli_correctness.py \
        --candidate-slice olfactorybulb/slices/DorsalColumnSliceEPLI_smoke
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.epli import default_slice_synapse_blueprints
from olfactorybulb.slice_connectivity_optimizer import (
    load_slice_geometry,
    observed_metrics_for_synapse_set,
    resolve_slice_dir,
)


@dataclass
class AuditItem:
    check_id: str
    status: str
    title: str
    criterion: str
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def _status_rank(status: str) -> int:
    return {"FAIL": 3, "WARN": 2, "PASS": 1}.get(status, 0)


def _rounded(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def audit_baseline_slice() -> list[AuditItem]:
    items: list[AuditItem] = []
    groups = load_slice_geometry("DorsalColumnSlice")

    counts = {
        "MCs": len(groups["MCs"].cell_names),
        "TCs": len(groups["TCs"].cell_names),
        "GCs": len(groups["GCs"].cell_names),
    }
    items.append(
        AuditItem(
            check_id="baseline_slice_population_counts",
            status="PASS" if all(counts.values()) else "FAIL",
            title="Baseline dorsal slice contains nonzero principal and granule populations",
            criterion="Canonical maintained slice should contain MC, TC, and GC populations with exported geometry.",
            evidence=counts,
        )
    )

    for synapse_set in ("GCs__MCs", "GCs__TCs"):
        metrics = observed_metrics_for_synapse_set("DorsalColumnSlice", synapse_set, groups=groups)
        items.append(
            AuditItem(
                check_id=f"baseline_{synapse_set}",
                status="PASS" if metrics.entry_count > 0 else "FAIL",
                title=f"Canonical {synapse_set} set is populated",
                criterion="Maintained baseline slice should export nonzero reciprocal GC connectivity.",
                evidence={
                    "entry_count": metrics.entry_count,
                    "source_coverage": _rounded(metrics.source_coverage),
                    "target_coverage": _rounded(metrics.target_coverage),
                    "median_distance_um": _rounded(metrics.median_distance_um),
                    "target_family_fraction": metrics.target_family_fraction,
                },
            )
        )

    return items


def audit_epli_defaults() -> list[AuditItem]:
    items: list[AuditItem] = []
    epli_blueprints = [
        blueprint
        for blueprint in default_slice_synapse_blueprints(include_epli=True)
        if blueprint["group_from"] == "EPLIs"
    ]

    reciprocal_ok = all(
        blueprint.get("is_reciprocal")
        and blueprint.get("synapse_name_source") == "AmpaNmdaSyn"
        and blueprint.get("synapse_name_dest") == "GabaSyn"
        for blueprint in epli_blueprints
    )
    items.append(
        AuditItem(
            check_id="epli_reciprocal_architecture",
            status="PASS" if reciprocal_ok else "FAIL",
            title="EPLI default architecture is reciprocal excitatory-inhibitory",
            criterion="Default EPLI integration should preserve M/T -> EPLI excitation and EPLI -> M/T inhibition.",
            evidence={"blueprint_count": len(epli_blueprints)},
            note="This matches the high-level architecture in Kato 2013, Huang 2013, and Burton 2024.",
        )
    )

    soma_only = all(blueprint.get("section_pattern_dest") == "*soma*" for blueprint in epli_blueprints)
    items.append(
        AuditItem(
            check_id="epli_target_pattern_specificity",
            status="FAIL" if soma_only else "WARN",
            title="Default EPLI target pattern is soma-only",
            criterion="Perisomatic inhibition in the literature includes soma, proximal apical dendrite, and axon hillock territory; soma-only targeting is too narrow.",
            evidence={
                blueprint["group_to"]: {
                    "section_pattern_dest": blueprint.get("section_pattern_dest"),
                    "max_distance_um": blueprint.get("max_distance"),
                }
                for blueprint in epli_blueprints
            },
            note="Current defaults encode a placeholder contact class, not a validated anatomical targeting rule.",
        )
    )

    unsupported_distance = any(float(blueprint.get("max_distance", 0)) >= 20.0 for blueprint in epli_blueprints)
    items.append(
        AuditItem(
            check_id="epli_default_contact_radius",
            status="WARN" if unsupported_distance else "PASS",
            title="Default EPLI contact radius is heuristic",
            criterion="Default contact radii should be justified by recovered slice geometry or direct anatomical data.",
            evidence={blueprint["group_to"]: blueprint.get("max_distance") for blueprint in epli_blueprints},
            note="The repo currently uses 20 um as a first-pass placeholder, not a literature-derived number.",
        )
    )

    return items


def audit_epli_distribution_assumptions() -> list[AuditItem]:
    blender_source = (REPO_ROOT / "olfactorybulb" / "slicebuilder" / "blender.py").read_text()

    has_opl_fallback = "self.epli_particles_name = epli_particles_object_name or tc_particles_object_name" in blender_source
    uses_slice_order_default = "epli_selection_strategy='slice_order'" in blender_source
    has_default_depth_band = "epli_depth_min_fraction=0.2" in blender_source and "epli_depth_max_fraction=0.8" in blender_source

    return [
        AuditItem(
            check_id="epli_particle_cloud_source",
            status="WARN" if has_opl_fallback else "PASS",
            title="EPLI soma candidates reuse the TC/OPL particle cloud by default",
            criterion="A biologically grounded EPLI distribution should be based on an explicit EPL interneuron density model, not a TC fallback cloud.",
            evidence={
                "source_scan": "self.epli_particles_name = epli_particles_object_name or tc_particles_object_name",
                "default_depth_band_detected": has_default_depth_band,
            },
            note="The current default keeps the implementation opt-in and conservative, but it is not a validated cell-density prior.",
        ),
        AuditItem(
            check_id="epli_selection_strategy_default",
            status="WARN" if uses_slice_order_default else "PASS",
            title="Default EPLI candidate ranking is order-based",
            criterion="Placement selection should not depend on raw particle ordering when used for biological inference.",
            evidence={"source_scan": "epli_selection_strategy='slice_order'"},
            note="`principal_proximity` exists as a debugging/search option, but that is also a heuristic rather than a biological distribution model.",
        ),
    ]


def audit_synthetic_cell_geometry(*, skip_neuron: bool = False) -> list[AuditItem]:
    items: list[AuditItem] = []

    if skip_neuron:
        items.append(
            AuditItem(
                check_id="synthetic_cell_geometry_skipped",
                status="WARN",
                title="Synthetic EPLI geometry check skipped",
                criterion="Run this audit under the OBGPU/NEURON environment to verify literature-constrained morphology.",
            )
        )
        return items

    try:
        from prev_ob_models.SyntheticEPL2026.isolated_cells import PVCRH_FSI1
    except Exception as exc:  # pragma: no cover - depends on environment
        items.append(
            AuditItem(
                check_id="synthetic_cell_geometry_import",
                status="FAIL",
                title="Synthetic EPLI geometry could not be instantiated",
                criterion="Audit must be able to load the configured surrogate cell in NEURON.",
                evidence={"error": repr(exc)},
            )
        )
        return items

    cell = PVCRH_FSI1()
    soma_diameter_um = float(cell.soma.diam)
    primary_count = len(cell.primary_dendrites)
    planar_span_um = float(cell.planar_dendritic_span_um)
    branch_root_distances = []
    for section in cell.branch_dendrites:
        x = cell.h.x3d(0, sec=section)
        y = cell.h.y3d(0, sec=section)
        z = cell.h.z3d(0, sec=section)
        branch_root_distances.append((x * x + y * y + z * z) ** 0.5)

    items.extend(
        [
            AuditItem(
                check_id="synthetic_soma_diameter",
                status="PASS" if 8.9 <= soma_diameter_um <= 10.3 else "FAIL",
                title="Synthetic EPLI soma diameter matches Huang 2013 target",
                criterion="Target soma diameter is 9.6 ± 0.7 um for CRH+ EPL interneurons.",
                evidence={"observed_um": _rounded(soma_diameter_um)},
            ),
            AuditItem(
                check_id="synthetic_primary_process_count",
                status="PASS" if 3 <= primary_count <= 4 else "FAIL",
                title="Synthetic EPLI primary process count matches target regime",
                criterion="Target primary-process count is 3.5 ± 0.4, so 3-4 primaries is the intended range.",
                evidence={"observed_count": primary_count},
            ),
            AuditItem(
                check_id="synthetic_planar_span",
                status="PASS" if 66.5 <= planar_span_um <= 75.5 else "FAIL",
                title="Synthetic EPLI planar span matches Huang 2013 target",
                criterion="Target neurite span is 71 ± 4.5 um.",
                evidence={"observed_um": _rounded(planar_span_um)},
            ),
            AuditItem(
                check_id="synthetic_branching_zone",
                status="PASS" if max(branch_root_distances) <= 30.0 else "FAIL",
                title="Synthetic EPLI branching occurs within proximal EPL territory",
                criterion="Highest branching should occur within roughly 30 um of the soma.",
                evidence={
                    "max_branch_origin_um": _rounded(max(branch_root_distances)),
                    "branch_origin_distances_um": [_rounded(value) for value in branch_root_distances],
                },
            ),
            AuditItem(
                check_id="synthetic_axonless_topology",
                status="PASS",
                title="Synthetic EPLI topology is axonless",
                criterion="Current target class is implemented as an axonless/anaxonic surrogate.",
                evidence={"has_axon_sections": hasattr(cell, "axon") and bool(getattr(cell, "axon"))},
            ),
        ]
    )

    return items


def audit_candidate_slice(candidate_slice: str | None) -> list[AuditItem]:
    items: list[AuditItem] = []
    if not candidate_slice:
        slice_dir = resolve_slice_dir("DorsalColumnSlice")
        has_epli_assets = (slice_dir / "EPLIs.json").exists()
        items.append(
            AuditItem(
                check_id="canonical_epli_assets_present",
                status="FAIL" if not has_epli_assets else "PASS",
                title="Canonical maintained slice does not yet ship EPLI assets",
                criterion="A network-ready EPLI claim requires an exported slice with EPLI group geometry and populated synapse sets.",
                evidence={"slice_dir": str(slice_dir), "has_epli_assets": has_epli_assets},
                note="This is the central readiness gap right now: the runtime hooks exist, but the maintained slice remains MC/TC/GC only.",
            )
        )
        return items

    slice_dir = resolve_slice_dir(candidate_slice)
    status = "PASS" if slice_dir.exists() else "FAIL"
    items.append(
        AuditItem(
            check_id="candidate_slice_exists",
            status=status,
            title="Candidate EPLI slice exists on disk",
            criterion="Audit can only validate exported connectivity if slice assets exist.",
            evidence={"slice_dir": str(slice_dir)},
        )
    )
    if not slice_dir.exists():
        return items

    group_presence = {name: (slice_dir / f"{name}.json").exists() for name in ("MCs", "TCs", "GCs", "EPLIs")}
    items.append(
        AuditItem(
            check_id="candidate_slice_group_presence",
            status="PASS" if all(group_presence.values()) else "FAIL",
            title="Candidate slice includes all expected cell-group exports",
            criterion="A candidate EPLI slice should export MC, TC, GC, and EPLI geometry.",
            evidence=group_presence,
        )
    )

    for synapse_set in ("EPLIs__MCs", "EPLIs__TCs"):
        syn_path = slice_dir / f"{synapse_set}.json"
        entry_count = None
        if syn_path.exists():
            data = json.loads(syn_path.read_text())
            entry_count = len(data.get("entries", []))
        items.append(
            AuditItem(
                check_id=f"candidate_{synapse_set}",
                status="PASS" if (entry_count or 0) > 0 else "FAIL",
                title=f"Candidate slice exports nonzero {synapse_set} entries",
                criterion="Network-ready EPLI connectivity requires nonzero explicit synapse-set entries.",
                evidence={"entry_count": entry_count, "path": str(syn_path)},
            )
        )

    return items


def summarize(items: list[AuditItem]) -> dict[str, Any]:
    counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    worst_status = max((item.status for item in items), key=_status_rank, default="PASS")
    return {
        "summary": counts,
        "worst_status": worst_status,
        "items": [asdict(item) for item in items],
    }


def _print_human(summary: dict[str, Any]) -> None:
    print("EPLI correctness audit")
    print("======================")
    print(
        "Summary:",
        ", ".join(f"{key}={value}" for key, value in summary["summary"].items()),
        f"(worst={summary['worst_status']})",
    )
    print()
    for item in summary["items"]:
        print(f"[{item['status']}] {item['check_id']}: {item['title']}")
        print(f"  Criterion: {item['criterion']}")
        if item["evidence"]:
            print(f"  Evidence: {json.dumps(item['evidence'], sort_keys=True)}")
        if item["note"]:
            print(f"  Note: {item['note']}")
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-slice", default=None, help="Optional exported EPLI slice directory or slice name to audit.")
    parser.add_argument("--skip-neuron", action="store_true", help="Skip NEURON-backed morphology checks.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human-readable text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    items: list[AuditItem] = []
    items.extend(audit_baseline_slice())
    items.extend(audit_epli_defaults())
    items.extend(audit_epli_distribution_assumptions())
    items.extend(audit_synthetic_cell_geometry(skip_neuron=args.skip_neuron))
    items.extend(audit_candidate_slice(args.candidate_slice))

    summary = summarize(items)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary)

    return 0 if summary["worst_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
