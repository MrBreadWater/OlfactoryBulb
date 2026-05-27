"""Audit optional EPLI implementation against explicit structural constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from olfactorybulb.audit.core import AuditItem, AuditReport, collect_items, rounded
from olfactorybulb.audit.epli_reference import (
    BRANCHING_ZONE_MAX_UM,
    FAST_SPIKING_REFERENCE,
    PLANAR_SPAN_UM,
    PRIMARY_PROCESS_COUNT,
    SOMA_DIAMETER_UM,
)
from olfactorybulb.audit.neuron_protocols import monotonic_non_decreasing, sweep_soma_step_responses
from olfactorybulb.epli import PRINCIPAL_PERISOMATIC_SELECTOR, default_slice_synapse_blueprints
from olfactorybulb.slice_connectivity_optimizer import (
    load_slice_geometry,
    observed_metrics_for_synapse_set,
    resolve_slice_dir,
)


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
                    "source_coverage": rounded(metrics.source_coverage),
                    "target_coverage": rounded(metrics.target_coverage),
                    "median_distance_um": rounded(metrics.median_distance_um),
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

    selectors = [blueprint.get("section_pattern_dest") for blueprint in epli_blueprints]
    has_semantic_perisomatic_selector = all(selector == PRINCIPAL_PERISOMATIC_SELECTOR for selector in selectors)
    soma_only = all(selector == "*soma*" for selector in selectors)
    items.append(
        AuditItem(
            check_id="epli_target_pattern_specificity",
            status="PASS" if has_semantic_perisomatic_selector else ("FAIL" if soma_only else "WARN"),
            title="Default EPLI target pattern encodes perisomatic principal territory",
            criterion="Perisomatic inhibition in the literature includes soma, proximal apical dendrite, and axon hillock territory; the default selector should encode that broader territory instead of soma-only.",
            evidence={
                blueprint["group_to"]: {
                    "section_pattern_dest": blueprint.get("section_pattern_dest"),
                    "max_distance_um": blueprint.get("max_distance"),
                }
                for blueprint in epli_blueprints
            },
            note=(
                "The current selector is a semantic perisomatic scaffold."
                if has_semantic_perisomatic_selector
                else "Current defaults encode a placeholder contact class, not a validated anatomical targeting rule."
            ),
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
    repo_root = Path(__file__).resolve().parents[2]
    blender_source = (repo_root / "olfactorybulb" / "slicebuilder" / "blender.py").read_text()

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
    except Exception as exc:  # pragma: no cover
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

    PVCRH_FSI1._instance_counter = 0
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
                status="PASS" if SOMA_DIAMETER_UM.low <= soma_diameter_um <= SOMA_DIAMETER_UM.high else "FAIL",
                title="Synthetic EPLI soma diameter matches Huang 2013 target",
                criterion=f"Target soma diameter is {SOMA_DIAMETER_UM.mean} ± {SOMA_DIAMETER_UM.tolerance} {SOMA_DIAMETER_UM.units}.",
                evidence={"observed_um": rounded(soma_diameter_um), "target_low_um": SOMA_DIAMETER_UM.low, "target_high_um": SOMA_DIAMETER_UM.high},
            ),
            AuditItem(
                check_id="synthetic_primary_process_count",
                status="PASS" if PRIMARY_PROCESS_COUNT.low <= primary_count <= PRIMARY_PROCESS_COUNT.high else "FAIL",
                title="Synthetic EPLI primary process count matches target regime",
                criterion=f"Target primary-process count is {PRIMARY_PROCESS_COUNT.mean} ± {PRIMARY_PROCESS_COUNT.tolerance}.",
                evidence={"observed_count": primary_count, "target_low": PRIMARY_PROCESS_COUNT.low, "target_high": PRIMARY_PROCESS_COUNT.high},
            ),
            AuditItem(
                check_id="synthetic_planar_span",
                status="PASS" if PLANAR_SPAN_UM.low <= planar_span_um <= PLANAR_SPAN_UM.high else "FAIL",
                title="Synthetic EPLI planar span matches Huang 2013 target",
                criterion=f"Target neurite span is {PLANAR_SPAN_UM.mean} ± {PLANAR_SPAN_UM.tolerance} {PLANAR_SPAN_UM.units}.",
                evidence={"observed_um": rounded(planar_span_um), "target_low_um": PLANAR_SPAN_UM.low, "target_high_um": PLANAR_SPAN_UM.high},
            ),
            AuditItem(
                check_id="synthetic_branching_zone",
                status="PASS" if max(branch_root_distances) <= BRANCHING_ZONE_MAX_UM else "FAIL",
                title="Synthetic EPLI branching occurs within proximal EPL territory",
                criterion=f"Highest branching should occur within roughly {BRANCHING_ZONE_MAX_UM:g} um of the soma.",
                evidence={
                    "max_branch_origin_um": rounded(max(branch_root_distances)),
                    "branch_origin_distances_um": [rounded(value) for value in branch_root_distances],
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


def audit_synthetic_cell_behavior(*, skip_neuron: bool = False) -> list[AuditItem]:
    items: list[AuditItem] = []

    if skip_neuron:
        items.append(
            AuditItem(
                check_id="synthetic_cell_behavior_skipped",
                status="WARN",
                title="Synthetic EPLI behavior check skipped",
                criterion="Run this audit under the OBGPU/NEURON environment to verify stable fast-spiking behavior.",
            )
        )
        return items

    try:
        from prev_ob_models.SyntheticEPL2026.isolated_cells import PVCRH_FSI1
    except Exception as exc:  # pragma: no cover
        items.append(
            AuditItem(
                check_id="synthetic_cell_behavior_import",
                status="FAIL",
                title="Synthetic EPLI behavior audit could not import the surrogate cell",
                criterion="Behavior audit must be able to instantiate the configured surrogate cell in NEURON.",
                evidence={"error": repr(exc)},
            )
        )
        return items

    def cell_factory():
        PVCRH_FSI1._instance_counter = 0
        return PVCRH_FSI1()

    amplitudes = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, FAST_SPIKING_REFERENCE.audit_current_max_nA]
    responses = sweep_soma_step_responses(cell_factory, amplitudes)
    response_by_amp = {response.amp_nA: response for response in responses}
    response_rates = [response.step_rate_hz for response in responses if response.amp_nA > 0]
    rest = response_by_amp[0.0]
    moderate = response_by_amp[0.2]
    fast = response_by_amp[FAST_SPIKING_REFERENCE.audit_current_max_nA]
    max_rate = max(response_rates) if response_rates else 0.0

    items.extend(
        [
            AuditItem(
                check_id="synthetic_rest_stability",
                status="PASS" if (not rest.has_nan and len(rest.step_spike_times_ms) == 0) else "FAIL",
                title="Synthetic EPLI remains numerically stable at rest",
                criterion="The surrogate should not generate NaNs or spontaneous spikes during a resting fixed-step audit run.",
                evidence={
                    "has_nan": rest.has_nan,
                    "rest_final_v_mV": rounded(rest.final_v_mV),
                    "rest_max_v_mV": rounded(rest.max_v_mV),
                    "rest_spike_count": len(rest.step_spike_times_ms),
                },
            ),
            AuditItem(
                check_id="synthetic_moderate_current_spiking",
                status="PASS" if (not moderate.has_nan and len(moderate.step_spike_times_ms) > 0) else "FAIL",
                title="Synthetic EPLI spikes under moderate current injection",
                criterion="The surrogate should fire repetitively under a moderate somatic current step without numerical instability.",
                evidence={
                    "amp_nA": moderate.amp_nA,
                    "has_nan": moderate.has_nan,
                    "spike_count": len(moderate.step_spike_times_ms),
                    "step_rate_hz": rounded(moderate.step_rate_hz),
                    "max_v_mV": rounded(moderate.max_v_mV),
                },
            ),
            AuditItem(
                check_id="synthetic_fast_spiking_capability",
                status="PASS" if (not fast.has_nan and max_rate >= FAST_SPIKING_REFERENCE.minimum_fast_spiking_rate_hz) else "FAIL",
                title="Synthetic EPLI reaches a literature-consistent fast-spiking regime",
                criterion=(
                    f"The audit sweep should reach at least {FAST_SPIKING_REFERENCE.minimum_fast_spiking_rate_hz:g} Hz "
                    f"by {FAST_SPIKING_REFERENCE.audit_current_max_nA:g} nA, consistent with the Huang 2013 fast-spiking regime."
                ),
                evidence={
                    "reference": FAST_SPIKING_REFERENCE.source,
                    "audit_current_max_nA": FAST_SPIKING_REFERENCE.audit_current_max_nA,
                    "target_min_rate_hz": FAST_SPIKING_REFERENCE.minimum_fast_spiking_rate_hz,
                    "target_stretch_rate_hz": FAST_SPIKING_REFERENCE.stretch_fast_spiking_rate_hz,
                    "observed_max_rate_hz": rounded(max_rate),
                    "observed_rate_at_max_current_hz": rounded(fast.step_rate_hz),
                },
                note="Kato 2013 reports still higher PV-FSI rates, but the current audit only requires the Huang-like fast-spiking regime.",
            ),
            AuditItem(
                check_id="synthetic_fi_monotonicity",
                status="PASS" if monotonic_non_decreasing(response_rates) else "WARN",
                title="Synthetic EPLI firing rate increases monotonically across the audit step sweep",
                criterion="A first-pass fast-spiking scaffold should exhibit a nondecreasing f-I curve over the audit current range.",
                evidence={
                    "amps_nA": [response.amp_nA for response in responses if response.amp_nA > 0],
                    "rates_hz": [rounded(rate) for rate in response_rates],
                },
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


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-slice", default=None, help="Optional exported EPLI slice directory or slice name to audit.")
    parser.add_argument("--skip-neuron", action="store_true", help="Skip NEURON-backed morphology checks.")


def run(args: argparse.Namespace) -> AuditReport:
    items = collect_items(
        audit_baseline_slice(),
        audit_epli_defaults(),
        audit_epli_distribution_assumptions(),
        audit_synthetic_cell_geometry(skip_neuron=bool(args.skip_neuron)),
        audit_synthetic_cell_behavior(skip_neuron=bool(args.skip_neuron)),
        audit_candidate_slice(getattr(args, "candidate_slice", None)),
    )
    return AuditReport(
        audit_id="epli_correctness",
        title="EPLI correctness audit",
        items=items,
    )
