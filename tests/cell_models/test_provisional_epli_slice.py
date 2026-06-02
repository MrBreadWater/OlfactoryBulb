"""Validate the checked-in provisional EPLI slice asset."""

from pathlib import Path

from olfactorybulb.slice_connectivity_optimizer import (
    load_slice_geometry,
    observed_metrics_for_synapse_set,
    resolve_slice_dir,
)


slice_dir = resolve_slice_dir("DorsalColumnSliceEPLIProvisional")
assert Path(slice_dir).exists()
for file_name in (
    "EPLIs.json",
    "EPLIs__TCs.json",
    "EPLIs__MCs.json",
    "MCs.json",
    "TCs.json",
    "GCs.json",
    "glom_cells.json",
):
    assert (slice_dir / file_name).exists(), file_name

groups = load_slice_geometry(slice_dir)
assert "EPLIs" in groups
assert len(groups["EPLIs"].cell_names) == 24

tc_metrics = observed_metrics_for_synapse_set(slice_dir, "EPLIs__TCs", groups=groups)
assert tc_metrics.entry_count == 71
assert tc_metrics.source_coverage == 1.0
assert tc_metrics.target_coverage == 1.0

mc_metrics = observed_metrics_for_synapse_set(slice_dir, "EPLIs__MCs", groups=groups)
assert mc_metrics.entry_count == 85
assert round(mc_metrics.source_coverage, 3) == 0.75
assert mc_metrics.target_coverage == 1.0

print("provisional EPLI slice smoke test: OK")
