"""Smoke tests for static slice connectivity inspection helpers.

Run with:
    MPLCONFIGDIR=/tmp/mpl conda run -n OBGPU python test_slice_connectivity_helpers.py
"""

from obgpu_experiment_helpers import (
    find_cell_drivers,
    load_slice_connectivity,
    summarize_slice_connectivity,
)


connectivity = load_slice_connectivity("DorsalColumnSlice")
assert "glom_cells" in connectivity
assert "cell_groups" in connectivity
assert "synapse_sets" in connectivity
assert set(connectivity["cell_groups"]) >= {"MCs", "TCs", "GCs"}
assert set(connectivity["synapse_sets"]) >= {"GCs__MCs", "GCs__TCs"}

summary = summarize_slice_connectivity("DorsalColumnSlice")
assert summary["cell_group_counts"]["MCs"] > 0
assert summary["cell_group_counts"]["TCs"] > 0
assert summary["cell_group_counts"]["GCs"] > 0
assert summary["synapse_set_counts"]["GCs__MCs"] > 0
assert summary["synapse_set_counts"]["GCs__TCs"] > 0

driver_info = find_cell_drivers("MC5[14]", slice_name="DorsalColumnSlice")
assert driver_info["target_type"] == "MC"
assert driver_info["reciprocal_synapse_set"] == "GCs__MCs"
assert "GCs__MCs" in driver_info["incoming_synapses_by_set"]

print("slice connectivity helper smoke test: OK")
