"""Positive-control tests for offline slice connectivity optimization.

Run with:
    MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_slice_connectivity_optimizer.py
"""

from olfactorybulb.slice_connectivity_optimizer import (
    ConnectivityMetrics,
    build_candidate_pool,
    grid_search_against_reference,
    load_slice_geometry,
    observed_metrics_for_synapse_set,
    score_epli_candidate,
)


groups = load_slice_geometry("DorsalColumnSlice")

for synapse_set, expected_count in [("GCs__MCs", 2238), ("GCs__TCs", 1751)]:
    metrics = observed_metrics_for_synapse_set("DorsalColumnSlice", synapse_set, groups=groups)
    assert metrics.entry_count == expected_count
    assert metrics.source_family_fraction == {"apic": 1.0}
    assert metrics.target_family_fraction == {"dend": 1.0}
    assert metrics.median_distance_um is not None and 2.5 <= metrics.median_distance_um <= 3.5
    assert metrics.p90_distance_um is not None and 4.0 <= metrics.p90_distance_um <= 5.5

gc_mc_pool = build_candidate_pool(
    groups["GCs"],
    groups["MCs"],
    source_patterns=["*apic*"],
    target_patterns=["*dend*"],
    max_distance_um=8,
    use_radius=True,
)
assert len(gc_mc_pool.pairs_sorted) >= 2000

for synapse_set in ["GCs__MCs", "GCs__TCs"]:
    results = grid_search_against_reference(
        "DorsalColumnSlice",
        reference_synapse_set=synapse_set,
        source_patterns=["*apic*", "*dend*", "*soma*"],
        target_patterns=["*dend*", "*apic*", "*soma*", "*axon*"],
        max_distances_um=[4, 5, 6],
        use_radii=[True, False],
        max_syns_per_pts=[1, 2, 3],
    )
    top = results[0]
    assert top.spec.source_pattern == "*apic*"
    assert top.spec.target_pattern == "*dend*"
    assert top.spec.max_distance_um == 5.0
    assert top.spec.use_radius is True
    assert top.spec.max_syns_per_pt == 2
    assert top.score > 0.99

reference = observed_metrics_for_synapse_set("DorsalColumnSlice", "GCs__MCs", groups=groups)
plausible_dendritic = ConnectivityMetrics(
    label="plausible",
    source_group="EPLIs",
    target_group="MCs",
    entry_count=12,
    total_source_cells=2,
    total_target_cells=4,
    connected_source_cells=2,
    connected_target_cells=3,
    source_coverage=1.0,
    target_coverage=0.75,
    mean_entries_per_source_total=6.0,
    mean_entries_per_source_connected=6.0,
    mean_entries_per_target_total=3.0,
    mean_entries_per_target_connected=4.0,
    median_distance_um=3.0,
    mean_distance_um=3.1,
    p90_distance_um=4.5,
    source_family_fraction={"dend": 1.0},
    target_family_fraction={"dend": 1.0},
)
plausible_split_dendritic = ConnectivityMetrics(
    label="plausible_split",
    source_group="EPLIs",
    target_group="TCs",
    entry_count=10,
    total_source_cells=2,
    total_target_cells=4,
    connected_source_cells=2,
    connected_target_cells=2,
    source_coverage=1.0,
    target_coverage=0.5,
    mean_entries_per_source_total=5.0,
    mean_entries_per_source_connected=5.0,
    mean_entries_per_target_total=2.5,
    mean_entries_per_target_connected=5.0,
    median_distance_um=4.0,
    mean_distance_um=4.2,
    p90_distance_um=6.0,
    source_family_fraction={"dend_primary": 0.8, "dend_branch": 0.2},
    target_family_fraction={"dend": 1.0},
)
implausible_somatic = ConnectivityMetrics(
    label="implausible",
    source_group="EPLIs",
    target_group="TCs",
    entry_count=12,
    total_source_cells=2,
    total_target_cells=4,
    connected_source_cells=2,
    connected_target_cells=4,
    source_coverage=1.0,
    target_coverage=1.0,
    mean_entries_per_source_total=6.0,
    mean_entries_per_source_connected=6.0,
    mean_entries_per_target_total=3.0,
    mean_entries_per_target_connected=3.0,
    median_distance_um=3.0,
    mean_distance_um=3.1,
    p90_distance_um=4.5,
    source_family_fraction={"soma": 1.0},
    target_family_fraction={"soma": 1.0},
)
plausible_score, _ = score_epli_candidate(plausible_dendritic, reference=reference)
plausible_split_score, _ = score_epli_candidate(plausible_split_dendritic, reference=reference)
implausible_score, _ = score_epli_candidate(implausible_somatic, reference=reference)
assert plausible_score > 1.0
assert plausible_split_score > 0.5
assert implausible_score == 0.0
assert plausible_score > implausible_score
assert plausible_split_score > implausible_score

print("slice connectivity optimizer positive control: OK")
