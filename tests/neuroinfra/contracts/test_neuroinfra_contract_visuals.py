"""Regression tests for generic visual contract helpers."""

from __future__ import annotations

from neuroinfra.contracts.visuals import (
    ConditionPairSpec,
    DashboardTabSpec,
    FrequencyGroupSpec,
    build_visual_contract_snapshot,
)


groups = (
    FrequencyGroupSpec(label="MT", display_label="Mitral Cell / Tufted Cell", cell_types=("MC", "TC")),
    FrequencyGroupSpec(label="GC", display_label="Granule Cell", cell_types=("GC",)),
)
pairs = (
    ConditionPairSpec(
        title="LFP spectrogram",
        control_file="control.png",
        ketamine_file="ketamine.png",
        dom_id_suffix="spectrogram",
        open_by_default=True,
    ),
)
tabs = (
    DashboardTabSpec(
        key="best",
        label="Best",
        table_heading="Top Candidates",
        packet_heading="Best Visual Packets",
    ),
    DashboardTabSpec(
        key="recent",
        label="Recent",
        table_heading="Most Recent Candidates",
        packet_heading="Recent Visual Packets",
        display_limit=5,
    ),
)

snapshot = build_visual_contract_snapshot(
    style_version=13,
    frequency_groups=groups,
    fixed_condition_pairs=pairs,
    dashboard_tabs=tabs,
    primary_psd_name_order=("03_psd_overlay.png",),
    packet_files=("03_psd_overlay.png", "04_spectrogram_control.png"),
    spectrogram_pipeline={"module": "packet_builder", "function": "save_spectrogram"},
    spectrogram_window_ms=1000.0,
    time_modulus_ms=200.0,
)

assert snapshot["style_version"] == 13
assert snapshot["frequency_groups"][0]["label"] == "MT"
assert snapshot["frequency_groups"][1]["cell_types"] == ("GC",)
assert snapshot["fixed_condition_pairs"][0]["ketamine_file"] == "ketamine.png"
assert snapshot["dashboard_tabs"][1]["display_limit"] == 5
assert snapshot["primary_psd_name_order"] == ["03_psd_overlay.png"]
assert snapshot["packet_files"] == ["03_psd_overlay.png", "04_spectrogram_control.png"]
assert snapshot["spectrogram_pipeline"]["function"] == "save_spectrogram"
assert snapshot["spectrogram_window_ms"] == 1000.0
assert snapshot["time_modulus_ms"] == 200.0
