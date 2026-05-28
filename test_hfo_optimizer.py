"""Regression tests for the HFO optimizer scoring helpers."""

from __future__ import annotations

import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import obgpu_experiment_helpers as hlp
from olfactorybulb.hfo_optimizer import (
    DEFAULT_CAMPAIGNS_BASE,
    PAIR_SCORE_VERSION,
    ParameterSpec,
    default_campaign_run_config,
    default_hfo_search_space,
    load_candidate_archive_rows,
    lfp_source_diagnostic_configs,
    parameter_plausibility_penalty,
    propose_elite_batch,
    score_candidate_pair,
    score_condition_result,
    score_hfo_batch,
    sustained_odor_schedule,
    window_result_for_condition,
    write_objective_filter,
)


home_checkout = Path.home() / "OlfactoryBulb"
if home_checkout.exists():
    assert DEFAULT_CAMPAIGNS_BASE == home_checkout / "results" / "notebook_runs" / "optimization"

default_paths = [spec.path for spec in default_hfo_search_space()]
for required_path in (
    "epli_ampa_weight_scale",
    "epli_gaba_weight_scale",
    "kar_osn_weight_scale",
    "kar_gc_weight_scale",
    "gc_ka_gbar_scale",
):
    assert required_path in default_paths
default_specs = {spec.path: spec for spec in default_hfo_search_space()}
assert default_specs["kar_mt_gmax"].high == 0.08
assert default_specs["kar_gc_gmax"].high == 0.025
assert default_specs["kar_osn_weight_scale"].high == 2.0
assert default_specs["kar_gc_weight_scale"].high == 4.0

schedule = sustained_odor_schedule(9000.0)
assert min(schedule) == 0
assert max(schedule) == 8800
assert len(schedule) == 45
campaign_config = default_campaign_run_config({}, tstop_ms=9000.0)
assert campaign_config["input_odors"] == schedule
assert campaign_config["inhale_duration_ms"] == 125.0
assert campaign_config["record_gc_output_events"] is False
assert campaign_config["save_soma_traces"] is False
assert campaign_config["save_voltage_summary"] is False
assert hlp.build_run_config()["remote_ssh_command_timeout_s"] == 300
assert hlp.build_run_config()["remote_ssh_exec_timeout_s"] == 30
assert hlp.build_run_config()["remote_poll_command_timeout_s"] == 60
assert hlp._remote_ssh_command_timeout_s({"remote_ssh_command_timeout_s": 0}) is None
assert hlp._remote_ssh_exec_timeout_s({"remote_ssh_exec_timeout_s": 0}) is None
assert hlp._remote_poll_command_timeout_s({"remote_poll_command_timeout_s": None}) == 300.0

lfp_filter_config = hlp.build_run_config(
    lfp_include_cell_types=["MC", "TC"],
    lfp_exclude_cell_types=["GC"],
    save_soma_traces=False,
    save_voltage_summary=False,
)
lfp_filter_overrides = hlp.build_param_overrides(lfp_filter_config)
assert lfp_filter_overrides["lfp_include_cell_types"] == ["MC", "TC"]
assert lfp_filter_overrides["lfp_exclude_cell_types"] == ["GC"]
assert lfp_filter_overrides["save_soma_traces"] is False
assert lfp_filter_overrides["save_voltage_summary"] is False
switch_config = hlp.build_run_config(
    ketamine_block=1.0,
    ketamine_switch_time_ms=4500.0,
    ketamine_block_after_switch=0.0,
)
switch_overrides = hlp.build_param_overrides(switch_config)
switch_synapse_overrides = switch_overrides["synapse_properties"]["AmpaNmdaSyn"]
assert switch_synapse_overrides["ketamine_block"] == 1.0
assert switch_synapse_overrides["ketamine_switch_time"] == 4500.0
assert switch_synapse_overrides["ketamine_block_after"] == 0.0
switch_default_after = hlp.build_param_overrides(
    hlp.build_run_config(ketamine_switch_time_ms=4500.0)
)["synapse_properties"]["AmpaNmdaSyn"]
assert switch_default_after["ketamine_block_after"] == 0.0
assert hlp.cell_type_of("PVCRH_FSI1[0].soma") == "EPLI"
lfp_diagnostics = lfp_source_diagnostic_configs(
    lfp_filter_config,
    shifted_locations=([116, 900, -61], [116, 1250, -61]),
)
assert lfp_diagnostics["exclude_gc_lfp"]["lfp_exclude_cell_types"] == ["GC"]
assert lfp_diagnostics["non_gc_sources_lfp"]["lfp_include_cell_types"][:2] == ["MC", "TC"]
assert lfp_diagnostics["probe_shift_00"]["lfp_electrode_location"] == [116.0, 900.0, -61.0]


def synthetic_result(
    *,
    freq_hz: float,
    amplitude: float = 1.0,
    noise_std: float = 0.15,
    duration_ms: float = 2000.0,
    dt_ms: float = 0.1,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_ms, dt_ms)
    y = amplitude * np.sin(2.0 * np.pi * float(freq_hz) * t / 1000.0)
    y = y + rng.normal(scale=noise_std, size=len(t))
    return {
        "lfp_t": t,
        "lfp": y,
        "soma_spikes": {"labels": [], "spike_times": [], "metadata": {}},
        "summary": {"params": {"tstop": duration_ms}},
    }


target = synthetic_result(freq_hz=180.0, amplitude=1.0, seed=1)
upper_target = synthetic_result(freq_hz=220.0, amplitude=1.0, seed=4)
lower_edge_target = synthetic_result(freq_hz=161.0, amplitude=1.0, seed=5)
off_target = synthetic_result(freq_hz=90.0, amplitude=1.0, seed=2)
flat = synthetic_result(freq_hz=40.0, amplitude=0.25, noise_std=0.25, seed=3)

switch_t = np.arange(0.0, 4000.0, 0.1)
switch_lfp = np.where(
    switch_t < 2000.0,
    0.25 * np.sin(2.0 * np.pi * 40.0 * switch_t / 1000.0),
    np.sin(2.0 * np.pi * 180.0 * switch_t / 1000.0),
)
switch_result = {
    "lfp_t": switch_t,
    "lfp": switch_lfp,
    "input_times": [("input", np.arange(0.0, 4000.0, 100.0))],
    "soma_spikes": {
        "labels": ["EPLI0[0].soma"],
        "spike_times": [np.arange(100.0, 3900.0, 100.0)],
        "metadata": {},
    },
    "summary": {"params": {"tstop": 4000.0}},
}
ketamine_window = window_result_for_condition(
    switch_result,
    start_ms=2200.0,
    stop_ms=4000.0,
    condition="ketamine",
)
assert ketamine_window["summary"]["params"]["tstop"] == 1800.0
assert np.isclose(ketamine_window["lfp_t"][0], 0.0)
assert np.all(ketamine_window["soma_spikes"]["spike_times"][0] >= 0.0)

with TemporaryDirectory() as tmpdir:
    switch_batch_plan = {
        "batch_name": "batch_switch",
        "strategy": "test",
        "stage": "test",
        "candidates": [{"optimizer_candidate_id": "Cswitch", "kar_mt_gmax": 0.02}],
    }
    switch_sweep = {
        "sweep_dir": tmpdir,
        "items": [
            {
                "value": {
                    "optimizer_candidate_id": "Cswitch",
                    "optimizer_pair_id": "Cswitch",
                    "optimizer_condition": "switch",
                    "ketamine_switch_time_ms": 2000.0,
                    "ketamine_switch_washout_ms": 200.0,
                },
                "label": "switch_item",
                "result": switch_result,
                "run": None,
            }
        ],
    }
    switch_scored = score_hfo_batch(
        tmpdir,
        batch_plan=switch_batch_plan,
        sweep=switch_sweep,
        target_hz=180.0,
        target_half_width_hz=20.0,
        switch_washout_ms=200.0,
    )
    assert [row["condition"] for row in switch_scored["item_rows"]] == ["control", "ketamine"]
    switch_candidate = switch_scored["candidate_rows"][0]
    assert switch_candidate["ketamine_metrics"]["relative_band_power"]["target_hfo"] > (
        switch_candidate["control_metrics"]["relative_band_power"]["target_hfo"]
    )
    assert math.isfinite(switch_candidate["pair_score"])

target_metrics = score_condition_result(target)
upper_target_metrics = score_condition_result(upper_target, target_hz=180.0, target_half_width_hz=20.0)
lower_edge_metrics = score_condition_result(lower_edge_target)
off_target_metrics = score_condition_result(off_target)
flat_metrics = score_condition_result(flat)

assert math.isfinite(target_metrics["condition_score"])
assert target_metrics["condition_score"] > off_target_metrics["condition_score"]
assert target_metrics["peak_hz"] > 150.0 and target_metrics["peak_hz"] < 210.0
assert target_metrics["target_peak_contrast"] > 1.0
assert target_metrics["target_centroid_match"] > lower_edge_metrics["target_centroid_match"]
assert upper_target_metrics["target_band_hz"] == [160.0, 230.0]
assert upper_target_metrics["peak_hz"] > 210.0 and upper_target_metrics["peak_hz"] < 230.0

active_epli_target_metrics = {**target_metrics, "epli_rate_hz": 5.0}
silent_epli_target_metrics = {**target_metrics, "epli_rate_hz": 0.0}
low_contrast_target_metrics = {**active_epli_target_metrics, "target_peak_contrast": 1.0, "peak_ratio": 1.0}
good_pair = score_candidate_pair(control_metrics=flat_metrics, ketamine_metrics=active_epli_target_metrics)
edge_pair = score_candidate_pair(control_metrics=flat_metrics, ketamine_metrics=lower_edge_metrics)
silent_pair = score_candidate_pair(control_metrics=flat_metrics, ketamine_metrics=silent_epli_target_metrics)
low_contrast_pair = score_candidate_pair(control_metrics=flat_metrics, ketamine_metrics=low_contrast_target_metrics)
bad_pair = score_candidate_pair(control_metrics=target_metrics, ketamine_metrics=target_metrics)
upper_bad_pair = score_candidate_pair(control_metrics=upper_target_metrics, ketamine_metrics=upper_target_metrics)
reversed_pair = score_candidate_pair(control_metrics=target_metrics, ketamine_metrics=flat_metrics)
artifact_control_metrics = {
    **flat_metrics,
    "condition_score": 2.0,
    "relative_band_power": {
        **flat_metrics["relative_band_power"],
        "target_hfo": 0.08,
        "supra_hfo": 0.35,
    },
}
artifact_pair = score_candidate_pair(control_metrics=artifact_control_metrics, ketamine_metrics=target_metrics)
leaky_control_metrics = {
    **flat_metrics,
    "relative_band_power": {
        **flat_metrics["relative_band_power"],
        "target_hfo": 0.14,
    },
}
leaky_pair = score_candidate_pair(control_metrics=leaky_control_metrics, ketamine_metrics=target_metrics)
missing_control_pair = score_candidate_pair(
    control_metrics={"condition_score": float("-inf"), "relative_band_power": {}, "peak_ratio": 0.0},
    ketamine_metrics=target_metrics,
)

assert good_pair["pair_score"] > bad_pair["pair_score"]
assert good_pair["pair_score"] > edge_pair["pair_score"]
assert good_pair["pair_score"] > silent_pair["pair_score"]
assert good_pair["pair_score"] > low_contrast_pair["pair_score"]
assert good_pair["pair_score"] > artifact_pair["pair_score"]
assert good_pair["control_target_excess_penalty"] == 0.0
assert bad_pair["pair_score"] > reversed_pair["pair_score"]
assert good_pair["target_contrast_log10"] > 0.0
assert good_pair["compound_contrast_log10"] > 0.0
assert bad_pair["same_peak_penalty"] > 0.0
assert upper_bad_pair["same_peak_penalty"] > 0.0
assert upper_bad_pair["pair_score_version"] == 6
assert PAIR_SCORE_VERSION == 6
assert "psd_shape_power" in target_metrics
assert len(target_metrics["psd_shape_power"]) > 10
assert good_pair["psd_template_loss"] < bad_pair["psd_template_loss"]
assert good_pair["psd_contrast_template_loss"] < bad_pair["psd_contrast_template_loss"]
assert bad_pair["control_hfo_template_similarity"] > good_pair["control_hfo_template_similarity"]
assert leaky_pair["control_target_excess_penalty"] > 0.0
assert edge_pair["ketamine_center_penalty"] > 0.0
assert silent_pair["ketamine_epli_silence_penalty"] > 0.0
assert low_contrast_pair["ketamine_peak_contrast_penalty"] > 0.0
assert bad_pair["target_delta"] == 0.0
assert artifact_pair["control_wrong_band_penalty"] > 0.0
assert reversed_pair["negative_delta_penalty"] > 0.0
assert missing_control_pair["pair_score"] == float("-inf")

plausible_penalty, plausible_components = parameter_plausibility_penalty(
    {"kar_mt_gmax": 0.03, "kar_osn_weight_scale": 1.0, "kar_gc_gmax": 0.008}
)
implausible_penalty, implausible_components = parameter_plausibility_penalty(
    {"kar_mt_gmax": 276.0, "kar_osn_weight_scale": 5.2, "kar_gc_gmax": 0.004}
)
assert plausible_penalty == 0.0
assert plausible_components == {}
assert implausible_penalty > 100.0
assert "kar_mt_effective_drive" in implausible_components

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_mt_gmax", low=0.01, high=100.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(16):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(16 - index),
                "parameters": {
                    "kar_mt_gmax": 0.02 + index,
                    "gaba_gmax": 0.2 + 0.2 * index,
                    "tc_input_weight": 0.5 + 0.02 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=8,
        seed=42,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["strategy"] == "elite_truncated_gaussian_plus_lhs"
    assert batch["local_source_ids"] == ["C00000", "C00001", "C00002", "C00003"]
    assert sum(batch["proposal_counts"].values()) == 8
    assert batch["local_detail_counts"]["tight_top"] >= 1
    assert batch["local_detail_counts"]["tight_top"] + batch["local_detail_counts"]["broad_weighted"] == batch["proposal_counts"]["local"]
    assert len(batch["candidates"]) == 8

with TemporaryDirectory() as tmpdir:
    rows = [
        {
            "batch_name": "batch_0001",
            "candidate_id": "C00001",
            "pair_score": 10.0,
            "parameters": {},
        },
        {
            "batch_name": "batch_0052",
            "candidate_id": "C00052",
            "pair_score": 1.0,
            "parameters": {},
        },
    ]
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_objective_filter(tmpdir, {"min_batch_index": 52})
    filtered_rows = load_candidate_archive_rows(tmpdir)
    assert [row["candidate_id"] for row in filtered_rows] == ["C00052"]

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_mt_gmax", low=0.01, high=100.0, scale="log"),
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(200):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(200 - index),
                "parameters": {
                    "kar_mt_gmax": 0.02 + 0.01 * index,
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=11,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 4
    assert batch["proposal_counts"]["explore"] == 2
    assert batch["targeted_detail"]["top_pair"] == ["C00000", "C00001"]
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="epli_ampa_weight_scale", low=0.1, high=8.0, scale="log", default=1.0),
        ParameterSpec(path="epli_gaba_weight_scale", low=0.1, high=8.0, scale="log", default=1.0),
        ParameterSpec(path="gap_tc", low=4.0, high=64.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(200):
        rows.append(
            {
                "batch_name": "batch_0052",
                "candidate_id": f"C{index:05d}",
                "pair_score": float(200 - index),
                "ketamine_metrics": {
                    "peak_hz": 180.0,
                    "relative_band_power": {"target_hfo": 0.08 + 0.0001 * index},
                    "target_peak_contrast": 0.2 + 0.001 * index,
                    "mean_firing_rate_by_type": {"EPLI": 3.0},
                },
                "control_metrics": {
                    "peak_hz": 120.0,
                    "relative_band_power": {"target_hfo": 0.05},
                    "target_peak_contrast": 0.1,
                    "mean_firing_rate_by_type": {"EPLI": 2.0},
                },
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "epli_ampa_weight_scale": 1.0,
                    "epli_gaba_weight_scale": 1.0,
                    "gap_tc": 8.0 + 0.05 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_objective_filter(tmpdir, {"min_batch_index": 52, "target_hfo_hz": [160.0, 230.0]})

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=111,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 12
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "frontier"
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_mt_gmax", low=0.01, high=100.0, scale="log"),
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(224):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(224 - index),
                "parameters": {
                    "kar_mt_gmax": 0.02 + 0.01 * index,
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=12,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 8
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "stencil"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 8
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_mt_gmax", low=0.01, high=100.0, scale="log"),
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="gap_tc", low=4.0, high=64.0, scale="log"),
        ParameterSpec(path="kar_gc_weight_scale", low=0.5, high=6.0, scale="log"),
        ParameterSpec(path="gc_ka_gbar_scale", low=0.25, high=3.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(256):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(256 - index),
                "parameters": {
                    "kar_mt_gmax": 0.02 + 0.01 * index,
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=13,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 10
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "combo"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 10
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="gap_tc", low=4.0, high=64.0, scale="log"),
        ParameterSpec(path="kar_gc_weight_scale", low=0.5, high=6.0, scale="log"),
        ParameterSpec(path="gc_ka_gbar_scale", low=0.25, high=3.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(288):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(288 - index),
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=14,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 12
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "micro"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 12
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="gap_tc", low=4.0, high=64.0, scale="log"),
        ParameterSpec(path="kar_gc_weight_scale", low=0.5, high=6.0, scale="log"),
        ParameterSpec(path="gc_ka_gbar_scale", low=0.25, high=3.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(320):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(320 - index),
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=15,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 11
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "ridge"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 11
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

    for index in range(320, 368):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(320 - index) / 10.0,
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=16,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 12
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "needle"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 12
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

    for index in range(368, 416):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(368 - index) / 10.0,
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=17,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 12
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "basin"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 12
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="epli_ampa_weight_scale", low=0.1, high=8.0, scale="log", default=1.0),
        ParameterSpec(path="epli_gaba_weight_scale", low=0.1, high=8.0, scale="log", default=1.0),
        ParameterSpec(path="gap_tc", low=4.0, high=64.0, scale="log"),
        ParameterSpec(path="kar_gc_weight_scale", low=0.5, high=6.0, scale="log"),
        ParameterSpec(path="gc_ka_gbar_scale", low=0.25, high=3.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(448):
        ket_peak = 180.0 if index % 3 else 161.0
        control_target = 0.06 + 0.0001 * (index % 100)
        ket_target = 0.13 + 0.0002 * (index % 100)
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(448 - index),
                "ketamine_metrics": {
                    "peak_hz": ket_peak,
                    "relative_band_power": {"target_hfo": ket_target},
                },
                "control_metrics": {
                    "peak_hz": 195.0,
                    "relative_band_power": {"target_hfo": control_target},
                },
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=18,
        method="elite_truncated_gaussian_plus_lhs",
    )
    assert batch["proposal_counts"]["targeted"] == 12
    assert batch["proposal_counts"]["explore"] == 1
    assert batch["targeted_detail"]["mode"] == "frontier"
    assert batch["targeted_detail"]["coordinate_probe_count"] == 12
    assert sum(batch["proposal_counts"].values()) == 16
    assert len(batch["candidates"]) == 16
    assert any(candidate["epli_ampa_weight_scale"] > 1.0 for candidate in batch["candidates"])
    assert any(candidate["epli_gaba_weight_scale"] > 1.0 for candidate in batch["candidates"])

with TemporaryDirectory() as tmpdir:
    search_space = [
        ParameterSpec(path="kar_gc_gmax", low=0.001, high=10.0, scale="log"),
        ParameterSpec(path="gaba_gmax", low=0.1, high=10.0, scale="log"),
        ParameterSpec(path="ampa_nmda_gmax", low=16.0, high=128.0, scale="log"),
        ParameterSpec(path="epli_ampa_weight_scale", low=0.1, high=8.0, scale="log", default=1.0),
        ParameterSpec(path="epli_gaba_weight_scale", low=0.1, high=8.0, scale="log", default=1.0),
        ParameterSpec(path="gap_tc", low=4.0, high=64.0, scale="log"),
        ParameterSpec(path="kar_gc_weight_scale", low=0.5, high=6.0, scale="log"),
        ParameterSpec(path="gc_ka_gbar_scale", low=0.25, high=3.0, scale="log"),
        ParameterSpec(path="tc_input_weight", low=0.4, high=1.2, scale="linear"),
    ]
    state_path = f"{tmpdir}/state.json"
    with open(state_path, "w") as handle:
        json.dump({"next_batch_index": 0, "next_candidate_index": 0, "completed_batches": []}, handle)
    rows = []
    for index in range(448):
        rows.append(
            {
                "candidate_id": f"C{index:05d}",
                "pair_score": float(448 - index),
                "parameters": {
                    "kar_gc_gmax": 0.002 + 0.001 * index,
                    "gaba_gmax": 0.2 + 0.02 * index,
                    "ampa_nmda_gmax": 20.0 + 0.1 * index,
                    "epli_ampa_weight_scale": 1.0,
                    "epli_gaba_weight_scale": 1.0,
                    "gap_tc": 8.0 + 0.05 * index,
                    "kar_gc_weight_scale": 0.6 + 0.01 * index,
                    "gc_ka_gbar_scale": 0.4 + 0.005 * index,
                    "tc_input_weight": 0.5 + 0.001 * index,
                },
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    first_batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=19,
        method="elite_truncated_gaussian_plus_lhs",
    )
    for candidate in first_batch["candidates"]:
        rows.append(
            {
                "candidate_id": candidate["optimizer_candidate_id"],
                "pair_score": 100.0,
                "parameters": candidate,
            }
        )
    with open(f"{tmpdir}/candidate_archive.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    second_batch = propose_elite_batch(
        tmpdir,
        search_space=search_space,
        n_candidates=16,
        seed=19,
        method="elite_truncated_gaussian_plus_lhs",
    )

    def signature(params):
        return tuple(round(spec.encode(params.get(spec.path, spec.default_value())), 10) for spec in search_space)

    archived_signatures = {signature(row["parameters"]) for row in rows}
    proposed_signatures = [signature(candidate) for candidate in second_batch["candidates"]]
    assert not archived_signatures.intersection(proposed_signatures)
    assert second_batch["targeted_detail"]["archive_duplicate_rows_dropped"] > 0

print("hfo optimizer scoring: OK")
