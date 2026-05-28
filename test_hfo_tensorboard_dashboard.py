"""Regression checks for the HFO TensorBoard sidecar exporter."""

from __future__ import annotations

from tools.analysis.hfo_tensorboard_dashboard import (
    _top_candidate_table_rows,
    collect_scalar_records,
)


row = {
    "candidate_id": "C00042",
    "batch_name": "batch_0003",
    "pair_score": 2.5,
    "target_delta": 0.07,
    "control_target_excess_penalty": 0.1,
    "parameters": {
        "kar_mt_gmax": 0.03,
        "gaba_gmax": 1.2,
        "optimizer_candidate_id": "C00042",
    },
    "control_metrics": {
        "condition_score": 0.4,
        "peak_hz": 95.0,
        "relative_band_power": {
            "target_hfo": 0.03,
            "high_gamma": 0.12,
        },
        "band_power": {
            "target_hfo": 1.0,
            "high_gamma": 2.0,
        },
        "mean_firing_rate_by_type": {
            "TC": 4.0,
            "EPLI": 2.0,
        },
    },
    "ketamine_metrics": {
        "condition_score": 2.0,
        "peak_hz": 190.0,
        "relative_band_power": {
            "target_hfo": 0.10,
            "high_gamma": 0.08,
        },
        "band_power": {
            "target_hfo": 4.0,
            "high_gamma": 1.5,
        },
        "mean_firing_rate_by_type": {
            "TC": 12.0,
            "EPLI": 9.0,
        },
    },
}

records = collect_scalar_records([row])
by_tag = {record.tag: record for record in records}

assert by_tag["score/pair_score"].step == 42
assert by_tag["score/pair_score"].value == 2.5
assert by_tag["band_relative/ketamine/target_hfo"].value == 0.10
assert by_tag["band_relative/control/high_gamma"].value == 0.12
assert by_tag["band_relative_delta/target_hfo"].value == 0.07
assert by_tag["rate_hz/ketamine/EPLI"].value == 9.0
assert by_tag["param/kar_mt_gmax"].value == 0.03
assert by_tag["best_so_far/pair_score"].value == 2.5

next_row = {**row, "candidate_id": "C00043", "pair_score": 3.0, "target_delta": 0.08}
incremental_records = collect_scalar_records([row, next_row], start_index=1)
assert {record.step for record in incremental_records} == {43}
incremental_by_tag = {record.tag: record for record in incremental_records}
assert incremental_by_tag["best_so_far/pair_score"].value == 3.0
assert incremental_by_tag["best_so_far/target_delta"].value == 0.08

table_rows = _top_candidate_table_rows([row], top_n=1)
assert table_rows[0]["candidate_id"] == "C00042"
assert table_rows[0]["ketamine_high_gamma_rel"] == 0.08
assert table_rows[0]["control_high_gamma_rel"] == 0.12
assert table_rows[0]["param_gaba_gmax"] == 1.2
