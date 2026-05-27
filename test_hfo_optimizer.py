"""Regression tests for the HFO optimizer scoring helpers."""

from __future__ import annotations

import math
import json
from tempfile import TemporaryDirectory

import numpy as np

from olfactorybulb.hfo_optimizer import (
    ParameterSpec,
    propose_elite_batch,
    score_candidate_pair,
    score_condition_result,
)


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
off_target = synthetic_result(freq_hz=90.0, amplitude=1.0, seed=2)
flat = synthetic_result(freq_hz=40.0, amplitude=0.25, noise_std=0.25, seed=3)

target_metrics = score_condition_result(target)
off_target_metrics = score_condition_result(off_target)
flat_metrics = score_condition_result(flat)

assert math.isfinite(target_metrics["condition_score"])
assert target_metrics["condition_score"] > off_target_metrics["condition_score"]
assert target_metrics["peak_hz"] > 150.0 and target_metrics["peak_hz"] < 210.0

good_pair = score_candidate_pair(control_metrics=flat_metrics, ketamine_metrics=target_metrics)
bad_pair = score_candidate_pair(control_metrics=target_metrics, ketamine_metrics=target_metrics)
reversed_pair = score_candidate_pair(control_metrics=target_metrics, ketamine_metrics=flat_metrics)
missing_control_pair = score_candidate_pair(
    control_metrics={"condition_score": float("-inf"), "relative_band_power": {}, "peak_ratio": 0.0},
    ketamine_metrics=target_metrics,
)

assert good_pair["pair_score"] > bad_pair["pair_score"]
assert bad_pair["pair_score"] > reversed_pair["pair_score"]
assert good_pair["target_contrast_log10"] > 0.0
assert good_pair["compound_contrast_log10"] > 0.0
assert bad_pair["same_peak_penalty"] > 0.0
assert bad_pair["target_delta"] == 0.0
assert reversed_pair["negative_delta_penalty"] > 0.0
assert missing_control_pair["pair_score"] == float("-inf")

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
    assert len(batch["candidates"]) == 8

print("hfo optimizer scoring: OK")
