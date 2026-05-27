"""Regression tests for the HFO optimizer scoring helpers."""

from __future__ import annotations

import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from olfactorybulb.hfo_optimizer import (
    DEFAULT_CAMPAIGNS_BASE,
    ParameterSpec,
    propose_elite_batch,
    score_candidate_pair,
    score_condition_result,
)


home_checkout = Path.home() / "OlfactoryBulb"
if home_checkout.exists():
    assert DEFAULT_CAMPAIGNS_BASE == home_checkout / "results" / "notebook_runs" / "optimization"


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
    assert batch["local_detail_counts"]["tight_top"] >= 1
    assert batch["local_detail_counts"]["tight_top"] + batch["local_detail_counts"]["broad_weighted"] == batch["proposal_counts"]["local"]
    assert len(batch["candidates"]) == 8

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

print("hfo optimizer scoring: OK")
