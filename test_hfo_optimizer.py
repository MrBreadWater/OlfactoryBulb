"""Regression tests for the HFO optimizer scoring helpers."""

from __future__ import annotations

import math

import numpy as np

from olfactorybulb.hfo_optimizer import score_candidate_pair, score_condition_result


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

assert good_pair["pair_score"] > bad_pair["pair_score"]
assert good_pair["target_contrast_log10"] > 0.0

print("hfo optimizer scoring: OK")
