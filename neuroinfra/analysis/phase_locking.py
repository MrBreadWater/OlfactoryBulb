"""Reusable phase-locking analysis helpers for resolved signals and spike rows."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import hilbert

from .spectral import compute_bandpassed_signal


def compute_phase_locking_from_spike_rows(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    spike_rows: list[tuple[str, np.ndarray | list[float]]],
    *,
    band: tuple[float, float] = (80.0, 130.0),
    dt_ms: float | None = None,
    order: int = 4,
) -> dict[str, Any]:
    """Measure phase locking between one signal and labeled spike-time rows."""
    bandpassed_t, bandpassed = compute_bandpassed_signal(
        signal_t,
        signal_y,
        dt_ms=dt_ms,
        lowcut_hz=float(band[0]),
        highcut_hz=float(band[1]),
        order=order,
    )
    if len(bandpassed_t) < 4:
        return {
            "band": tuple(float(value) for value in band),
            "n_spikes": 0,
            "vector_strength": 0.0,
            "mean_phase_rad": np.nan,
            "per_row": [],
        }

    phase = np.angle(hilbert(np.asarray(bandpassed, dtype=float)))
    unwrapped_phase = np.unwrap(phase)

    all_vectors: list[np.ndarray] = []
    per_row: list[dict[str, Any]] = []
    for label, spikes in spike_rows:
        spike_times = np.asarray(spikes, dtype=float)
        spike_times = spike_times[np.isfinite(spike_times)]
        spike_times = spike_times[
            (spike_times >= float(bandpassed_t[0]))
            & (spike_times <= float(bandpassed_t[-1]))
        ]
        if len(spike_times) == 0:
            continue

        spike_phase = np.angle(
            np.exp(1j * np.interp(spike_times, bandpassed_t, unwrapped_phase))
        )
        vectors = np.exp(1j * spike_phase)
        row_vector = np.mean(vectors)
        per_row.append(
            {
                "label": str(label),
                "n_spikes": int(len(spike_times)),
                "vector_strength": float(np.abs(row_vector)),
                "mean_phase_rad": float(np.angle(row_vector)),
            }
        )
        all_vectors.append(vectors)

    if all_vectors:
        combined = np.concatenate(all_vectors)
        mean_vector = np.mean(combined)
        vector_strength = float(np.abs(mean_vector))
        mean_phase_rad = float(np.angle(mean_vector))
        n_spikes = int(len(combined))
    else:
        vector_strength = 0.0
        mean_phase_rad = np.nan
        n_spikes = 0

    return {
        "band": tuple(float(value) for value in band),
        "n_spikes": n_spikes,
        "vector_strength": vector_strength,
        "mean_phase_rad": mean_phase_rad,
        "per_row": per_row,
    }
