"""Smoke tests for extracted phase-locking analysis helpers."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-phase-locking-"))

import numpy as np

from neuroinfra.analysis.phase_locking import compute_phase_locking_from_spike_rows


def _demo_signal() -> tuple[np.ndarray, np.ndarray]:
    t_ms = np.arange(0.0, 1000.0, 0.1, dtype=float)
    y = np.sin(2.0 * np.pi * 100.0 * t_ms / 1000.0)
    return t_ms, y


def main() -> None:
    t_ms, signal_y = _demo_signal()
    peak_spikes = np.arange(52.5, 950.0, 10.0, dtype=float)
    trough_spikes = np.arange(57.5, 955.0, 10.0, dtype=float)

    aligned = compute_phase_locking_from_spike_rows(
        t_ms,
        signal_y,
        [("cell_a", peak_spikes), ("cell_b", peak_spikes + 10.0)],
        band=(80.0, 120.0),
        dt_ms=0.1,
    )
    assert aligned["n_spikes"] == int(len(peak_spikes) * 2)
    assert aligned["vector_strength"] > 0.95
    assert len(aligned["per_row"]) == 2
    assert all(row["vector_strength"] > 0.95 for row in aligned["per_row"])

    mixed = compute_phase_locking_from_spike_rows(
        t_ms,
        signal_y,
        [("peak", peak_spikes), ("trough", trough_spikes)],
        band=(80.0, 120.0),
        dt_ms=0.1,
    )
    assert mixed["n_spikes"] == int(len(peak_spikes) + len(trough_spikes))
    assert mixed["vector_strength"] < 0.25
    assert len(mixed["per_row"]) == 2
    assert all(row["vector_strength"] > 0.95 for row in mixed["per_row"])

    empty = compute_phase_locking_from_spike_rows(
        t_ms,
        signal_y,
        [("silent", np.array([], dtype=float))],
        band=(80.0, 120.0),
        dt_ms=0.1,
    )
    assert empty["n_spikes"] == 0
    assert empty["vector_strength"] == 0.0
    assert np.isnan(empty["mean_phase_rad"])
    assert empty["per_row"] == []

    print("analysis phase-locking helpers: OK")


if __name__ == "__main__":
    main()
