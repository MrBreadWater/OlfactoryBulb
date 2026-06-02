"""Smoke tests for extracted spectral-analysis helpers."""

from __future__ import annotations

import numpy as np

from neuroinfra.analysis.spectral import (
    butter_bandpass_filter,
    compute_band_power_summary,
    compute_spectrogram,
    compute_wavelet_band_power,
    fold_time_matrix_by_modulus,
    fold_time_series_by_modulus,
    normalize_time_modulus,
    uniform_trace,
)


def main() -> None:
    t_uniform, y_uniform = uniform_trace([0.0, 1.0, 3.0], [0.0, 1.0, 1.0], dt_ms=1.0)
    assert np.allclose(t_uniform, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(y_uniform, np.array([0.0, 1.0, 1.0, 1.0]))

    assert normalize_time_modulus(None) is None
    assert normalize_time_modulus(-1.0) is None
    assert normalize_time_modulus(100.0) == 100.0

    folded_t, folded_y = fold_time_series_by_modulus(
        np.array([0.0, 50.0, 100.0, 150.0]),
        np.array([1.0, 3.0, 1.0, 3.0]),
        100.0,
        dt_ms=50.0,
    )
    assert np.allclose(folded_t, np.array([25.0, 75.0]))
    assert np.allclose(folded_y, np.array([1.0, 3.0]))

    folded_mt, folded_matrix = fold_time_matrix_by_modulus(
        np.array([0.0, 50.0, 100.0, 150.0]),
        np.array([[1.0, 3.0, 1.0, 3.0], [2.0, 4.0, 2.0, 4.0]]),
        100.0,
    )
    assert np.allclose(folded_mt, np.array([25.0, 75.0]))
    assert np.allclose(folded_matrix, np.array([[1.0, 3.0], [2.0, 4.0]]))

    t_ms = np.arange(0.0, 1000.0, 0.5, dtype=float)
    sine_100hz = np.sin(2.0 * np.pi * 100.0 * t_ms / 1000.0)
    filtered = butter_bandpass_filter(sine_100hz, 80.0, 120.0, fs_hz=2000.0, order=4)
    assert len(filtered) == len(sine_100hz)

    times_ms, freqs_hz, power = compute_spectrogram(
        t_ms,
        sine_100hz,
        dt_ms=0.5,
        max_freq_hz=200.0,
        nperseg=256,
        noverlap=192,
    )
    assert len(times_ms) >= 2
    assert np.max(freqs_hz) <= 200.0
    assert power.shape == (len(freqs_hz), len(times_ms))

    summary = compute_band_power_summary(
        t_ms,
        sine_100hz,
        dt_ms=0.5,
        bands={
            "off_target": (20.0, 60.0),
            "target": (90.0, 110.0),
        },
        relative_band=(20.0, 150.0),
    )
    assert summary["band_power"]["target"] > summary["band_power"]["off_target"]
    assert summary["relative_band_power"]["target"] > summary["relative_band_power"]["off_target"]

    try:
        wt_t, wt_freqs, wt_power, wt_traces = compute_wavelet_band_power(t_ms, sine_100hz, dt_ms=0.5)
    except ModuleNotFoundError:
        pass
    else:
        assert len(wt_t) > 0
        assert wt_power.shape[1] == len(wt_t)
        assert set(wt_traces) == {"beta", "low_gamma", "high_gamma"}
        assert all(len(values) == len(wt_t) for values in wt_traces.values())
        assert len(wt_freqs) == wt_power.shape[0]

    print("analysis spectral helpers: OK")


if __name__ == "__main__":
    main()
