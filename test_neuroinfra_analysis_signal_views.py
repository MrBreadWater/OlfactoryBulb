"""Smoke tests for extracted named-signal view helpers."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-signal-views-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.signal_views import (
    SignalPsdOverlay,
    compute_resolved_bandpassed_signal,
    compute_resolved_band_power_summary,
    log_spectrogram_display_power,
    plot_resolved_band_power_summary,
    plot_resolved_signal,
    plot_resolved_signal_psd_overview,
    plot_resolved_spectrogram,
    plot_resolved_wavelet,
    plot_resolved_wavelet_band_power,
)


def _resolve_signal(result: dict[str, object], signal: str, dt_ms: float | None) -> tuple[np.ndarray, np.ndarray]:
    assert signal == "demo"
    del dt_ms
    return np.asarray(result["t"], dtype=float), np.asarray(result["y"], dtype=float)


def main() -> None:
    power = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=float)
    display = log_spectrogram_display_power(power)
    assert display.shape == power.shape
    assert np.isclose(display.min(), 0.0)

    result = {
        "t": np.arange(0.0, 400.0, 1.0),
        "y": np.sin(np.linspace(0.0, 16.0 * np.pi, 400)),
    }

    bp_t, bp_y = compute_resolved_bandpassed_signal(
        result,
        signal="demo",
        resolve_signal_fn=_resolve_signal,
        dt_ms=1.0,
        lowcut_hz=10.0,
        highcut_hz=80.0,
    )
    assert len(bp_t) == len(result["t"])
    assert len(bp_y) == len(result["y"])

    summary = compute_resolved_band_power_summary(
        result,
        signal="demo",
        resolve_signal_fn=_resolve_signal,
        dt_ms=1.0,
        bands={"low": (5.0, 20.0), "high": (20.0, 80.0)},
        relative_band=(5.0, 100.0),
    )
    assert summary["signal"] == "demo"
    assert set(summary["band_power"]) == {"low", "high"}

    signal_ax = plot_resolved_signal(result, signal="demo", resolve_signal_fn=_resolve_signal, dt_ms=1.0)
    try:
        assert signal_ax.get_title() == "demo Trace"
        assert len(signal_ax.lines) == 1
    finally:
        plt.close(signal_ax.figure)

    overview_fig, overview_axes = plot_resolved_signal_psd_overview(
        result,
        signal="demo",
        resolve_signal_fn=_resolve_signal,
        dt_ms=1.0,
        lowcut_hz=10.0,
        highcut_hz=80.0,
        signal_label="Demo Signal",
        psd_overlay_builder=lambda freqs, power: [
            SignalPsdOverlay(
                freqs_hz=freqs,
                power=power * 0.5,
                label="reference",
                color="tab:orange",
            )
        ],
    )
    try:
        assert len(overview_axes) == 3
        assert overview_axes[0].get_title() == "Raw Demo Signal"
        assert overview_axes[1].get_title() == "Band-passed Demo Signal (10-80 Hz)"
        legend = overview_axes[2].get_legend()
        assert legend is not None
        assert {text.get_text() for text in legend.texts} == {"Measured PSD", "reference"}
    finally:
        plt.close(overview_fig)

    spectrogram_ax = plot_resolved_spectrogram(
        result,
        signal="demo",
        resolve_signal_fn=_resolve_signal,
        dt_ms=1.0,
        nperseg=64,
        noverlap=48,
    )
    try:
        assert spectrogram_ax.get_title() == "DEMO Spectrogram"
        assert len(spectrogram_ax.collections) > 0
    finally:
        plt.close(spectrogram_ax.figure)

    try:
        wavelet_ax = plot_resolved_wavelet(
            result,
            signal="demo",
            resolve_signal_fn=_resolve_signal,
            dt_ms=1.0,
        )
    except ModuleNotFoundError as exc:
        assert "PyWavelets" in str(exc)
    else:
        try:
            assert wavelet_ax.get_title() == "DEMO Wavelet Power"
            assert len(wavelet_ax.collections) > 0
        finally:
            plt.close(wavelet_ax.figure)

    try:
        band_power_ax = plot_resolved_wavelet_band_power(
            result,
            signal="demo",
            resolve_signal_fn=_resolve_signal,
            dt_ms=1.0,
        )
    except ModuleNotFoundError as exc:
        assert "PyWavelets" in str(exc)
    else:
        try:
            assert band_power_ax.get_title() == "Band Power Over Time"
            assert len(band_power_ax.lines) > 0
        finally:
            plt.close(band_power_ax.figure)

    band_fig, band_axes, plotted_summary = plot_resolved_band_power_summary(
        result,
        signal="demo",
        resolve_signal_fn=_resolve_signal,
        dt_ms=1.0,
        bands={"low": (5.0, 20.0), "high": (20.0, 80.0)},
        relative_band=(5.0, 100.0),
        signal_label="demo",
    )
    try:
        assert plotted_summary["signal"] == "demo"
        assert band_axes[0].get_title() == "demo Band Power"
        assert band_axes[1].get_title() == "Relative Band Power"
    finally:
        plt.close(band_fig)

    print("analysis signal view helpers: OK")


if __name__ == "__main__":
    main()
