"""Smoke tests for extracted frequency-sample plotting helpers."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-frequency-plots-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.frequency_plots import (
    FrequencyPlotConfig,
    coerce_frequency_plot_config,
    frequency_plot_config_with_modulus,
    plot_frequency_kde_1d_from_samples,
    plot_frequency_kde_2d_from_samples,
    plot_frequency_time_binned_from_samples,
)


def main() -> None:
    base = coerce_frequency_plot_config({"max_freq_hz": 150.0, "num_time_bins": 16}, dot_alpha=0.4)
    assert isinstance(base, FrequencyPlotConfig)
    assert base.max_freq_hz == 150.0
    assert base.num_time_bins == 16
    assert base.dot_alpha == 0.4

    modded = frequency_plot_config_with_modulus(base, 250.0)
    assert modded.modulus == 250.0
    assert modded.max_freq_hz == base.max_freq_hz

    freqs = np.array([40.0, 42.0, 44.0, 80.0, 82.0, 120.0], dtype=float)
    times = np.array([10.0, 15.0, 20.0, 60.0, 70.0, 120.0], dtype=float)

    fig, ax = plt.subplots()
    try:
        plot_frequency_kde_1d_from_samples(freqs, config=base, title="1D KDE", ax=ax)
        assert ax.get_title() == "1D KDE"
        assert ax.get_xlabel() == "Frequency (Hz)"
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_frequency_kde_2d_from_samples(times, freqs, config=base, title="2D KDE", ax=ax)
        assert ax.get_title() == "2D KDE"
        assert ax.get_xlabel() == "Time (ms)"
        assert ax.get_ylabel() == "Frequency (Hz)"
        assert len(ax.figure.axes) == 2
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_frequency_time_binned_from_samples(
            times,
            freqs,
            config=base,
            title="Time Binned",
            ax=ax,
            show_dots=True,
            show_ridgeline_kde=True,
        )
        assert ax.get_title() == "Time Binned"
        assert ax.get_xlabel() == "Time (ms)"
        assert ax.get_ylabel() == "Frequency (Hz)"
        assert len(ax.collections) > 0
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_frequency_kde_1d_from_samples(np.array([], dtype=float), config=base, title="Empty", ax=ax)
        assert ax.get_title() == "Empty"
        assert any(text.get_text() == "No frequency samples" for text in ax.texts)
    finally:
        plt.close(fig)

    print("analysis frequency plot helpers: OK")


if __name__ == "__main__":
    main()
