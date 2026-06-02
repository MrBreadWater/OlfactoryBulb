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
    ResultFrequencyPlotFamily,
    coerce_frequency_plot_config,
    frequency_plot_config_with_modulus,
    plot_frequency_kde_1d_from_samples,
    plot_frequency_kde_2d_from_samples,
    plot_frequency_time_binned_from_samples,
    plot_result_frequency_kde_1d,
    plot_result_frequency_kde_2d,
    plot_result_frequency_time_binned,
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

    family = ResultFrequencyPlotFamily(
        collect_samples_fn=lambda _result, *, modulus=None, selection=None: {
            "times": np.array([10.0, 20.0, 30.0]) if modulus is None else np.mod([10.0, 20.0, 30.0], modulus),
            "freqs": np.array([40.0, 80.0, 120.0]),
        },
        selection_label_fn=lambda selection: "all" if not selection else "+".join(selection),
        title_1d="Family 1D",
        title_2d="Family 2D",
        title_time_binned="Family Binned",
    )

    fig, ax = plt.subplots()
    try:
        plot_result_frequency_kde_1d(
            {"demo": True},
            family,
            config={"max_freq_hz": 160.0, "modulus": 50.0},
            ax=ax,
            selection=("MC", "TC"),
            collector_kwargs={"selection": ("MC", "TC")},
        )
        assert ax.get_title() == "Family 1D (MC+TC)"
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_result_frequency_kde_2d(
            {"demo": True},
            family,
            config={"max_freq_hz": 160.0, "modulus": 50.0},
            ax=ax,
            selection=None,
        )
        assert ax.get_title() == "Family 2D (all)"
        assert ax.get_xlabel() == "Time (ms)"
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_result_frequency_time_binned(
            {"demo": True},
            family,
            config={"max_freq_hz": 160.0, "num_time_bins": 8},
            ax=ax,
            selection=("MC",),
        )
        assert ax.get_title() == "Family Binned (MC)"
    finally:
        plt.close(fig)

    print("analysis frequency plot helpers: OK")


if __name__ == "__main__":
    main()
