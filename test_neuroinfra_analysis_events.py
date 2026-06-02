"""Smoke tests for extracted event-series analysis helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.events import (
    binned_event_rate,
    calculate_event_frequency,
    ensure_raster_axis,
    fit_raster_labels,
    plot_event_raster_rows,
    rate_series_label,
    recommended_raster_fontsize,
    recommended_raster_height,
    smooth_rate_series,
    style_raster_axis,
)


def main() -> None:
    t_freq, event_hz = calculate_event_frequency([0.0, 50.0, 100.0])
    assert np.allclose(t_freq, [25.0, 75.0])
    assert np.allclose(event_hz, [20.0, 20.0])

    raw_rate = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=float)
    smoothed = smooth_rate_series(raw_rate, bin_ms=5.0, smooth_sigma_ms=10.0)
    assert smoothed.shape == raw_rate.shape
    assert float(smoothed[2]) < 10.0
    assert float(smoothed[1]) > 0.0

    centers, rate_hz = binned_event_rate(
        [np.array([10.0, 20.0, 30.0]), np.array([15.0, 25.0])],
        t_stop=40.0,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        denominator=2.0,
    )
    assert np.allclose(centers, [5.0, 15.0, 25.0, 35.0])
    assert np.allclose(rate_hz, [0.0, 100.0, 100.0, 50.0])

    assert rate_series_label("Inputs", {"normalization": "per_target_cell", "n_target_cells": 7}) == "Inputs (n=7 cells)"
    assert recommended_raster_fontsize(40) == 7.0
    assert recommended_raster_fontsize(100) == 6.0
    assert recommended_raster_height(0, min_height=4.0) == 4.0
    assert recommended_raster_height(100, min_height=4.0) > 4.0

    fig, ax = plt.subplots(figsize=(4, 3))
    offsets = style_raster_axis(
        ax,
        ["cell_a", "cell_b"],
        ylabel="Cell",
        title="Raster Demo",
        fontsize=7.0,
        line_spacing=1.5,
    )
    assert np.allclose(offsets, [0.0, 1.5])
    fit_raster_labels(ax, offsets, min_height=3.0)
    plt.close(fig)

    auto_ax = ensure_raster_axis(None, 5, width=6.0, min_height=3.5, per_row_height=0.2)
    try:
        auto_fig = auto_ax.figure
        width, height = auto_fig.get_size_inches()
        assert np.isclose(width, 6.0)
        assert height >= 3.5
    finally:
        plt.close(auto_ax.figure)

    rows = [
        ("cell_a", np.array([0.0, 20.0, 40.0])),
        ("cell_b", np.array([10.0, 30.0, 50.0])),
    ]
    raster_ax = plot_event_raster_rows(
        rows,
        ylabel="Cell",
        title="Generic Raster",
        modulus=50.0,
        width=5.0,
        min_height=3.0,
        per_row_height=0.15,
        colors=["tab:blue", "tab:red"],
    )
    try:
        assert raster_ax.get_title() == "Generic Raster"
        assert "modulo 50" in raster_ax.get_xlabel()
        assert len(raster_ax.collections) > 0
    finally:
        plt.close(raster_ax.figure)

    empty_ax = plot_event_raster_rows([], title="Unused", no_data_message="No rows")
    try:
        assert empty_ax.get_title() == "No rows"
    finally:
        plt.close(empty_ax.figure)

    print("analysis event helpers: OK")


if __name__ == "__main__":
    main()
