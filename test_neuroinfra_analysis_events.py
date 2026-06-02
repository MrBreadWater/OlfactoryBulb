"""Smoke tests for extracted event-series analysis helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.events import (
    EventRateTrace,
    FrequencySampleCollection,
    binned_event_rate,
    build_event_overview_layout,
    calculate_event_frequency,
    collect_frequency_samples_from_rows,
    ensure_raster_axis,
    fit_raster_labels,
    overview_left_margin,
    plot_event_overview,
    plot_event_rate_traces,
    plot_event_raster_rows,
    rate_series_label,
    recommended_raster_fontsize,
    recommended_raster_height,
    recommended_raster_line_spacing,
    smooth_rate_series,
    style_raster_axis,
)


def main() -> None:
    t_freq, event_hz = calculate_event_frequency([0.0, 50.0, 100.0])
    assert np.allclose(t_freq, [25.0, 75.0])
    assert np.allclose(event_hz, [20.0, 20.0])

    freq_samples = collect_frequency_samples_from_rows(
        [
            ("MC0", np.array([0.0, 50.0, 100.0])),
            ("TC0", np.array([20.0])),
            ("MC1", np.array([10.0, 40.0, 70.0])),
        ],
        label_fn=lambda row: row[0],
        times_fn=lambda row: row[1],
        include_prefixes=("MC",),
        modulus=60.0,
    )
    assert isinstance(freq_samples, FrequencySampleCollection)
    assert freq_samples.labels == ("MC0", "MC1")
    assert len(freq_samples.rows) == 2
    assert np.allclose(freq_samples.times_ms, [25.0, 15.0, 25.0, 55.0])
    assert np.allclose(freq_samples.freqs_hz, [20.0, 20.0, 33.3333333333, 33.3333333333])

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
    assert recommended_raster_line_spacing(20) == 1.4
    assert recommended_raster_line_spacing(120) == 1.6
    assert overview_left_margin(30, per_char=0.006) > 0.22

    layout = build_event_overview_layout(
        n_rows=90,
        max_label_len=18,
        raster_min_height=4.5,
        rate_height=4.0,
    )
    assert layout.n_rows == 90
    assert layout.line_spacing == 1.6
    assert layout.total_height > layout.rate_height

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

    rate_ax = plot_event_rate_traces(
        [
            EventRateTrace(
                base_label="All inputs",
                times_ms=np.array([5.0, 15.0, 25.0]),
                rate_hz=np.array([1.0, 2.0, 1.5]),
                metadata={"normalization": "per_target_cell", "n_target_cells": 4, "unit": "events/s per target cell"},
                color="tab:blue",
            ),
            EventRateTrace(
                base_label="To MCs",
                times_ms=np.array([]),
                rate_hz=np.array([]),
                metadata={"normalization": "per_target_cell", "n_target_cells": 2, "unit": "events/s per target cell"},
                color="tab:red",
            ),
        ],
        title="Event Rate Demo",
    )
    try:
        assert rate_ax.get_title() == "Event Rate Demo"
        assert len(rate_ax.lines) == 1
        legend_texts = [text.get_text() for text in rate_ax.get_legend().get_texts()]
        assert "All inputs (n=4 cells)" in legend_texts
    finally:
        plt.close(rate_ax.figure)

    empty_rate_ax = plot_event_rate_traces([], title="Empty Rate", no_data_message="Nothing here")
    try:
        assert empty_rate_ax.get_title() == "Empty Rate"
        assert any(text.get_text() == "Nothing here" for text in empty_rate_ax.texts)
    finally:
        plt.close(empty_rate_ax.figure)

    overview_fig, overview_axes = plot_event_overview(
        layout=layout,
        raster_plotter=lambda axis, current_layout: plot_event_raster_rows(
            rows,
            ax=axis,
            title=f"Raster {current_layout.line_spacing}",
        ),
        rate_plotter=lambda axis, _layout: plot_event_rate_traces(
            [
                EventRateTrace(
                    base_label="All inputs",
                    times_ms=np.array([5.0, 15.0, 25.0]),
                    rate_hz=np.array([1.0, 2.0, 1.5]),
                    metadata={"normalization": "per_target_cell", "n_target_cells": 4, "unit": "events/s"},
                )
            ],
            ax=axis,
            title="Rate Panel",
        ),
    )
    try:
        assert overview_axes.shape == (2,)
        assert overview_axes[0].get_title().startswith("Raster")
        assert overview_axes[1].get_title() == "Rate Panel"
    finally:
        plt.close(overview_fig)

    print("analysis event helpers: OK")


if __name__ == "__main__":
    main()
