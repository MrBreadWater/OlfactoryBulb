"""Smoke tests for extracted event-series analysis helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.events import (
    EventRateTrace,
    EventRateSeriesSpec,
    EventRateNormalizationRule,
    FrequencySampleCollection,
    PreparedEventRows,
    ResultEventFamilySpec,
    binned_event_rate,
    build_event_overview_layout,
    build_event_overview_layout_for_rows,
    build_event_rate_trace_series,
    calculate_event_frequency,
    calculate_trace_event_frequency,
    compute_event_rate_from_rows,
    compute_result_event_family_rate,
    collect_frequency_samples_from_rows,
    collect_frequency_samples_from_trace_rows,
    collect_result_event_family_samples,
    ensure_raster_axis,
    filter_result_event_family_rows,
    filter_rows_by_label_prefix,
    fit_raster_labels,
    overview_left_margin,
    prepare_event_display_rows,
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

    trace_t_freq, trace_event_hz = calculate_trace_event_frequency(
        [0.0, 10.0, 20.0, 30.0, 40.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        event_times_fn=lambda t, values: t[np.asarray(values) > 0.5],
    )
    assert np.allclose(trace_t_freq, [20.0])
    assert np.allclose(trace_event_hz, [50.0])

    trace_freq_samples = collect_frequency_samples_from_trace_rows(
        [
            ("MC0", np.array([0.0, 10.0, 20.0, 30.0, 40.0]), np.array([0.0, 1.0, 0.0, 1.0, 0.0])),
            ("TC0", np.array([0.0, 10.0, 20.0]), np.array([0.0, 0.0, 0.0])),
            ("MC1", np.array([0.0, 15.0, 30.0, 45.0]), np.array([0.0, 1.0, 0.0, 1.0])),
        ],
        label_fn=lambda row: row[0],
        time_fn=lambda row: row[1],
        value_fn=lambda row: row[2],
        event_times_fn=lambda t, values: t[np.asarray(values) > 0.5],
        include_prefixes=("MC",),
        modulus=35.0,
    )
    assert trace_freq_samples.labels == ("MC0", "MC1")
    assert np.allclose(trace_freq_samples.times_ms, [20.0, 30.0])
    assert np.allclose(trace_freq_samples.freqs_hz, [50.0, 33.3333333333])

    filtered_rows = filter_rows_by_label_prefix(
        [("MC0", [0.0]), ("TC0", [1.0]), ("MC1", [2.0])],
        label_fn=lambda row: row[0],
        include_prefixes=("MC",),
        normalize_label_fn=str,
    )
    assert [row[0] for row in filtered_rows] == ["MC0", "MC1"]

    centers, rate_hz, metadata = compute_event_rate_from_rows(
        [("MC0", [10.0, 20.0, 30.0]), ("MC1", [15.0, 25.0])],
        times_fn=lambda row: row[1],
        t_stop=40.0,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        normalization="per_cell",
        default_normalization="per_target_cell",
        normalization_rules={
            "total": EventRateNormalizationRule(
                unit="events/s",
                aliases=(),
                denominator_fn=lambda rows: 1.0,
                metadata_fn=lambda rows: {"n_rows": len(rows)},
            ),
            "per_target_cell": EventRateNormalizationRule(
                unit="events/s per cell",
                aliases=("per_cell",),
                denominator_fn=lambda rows: float(len(rows)),
                metadata_fn=lambda rows: {"n_rows": len(rows)},
            ),
        },
        return_metadata=True,
    )
    assert np.allclose(centers, [5.0, 15.0, 25.0, 35.0])
    assert np.allclose(rate_hz, [0.0, 100.0, 100.0, 50.0])
    assert metadata["normalization"] == "per_target_cell"
    assert metadata["unit"] == "events/s per cell"
    assert metadata["denominator"] == 2.0
    assert metadata["n_rows"] == 2

    traces = build_event_rate_trace_series(
        {"tag": "demo"},
        [
            EventRateSeriesSpec("All rows", None, "black"),
            EventRateSeriesSpec("Subset", ["MC"], "tab:blue"),
        ],
        compute_rate_fn=lambda _result, *, return_metadata, target_types, **_kwargs: (
            np.array([5.0, 15.0]),
            np.array([1.0, 2.0]) if target_types is None else np.array([0.5, 1.5]),
            {
                "normalization": "per_target_cell",
                "n_target_cells": 3 if target_types is None else 1,
                "unit": "events/s per cell",
            },
        ),
        selection_kwarg="target_types",
        compute_rate_kwargs={"bin_ms": 10.0, "smooth_sigma_ms": 0.0, "normalization": "per_target_cell"},
    )
    assert [trace.base_label for trace in traces] == ["All rows", "Subset"]
    assert np.allclose(traces[0].rate_hz, [1.0, 2.0])
    assert np.allclose(traces[1].rate_hz, [0.5, 1.5])
    assert traces[1].metadata["n_target_cells"] == 1

    family_spec = ResultEventFamilySpec(
        rows_from_result_fn=lambda result: list(result["rows"]),
        filter_label_fn=lambda row: row["dest"],
        times_fn=lambda row: row["times"],
        sample_label_fn=lambda row: f"{row['src']}->{row['dest']}",
        normalize_label_fn=str,
        normalization_rules={
            "total": EventRateNormalizationRule(
                unit="events/s",
                aliases=(),
                denominator_fn=lambda rows: 1.0,
                metadata_fn=lambda rows: {"n_rows": len(rows)},
            ),
            "per_target_cell": EventRateNormalizationRule(
                unit="events/s per cell",
                aliases=("per_cell",),
                denominator_fn=lambda rows: float(len({row["dest"] for row in rows})),
                metadata_fn=lambda rows: {"n_targets": len({row["dest"] for row in rows})},
            ),
        },
        default_normalization="per_target_cell",
    )
    family_result = {
        "rows": [
            {"src": "GC0", "dest": "MC0", "times": np.array([0.0, 20.0, 40.0])},
            {"src": "GC1", "dest": "TC0", "times": np.array([10.0, 30.0])},
        ]
    }
    filtered_family_rows = filter_result_event_family_rows(
        family_result,
        family_spec,
        include_prefixes=("MC",),
    )
    assert len(filtered_family_rows) == 1
    family_samples = collect_result_event_family_samples(
        family_result,
        family_spec,
        include_prefixes=("MC",),
        modulus=25.0,
    )
    assert family_samples.labels == ("GC0->MC0",)
    assert np.allclose(family_samples.times_ms, [10.0, 5.0])
    assert np.allclose(family_samples.freqs_hz, [50.0, 50.0])
    family_rate_t, family_rate_hz, family_rate_meta = compute_result_event_family_rate(
        family_result,
        family_spec,
        t_stop=50.0,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        normalization="per_cell",
        return_metadata=True,
    )
    assert family_rate_meta["normalization"] == "per_target_cell"
    assert family_rate_meta["n_targets"] == 2
    np.testing.assert_allclose(family_rate_t, [5.0, 15.0, 25.0, 35.0, 45.0])
    np.testing.assert_allclose(family_rate_hz, [50.0, 50.0, 50.0, 50.0, 50.0])

    prepared_rows = prepare_event_display_rows(
        [
            ("b", [5.0, 15.0]),
            ("a", [0.0, 10.0]),
            ("long_name", [20.0]),
        ],
        label_fn=lambda row: row[0],
        times_fn=lambda row: row[1],
        sort_key_fn=lambda row: row[0],
        limit=2,
        label_transform_fn=lambda label: label.upper(),
    )
    assert isinstance(prepared_rows, PreparedEventRows)
    assert [label for label, _times in prepared_rows.rows] == ["A", "B"]
    assert prepared_rows.max_label_length == 1

    prepared_layout = build_event_overview_layout_for_rows(
        prepared_rows,
        raster_min_height=4.5,
        rate_height=4.0,
        left_margin_per_char=0.01,
    )
    assert prepared_layout.n_rows == 2
    assert prepared_layout.left_margin >= 0.22

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
