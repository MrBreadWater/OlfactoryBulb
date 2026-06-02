"""Smoke tests for extracted analysis plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.plotting import (
    plot_band_power_summary,
    plot_named_time_series,
    plot_stacked_labeled_traces,
    plot_time_frequency_map,
    plot_time_series,
    time_axis_label,
)


def main() -> None:
    assert time_axis_label(None) == "Time (ms)"
    assert time_axis_label(100.0) == "Time modulo 100 ms"

    fig, ax = plt.subplots()
    try:
        plot_time_series(
            np.array([0.0, 50.0, 100.0, 150.0]),
            np.array([1.0, 3.0, 1.0, 3.0]),
            ax=ax,
            modulus=100.0,
            dt_ms=50.0,
            title="Folded Trace",
            ylabel="voltage",
        )
        assert ax.get_xlabel() == "Time modulo 100 ms"
        assert ax.get_ylabel() == "voltage"
        assert ax.get_title() == "Folded Trace"
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_named_time_series(
            np.array([0.0, 50.0, 100.0, 150.0]),
            {"alpha": np.array([1.0, 2.0, 1.0, 2.0]), "beta": np.array([0.5, 1.5, 0.5, 1.5])},
            ax=ax,
            modulus=100.0,
            dt_ms=50.0,
            title="Named Traces",
            ylabel="rate",
        )
        assert ax.get_xlabel() == "Time modulo 100 ms"
        assert ax.get_title() == "Named Traces"
        assert len(ax.lines) == 2
        legend = ax.get_legend()
        assert legend is not None
        assert {text.get_text() for text in legend.texts} == {"alpha", "beta"}
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_stacked_labeled_traces(
            [
                ("MC0", np.array([0.0, 1.0]), np.array([1.0, 2.0])),
                ("GC0", np.array([0.0, 1.0]), np.array([0.5, 0.75])),
            ],
            ax=ax,
            title="Stacked",
            ylabel="offset voltage",
            line_kwargs_fn=lambda row: {"color": "tab:blue" if row[0].startswith("MC") else "tab:green"},
            offset_step_fn=lambda row: 40.0 if row[0].startswith("GC") else 120.0,
        )
        assert ax.get_title() == "Stacked"
        assert ax.get_ylabel() == "offset voltage"
        assert len(ax.lines) == 2
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.texts] == ["MC0", "GC0"]
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [120.5, 120.75])
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_time_frequency_map(
            np.array([0.0, 50.0, 100.0, 150.0]),
            np.array([10.0, 20.0]),
            np.array([[1.0, 2.0, 1.0, 2.0], [0.5, 1.0, 0.5, 1.0]]),
            ax=ax,
            modulus=100.0,
            title="Map",
            colorbar_label="density",
            power_transform=lambda power: np.log1p(power),
        )
        assert ax.get_xlabel() == "Time modulo 100 ms"
        assert ax.get_ylabel() == "Frequency (Hz)"
        assert ax.get_title() == "Map"
        assert len(ax.figure.axes) == 2  # plot axis + colorbar axis
    finally:
        plt.close(fig)

    summary = {
        "band_power": {"low": 1.0, "high": 3.0},
        "relative_band_power": {"low": 0.25, "high": 0.75},
    }
    fig, axes = plot_band_power_summary(summary, signal_label="lfp")
    try:
        assert axes[0].get_title() == "lfp Band Power"
        assert axes[1].get_title() == "Relative Band Power"
        assert len(axes[0].patches) == 2
        assert len(axes[1].patches) == 2
    finally:
        plt.close(fig)

    print("analysis plotting helpers: OK")


if __name__ == "__main__":
    main()
