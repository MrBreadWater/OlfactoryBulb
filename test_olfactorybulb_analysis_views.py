"""Smoke tests for the concrete olfactory-bulb grouped analysis views."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from olfactorybulb.analysis_views import (
    build_soma_spike_raster_suite,
    build_soma_voltage_plot_suite,
    cell_color,
    display_group_for_cell_type,
    ordered_display_groups,
    saved_soma_spike_rows_for_display,
)


def main() -> None:
    trace_rows = [
        ("MC0.soma", np.array([0.0, 1.0]), np.array([-65.0, -60.0])),
        ("MC1.soma", np.array([0.0, 1.0]), np.array([-66.0, -61.0])),
        ("TC0.soma", np.array([0.0, 1.0]), np.array([-64.0, -59.0])),
        ("GC0.soma", np.array([0.0, 1.0]), np.array([-70.0, -68.0])),
    ]
    result = {
        "soma_spikes": {
            "labels": ["MC0.soma", "MC1.soma", "TC0.soma", "GC0.soma"],
            "spike_times": [
                np.array([5.0, 25.0]),
                np.array([10.0, 30.0]),
                np.array([15.0, 35.0]),
                np.array([20.0, 40.0]),
            ],
        }
    }

    assert display_group_for_cell_type("MC", combine_mt=True) == "MT"
    assert display_group_for_cell_type("TC", combine_mt=False) == "TC"
    assert ordered_display_groups({"GC", "TC", "MC"}) == ["MT", "GC"]
    assert cell_color("MC") == "tab:blue"
    assert cell_color("unknown") == "tab:purple"

    grouped_saved_rows = saved_soma_spike_rows_for_display(
        result,
        max_cells_per_type=3,
        combine_mt=True,
    )
    assert grouped_saved_rows is not None
    assert [row[0] for row in grouped_saved_rows] == ["MC0.soma", "TC0.soma", "MC1.soma", "GC0.soma"]

    trace_suite = build_soma_voltage_plot_suite(max_per_type=2, combine_mt=True)
    prepared_trace_rows = trace_suite.prepare_rows(trace_rows)
    assert [row[0] for row in prepared_trace_rows] == ["MC0.soma", "TC0.soma", "GC0.soma"]

    fig, ax = plt.subplots()
    try:
        trace_suite.plot(trace_rows, ax=ax)
        assert ax.get_title() == "Sample Soma Voltages (MT grouped)"
        assert ax.get_ylabel() == "Offset Voltage"
        assert len(ax.lines) == 3
        assert ax.lines[0].get_color() == "tab:blue"
        assert ax.lines[-1].get_color() == "tab:orange"
    finally:
        plt.close(fig)

    raster_suite = build_soma_spike_raster_suite(max_cells_per_type=3, combine_mt=True)
    prepared_raster_rows = raster_suite.prepare_rows(grouped_saved_rows)
    assert [row[0] for row in prepared_raster_rows] == ["MC0.soma", "TC0.soma", "MC1.soma", "GC0.soma"]

    fig, ax = plt.subplots()
    try:
        raster_suite.plot_prepared_rows(prepared_raster_rows, ax=ax, modulus=20.0)
        assert ax.get_title() == "Detected Soma Spike Raster (MT grouped)"
        assert ax.get_ylabel() == "Cell"
        assert ax.get_xlabel() == "Time modulo 20 ms"
        assert len(ax.collections) == 4
    finally:
        plt.close(fig)

    print("olfactorybulb analysis views: OK")


if __name__ == "__main__":
    main()
