"""Smoke tests for extracted grouped trace and raster display helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuroinfra.analysis.catalog import ordered_names, round_robin_limit_by_subgroup
from neuroinfra.analysis.grouped_views import (
    GroupedEventRasterSuite,
    GroupedRowPolicy,
    GroupedTracePlotSuite,
)


def _mt_group(label: str) -> str:
    return "MT" if str(label).startswith(("MC", "TC")) else "GC"


def _display_policy(mt_limit: int = 2, gc_limit: int = 1) -> GroupedRowPolicy:
    return GroupedRowPolicy(
        bucket_fn=lambda row: _mt_group(row[0]),
        order_buckets_fn=lambda buckets: ordered_names(
            buckets,
            preferred_order=("MT", "GC"),
            unknown_name="other",
        ),
        limit_bucket_rows_fn=lambda bucket, bucket_rows: (
            round_robin_limit_by_subgroup(
                bucket_rows,
                subgroup_fn=lambda row: row[0][:2],
                max_rows=mt_limit,
            )
            if bucket == "MT"
            else list(bucket_rows[:gc_limit])
        ),
    )


def main() -> None:
    trace_rows = [
        ("MC0.soma", np.array([0.0, 1.0]), np.array([-65.0, -60.0])),
        ("MC1.soma", np.array([0.0, 1.0]), np.array([-66.0, -61.0])),
        ("TC0.soma", np.array([0.0, 1.0]), np.array([-64.0, -59.0])),
        ("GC0.soma", np.array([0.0, 1.0]), np.array([-70.0, -68.0])),
    ]
    event_rows = [
        ("MC0.soma", np.array([5.0, 25.0])),
        ("MC1.soma", np.array([10.0, 30.0])),
        ("TC0.soma", np.array([15.0, 35.0])),
        ("GC0.soma", np.array([20.0, 40.0])),
    ]

    policy = _display_policy()
    prepared_traces = policy.prepare_rows(trace_rows)
    assert [row[0] for row in prepared_traces] == ["MC0.soma", "TC0.soma", "GC0.soma"]

    trace_suite = GroupedTracePlotSuite(
        grouping=policy,
        title="Grouped traces",
        ylabel="Offset Voltage",
        line_kwargs_fn=lambda row: {
            "color": "tab:blue" if row[0].startswith(("MC", "TC")) else "tab:green"
        },
        offset_step_fn=lambda row: 40.0 if row[0].startswith("GC") else 120.0,
    )
    fig, ax = plt.subplots()
    try:
        trace_suite.plot(trace_rows, ax=ax)
        assert ax.get_title() == "Grouped traces"
        assert ax.get_ylabel() == "Offset Voltage"
        assert len(ax.lines) == 3
        assert ax.lines[0].get_color() == "tab:blue"
        assert ax.lines[-1].get_color() == "tab:green"
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.texts] == ["MC0.soma", "TC0.soma", "GC0.soma"]
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [56.0, 61.0])
        np.testing.assert_allclose(ax.lines[2].get_ydata(), [170.0, 172.0])
    finally:
        plt.close(fig)

    raster_suite = GroupedEventRasterSuite(
        grouping=policy,
        title="Grouped raster",
        ylabel="Cell",
        line_spacing=1.3,
        colors_fn=lambda rows: [
            "tab:blue" if label.startswith(("MC", "TC")) else "tab:green"
            for label, _times in rows
        ],
        no_data_message="No spikes saved",
    )
    fig, ax = plt.subplots()
    try:
        raster_suite.plot(event_rows, ax=ax, modulus=20.0)
        assert ax.get_title() == "Grouped raster"
        assert ax.get_ylabel() == "Cell"
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert labels == ["MC0.soma", "TC0.soma", "GC0.soma"]
        assert len(ax.collections) == 3
        assert ax.get_xlabel() == "Time modulo 20 ms"
    finally:
        plt.close(fig)

    print("analysis grouped views: OK")


if __name__ == "__main__":
    main()
