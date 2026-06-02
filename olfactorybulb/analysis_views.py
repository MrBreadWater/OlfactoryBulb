"""Concrete olfactory-bulb analysis view adapters built on neuroinfra."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from neuroinfra.analysis import (
    GroupedEventRasterSuite,
    GroupedRowPolicy,
    GroupedTracePlotSuite,
    ordered_names,
    round_robin_limit_by_subgroup,
)
from olfactorybulb.analysis_data import (
    cell_type_of,
    normalize_cell_name,
    ordered_cell_types,
    saved_soma_spike_rows,
)

PLOT_DISPLAY_CELL_GROUPS = ("MT", "GC", "EPLI", "other")
CELL_TYPE_COLORS = {
    "MC": "tab:blue",
    "TC": "tab:red",
    "GC": "tab:orange",
    "EPLI": "tab:green",
    "other": "tab:gray",
}


def display_group_for_cell_type(cell_type: str, *, combine_mt: bool = True) -> str:
    """Map notebook cell-family labels to a small display bucket."""
    cell_type = str(cell_type)
    if combine_mt and cell_type in {"MC", "TC"}:
        return "MT"
    return cell_type


def ordered_display_groups(
    groups: list[str] | tuple[str, ...] | set[str],
    *,
    combine_mt: bool = True,
) -> list[str]:
    """Return display buckets in a stable order for plots."""
    raw_seen = {str(group) for group in groups}
    if combine_mt:
        seen = {display_group_for_cell_type(group, combine_mt=True) for group in raw_seen}
    else:
        return ordered_cell_types(raw_seen)
    return ordered_names(
        seen,
        preferred_order=PLOT_DISPLAY_CELL_GROUPS,
        unknown_name="other",
    )


def truncate_display_rows_for_group(
    rows: list[tuple[str, Any]],
    max_rows: int,
    *,
    combine_mt: bool,
    display_group: str,
) -> list[tuple[str, Any]]:
    """Limit rows per display bucket with fair MT sampling when MC/TC are merged."""
    if not rows or max_rows <= 0:
        return []
    if not combine_mt or display_group != "MT":
        return rows[:max_rows]
    return round_robin_limit_by_subgroup(
        rows,
        subgroup_fn=lambda row: cell_type_of(str(row[0])),
        max_rows=max_rows,
        unknown_subgroup="other",
    )


def build_soma_display_grouping(
    *,
    max_rows_per_group: int,
    combine_mt: bool,
) -> GroupedRowPolicy:
    """Build the stable soma display grouping used by trace and raster views."""
    return GroupedRowPolicy(
        bucket_fn=lambda row: display_group_for_cell_type(cell_type_of(row[0]), combine_mt=combine_mt),
        order_buckets_fn=lambda buckets: ordered_display_groups(buckets, combine_mt=combine_mt),
        limit_bucket_rows_fn=lambda bucket, bucket_rows: truncate_display_rows_for_group(
            list(bucket_rows),
            max_rows_per_group,
            combine_mt=combine_mt,
            display_group=bucket,
        ),
        unknown_bucket="other",
    )


def cell_color(cell_type: str) -> str:
    """Return a stable plotting color for one cell family."""
    return CELL_TYPE_COLORS.get(str(cell_type), "tab:purple")


def soma_trace_line_kwargs(row: tuple[str, Any, Any]) -> dict[str, Any]:
    """Return plotting color for one saved soma trace row."""
    label = row[0]
    try:
        color_key = cell_type_of(label)
    except ValueError:
        color_key = "other"
    return {"color": cell_color(color_key)}


def soma_trace_offset_step(row: tuple[str, Any, Any], *, combine_mt: bool) -> float:
    """Return the vertical offset spacing for one grouped soma trace row."""
    label = row[0]
    try:
        display_group = display_group_for_cell_type(cell_type_of(label), combine_mt=combine_mt)
    except ValueError:
        display_group = "other"
    return 40.0 if display_group == "GC" else 120.0


def soma_spike_raster_colors(rows: list[tuple[str, np.ndarray]]) -> list[str]:
    """Return stable colors for one grouped soma-spike raster."""
    return [
        cell_color(cell_type_of(label) if re.match(r"([A-Z]+)", normalize_cell_name(label)) else "other")
        for label, _spikes in rows
    ]


def build_soma_voltage_plot_suite(
    *,
    max_per_type: int,
    combine_mt: bool,
) -> GroupedTracePlotSuite:
    """Build the grouped soma-voltage plot suite for one notebook call."""
    return GroupedTracePlotSuite(
        grouping=build_soma_display_grouping(
            max_rows_per_group=max_per_type,
            combine_mt=combine_mt,
        ),
        title="Sample Soma Voltages" + (" (MT grouped)" if combine_mt else ""),
        ylabel="Offset Voltage",
        line_kwargs_fn=soma_trace_line_kwargs,
        offset_step_fn=lambda row: soma_trace_offset_step(row, combine_mt=combine_mt),
        legend_loc="upper right",
        legend_fontsize=8.0,
        legend_ncol=2,
    )


def build_soma_spike_raster_suite(
    *,
    max_cells_per_type: int,
    combine_mt: bool,
) -> GroupedEventRasterSuite:
    """Build the grouped soma-spike raster suite for one notebook call."""
    return GroupedEventRasterSuite(
        grouping=build_soma_display_grouping(
            max_rows_per_group=max_cells_per_type,
            combine_mt=combine_mt,
        ),
        ylabel="Cell",
        title="Detected Soma Spike Raster" + (" (MT grouped)" if combine_mt else ""),
        width=14.0,
        min_height=4.5,
        per_row_height=0.10,
        default_fontsize=7.0,
        line_spacing=1.3,
        colors_fn=lambda rows: soma_spike_raster_colors(list(rows)),
        no_data_message="No soma spikes saved",
    )


def infer_grouped_cell_types_from_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    """Infer ordered cell-family buckets from one list of saved labels."""
    inferred = []
    for label in labels:
        try:
            inferred.append(cell_type_of(label))
        except ValueError:
            inferred.append("other")
    return ordered_cell_types(inferred)


def saved_soma_spike_rows_for_display(
    result: dict[str, Any],
    *,
    max_cells_per_type: int,
    threshold: float | None = None,
    combine_mt: bool = True,
) -> list[tuple[str, np.ndarray]] | None:
    """Return saved spike rows grouped in stable family display order for rasters."""
    rows = saved_soma_spike_rows(result, threshold=threshold)
    if rows is None:
        return None
    return build_soma_display_grouping(
        max_rows_per_group=max_cells_per_type,
        combine_mt=combine_mt,
    ).prepare_rows(rows)
