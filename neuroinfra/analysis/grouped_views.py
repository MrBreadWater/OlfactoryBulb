"""Reusable grouped trace and raster display helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .catalog import ordered_group_rows
from .events import plot_event_raster_rows, recommended_raster_fontsize
from .plotting import plot_stacked_labeled_traces


@dataclass(frozen=True)
class GroupedRowPolicy:
    """Stable bucket/ordering/limiting policy for grouped display rows."""

    bucket_fn: Callable[[Any], str]
    order_buckets_fn: Callable[[Iterable[str]], list[str]]
    limit_bucket_rows_fn: Callable[[str, list[Any]], Sequence[Any]] | None = None
    unknown_bucket: str = "other"

    def prepare_rows(self, rows: Sequence[Any]) -> list[Any]:
        """Bucket rows, order the buckets, and flatten them for display."""
        return ordered_group_rows(
            rows,
            bucket_fn=self.bucket_fn,
            order_buckets_fn=self.order_buckets_fn,
            limit_bucket_rows_fn=self.limit_bucket_rows_fn,
            unknown_bucket=self.unknown_bucket,
        )


@dataclass(frozen=True)
class GroupedTracePlotSuite:
    """Grouped stacked-trace plotting on top of a reusable row policy."""

    grouping: GroupedRowPolicy
    title: str = "Stacked Traces"
    xlabel: str = "Time (ms)"
    ylabel: str = "Offset Value"
    linewidth: float = 1.0
    legend_loc: str = "upper right"
    legend_fontsize: float = 8.0
    legend_ncol: int = 2
    line_kwargs_fn: Callable[[tuple[str, Any, Any]], dict[str, Any]] | None = None
    offset_step_fn: Callable[[tuple[str, Any, Any]], float] | None = None

    def prepare_rows(self, rows: Sequence[tuple[str, Any, Any]]) -> list[tuple[str, Any, Any]]:
        """Prepare trace rows in grouped display order."""
        return list(self.grouping.prepare_rows(rows))

    def plot_prepared_rows(
        self,
        rows: Sequence[tuple[str, Any, Any]],
        *,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot trace rows that are already prepared in display order."""
        return plot_stacked_labeled_traces(
            rows,
            ax=ax,
            title=title or self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            linewidth=self.linewidth,
            line_kwargs_fn=self.line_kwargs_fn,
            offset_step_fn=self.offset_step_fn,
            legend_loc=self.legend_loc,
            legend_fontsize=self.legend_fontsize,
            legend_ncol=self.legend_ncol,
        )

    def plot(
        self,
        rows: Sequence[tuple[str, Any, Any]],
        *,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Prepare and plot one grouped stacked-trace view."""
        return self.plot_prepared_rows(
            self.prepare_rows(rows),
            ax=ax,
            title=title,
        )


@dataclass(frozen=True)
class GroupedEventRasterSuite:
    """Grouped event-raster plotting on top of a reusable row policy."""

    grouping: GroupedRowPolicy
    ylabel: str = "Row"
    title: str = "Event Raster"
    width: float = 14.0
    min_height: float = 4.5
    per_row_height: float = 0.10
    default_fontsize: float = 7.0
    line_spacing: float = 1.4
    colors_fn: Callable[[Sequence[tuple[str, Any]]], Sequence[Any] | Any] | None = None
    linelengths: float | Sequence[float] = 1.0
    no_data_message: str = "No events saved"

    def prepare_rows(self, rows: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
        """Prepare event rows in grouped display order."""
        return list(self.grouping.prepare_rows(rows))

    def plot_prepared_rows(
        self,
        rows: Sequence[tuple[str, Any]],
        *,
        ax: Any = None,
        modulus: float | int | None = None,
        fontsize: float | None = None,
        line_spacing: float | None = None,
        title: str | None = None,
    ) -> Any:
        """Plot event rows that are already prepared in display order."""
        resolved_fontsize = (
            recommended_raster_fontsize(len(rows), default=self.default_fontsize)
            if fontsize is None
            else float(fontsize)
        )
        resolved_colors: Sequence[Any] | Any = "black"
        if self.colors_fn is not None:
            resolved_colors = self.colors_fn(rows)
        return plot_event_raster_rows(
            rows,
            ax=ax,
            ylabel=self.ylabel,
            title=title or self.title,
            width=self.width,
            min_height=self.min_height,
            per_row_height=self.per_row_height,
            fontsize=resolved_fontsize,
            line_spacing=float(line_spacing if line_spacing is not None else self.line_spacing),
            modulus=modulus,
            colors=resolved_colors,
            linelengths=self.linelengths,
            no_data_message=self.no_data_message,
        )

    def plot(
        self,
        rows: Sequence[tuple[str, Any]],
        *,
        ax: Any = None,
        modulus: float | int | None = None,
        fontsize: float | None = None,
        line_spacing: float | None = None,
        title: str | None = None,
    ) -> Any:
        """Prepare and plot one grouped event-raster view."""
        return self.plot_prepared_rows(
            self.prepare_rows(rows),
            ax=ax,
            modulus=modulus,
            fontsize=fontsize,
            line_spacing=line_spacing,
            title=title,
        )
