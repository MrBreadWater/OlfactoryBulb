"""Reusable event-series analysis and raster plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .spectral import normalize_time_modulus


@dataclass(frozen=True)
class EventRateTrace:
    """One named event-rate series prepared for plotting."""

    base_label: str
    times_ms: np.ndarray | list[float]
    rate_hz: np.ndarray | list[float]
    metadata: dict[str, Any]
    color: Any = "black"


@dataclass(frozen=True)
class EventOverviewLayout:
    """Layout hints for a raster-plus-rate overview figure."""

    n_rows: int
    label_fontsize: float
    line_spacing: float
    raster_height: float
    rate_height: float
    total_height: float
    left_margin: float


def calculate_event_frequency(times: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert event times into midpoint/frequency samples."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.array([]), np.array([])
    t_freq = (times[:-1] + times[1:]) / 2.0
    event_hz = 1000.0 / np.diff(times)
    return t_freq, event_hz


def smooth_rate_series(
    rate_hz: np.ndarray,
    *,
    bin_ms: float,
    smooth_sigma_ms: float,
) -> np.ndarray:
    """Gaussian-smooth a binned rate trace."""
    if smooth_sigma_ms and smooth_sigma_ms > 0:
        sigma_bins = float(smooth_sigma_ms) / float(bin_ms)
        radius = max(1, int(round(4.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        smoothed = np.convolve(rate_hz, kernel, mode="same")
        if smoothed.shape != rate_hz.shape:
            extra = smoothed.shape[0] - rate_hz.shape[0]
            start = max(extra // 2, 0)
            stop = start + rate_hz.shape[0]
            smoothed = smoothed[start:stop]
        rate_hz = smoothed
    return rate_hz


def binned_event_rate(
    event_series: Sequence[np.ndarray | list[float]],
    *,
    t_stop: float,
    bin_ms: float,
    smooth_sigma_ms: float,
    denominator: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin one or more event series into a smoothed rate trace."""
    if t_stop <= 0.0:
        return np.array([]), np.array([])

    edges = np.arange(0.0, t_stop + float(bin_ms), float(bin_ms))
    if edges.size < 2:
        edges = np.array([0.0, float(bin_ms)], dtype=float)

    flat_times = []
    for times in event_series:
        times = np.asarray(times, dtype=float)
        if times.size:
            flat_times.append(times)

    if flat_times:
        counts, _edges = np.histogram(np.concatenate(flat_times), bins=edges)
    else:
        counts = np.zeros(len(edges) - 1, dtype=float)

    rate_hz = counts.astype(float) / (float(bin_ms) / 1000.0)
    denom = max(float(denominator), 1.0)
    rate_hz /= denom
    rate_hz = smooth_rate_series(rate_hz, bin_ms=bin_ms, smooth_sigma_ms=smooth_sigma_ms)
    centers = edges[:-1] + float(bin_ms) / 2.0
    return centers, rate_hz


def rate_series_label(base_label: str, metadata: dict[str, Any]) -> str:
    """Append denominator information to a plotted rate-series label."""
    normalization = str(metadata.get("normalization", ""))
    if normalization == "per_target_cell":
        return f"{base_label} (n={metadata.get('n_target_cells', 0)} cells)"
    if normalization == "per_source_cell":
        return f"{base_label} (n={metadata.get('n_source_cells', 0)} sources)"
    if normalization == "per_connection":
        return f"{base_label} (n={metadata.get('n_connections', 0)} connections)"
    if normalization == "per_cell":
        return f"{base_label} (n={metadata.get('n_target_cells', 0)} cells)"
    if normalization in {"per_segment", "per_input_segment"}:
        return f"{base_label} (n={metadata.get('n_segments', 0)} segments)"
    return base_label


def recommended_raster_fontsize(n_rows: int, *, default: float = 7.0) -> float:
    """Choose a compact but readable y-label font size for dense rasters."""
    if n_rows >= 140:
        return 5.0
    if n_rows >= 80:
        return 6.0
    return float(default)


def recommended_raster_height(n_rows: int, *, min_height: float = 4.0) -> float:
    """Estimate a reasonable figure height for a raster plot."""
    if n_rows <= 0:
        return float(min_height)
    return max(float(min_height), 0.06 * float(n_rows) + 1.5)


def recommended_raster_line_spacing(
    n_rows: int,
    *,
    threshold: int = 80,
    dense_value: float = 1.6,
    default: float = 1.4,
) -> float:
    """Pick a slightly wider line spacing for dense rasters."""
    return float(dense_value if int(n_rows) > int(threshold) else default)


def overview_left_margin(
    max_label_len: int,
    *,
    min_margin: float = 0.22,
    max_margin: float = 0.5,
    base: float = 0.15,
    per_char: float = 0.006,
) -> float:
    """Estimate a figure left margin from the longest raster label."""
    return min(float(max_margin), max(float(min_margin), float(base) + float(per_char) * float(max_label_len)))


def build_event_overview_layout(
    *,
    n_rows: int,
    max_label_len: int,
    raster_min_height: float = 4.5,
    rate_height: float = 4.0,
    default_label_fontsize: float = 7.0,
    dense_line_spacing_threshold: int = 80,
    dense_line_spacing: float = 1.6,
    default_line_spacing: float = 1.4,
    left_margin_min: float = 0.22,
    left_margin_max: float = 0.5,
    left_margin_base: float = 0.15,
    left_margin_per_char: float = 0.006,
) -> EventOverviewLayout:
    """Build shared layout hints for a raster-plus-rate overview figure."""
    label_fontsize = recommended_raster_fontsize(n_rows, default=default_label_fontsize)
    line_spacing = recommended_raster_line_spacing(
        n_rows,
        threshold=dense_line_spacing_threshold,
        dense_value=dense_line_spacing,
        default=default_line_spacing,
    )
    raster_height = recommended_raster_height(n_rows, min_height=raster_min_height)
    left_margin = overview_left_margin(
        max_label_len,
        min_margin=left_margin_min,
        max_margin=left_margin_max,
        base=left_margin_base,
        per_char=left_margin_per_char,
    )
    return EventOverviewLayout(
        n_rows=int(n_rows),
        label_fontsize=float(label_fontsize),
        line_spacing=float(line_spacing),
        raster_height=float(raster_height),
        rate_height=float(rate_height),
        total_height=float(raster_height + rate_height),
        left_margin=float(left_margin),
    )


def ensure_raster_axis(
    ax: Any,
    n_rows: int,
    *,
    width: float = 14.0,
    min_height: float = 4.0,
    per_row_height: float = 0.22,
) -> Any:
    """Create a raster axis sized to the current row count when needed."""
    if ax is None:
        height = max(min_height, per_row_height * max(int(n_rows), 1) + 1.0)
        _fig, ax = plt.subplots(figsize=(width, height))
    return ax


def style_raster_axis(
    ax: Any,
    labels: list[str],
    *,
    ylabel: str,
    title: str,
    fontsize: float = 7.0,
    line_spacing: float = 1.4,
) -> np.ndarray:
    """Apply shared styling and row offsets to a raster axis."""
    n_rows = len(labels)
    offsets = np.arange(n_rows, dtype=float) * float(line_spacing)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels, fontsize=fontsize)
    if n_rows:
        pad = max(0.7, line_spacing)
        ax.set_ylim(offsets[0] - pad, offsets[-1] + pad)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return offsets


def fit_raster_labels(
    ax: Any,
    offsets: np.ndarray,
    *,
    min_fontsize: float = 4.5,
    target_ratio: float = 0.9,
    min_height: float = 4.0,
    max_iter: int = 8,
) -> Any:
    """Shrink labels or grow the figure until label height fits the row spacing."""
    if len(offsets) < 2:
        return ax

    fig = ax.figure
    labels = [label for label in ax.get_yticklabels() if label.get_text()]
    if not labels:
        return ax

    for _ in range(max_iter):
        fig.canvas.draw()
        labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if not labels:
            return ax

        renderer = fig.canvas.get_renderer()
        max_label_height_px = max(label.get_window_extent(renderer=renderer).height for label in labels)
        p0 = ax.transData.transform((0.0, float(offsets[0])))[1]
        p1 = ax.transData.transform((0.0, float(offsets[1])))[1]
        spacing_px = abs(float(p1 - p0))
        if spacing_px <= 0:
            return ax

        ratio = max_label_height_px / spacing_px
        if ratio > target_ratio:
            current_font = labels[0].get_fontsize()
            if current_font > min_fontsize + 0.05:
                scale = max(target_ratio / ratio * 0.98, min_fontsize / current_font)
                new_font = max(min_fontsize, current_font * scale)
                for label in labels:
                    label.set_fontsize(new_font)
                continue

            width, height = fig.get_size_inches()
            new_height = max(float(min_height), height * (ratio / target_ratio) * 1.02)
            if abs(new_height - height) < 0.05:
                break
            fig.set_size_inches(width, new_height, forward=True)
            continue

        if ratio < target_ratio * 0.65:
            width, height = fig.get_size_inches()
            shrink = max(ratio / target_ratio, 0.75)
            new_height = max(float(min_height), height * shrink)
            if abs(new_height - height) < 0.05:
                break
            fig.set_size_inches(width, new_height, forward=True)
            continue

        break

    return ax


def plot_event_raster_rows(
    rows: Sequence[tuple[str, np.ndarray | list[float]]],
    *,
    ax: Any = None,
    ylabel: str = "Row",
    title: str = "Event Raster",
    width: float = 14.0,
    min_height: float = 4.0,
    per_row_height: float = 0.10,
    fontsize: float | None = None,
    line_spacing: float = 1.4,
    modulus: float | int | None = None,
    colors: Sequence[Any] | Any = "black",
    linelengths: float = 1.0,
    no_data_message: str = "No events saved",
) -> Any:
    """Plot a generic event raster from labeled event-time rows."""
    ax = ensure_raster_axis(
        ax,
        len(rows),
        width=width,
        min_height=min_height,
        per_row_height=per_row_height,
    )
    if not rows:
        ax.set_title(no_data_message)
        return ax

    modulus_value = normalize_time_modulus(modulus)
    times = [
        np.mod(np.asarray(times, dtype=float), modulus_value)
        if modulus_value is not None
        else np.asarray(times, dtype=float)
        for _label, times in rows
    ]
    labels = [str(label) for label, _times in rows]
    font_value = recommended_raster_fontsize(len(rows)) if fontsize is None else min(
        float(fontsize),
        recommended_raster_fontsize(len(rows), default=float(fontsize)),
    )
    offsets = style_raster_axis(
        ax,
        labels,
        ylabel=ylabel,
        title=title,
        fontsize=font_value,
        line_spacing=line_spacing,
    )
    ax.eventplot(times, lineoffsets=offsets, linelengths=linelengths, colors=colors)
    if modulus_value is not None:
        ax.set_xlim(0.0, modulus_value)
        ax.set_xlabel(f"Time modulo {modulus_value:g} ms")
    fit_raster_labels(ax, offsets, min_height=min_height)
    return ax


def plot_event_rate_traces(
    traces: Sequence[EventRateTrace],
    *,
    ax: Any = None,
    title: str = "Event Rate",
    ylabel_fallback: str = "events/s",
    xlabel: str = "Time (ms)",
    linewidth: float = 1.2,
    legend_loc: str = "upper right",
    no_data_message: str = "No events saved",
    label_formatter: Callable[[str, dict[str, Any]], str] = rate_series_label,
) -> Any:
    """Plot one or more event-rate traces with consistent label handling."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    plotted = False
    ylabel = None
    for trace in traces:
        times = np.asarray(trace.times_ms, dtype=float)
        rate = np.asarray(trace.rate_hz, dtype=float)
        if len(times) == 0 or len(rate) == 0:
            continue
        ylabel = str(trace.metadata.get("unit", ylabel_fallback))
        ax.plot(
            times,
            rate,
            color=trace.color,
            linewidth=linewidth,
            label=label_formatter(trace.base_label, trace.metadata),
        )
        plotted = True

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or ylabel_fallback)
    ax.set_title(title)
    if plotted:
        ax.legend(loc=legend_loc, fontsize=8)
    else:
        ax.text(0.5, 0.5, no_data_message, ha="center", va="center", transform=ax.transAxes)
    return ax


def plot_event_overview(
    *,
    layout: EventOverviewLayout,
    raster_plotter: Callable[[Any, EventOverviewLayout], Any],
    rate_plotter: Callable[[Any, EventOverviewLayout], Any],
    figure_width: float = 16.0,
    hspace: float = 0.25,
) -> tuple[Any, Any]:
    """Render a two-row raster-plus-rate overview using a shared layout."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, layout.total_height),
        sharex=False,
        gridspec_kw={"height_ratios": [layout.raster_height, layout.rate_height]},
    )
    raster_plotter(axes[0], layout)
    rate_plotter(axes[1], layout)
    fig.subplots_adjust(left=layout.left_margin, hspace=hspace)
    return fig, axes
