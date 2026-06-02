"""Reusable sweep-plot and frame-rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SweepPlotSpec:
    """One sweep-animation artifact generated from an existing sweep."""

    name: str
    plot: str | Any
    plot_kwargs: dict[str, Any] | None = None
    filename: str | None = None
    figsize: tuple[float, float] = (12.0, 5.0)
    interval: int = 100
    fps: int = 10
    title_fn: Any = None


def make_sweep_plot_spec(
    plot: str | Any,
    *,
    name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    fps: int = 10,
    title_fn: Any = None,
) -> SweepPlotSpec:
    """Build a sweep-plot spec from a plot name or custom callable."""
    if name is None:
        if isinstance(plot, str):
            name = plot
        else:
            name = getattr(plot, "__name__", "custom_plot")
    return SweepPlotSpec(
        name=str(name),
        plot=plot,
        plot_kwargs=dict(plot_kwargs or {}),
        filename=filename,
        figsize=figsize,
        interval=int(interval),
        fps=int(fps),
        title_fn=title_fn,
    )


def normalize_sweep_plot_spec(plot_spec: SweepPlotSpec | str | Any | dict[str, Any]) -> SweepPlotSpec:
    """Accept ergonomic plot-spec forms and normalize them."""
    if isinstance(plot_spec, SweepPlotSpec):
        return make_sweep_plot_spec(
            plot_spec.plot,
            name=plot_spec.name,
            plot_kwargs=plot_spec.plot_kwargs,
            filename=plot_spec.filename,
            figsize=plot_spec.figsize,
            interval=plot_spec.interval,
            fps=plot_spec.fps,
            title_fn=plot_spec.title_fn,
        )
    if isinstance(plot_spec, str) or callable(plot_spec):
        return make_sweep_plot_spec(plot_spec)
    if isinstance(plot_spec, dict):
        plot = plot_spec.get("plot")
        if plot is None:
            if "plot_fn" in plot_spec:
                plot = plot_spec["plot_fn"]
            elif "name" in plot_spec:
                plot = plot_spec["name"]
            else:
                raise ValueError("Plot-spec dict must include 'plot', 'plot_fn', or 'name'")
        return make_sweep_plot_spec(
            plot,
            name=plot_spec.get("name"),
            plot_kwargs=plot_spec.get("plot_kwargs"),
            filename=plot_spec.get("filename"),
            figsize=tuple(plot_spec.get("figsize", (12.0, 5.0))),
            interval=int(plot_spec.get("interval", 100)),
            fps=int(plot_spec.get("fps", 10)),
            title_fn=plot_spec.get("title_fn"),
        )
    raise TypeError(f"Unsupported sweep-plot spec type {type(plot_spec)!r}")


def extract_figure_from_plot_result(plot_result: Any) -> Any:
    """Best-effort extraction of a Matplotlib figure from a plot return value."""
    if plot_result is None:
        return plt.gcf()
    if hasattr(plot_result, "savefig"):
        return plot_result
    if hasattr(plot_result, "figure"):
        return plot_result.figure
    if isinstance(plot_result, tuple):
        for item in plot_result:
            if hasattr(item, "savefig"):
                return item
            if hasattr(item, "figure"):
                return item.figure
    return plt.gcf()


def build_sweep_plot_callable(
    spec: SweepPlotSpec,
    *,
    plot_resolver: Callable[[str], Any],
) -> Any:
    """Resolve one plot spec into a figure-producing callable."""
    plot_kwargs = dict(spec.plot_kwargs or {})
    if isinstance(spec.plot, str):
        plot_fn = plot_resolver(spec.plot)
    else:
        plot_fn = spec.plot

    def _wrapped(result: dict[str, Any]) -> Any:
        return extract_figure_from_plot_result(plot_fn(result, **plot_kwargs))

    return _wrapped


def is_deprecated_sweep_animation_spec(
    spec: SweepPlotSpec,
    *,
    deprecated_names: Iterable[str],
) -> bool:
    """Return whether a sweep spec names a retired animation plot."""
    names = {
        str(spec.name or ""),
        str(spec.filename or ""),
    }
    if isinstance(spec.plot, str):
        names.add(spec.plot)
    deprecated = {str(name) for name in deprecated_names}
    return any(name in deprecated for name in names)


def format_sweep_value(value: Any) -> str:
    """Format a sweep value compactly for figure titles."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_sweep_value_label(sweep: dict[str, Any], value: Any) -> str:
    """Format one sweep-path/value label for animation titles."""
    path = sweep.get("path", "")
    if isinstance(path, dict):
        if isinstance(value, dict):
            return ", ".join(f"{key}={format_sweep_value(val)}" for key, val in value.items())
        return ", ".join(f"{key}={format_sweep_value(path_value)}" for key, path_value in path.items())
    if isinstance(path, (list, tuple)):
        if isinstance(value, (list, tuple)):
            pairs = zip(path, value)
            return ", ".join(f"{key}={format_sweep_value(val)}" for key, val in pairs)
        return ", ".join(str(key) for key in path)
    return f"{path} = {format_sweep_value(value)}"


def format_sweep_progress_label(frame_index: int, total_frames: int, *, width: int = 12) -> str:
    """Format one compact sweep-progress label without ASCII bar glyphs."""
    del width
    total = max(int(total_frames), 1)
    current = max(1, min(int(frame_index) + 1, total))
    percent = (current / total) * 100.0
    return f"{current}/{total} ({percent:.1f}%)"


def format_sweep_frame_title(sweep: dict[str, Any], value: Any, frame_index: int, total_frames: int) -> str:
    """Build one default animation title with value and sweep progress."""
    return (
        f"{format_sweep_value_label(sweep, value)}"
        f" | {format_sweep_progress_label(frame_index, total_frames)}"
    )


def describe_unavailable_sweep_item(item: dict[str, Any]) -> str:
    """Return a compact reason for a missing/unrenderable sweep frame."""
    load_error = item.get("load_error")
    if load_error:
        return f"Load failed: {load_error}"
    status = item.get("status")
    if isinstance(status, dict):
        for key in ("error", "reason", "state"):
            value = status.get(key)
            if value not in (None, ""):
                return f"{key}: {value}"
        if status.get("ok") is False:
            return "Run did not produce a usable local result payload."
    return "No local result payload was recovered for this sweep item."


def make_sweep_placeholder_figure(
    sweep: dict[str, Any],
    item: dict[str, Any],
    frame_index: int,
    total_frames: int,
    *,
    reason: str,
    figsize: tuple[float, float],
    title_formatter: Callable[[dict[str, Any], Any, int, int], str] = format_sweep_frame_title,
) -> Any:
    """Render an explicit placeholder frame instead of aborting partial sweeps."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    value = item.get("value")
    label = str(item.get("label") or f"item_{frame_index:03d}")
    title = title_formatter(sweep, value, frame_index, total_frames)
    ax.text(
        0.5,
        0.62,
        "Sweep item unavailable",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.46,
        f"{label}\n{title}",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.24,
        str(reason),
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    return fig


def fig_to_rgb_array(fig: Any) -> np.ndarray:
    """Render a matplotlib figure to an H x W x 3 uint8 numpy array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def render_sweep_frame(
    sweep: dict[str, Any],
    item: dict[str, Any],
    frame_index: int,
    total_frames: int,
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
    title_formatter: Callable[[dict[str, Any], Any, int, int], str] = format_sweep_frame_title,
) -> tuple[np.ndarray, str]:
    """Render one sweep item to a frame array and title."""
    result = item.get("result") if isinstance(item, dict) else None
    value = item.get("value") if isinstance(item, dict) else None
    if title_fn is not None:
        try:
            title = title_fn(
                value,
                frame_index=frame_index,
                total_frames=total_frames,
                sweep=sweep,
            )
        except TypeError:
            title = title_fn(value)
    else:
        title = title_formatter(sweep, value, frame_index, total_frames)

    if result is None:
        fig = make_sweep_placeholder_figure(
            sweep,
            item,
            frame_index,
            total_frames,
            reason=describe_unavailable_sweep_item(item),
            figsize=figsize,
            title_formatter=title_formatter,
        )
    else:
        before_figs = set(plt.get_fignums())
        try:
            returned = plot_fn(result)
            fig = extract_figure_from_plot_result(returned)
        except Exception as exc:
            if close_frames:
                for fignum in set(plt.get_fignums()) - before_figs:
                    plt.close(fignum)
            fig = make_sweep_placeholder_figure(
                sweep,
                item,
                frame_index,
                total_frames,
                reason=f"Plot failed: {type(exc).__name__}: {exc}",
                figsize=figsize,
                title_formatter=title_formatter,
            )

    frame_rgb = fig_to_rgb_array(fig)

    if close_frames:
        plt.close(fig)

    return frame_rgb, str(title)
