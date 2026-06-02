"""Reusable plotting helpers for notebook-facing analysis workflows."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .spectral import (
    fold_time_matrix_by_modulus,
    fold_time_series_by_modulus,
    normalize_time_modulus,
)


def time_axis_label(modulus: float | int | None) -> str:
    """Return a time-axis label for either full or folded traces."""
    modulus_value = normalize_time_modulus(modulus)
    if modulus_value is None:
        return "Time (ms)"
    return f"Time modulo {float(modulus_value):g} ms"


def plot_time_series(
    t: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    ax: Any = None,
    modulus: float | int | None = None,
    dt_ms: float | None = None,
    title: str = "Signal Trace",
    ylabel: str = "Signal",
    linewidth: float = 1.0,
    **plot_kwargs: Any,
) -> Any:
    """Plot one time/value trace, optionally folded by a time modulus."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    plot_t, plot_y = fold_time_series_by_modulus(t, y, modulus, dt_ms=dt_ms)
    ax.plot(plot_t, plot_y, linewidth=linewidth, **plot_kwargs)
    ax.set_xlabel(time_axis_label(modulus))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_named_time_series(
    t: np.ndarray | list[float],
    traces: dict[str, np.ndarray | list[float]],
    *,
    ax: Any = None,
    modulus: float | int | None = None,
    dt_ms: float | None = None,
    title: str = "Signals Over Time",
    ylabel: str = "Signal",
    legend_loc: str = "upper right",
    line_kwargs_by_name: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Plot multiple named traces that share the same time base."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    line_kwargs_by_name = dict(line_kwargs_by_name or {})
    for name, values in traces.items():
        plot_t, plot_values = fold_time_series_by_modulus(t, values, modulus, dt_ms=dt_ms)
        ax.plot(plot_t, plot_values, linewidth=1.2, label=name, **line_kwargs_by_name.get(name, {}))
    ax.set_xlabel(time_axis_label(modulus))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if traces:
        ax.legend(loc=legend_loc)
    return ax


def plot_time_frequency_map(
    times_ms: np.ndarray | list[float],
    freqs_hz: np.ndarray | list[float],
    power: np.ndarray,
    *,
    ax: Any = None,
    modulus: float | int | None = None,
    title: str = "Time/Frequency Map",
    ylabel: str = "Frequency (Hz)",
    colorbar_label: str = "Power",
    power_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Plot a time/frequency power matrix, optionally folded by time modulus."""
    ax = ax or plt.subplots(figsize=(14, 5))[1]
    plot_t, plot_power = fold_time_matrix_by_modulus(times_ms, power, modulus)
    if power_transform is not None:
        plot_power = np.asarray(power_transform(plot_power), dtype=float)
    mesh = ax.pcolormesh(
        plot_t,
        np.asarray(freqs_hz, dtype=float),
        plot_power,
        shading="auto",
        **dict(pcolormesh_kwargs or {}),
    )
    ax.set_xlabel(time_axis_label(modulus))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label=colorbar_label)
    return ax


def plot_band_power_summary(
    summary: dict[str, Any],
    *,
    signal_label: str | None = None,
    figsize: tuple[float, float] = (12, 4),
    absolute_color: str = "tab:blue",
    relative_color: str = "tab:green",
) -> tuple[Any, Any]:
    """Plot absolute and relative band-power bars from one computed summary."""
    names = list(summary["band_power"].keys())
    absolute = [summary["band_power"][name] for name in names]
    relative = [summary["relative_band_power"][name] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
    axes[0].bar(names, absolute, color=absolute_color)
    axes[0].set_title(f"{signal_label} Band Power" if signal_label else "Band Power")
    axes[0].set_ylabel("Integrated PSD")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, relative, color=relative_color)
    axes[1].set_title("Relative Band Power")
    axes[1].set_ylabel("Fraction")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig, axes


def plot_stacked_labeled_traces(
    rows: Sequence[tuple[str, Any, Any]],
    *,
    ax: Any = None,
    title: str = "Stacked Traces",
    xlabel: str = "Time (ms)",
    ylabel: str = "Offset Value",
    linewidth: float = 1.0,
    line_kwargs_fn: Callable[[tuple[str, Any, Any]], dict[str, Any]] | None = None,
    offset_step_fn: Callable[[tuple[str, Any, Any]], float] | None = None,
    legend_loc: str = "upper right",
    legend_fontsize: float = 8.0,
    legend_ncol: int = 2,
) -> Any:
    """Plot labeled traces stacked vertically with configurable offsets."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    line_kwargs_fn = line_kwargs_fn or (lambda _row: {})
    offset_step_fn = offset_step_fn or (lambda _row: 1.0)

    offset = 0.0
    for row in rows:
        label, t, values = row
        plot_t = np.asarray(t, dtype=float)
        plot_values = np.asarray(values, dtype=float) + float(offset)
        ax.plot(
            plot_t,
            plot_values,
            linewidth=linewidth,
            label=str(label),
            **dict(line_kwargs_fn(row)),
        )
        offset += float(offset_step_fn(row))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ax.lines:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, ncol=legend_ncol)
    return ax
