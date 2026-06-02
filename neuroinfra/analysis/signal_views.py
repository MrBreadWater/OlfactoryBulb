"""Reusable plotting helpers built on top of named-signal resolvers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .plotting import (
    plot_named_time_series,
    plot_time_frequency_map,
    plot_time_series,
)
from .spectral import (
    compute_spectrogram,
    compute_wavelet_band_power,
    compute_wavelet_map,
)


def log_spectrogram_display_power(
    power: np.ndarray,
    *,
    floor: float = 1e-8,
) -> np.ndarray:
    """Normalize spectrogram power into a stable display range."""
    values = np.log(np.asarray(power, dtype=float) + float(floor))
    values -= values.min()
    return values


def plot_resolved_signal(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    dt_ms: float = 0.1,
    ax: Any = None,
    modulus: float | None = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Plot one named signal using a caller-provided resolver."""
    t, y = resolve_signal_fn(result, signal, dt_ms)
    return plot_time_series(
        t,
        y,
        ax=ax,
        modulus=modulus,
        dt_ms=dt_ms,
        title=title or f"{signal} Trace",
        ylabel=ylabel or signal,
    )


def plot_resolved_spectrogram(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    dt_ms: float = 0.1,
    max_freq_hz: float = 250.0,
    nperseg: int = 256,
    noverlap: int = 192,
    ax: Any = None,
    modulus: float | None = None,
    title: str | None = None,
    colorbar_label: str = "Power (dB)",
    power_transform: Callable[[np.ndarray], np.ndarray] = log_spectrogram_display_power,
) -> Any:
    """Plot a spectrogram for one resolved named signal."""
    signal_t, signal_y = resolve_signal_fn(result, signal, dt_ms)
    times_ms, freqs, power = compute_spectrogram(
        signal_t,
        signal_y,
        dt_ms=dt_ms,
        max_freq_hz=max_freq_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return plot_time_frequency_map(
        times_ms,
        freqs,
        power,
        ax=ax,
        modulus=modulus,
        title=title or f"{signal.upper()} Spectrogram",
        colorbar_label=colorbar_label,
        power_transform=power_transform,
    )


def plot_resolved_wavelet(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    dt_ms: float = 0.1,
    ax: Any = None,
    modulus: float | None = None,
    title: str | None = None,
    colorbar_label: str = "log(1 + |cwt|)",
) -> Any:
    """Plot the continuous wavelet power map for one resolved named signal."""
    signal_t, signal_y = resolve_signal_fn(result, signal, dt_ms)
    t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
    return plot_time_frequency_map(
        t,
        freqs,
        power,
        ax=ax,
        modulus=modulus,
        title=title or f"{signal.upper()} Wavelet Power",
        colorbar_label=colorbar_label,
    )


def plot_resolved_wavelet_band_power(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    dt_ms: float = 0.1,
    bands: dict[str, tuple[float, float]] | None = None,
    ax: Any = None,
    modulus: float | None = None,
    title: str = "Band Power Over Time",
    ylabel: str = "Mean Wavelet Power",
) -> Any:
    """Plot band-collapsed wavelet power traces for one resolved named signal."""
    signal_t, signal_y = resolve_signal_fn(result, signal, dt_ms)
    t, _freqs, _power, traces = compute_wavelet_band_power(
        signal_t,
        signal_y,
        bands=bands,
        dt_ms=dt_ms,
    )
    return plot_named_time_series(
        t,
        traces,
        ax=ax,
        modulus=modulus,
        dt_ms=dt_ms,
        title=title,
        ylabel=ylabel,
    )
