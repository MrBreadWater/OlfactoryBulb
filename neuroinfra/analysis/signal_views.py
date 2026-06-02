"""Reusable plotting helpers built on top of named-signal resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .plotting import (
    plot_band_power_summary,
    plot_named_time_series,
    plot_time_frequency_map,
    plot_time_series,
)
from .spectral import (
    compute_band_power_summary,
    compute_bandpassed_signal,
    compute_spectrogram,
    compute_welch_psd,
    compute_wavelet_band_power,
    compute_wavelet_map,
)


@dataclass(frozen=True)
class SignalPsdOverlay:
    """One additional PSD curve to render on a resolved-signal overview."""

    freqs_hz: np.ndarray
    power: np.ndarray
    label: str | None = None
    color: str | None = None
    linewidth: float = 1.0
    linestyle: str = "--"


def log_spectrogram_display_power(
    power: np.ndarray,
    *,
    floor: float = 1e-8,
) -> np.ndarray:
    """Normalize spectrogram power into a stable display range."""
    values = np.log(np.asarray(power, dtype=float) + float(floor))
    values -= values.min()
    return values


def compute_resolved_bandpassed_signal(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    dt_ms: float | None = 0.1,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 120.0,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one named signal and return a band-passed copy."""
    signal_t, signal_y = resolve_signal_fn(result, signal, dt_ms)
    return compute_bandpassed_signal(
        signal_t,
        signal_y,
        dt_ms=dt_ms,
        lowcut_hz=lowcut_hz,
        highcut_hz=highcut_hz,
        order=order,
    )


def compute_resolved_band_power_summary(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float | None = 0.1,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
) -> dict[str, Any]:
    """Compute named-band Welch power metrics for one resolved signal."""
    signal_t, signal_y = resolve_signal_fn(result, signal, dt_ms)
    summary = compute_band_power_summary(
        signal_t,
        signal_y,
        bands=bands,
        dt_ms=dt_ms,
        relative_band=relative_band,
    )
    summary["signal"] = signal
    return summary


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


def plot_resolved_signal_psd_overview(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    dt_ms: float = 0.1,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 300.0,
    order: int = 4,
    psd_xlim_hz: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14.0, 10.0),
    signal_label: str | None = None,
    psd_overlay_builder: Callable[[np.ndarray, np.ndarray], Sequence[SignalPsdOverlay] | None] | None = None,
) -> tuple[Any, Any]:
    """Plot raw trace, band-passed trace, and Welch PSD for one resolved signal."""
    resolved_label = str(signal_label or signal).strip() or signal
    signal_t, signal_y = resolve_signal_fn(result, signal, dt_ms)
    bp_t, bp_y = compute_resolved_bandpassed_signal(
        result,
        signal=signal,
        resolve_signal_fn=resolve_signal_fn,
        dt_ms=dt_ms,
        lowcut_hz=lowcut_hz,
        highcut_hz=highcut_hz,
        order=order,
    )
    freqs, power = compute_welch_psd(
        bp_t,
        bp_y,
        dt_ms=dt_ms,
        nperseg=2048,
        remove_mean=False,
    )

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)
    axes[0].plot(signal_t, signal_y, color="black", linewidth=1.0)
    axes[0].set_title(f"Raw {resolved_label}")
    axes[0].set_ylabel(resolved_label)

    axes[1].plot(bp_t, bp_y, color="tab:purple", linewidth=1.0)
    axes[1].set_title(f"Band-passed {resolved_label} ({lowcut_hz:.0f}-{highcut_hz:.0f} Hz)")
    axes[1].set_ylabel(f"Filtered {resolved_label}")

    axes[2].plot(freqs, power, color="tab:green", linewidth=1.0, label="Measured PSD")
    overlays = list(psd_overlay_builder(freqs, power) or []) if psd_overlay_builder is not None else []
    for overlay in overlays:
        axes[2].plot(
            np.asarray(overlay.freqs_hz, dtype=float),
            np.asarray(overlay.power, dtype=float),
            color=overlay.color,
            linewidth=float(overlay.linewidth),
            linestyle=str(overlay.linestyle),
            label=overlay.label,
        )
    if overlays:
        axes[2].legend(loc="upper right", fontsize=9)

    if psd_xlim_hz is None:
        psd_xlim_hz = (0.0, float(highcut_hz))
    axes[2].set_xlim(float(psd_xlim_hz[0]), float(psd_xlim_hz[1]))
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("PSD")
    axes[2].set_title("Welch Power Spectrum")
    fig.tight_layout()
    return fig, axes


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


def plot_resolved_band_power_summary(
    result: dict[str, Any],
    *,
    signal: str,
    resolve_signal_fn: Callable[[dict[str, Any], str, float | None], tuple[np.ndarray, np.ndarray]],
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float | None = 0.1,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
    signal_label: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot absolute and relative band-power summaries for one resolved signal."""
    summary = compute_resolved_band_power_summary(
        result,
        signal=signal,
        resolve_signal_fn=resolve_signal_fn,
        bands=bands,
        dt_ms=dt_ms,
        relative_band=relative_band,
    )
    fig, axes = plot_band_power_summary(summary, signal_label=signal_label or signal)
    return fig, axes, summary
