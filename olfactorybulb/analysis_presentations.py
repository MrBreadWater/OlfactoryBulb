"""Concrete olfactory-bulb notebook presentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from olfactorybulb.analysis_hfo_views import (
    DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ,
    DEFAULT_PSD_TEMPLATE_FLOOR,
)


@dataclass(frozen=True)
class SweepAnimationHooks:
    """Plotting hooks for notebook sweep animation presets."""

    animate_sweep_fn: Callable[..., Any]
    plot_named_signal_fn: Callable[..., Any]
    plot_lfp_overview_fn: Callable[..., Any]
    plot_spectrogram_fn: Callable[..., Any]
    plot_wavelet_fn: Callable[..., Any]
    get_named_signal_fn: Callable[..., tuple[np.ndarray, np.ndarray]]
    compute_wavelet_map_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    plt_module: Any


@dataclass(frozen=True)
class StandardOutputHooks:
    """Plotting hooks for the standard notebook output bundle."""

    plot_input_overview_fn: Callable[..., Any]
    plot_voltage_traces_fn: Callable[..., Any]
    plot_spike_raster_fn: Callable[..., Any]
    plot_gc_output_overview_fn: Callable[..., Any]
    plot_lfp_overview_fn: Callable[..., Any]
    plot_spectrogram_fn: Callable[..., Any]
    plot_wavelet_fn: Callable[..., Any]
    plot_wavelet_band_power_fn: Callable[..., Any]
    plt_show_fn: Callable[[], None]


def animate_lfp_sweep(
    hooks: SweepAnimationHooks,
    sweep: dict[str, Any],
    *,
    signal: str = "lfp",
    dt_ms: float = 0.1,
    interval: int = 100,
) -> Any:
    """Animate trace-style outputs across a one-parameter sweep."""
    if signal != "lfp":
        return hooks.animate_sweep_fn(
            sweep,
            lambda result: hooks.plot_named_signal_fn(result, signal=signal, dt_ms=dt_ms),
            figsize=(12, 4),
            interval=interval,
        )

    return hooks.animate_sweep_fn(
        sweep,
        lambda result: hooks.plot_lfp_overview_fn(result, dt_ms=dt_ms),
        figsize=(12, 7),
        interval=interval,
    )


def animate_spectrogram_sweep(
    hooks: SweepAnimationHooks,
    sweep: dict[str, Any],
    *,
    signal: str = "lfp",
    dt_ms: float = 0.1,
    max_freq_hz: float = 250.0,
    nperseg: int = 256,
    noverlap: int = 192,
    interval: int = 100,
) -> Any:
    """Animate spectrograms across a one-parameter sweep."""
    return hooks.animate_sweep_fn(
        sweep,
        lambda result: hooks.plot_spectrogram_fn(
            result,
            signal=signal,
            dt_ms=dt_ms,
            max_freq_hz=max_freq_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        ),
        figsize=(12, 4),
        interval=interval,
    )


def animate_wavelet_sweep(
    hooks: SweepAnimationHooks,
    sweep: dict[str, Any],
    *,
    signal: str = "lfp",
    dt_ms: float = 0.1,
    interval: int = 100,
) -> Any:
    """Animate wavelet maps across a one-parameter sweep."""
    return hooks.animate_sweep_fn(
        sweep,
        lambda result: hooks.plot_wavelet_fn(result, signal=signal, dt_ms=dt_ms),
        figsize=(12, 4),
        interval=interval,
    )


def animate_sniff_average_sweep(
    hooks: SweepAnimationHooks,
    sweep: dict[str, Any],
    *,
    dt_ms: float = 0.1,
    sniff_count: int = 8,
    interval: int = 100,
) -> Any:
    """Animate sniff-averaged wavelet views across a sweep."""

    def _plot(result: dict[str, Any]) -> Any:
        signal_t, signal_y = hooks.get_named_signal_fn(result, signal="lfp", dt_ms=dt_ms)
        _t, _bp, freqs, power = hooks.compute_wavelet_map_fn(signal_t, signal_y, dt_ms=dt_ms)
        sniff_duration_ms = 200.0
        skip_first_n_sniffs = 1
        step = max(1, int(round(sniff_duration_ms / dt_ms)))
        start_index = step * skip_first_n_sniffs
        available_columns = max(0, power.shape[1] - start_index)
        chunk_count = min(int(sniff_count), available_columns // step)
        if chunk_count > 0:
            chunks = [
                power[:, start_index + i * step : start_index + (i + 1) * step]
                for i in range(chunk_count)
            ]
            averaged = np.mean(np.asarray(chunks, dtype=float), axis=0)
        else:
            averaged = power[:, :step]
        plot_t = np.arange(averaged.shape[1], dtype=float) * dt_ms
        fig, ax = hooks.plt_module.subplots(figsize=(5, 5))
        ax.contourf(
            plot_t,
            freqs,
            averaged,
            256,
            cmap="jet",
        )
        ax.set_ylim((20, 140))
        ax.set_xlabel("Time Since Sniff Onset [ms]")
        ax.set_ylabel("Frequency [Hz]")
        return fig

    return hooks.animate_sweep_fn(sweep, _plot, figsize=(5, 5), interval=interval)


def show_all_outputs(
    hooks: StandardOutputHooks,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Render the standard notebook figure set for one loaded result."""
    config = config or {}
    dt_ms = float(config.get("analysis_dt_ms", 0.1))
    input_bin_ms = float(config.get("input_bin_ms", 5.0))
    input_smooth_ms = float(config.get("input_smooth_sigma_ms", 10.0))
    input_max_segments = int(config.get("input_max_segments", 120))
    input_norm = str(config.get("input_rate_normalization", "per_target_cell"))
    max_voltage = int(config.get("max_voltage_traces_per_type", 4))
    max_raster = int(config.get("max_spike_raster_cells_per_type", 24))
    gc_bin_ms = float(config.get("gc_output_bin_ms", 5.0))
    gc_smooth_ms = float(config.get("gc_output_smooth_sigma_ms", 10.0))
    gc_max_connections = int(config.get("gc_output_max_connections", 120))
    gc_norm = str(config.get("gc_output_rate_normalization", "per_target_cell"))
    show_raw_voltage_traces = bool(config.get("show_voltage_traces", False))
    show_psd_template = bool(config.get("lfp_show_psd_target_template", True))
    psd_template_kind = str(config.get("lfp_psd_template_kind", "ketamine"))
    psd_template_fit = config.get("lfp_psd_template_fit_band_hz", DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ)
    psd_template_floor = config.get("lfp_psd_template_floor", DEFAULT_PSD_TEMPLATE_FLOOR)
    if isinstance(psd_template_fit, (list, tuple)) and len(psd_template_fit) == 2:
        psd_template_fit = (float(psd_template_fit[0]), float(psd_template_fit[1]))
    else:
        psd_template_fit = DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ
    try:
        psd_template_floor = float(psd_template_floor)
    except (TypeError, ValueError):
        psd_template_floor = DEFAULT_PSD_TEMPLATE_FLOOR
    psd_xlim_hz = config.get("lfp_psd_xlim_hz", (0.0, 300.0))
    if isinstance(psd_xlim_hz, (list, tuple)) and len(psd_xlim_hz) == 2:
        psd_xlim_hz = (float(psd_xlim_hz[0]), float(psd_xlim_hz[1]))
    else:
        psd_xlim_hz = None
    spectrogram_max_freq_hz = float(config.get("spectrogram_max_freq_hz", 250.0))
    spectrogram_nperseg = int(config.get("spectrogram_nperseg", 256))
    spectrogram_noverlap = int(config.get("spectrogram_noverlap", 192))

    hooks.plot_input_overview_fn(
        result,
        bin_ms=input_bin_ms,
        smooth_sigma_ms=input_smooth_ms,
        max_segments=input_max_segments,
        normalization=input_norm,
    )
    hooks.plt_show_fn()

    if show_raw_voltage_traces:
        hooks.plot_voltage_traces_fn(result, max_per_type=max_voltage)
        hooks.plt_show_fn()

    hooks.plot_spike_raster_fn(result, max_cells_per_type=max_raster)
    hooks.plt_show_fn()

    hooks.plot_gc_output_overview_fn(
        result,
        bin_ms=gc_bin_ms,
        smooth_sigma_ms=gc_smooth_ms,
        max_connections=gc_max_connections,
        normalization=gc_norm,
    )
    hooks.plt_show_fn()

    hooks.plot_lfp_overview_fn(
        result,
        dt_ms=dt_ms,
        show_psd_target_template=show_psd_template,
        psd_template_kind=psd_template_kind,
        psd_template_fit_band_hz=psd_template_fit,
        psd_template_floor=psd_template_floor,
        psd_xlim_hz=psd_xlim_hz,
    )
    hooks.plt_show_fn()

    hooks.plot_spectrogram_fn(
        result,
        signal=config.get("spectrogram_signal", "lfp"),
        dt_ms=dt_ms,
        max_freq_hz=spectrogram_max_freq_hz,
        nperseg=spectrogram_nperseg,
        noverlap=spectrogram_noverlap,
    )
    hooks.plt_show_fn()

    hooks.plot_wavelet_fn(result, signal=config.get("wavelet_signal", "lfp"), dt_ms=dt_ms)
    hooks.plt_show_fn()

    hooks.plot_wavelet_band_power_fn(result, signal=config.get("wavelet_signal", "lfp"), dt_ms=dt_ms)
    hooks.plt_show_fn()
