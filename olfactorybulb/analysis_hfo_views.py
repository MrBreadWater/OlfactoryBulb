"""Concrete olfactory-bulb HFO/LFP view adapters built on neuroinfra."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from neuroinfra.analysis.signal_views import SignalPsdOverlay

DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ = (130.0, 230.0)
DEFAULT_PSD_TEMPLATE_FLOOR = 1e-5


def build_psd_template_overlays(
    *,
    psd_template_kind: str,
    psd_template_fit_band_hz: tuple[float, float],
    psd_template_scale_method: str,
    psd_template_floor: float,
    psd_template_color: str,
    scaled_curve_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> Callable[[np.ndarray, np.ndarray], list[SignalPsdOverlay]]:
    """Return a builder that maps one measured PSD onto overlay curves."""

    def _builder(freqs: np.ndarray, power: np.ndarray) -> list[SignalPsdOverlay]:
        try:
            resolver = scaled_curve_fn
            if resolver is None:
                from olfactorybulb.hfo_optimizer import scaled_psd_template_curve

                resolver = scaled_psd_template_curve
            template_freqs, template_power = resolver(
                psd_template_kind,
                freqs,
                power,
                fit_band_hz=psd_template_fit_band_hz,
                method=psd_template_scale_method,
                floor=psd_template_floor,
            )
        except Exception:
            return []
        return [
            SignalPsdOverlay(
                freqs_hz=np.asarray(template_freqs, dtype=float),
                power=np.asarray(template_power, dtype=float),
                label=f"Template ({psd_template_kind})",
                color=psd_template_color,
                linewidth=1.0,
                linestyle="--",
            )
        ]

    return _builder


def plot_lfp_overview(
    profile: Any,
    result: dict[str, Any],
    *,
    dt_ms: float = 0.1,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 300.0,
    psd_xlim_hz: tuple[float, float] | None = None,
    show_psd_target_template: bool = True,
    psd_template_kind: str = "ketamine",
    psd_template_fit_band_hz: tuple[float, float] = DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ,
    psd_template_scale_method: str = "area",
    psd_template_floor: float = DEFAULT_PSD_TEMPLATE_FLOOR,
    psd_template_color: str = "tab:orange",
) -> tuple[Any, Any]:
    """Plot raw LFP, band-passed LFP, and a Welch PSD summary."""
    overlay_builder = None
    if show_psd_target_template:
        overlay_builder = build_psd_template_overlays(
            psd_template_kind=psd_template_kind,
            psd_template_fit_band_hz=psd_template_fit_band_hz,
            psd_template_scale_method=psd_template_scale_method,
            psd_template_floor=psd_template_floor,
            psd_template_color=psd_template_color,
        )
    return profile.require_signal_views().plot_signal_psd_overview(
        result,
        signal="lfp",
        dt_ms=dt_ms,
        lowcut_hz=lowcut_hz,
        highcut_hz=highcut_hz,
        psd_xlim_hz=psd_xlim_hz,
        signal_label="LFP",
        psd_overlay_builder=overlay_builder,
    )


def plot_hfo_power_summary(
    profile: Any,
    result: dict[str, Any],
    *,
    signal: str = "lfp",
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float = 0.1,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot absolute and relative HFO band power for a named signal."""
    return profile.require_signal_views().plot_band_power_summary(
        result,
        signal=signal,
        bands=bands,
        dt_ms=dt_ms,
        relative_band=relative_band,
    )
