"""Reusable frequency-sample plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde

from .spectral import normalize_time_modulus


@dataclass
class FrequencyPlotConfig:
    """Shared rendering controls for spike/event frequency distribution plots."""

    modulus: float | None = 1e8
    max_freq_hz: float = 200.0
    kde_bw_method: str | float = "scott"
    kde1d_engine: str = "histogram"
    kde_bw_x: float = 0.15
    kde_bw_y: float = 0.2
    kde2d_engine: str = "histogram"
    kde_resolution_t: int = 100
    kde_resolution_f: int = 100
    kde_f_resolution: int = 1600
    num_time_bins: int = 32
    bin_alpha: float = 0.5
    kde_cmap: str = "inferno"
    dot_size: float = 5.0
    dot_alpha: float = 0.2
    strip_plot: bool = True
    guide_line_spacing_ms: float = 0.0


@dataclass(frozen=True)
class ResultFrequencyPlotFamily:
    """One result-backed frequency-sample family plus its default plot titles."""

    collect_samples_fn: Any
    selection_label_fn: Any
    title_1d: str
    title_2d: str
    title_time_binned: str


@dataclass(frozen=True)
class ResultFrequencyPlotSuite:
    """Behavioral wrapper around one reusable result-backed frequency family."""

    family: ResultFrequencyPlotFamily

    def plot_kde_1d(
        self,
        result: dict[str, Any],
        *,
        config: "FrequencyPlotConfig | dict[str, Any] | None" = None,
        ax: Any = None,
        title: str | None = None,
        selection: Any = None,
        collector_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Render the 1D KDE view for this family."""
        return plot_result_frequency_kde_1d(
            result,
            self.family,
            config=config,
            ax=ax,
            title=title,
            selection=selection,
            collector_kwargs=collector_kwargs,
        )

    def plot_kde_2d(
        self,
        result: dict[str, Any],
        *,
        config: "FrequencyPlotConfig | dict[str, Any] | None" = None,
        ax: Any = None,
        title: str | None = None,
        selection: Any = None,
        collector_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Render the 2D time/frequency KDE view for this family."""
        return plot_result_frequency_kde_2d(
            result,
            self.family,
            config=config,
            ax=ax,
            title=title,
            selection=selection,
            collector_kwargs=collector_kwargs,
        )

    def plot_time_binned(
        self,
        result: dict[str, Any],
        *,
        config: "FrequencyPlotConfig | dict[str, Any] | None" = None,
        ax: Any = None,
        title: str | None = None,
        selection: Any = None,
        collector_kwargs: Mapping[str, Any] | None = None,
        show_dots: bool = True,
        show_ridgeline_kde: bool = False,
    ) -> Any:
        """Render the time-binned distribution view for this family."""
        return plot_result_frequency_time_binned(
            result,
            self.family,
            config=config,
            ax=ax,
            title=title,
            selection=selection,
            collector_kwargs=collector_kwargs,
            show_dots=show_dots,
            show_ridgeline_kde=show_ridgeline_kde,
        )


def coerce_frequency_plot_config(
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    **overrides: Any,
) -> FrequencyPlotConfig:
    """Normalize one frequency-plot config input into a dataclass instance."""
    if config is None:
        base = FrequencyPlotConfig()
    elif isinstance(config, FrequencyPlotConfig):
        base = FrequencyPlotConfig(**vars(config))
    elif isinstance(config, dict):
        base = FrequencyPlotConfig(**config)
    else:
        raise TypeError(f"Unsupported frequency-plot config type {type(config)!r}")

    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    return base


def frequency_plot_config_with_modulus(
    config: FrequencyPlotConfig | dict[str, Any] | None,
    modulus: float | int | None,
) -> FrequencyPlotConfig:
    """Copy one frequency plot config while replacing its time modulus."""
    copied = replace(coerce_frequency_plot_config(config))
    copied.modulus = normalize_time_modulus(modulus)
    return copied


def plot_result_frequency_kde_1d(
    result: dict[str, Any],
    family: ResultFrequencyPlotFamily,
    *,
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
    selection: Any = None,
    collector_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Collect one result-backed frequency family and render its 1D KDE view."""
    plot_config = coerce_frequency_plot_config(config)
    data = family.collect_samples_fn(
        result,
        modulus=plot_config.modulus,
        **dict(collector_kwargs or {}),
    )
    label = family.selection_label_fn(selection)
    return plot_frequency_kde_1d_from_samples(
        np.asarray(data["freqs"], dtype=float),
        config=plot_config,
        title=title or f"{family.title_1d} ({label})",
        ax=ax,
    )


def plot_result_frequency_kde_2d(
    result: dict[str, Any],
    family: ResultFrequencyPlotFamily,
    *,
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
    selection: Any = None,
    collector_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Collect one result-backed frequency family and render its 2D KDE view."""
    plot_config = coerce_frequency_plot_config(config)
    data = family.collect_samples_fn(
        result,
        modulus=plot_config.modulus,
        **dict(collector_kwargs or {}),
    )
    label = family.selection_label_fn(selection)
    return plot_frequency_kde_2d_from_samples(
        np.asarray(data["times"], dtype=float),
        np.asarray(data["freqs"], dtype=float),
        config=plot_config,
        title=title or f"{family.title_2d} ({label})",
        ax=ax,
    )


def plot_result_frequency_time_binned(
    result: dict[str, Any],
    family: ResultFrequencyPlotFamily,
    *,
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
    selection: Any = None,
    collector_kwargs: Mapping[str, Any] | None = None,
    show_dots: bool = True,
    show_ridgeline_kde: bool = False,
) -> Any:
    """Collect one result-backed frequency family and render its time-binned view."""
    plot_config = coerce_frequency_plot_config(config)
    data = family.collect_samples_fn(
        result,
        modulus=plot_config.modulus,
        **dict(collector_kwargs or {}),
    )
    label = family.selection_label_fn(selection)
    return plot_frequency_time_binned_from_samples(
        np.asarray(data["times"], dtype=float),
        np.asarray(data["freqs"], dtype=float),
        config=plot_config,
        title=title or f"{family.title_time_binned} ({label})",
        ax=ax,
        show_dots=show_dots,
        show_ridgeline_kde=show_ridgeline_kde,
    )


def _apply_frequency_kde_y_scale(kde: Any, scale_y: float) -> None:
    """Rescale a 1D KDE in-place along its frequency axis."""
    if float(scale_y) == 1.0:
        return
    kde.covariance *= float(scale_y) ** 2
    kde.cho_cov = np.linalg.cholesky(kde.covariance)
    kde.log_det = 2 * np.log(np.diag(kde.cho_cov * np.sqrt(2 * np.pi))).sum()


def _apply_frequency_kde_xy_scale(kernel: Any, scale_x: float, scale_y: float) -> None:
    """Rescale a 2D time/frequency KDE in-place."""
    if float(scale_x) == 1.0 and float(scale_y) == 1.0:
        return
    kernel.covariance[0, 0] *= float(scale_x) ** 2
    kernel.covariance[1, 1] *= float(scale_y) ** 2
    kernel.covariance[0, 1] *= float(scale_x) * float(scale_y)
    kernel.covariance[1, 0] *= float(scale_x) * float(scale_y)
    kernel.cho_cov = np.linalg.cholesky(kernel.covariance)
    kernel.log_det = 2 * np.log(np.diag(kernel.cho_cov * np.sqrt(2 * np.pi))).sum()


def plot_frequency_kde_1d_from_samples(
    freqs: np.ndarray,
    *,
    config: FrequencyPlotConfig,
    title: str,
    ax: Any = None,
) -> Any:
    """Plot a 1D KDE from frequency samples."""
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[np.isfinite(freqs)]
    if len(freqs) == 0:
        ax.text(0.5, 0.5, "No frequency samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, float(config.max_freq_hz))
        return ax

    f_upper = max(float(config.max_freq_hz), float(np.max(freqs)) * 1.1)
    engine = str(getattr(config, "kde1d_engine", "histogram")).strip().lower()
    if engine in {"exact", "gaussian", "gaussian_kde", "scipy"}:
        kde = gaussian_kde(freqs, bw_method=config.kde_bw_method)
        _apply_frequency_kde_y_scale(kde, config.kde_bw_y)
        f_range = np.linspace(0.0, f_upper, int(config.kde_f_resolution))
        density = kde(f_range)
    else:
        bins = max(16, int(config.kde_f_resolution))
        clipped = freqs[(freqs >= 0.0) & (freqs <= f_upper)]
        if len(clipped) == 0:
            clipped = freqs
        density, edges = np.histogram(clipped, bins=bins, range=(0.0, f_upper), density=True)
        sigma = max(0.0, float(config.kde_bw_y) * 8.0)
        density = gaussian_filter1d(density, sigma=sigma, mode="nearest")
        f_range = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(f_range, density)
    ax.fill_between(f_range, density, alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0, float(config.max_freq_hz))
    return ax


def plot_frequency_kde_2d_from_samples(
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    config: FrequencyPlotConfig,
    title: str,
    ax: Any = None,
) -> Any:
    """Plot a 2D time/frequency KDE from samples."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    finite = np.isfinite(times) & np.isfinite(freqs)
    times = times[finite]
    freqs = freqs[finite]
    if len(times) < 2 or len(freqs) < 2:
        ax.text(0.5, 0.5, "Not enough frequency samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, float(config.max_freq_hz))
        return ax

    tstop = float(np.max(times))
    max_freq_hz = float(config.max_freq_hz)
    engine = str(getattr(config, "kde2d_engine", "histogram")).strip().lower()
    if engine in {"exact", "gaussian", "gaussian_kde", "scipy"}:
        kernel = gaussian_kde(np.vstack([times, freqs]), bw_method=config.kde_bw_method)
        _apply_frequency_kde_xy_scale(kernel, config.kde_bw_x, config.kde_bw_y)
        t_grid = np.linspace(0.0, tstop, int(config.kde_resolution_t))
        f_grid = np.linspace(0.0, max_freq_hz, int(config.kde_resolution_f))
        t_mesh, f_mesh = np.meshgrid(t_grid, f_grid)
        positions = np.vstack([t_mesh.ravel(), f_mesh.ravel()])
        density = np.reshape(kernel(positions).T, t_mesh.shape)
    else:
        density, _t_edges, _f_edges = np.histogram2d(
            times,
            freqs,
            bins=(int(config.kde_resolution_t), int(config.kde_resolution_f)),
            range=((0.0, tstop), (0.0, max_freq_hz)),
        )
        sigma_t = max(0.0, float(config.kde_bw_x) * 6.0)
        sigma_f = max(0.0, float(config.kde_bw_y) * 6.0)
        density = gaussian_filter(density.T, sigma=(sigma_f, sigma_t), mode="nearest")

    im = ax.imshow(
        density,
        origin="lower",
        extent=[0, tstop, 0, max_freq_hz],
        aspect="auto",
        cmap=config.kde_cmap,
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Density (KDE)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return ax


def plot_frequency_time_binned_from_samples(
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    config: FrequencyPlotConfig,
    title: str,
    ax: Any = None,
    show_dots: bool = True,
    show_ridgeline_kde: bool = False,
) -> Any:
    """Plot time-binned frequency distributions from midpoint/frequency samples."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if len(times) == 0 or len(freqs) == 0:
        ax.text(0.5, 0.5, "No frequency samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, float(config.max_freq_hz))
        return ax

    tstop = float(np.max(times))
    t_bins = np.linspace(0.0, tstop, int(config.num_time_bins) + 1)
    if len(t_bins) < 2:
        t_bins = np.array([0.0, max(tstop, 1.0)], dtype=float)
    bin_width = float(t_bins[1] - t_bins[0])

    for i in range(len(t_bins) - 1):
        t_start, t_end = float(t_bins[i]), float(t_bins[i + 1])
        mask = (times >= t_start) & (times < t_end)
        if not np.any(mask):
            continue
        bin_f = freqs[mask]

        if show_dots:
            if bool(config.strip_plot):
                x_pos = np.full_like(bin_f, t_start + bin_width / 2.0)
            else:
                jitter = np.random.uniform(0.0, bin_width * 0.8, size=len(bin_f))
                x_pos = t_start + jitter
            ax.scatter(
                x_pos,
                bin_f,
                s=float(config.dot_size),
                alpha=float(config.dot_alpha),
                color="black",
                edgecolors="none",
            )

        if show_ridgeline_kde and len(bin_f) > 2:
            kde = gaussian_kde(bin_f, bw_method=config.kde_bw_method)
            _apply_frequency_kde_y_scale(kde, config.kde_bw_y)
            f_range = np.linspace(
                0.0,
                max(float(np.max(freqs)) * 1.1, float(config.max_freq_hz)),
                int(config.kde_f_resolution),
            )
            density = kde(f_range)
            if float(np.max(density)) > 0:
                density = density / float(np.max(density))
            ax.fill_betweenx(
                f_range,
                t_start,
                t_start + density * bin_width * 0.8,
                alpha=float(config.bin_alpha),
            )
            ax.plot(
                t_start + density * bin_width * 0.8,
                f_range,
                linewidth=1.0,
                color="black",
                alpha=0.3,
            )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim(0, float(config.max_freq_hz))
    ax.grid(True, alpha=0.3)
    return ax
