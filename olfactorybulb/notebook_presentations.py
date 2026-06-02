"""Concrete olfactory-bulb notebook presentation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from olfactorybulb.analysis_presentations import (
    StandardOutputHooks,
    show_all_outputs as _render_standard_outputs,
)
from olfactorybulb.notebook_reports import (
    NotebookReportHooks,
    print_run_summary as _render_run_summary,
)


@dataclass(frozen=True)
class NotebookPresentationHooks:
    """Hooks for notebook-facing figure, output-bundle, and summary adapters."""

    default_results_base: str | Path
    make_timestamp_fn: Callable[[], str]
    safe_name_fn: Callable[[Any], str]
    plt_module: Any
    save_figure_fn: Callable[..., Path]
    plot_input_overview_fn: Callable[..., Any]
    plot_voltage_traces_fn: Callable[..., Any]
    plot_spike_raster_fn: Callable[..., Any]
    plot_gc_output_overview_fn: Callable[..., Any]
    plot_lfp_overview_fn: Callable[..., Any]
    plot_spectrogram_fn: Callable[..., Any]
    plot_wavelet_fn: Callable[..., Any]
    plot_wavelet_band_power_fn: Callable[..., Any]
    result_overview_fn: Callable[[dict[str, Any]], dict[str, Any]]
    build_run_config_fn: Callable[..., dict[str, Any]]
    resolve_effective_params_fn: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_paramset_defaults_fn: Callable[[str], dict[str, Any]]
    diff_values_fn: Callable[[Any, Any], list[dict[str, Any]]]
    extract_runtime_control_snapshot_fn: Callable[[dict[str, Any]], dict[str, Any]]
    print_diff_section_fn: Callable[[str, list[dict[str, Any]], int | None], None]
    write_fn: Callable[[str], None] = print


def save_figure(
    hooks: NotebookPresentationHooks,
    name: str,
    *,
    fig: Any = None,
    run_or_result: Any = None,
    output_dir: str | Path | None = None,
    sweep: dict[str, Any] | None = None,
    dpi: int = 200,
    close: bool = False,
) -> Path:
    """Save one notebook figure using the standard olfactory-bulb output policy."""
    return hooks.save_figure_fn(
        name,
        fig=fig or hooks.plt_module.gcf(),
        safe_name_fn=hooks.safe_name_fn,
        default_output_dir_factory=lambda: Path(hooks.default_results_base) / "figures" / hooks.make_timestamp_fn(),
        close_figure_fn=hooks.plt_module.close,
        run_or_result=run_or_result,
        output_dir=output_dir,
        sweep=sweep,
        dpi=dpi,
        close=close,
    )


def show_all_outputs(
    hooks: NotebookPresentationHooks,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Render the standard notebook output bundle with concrete plotting hooks."""
    return _render_standard_outputs(
        StandardOutputHooks(
            plot_input_overview_fn=hooks.plot_input_overview_fn,
            plot_voltage_traces_fn=hooks.plot_voltage_traces_fn,
            plot_spike_raster_fn=hooks.plot_spike_raster_fn,
            plot_gc_output_overview_fn=hooks.plot_gc_output_overview_fn,
            plot_lfp_overview_fn=hooks.plot_lfp_overview_fn,
            plot_spectrogram_fn=hooks.plot_spectrogram_fn,
            plot_wavelet_fn=hooks.plot_wavelet_fn,
            plot_wavelet_band_power_fn=hooks.plot_wavelet_band_power_fn,
            plt_show_fn=hooks.plt_module.show,
        ),
        result,
        config=config,
    )


def print_run_summary(
    hooks: NotebookPresentationHooks,
    run: Any,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Print the standard olfactory-bulb notebook run summary."""
    return _render_run_summary(
        NotebookReportHooks(
            result_overview_fn=hooks.result_overview_fn,
            build_run_config_fn=hooks.build_run_config_fn,
            resolve_effective_params_fn=hooks.resolve_effective_params_fn,
            resolve_paramset_defaults_fn=hooks.resolve_paramset_defaults_fn,
            diff_values_fn=hooks.diff_values_fn,
            extract_runtime_control_snapshot_fn=hooks.extract_runtime_control_snapshot_fn,
            print_diff_section_fn=hooks.print_diff_section_fn,
            write_fn=hooks.write_fn,
        ),
        run,
        result,
        config=config,
    )
