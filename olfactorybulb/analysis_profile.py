"""Concrete olfactory-bulb analysis profile built on neuroinfra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from olfactorybulb.analysis_data import (
    OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS,
    collect_spike_frequency_samples,
    list_available_cell_types,
    list_available_soma_labels,
    normalize_cell_name,
    saved_voltage_summary_signal,
    split_traces_by_type,
)
from neuroinfra.analysis import (
    EventRateNormalizationRule,
    EventRateSeriesSpec,
    ResultAnalysisProfile,
    ResultEventFamilySpec,
    ResultEventFamilySuite,
    ResultEventPlotSuite,
    ResultFrequencyPlotFamily,
    ResultFrequencyPlotSuite,
    ResultSignalRegistry,
    ResultSignalViewSuite,
    SweepPlotRegistry,
    keyed_trace_signal_provider,
    labeled_trace_signal_provider,
    mean_aligned_row_trace,
    pattern_result_signal_provider,
    suffix_variant_signal_provider,
)
from neuroinfra.analysis.spectral import uniform_trace


@dataclass(frozen=True)
class OlfactoryBulbAnalysisProfileHooks:
    """Domain hooks required to assemble the concrete olfactory-bulb profile."""

    plot_voltage_traces_fn: Callable[..., Any]
    plot_spike_raster_fn: Callable[..., Any]
    plot_hfo_power_summary_fn: Callable[..., Any]
    plot_named_signal_fn: Callable[..., Any]
    plot_spectrogram_fn: Callable[..., Any]
    plot_wavelet_fn: Callable[..., Any]
    plot_wavelet_band_power_fn: Callable[..., Any]


def _resolve_event_tstop(result: dict[str, Any], event_series: list[np.ndarray]) -> float:
    """Infer the latest relevant time from LFP, soma traces, or event series."""
    if len(result.get("lfp_t", [])) > 0:
        return float(result["lfp_t"][-1])

    t_stop = 0.0
    for _label, t, _v in result.get("soma_vs", []):
        if len(t) > 0:
            t_stop = max(t_stop, float(t[-1]))
    for times in event_series:
        if len(times) > 0:
            t_stop = max(t_stop, float(times[-1]))
    return t_stop


def _gc_output_rate_normalization_rules(
) -> dict[str, EventRateNormalizationRule]:
    """Return reusable normalization rules for GC-output event rates."""
    return {
        "total": EventRateNormalizationRule(
            unit="events/s",
            aliases=(),
            denominator_fn=lambda events: 1.0,
            metadata_fn=lambda events: {
                "n_connections": len(events),
                "n_source_cells": len({normalize_cell_name(entry.get("source_section", "")) for entry in events}),
                "n_target_cells": len({normalize_cell_name(entry.get("dest_section", "")) for entry in events}),
            },
        ),
        "per_connection": EventRateNormalizationRule(
            unit="events/s per connection",
            aliases=(),
            denominator_fn=lambda events: float(len(events)),
            metadata_fn=lambda events: {
                "n_connections": len(events),
                "n_source_cells": len({normalize_cell_name(entry.get("source_section", "")) for entry in events}),
                "n_target_cells": len({normalize_cell_name(entry.get("dest_section", "")) for entry in events}),
            },
        ),
        "per_source_cell": EventRateNormalizationRule(
            unit="events/s per source GC",
            aliases=(),
            denominator_fn=lambda events: float(
                len({normalize_cell_name(entry.get("source_section", "")) for entry in events})
            ),
            metadata_fn=lambda events: {
                "n_connections": len(events),
                "n_source_cells": len({normalize_cell_name(entry.get("source_section", "")) for entry in events}),
                "n_target_cells": len({normalize_cell_name(entry.get("dest_section", "")) for entry in events}),
            },
        ),
        "per_target_cell": EventRateNormalizationRule(
            unit="events/s per target cell",
            aliases=(),
            denominator_fn=lambda events: float(
                len({normalize_cell_name(entry.get("dest_section", "")) for entry in events})
            ),
            metadata_fn=lambda events: {
                "n_connections": len(events),
                "n_source_cells": len({normalize_cell_name(entry.get("source_section", "")) for entry in events}),
                "n_target_cells": len({normalize_cell_name(entry.get("dest_section", "")) for entry in events}),
            },
        ),
    }


def _input_rate_normalization_rules(
) -> dict[str, EventRateNormalizationRule]:
    """Return reusable normalization rules for odor-input event rates."""
    return {
        "total": EventRateNormalizationRule(
            unit="events/s",
            aliases=(),
            denominator_fn=lambda rows: 1.0,
            metadata_fn=lambda rows: {
                "n_segments": len(rows),
                "n_target_cells": len({normalize_cell_name(section_name) for section_name, _times in rows}),
            },
        ),
        "per_segment": EventRateNormalizationRule(
            unit="events/s per input segment",
            aliases=("per_input_segment",),
            denominator_fn=lambda rows: float(len(rows)),
            metadata_fn=lambda rows: {
                "n_segments": len(rows),
                "n_target_cells": len({normalize_cell_name(section_name) for section_name, _times in rows}),
            },
        ),
        "per_target_cell": EventRateNormalizationRule(
            unit="events/s per target cell",
            aliases=("per_cell",),
            denominator_fn=lambda rows: float(
                len({normalize_cell_name(section_name) for section_name, _times in rows})
            ),
            metadata_fn=lambda rows: {
                "n_segments": len(rows),
                "n_target_cells": len({normalize_cell_name(section_name) for section_name, _times in rows}),
            },
        ),
    }


def _build_gc_output_event_family(
    hooks: OlfactoryBulbAnalysisProfileHooks,
) -> ResultEventFamilySuite:
    """Build the GC inhibitory-output event family suite."""
    spec = ResultEventFamilySpec(
        rows_from_result_fn=lambda result: list(result.get("gc_output_events", [])),
        filter_label_fn=lambda entry: entry.get("dest_section", ""),
        times_fn=lambda entry: entry.get("times", []),
        sample_label_fn=lambda entry: (
            f"{normalize_cell_name(entry.get('source_section', 'GC'))}->"
            f"{normalize_cell_name(entry.get('dest_section', 'cell'))}"
        ),
        normalize_label_fn=normalize_cell_name,
        normalization_rules=_gc_output_rate_normalization_rules(),
        default_normalization="per_target_cell",
    )
    return ResultEventFamilySuite(
        spec=spec,
        infer_t_stop_fn=lambda result, rows: _resolve_event_tstop(
            result,
            [np.asarray(spec.times_fn(row), dtype=float) for row in rows],
        ),
    )


def _build_input_event_family(
    hooks: OlfactoryBulbAnalysisProfileHooks,
) -> ResultEventFamilySuite:
    """Build the odor-input event family suite."""
    spec = ResultEventFamilySpec(
        rows_from_result_fn=lambda result: list(result.get("input_times", [])),
        filter_label_fn=lambda row: row[0],
        times_fn=lambda row: row[1],
        normalize_label_fn=normalize_cell_name,
        normalization_rules=_input_rate_normalization_rules(),
        default_normalization="per_target_cell",
    )
    return ResultEventFamilySuite(
        spec=spec,
        infer_t_stop_fn=lambda result, rows: _resolve_event_tstop(
            result,
            [np.asarray(spec.times_fn(row), dtype=float) for row in rows],
        ),
    )


def _lfp_signal_provider(hooks: OlfactoryBulbAnalysisProfileHooks) -> Any:
    """Provide the standard LFP named signal when it is present."""
    return keyed_trace_signal_provider(
        "lfp",
        time_key="lfp_t",
        value_key="lfp",
        uniform_trace_fn=uniform_trace,
    )


def _gc_output_rate_signal_provider(gc_output_event_family: ResultEventFamilySuite) -> Any:
    """Provide named GC inhibitory-output rate signals."""
    return suffix_variant_signal_provider(
        base_name="gc_output_rate",
        suffix_payloads={
            "": None,
            "_MC": ["MC"],
            "_TC": ["TC"],
        },
        availability_fn=lambda result, _context: len(result.get("gc_output_events") or []) > 0,
        resolve_variant_fn=lambda result, target_types, context: gc_output_event_family.compute_rate(
            result,
            bin_ms=5.0 if context.get("dt_ms") is None else float(context["dt_ms"]),
            smooth_sigma_ms=max(
                2.0 * (5.0 if context.get("dt_ms") is None else float(context["dt_ms"])),
                5.0,
            ),
            include_prefixes=target_types,
            normalization="per_target_cell",
        ),
    )


def _input_rate_signal_provider(input_event_family: ResultEventFamilySuite) -> Any:
    """Provide named odor-input rate signals."""
    return suffix_variant_signal_provider(
        base_name="input_rate",
        suffix_payloads={
            "": None,
            "_MC": ["MC"],
            "_TC": ["TC"],
        },
        availability_fn=lambda result, _context: len(result.get("input_times") or []) > 0,
        resolve_variant_fn=lambda result, target_types, context: input_event_family.compute_rate(
            result,
            bin_ms=5.0 if context.get("dt_ms") is None else float(context["dt_ms"]),
            smooth_sigma_ms=max(
                2.0 * (5.0 if context.get("dt_ms") is None else float(context["dt_ms"])),
                5.0,
            ),
            include_prefixes=target_types,
            normalization="per_target_cell",
        ),
    )


def _mean_voltage_signal_provider(hooks: OlfactoryBulbAnalysisProfileHooks) -> Any:
    """Provide dynamic per-cell-type mean-voltage signals."""

    def _resolve_mean_voltage(
        result: dict[str, Any],
        cell_type: str,
        dt_ms: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        saved_signal = saved_voltage_summary_signal(
            result,
            cell_type=cell_type,
            moment="mean",
            dt_ms=dt_ms,
        )
        if saved_signal is not None:
            return saved_signal
        grouped = split_traces_by_type(result)
        traces = grouped.get(cell_type, [])
        if not traces:
            raise KeyError(f"No soma traces found for {cell_type}")
        return mean_aligned_row_trace(
            traces,
            time_fn=lambda row: row[1],
            value_fn=lambda row: row[2],
            uniform_trace_fn=uniform_trace,
            dt_ms=dt_ms,
        )

    return pattern_result_signal_provider(
        r"mean_([A-Z]+)_voltage",
        list_names_fn=lambda result, _context: [
            f"mean_{cell_type}_voltage"
            for cell_type in list_available_cell_types(result)
        ],
        resolve_match_fn=lambda result, match, context: _resolve_mean_voltage(
            result,
            match.group(1),
            context.get("dt_ms"),
        ),
    )


def _soma_label_signal_provider(hooks: OlfactoryBulbAnalysisProfileHooks) -> Any:
    """Provide direct per-soma trace signals by saved label."""
    return labeled_trace_signal_provider(
        include_context_key="include_soma_labels",
        list_labels_fn=list_available_soma_labels,
        iter_rows_fn=lambda result: result.get("soma_vs", []),
        label_fn=lambda row: row[0],
        time_fn=lambda row: row[1],
        value_fn=lambda row: row[2],
        uniform_trace_fn=uniform_trace,
    )


def build_olfactorybulb_analysis_profile(
    hooks: OlfactoryBulbAnalysisProfileHooks,
) -> ResultAnalysisProfile:
    """Assemble the concrete olfactory-bulb analysis profile."""
    input_event_family = _build_input_event_family(hooks)
    gc_output_event_family = _build_gc_output_event_family(hooks)

    signal_registry = ResultSignalRegistry(
        providers=(
            _lfp_signal_provider(hooks),
            _gc_output_rate_signal_provider(gc_output_event_family),
            _input_rate_signal_provider(input_event_family),
            _mean_voltage_signal_provider(hooks),
            _soma_label_signal_provider(hooks),
        ),
    )
    signal_views = ResultSignalViewSuite(signal_registry)

    def _collect_gc_output_frequency_samples(
        result: dict[str, Any],
        *,
        indices: list[int] | range | None = None,
        target_types: list[str] | tuple[str, ...] | None = None,
        modulus: float | None = None,
    ) -> dict[str, Any]:
        sample_collection = gc_output_event_family.collect_samples(
            result,
            indices=indices,
            include_prefixes=target_types,
            modulus=modulus,
        )
        return {
            "times": sample_collection.times_ms,
            "freqs": sample_collection.freqs_hz,
            "events": list(sample_collection.rows),
            "n_events": len(sample_collection.rows),
        }

    gc_output_frequency_suite = ResultFrequencyPlotSuite(
        ResultFrequencyPlotFamily(
            collect_samples_fn=_collect_gc_output_frequency_samples,
            selection_label_fn=lambda target_types: "all" if not target_types else "_".join(str(name) for name in target_types),
            title_1d="GC Inhibitory Output Frequency Distribution",
            title_2d="GC Inhibitory Output Time/Frequency KDE",
            title_time_binned="GC Inhibitory Output Frequency Distributions",
        )
    )

    spike_frequency_suite = ResultFrequencyPlotSuite(
        ResultFrequencyPlotFamily(
            collect_samples_fn=collect_spike_frequency_samples,
            selection_label_fn=lambda cell_types: "all" if not cell_types else "+".join(str(name) for name in cell_types),
            title_1d="Soma Spike Frequency Distribution",
            title_2d="Soma Spike Time/Frequency KDE",
            title_time_binned="Soma Spike Frequency Distributions",
        )
    )

    input_event_plots = ResultEventPlotSuite(
        family=input_event_family,
        row_label_fn=lambda row: row[0],
        sort_key_fn=lambda row: row[0],
        label_transform_fn=lambda label: label.replace("h.", ""),
        rate_series_specs=(
            EventRateSeriesSpec("All inputs", None, "black"),
            EventRateSeriesSpec("To MCs", ["MC"], "tab:blue"),
            EventRateSeriesSpec("To TCs", ["TC"], "tab:red"),
        ),
        raster_ylabel="Input Segment",
        raster_title="Odor Input Raster",
        raster_width=14.0,
        raster_min_height=4.0,
        raster_per_row_height=0.10,
        raster_line_spacing=1.4,
        raster_colors="black",
        raster_no_data_message="No input events saved",
        rate_title="Odor Input Event Rate",
        rate_no_data_message="No input events saved",
        overview_figure_width=16.0,
        overview_raster_min_height=4.5,
        overview_rate_height=4.0,
        overview_left_margin_per_char=0.006,
    )
    gc_output_event_plots = ResultEventPlotSuite(
        family=gc_output_event_family,
        row_label_fn=lambda row: (
            f"{normalize_cell_name(row.get('source_section', 'GC'))}->"
            f"{normalize_cell_name(row.get('dest_section', 'cell'))}"
        ),
        rate_series_specs=(
            EventRateSeriesSpec("All targets", None, "black"),
            EventRateSeriesSpec("To MCs", ["MC"], "tab:blue"),
            EventRateSeriesSpec("To TCs", ["TC"], "tab:red"),
        ),
        raster_ylabel="Reciprocal GABA Connection",
        raster_title="GC Inhibitory Output Events",
        raster_width=16.0,
        raster_min_height=4.5,
        raster_per_row_height=0.10,
        raster_line_spacing=1.4,
        raster_colors="black",
        raster_no_data_message="No GC inhibitory-output events saved",
        rate_title="GC Inhibitory Output Rate",
        rate_no_data_message="No GC inhibitory-output events saved",
        overview_figure_width=16.0,
        overview_raster_min_height=4.5,
        overview_rate_height=4.0,
        overview_left_margin_per_char=0.007,
    )

    sweep_plot_registry = SweepPlotRegistry(
        plots={
            "voltage_traces": hooks.plot_voltage_traces_fn,
            "spike_raster": hooks.plot_spike_raster_fn,
            "hfo_power_summary": hooks.plot_hfo_power_summary_fn,
            "named_signal": hooks.plot_named_signal_fn,
            "spectrogram": hooks.plot_spectrogram_fn,
            "wavelet": hooks.plot_wavelet_fn,
            "wavelet_band_power": hooks.plot_wavelet_band_power_fn,
            "spike_frequency_kde_1d": spike_frequency_suite.plot_kde_1d,
            "spike_frequency_kde_2d": spike_frequency_suite.plot_kde_2d,
            "spike_frequency_time_binned": spike_frequency_suite.plot_time_binned,
            "gc_output_frequency_kde_1d": gc_output_frequency_suite.plot_kde_1d,
            "gc_output_frequency_kde_2d": gc_output_frequency_suite.plot_kde_2d,
            "gc_output_frequency_time_binned": gc_output_frequency_suite.plot_time_binned,
        },
        deprecated_names=frozenset(
            {
                "gc_output_frequency_overview",
                "gc_output_overview",
                "input_overview",
                "lfp_overview",
                "spike_frequency_overview",
            }
        ),
    )

    return ResultAnalysisProfile(
        category_hooks=OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS,
        signal_registry=signal_registry,
        signal_views=signal_views,
        event_families={
            "input": input_event_family,
            "gc_output": gc_output_event_family,
        },
        event_plots={
            "input": input_event_plots,
            "gc_output": gc_output_event_plots,
        },
        frequency_plots={
            "spike": spike_frequency_suite,
            "gc_output": gc_output_frequency_suite,
        },
        sweep_plot_registry=sweep_plot_registry,
    )
