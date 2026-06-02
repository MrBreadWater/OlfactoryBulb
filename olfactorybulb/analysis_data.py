"""Concrete olfactory-bulb result-analysis data adapters built on neuroinfra."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from neuroinfra.analysis import CategoryCatalogHooks
from neuroinfra.analysis.catalog import (
    group_rows_by_category,
    list_available_categories,
    list_unique_labels,
    ordered_names,
)
from neuroinfra.analysis.events import (
    calculate_trace_event_frequency,
    collect_frequency_samples_from_rows,
    collect_frequency_samples_from_trace_rows,
)
from neuroinfra.analysis.spectral import uniform_trace
from olfactorybulb.result_artifacts import detect_soma_spikes

PRIMARY_CELL_TYPE_ORDER = ("MC", "TC", "GC", "EPLI")
CELL_TYPE_ALIASES = {
    # The optional EPLI population currently uses the synthetic PVCRH_FSI1
    # model class. Saved section labels expose that class name, but notebook
    # summaries should report the runtime population, not a second cell type.
    "PVCRH": "EPLI",
}


def normalize_cell_name(name: Any) -> str:
    """Strip HOC prefixes and section suffixes down to a canonical cell label."""
    return str(name).removeprefix("h.").split(".", 1)[0]


def cell_type_of(name: Any) -> str:
    """Infer the cell family prefix such as ``MC`` or ``GC`` from a label."""
    match = re.match(r"([A-Z]+)", normalize_cell_name(name))
    if not match:
        raise ValueError(f"Could not infer cell type from {name!r}")
    cell_type = match.group(1)
    return CELL_TYPE_ALIASES.get(cell_type, cell_type)


def ordered_cell_types(cell_types: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    """Return cell types in stable olfactory-bulb display order."""
    return ordered_names(
        cell_types,
        preferred_order=PRIMARY_CELL_TYPE_ORDER,
        unknown_name="other",
    )


OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS = CategoryCatalogHooks(
    categorize_label_fn=lambda label: cell_type_of(label),
    order_categories_fn=lambda cell_types: ordered_cell_types(set(cell_types)),
    unknown_category="other",
)


def detect_spikes(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float | None = None,
    *,
    min_prominence_mv: float = 3.0,
    refractory_ms: float = 1.0,
) -> np.ndarray:
    """Detect spike peaks from a soma trace using prominence and a refractory window."""
    return detect_soma_spikes(
        t,
        v,
        threshold=threshold,
        min_prominence_mv=min_prominence_mv,
        refractory_ms=refractory_ms,
    )


def calculate_instantaneous_frequency(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spike times from one trace into instantaneous frequency samples."""
    return calculate_trace_event_frequency(
        t,
        v,
        event_times_fn=lambda trace_t, trace_v: detect_spikes(trace_t, trace_v, threshold=threshold),
    )


def saved_soma_spikes_match_threshold(result: dict[str, Any], threshold: float | None) -> bool:
    """Return whether saved soma spikes can satisfy one requested threshold."""
    if threshold is None:
        return True
    metadata = (dict.get(result, "soma_spikes") or {}).get("metadata", {})
    saved_threshold = metadata.get("threshold_mv")
    return saved_threshold is not None and np.isclose(float(saved_threshold), float(threshold))


def saved_soma_spike_rows(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = None,
    threshold: float | None = None,
) -> list[tuple[str, np.ndarray]] | None:
    """Return saved ``(label, spike_times)`` rows, or None when raw traces are required."""
    soma_spikes = dict.get(result, "soma_spikes") or {}
    labels = soma_spikes.get("labels")
    spike_times = soma_spikes.get("spike_times")
    if not labels or spike_times is None:
        return None
    if not saved_soma_spikes_match_threshold(result, threshold):
        return None

    prefixes = tuple(str(name) for name in cell_types) if cell_types else None
    if indices is None:
        indices = range(len(labels))

    rows = []
    for index in indices:
        if index >= len(labels):
            break
        label = str(labels[index])
        if prefixes is not None and not any(label.startswith(prefix) for prefix in prefixes):
            continue
        rows.append((label, np.asarray(spike_times[index], dtype=float)))
    return rows


def saved_voltage_summary_signal(
    result: dict[str, Any],
    *,
    cell_type: str,
    moment: str = "mean",
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one saved voltage-summary moment, or None when raw traces are required."""
    voltage_summary = dict.get(result, "voltage_summary") or {}
    t_by_type = voltage_summary.get("t_by_type") or {}
    values_by_type = voltage_summary.get(f"{moment}_by_type") or {}
    if cell_type not in t_by_type or cell_type not in values_by_type:
        return None
    return uniform_trace(t_by_type[cell_type], values_by_type[cell_type], dt_ms=dt_ms)


def split_traces_by_type(result: dict[str, Any]) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Group saved soma traces by cell family prefix."""
    return group_rows_by_category(
        list(result["soma_vs"]),
        label_fn=lambda row: row[0],
        transform_row_fn=lambda row: (
            row[0],
            np.asarray(row[1], dtype=float),
            np.asarray(row[2], dtype=float),
        ),
        hooks=OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS,
    )


def list_available_cell_types(result: dict[str, Any]) -> list[str]:
    """List saved cell families available for analysis in stable display order."""
    voltage_summary = dict.get(result, "voltage_summary") or {}
    soma_spikes = dict.get(result, "soma_spikes") or {}
    return list_available_categories(
        label_sources=(
            (label for label, _t, _v in result.get("soma_vs", [])),
            (str(cell_type) for cell_type in voltage_summary.get("cell_types", []) or []),
            (label for label in soma_spikes.get("labels", []) or []),
        ),
        hooks=OLFACTORY_BULB_CATEGORY_CATALOG_HOOKS,
    )


def list_available_soma_labels(result: dict[str, Any]) -> list[str]:
    """List saved soma labels from raw traces or compact spike artifacts."""
    soma_spikes = dict.get(result, "soma_spikes") or {}
    return list_unique_labels(
        (label for label, _t, _v in result.get("soma_vs", [])),
        (label for label in soma_spikes.get("labels", []) or []),
    )


def collect_spike_frequency_samples(
    result: dict[str, Any],
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = ("TC", "MC"),
    modulus: float | None = 1e8,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Collect midpoint/frequency samples from detected soma spikes."""
    prefixes = tuple(str(name) for name in cell_types) if cell_types else None

    saved_rows = saved_soma_spike_rows(
        result,
        indices=indices,
        cell_types=cell_types,
        threshold=threshold,
    )
    if saved_rows is not None:
        trace_rows = [(label, spike_times) for label, spike_times in saved_rows]
    else:
        soma_vs = list(result.get("soma_vs", []))
        sample_collection = collect_frequency_samples_from_trace_rows(
            soma_vs,
            label_fn=lambda row: row[0],
            time_fn=lambda row: row[1],
            value_fn=lambda row: row[2],
            event_times_fn=lambda trace_t, trace_v: detect_spikes(trace_t, trace_v, threshold=threshold),
            indices=indices,
            include_prefixes=prefixes,
            modulus=modulus,
        )
        return {
            "times": sample_collection.times_ms,
            "freqs": sample_collection.freqs_hz,
            "labels": list(sample_collection.labels),
            "n_traces": len(sample_collection.labels),
            "cell_types": list(prefixes) if prefixes is not None else None,
        }

    sample_collection = collect_frequency_samples_from_rows(
        trace_rows,
        label_fn=lambda row: row[0],
        times_fn=lambda row: row[1],
        modulus=modulus,
    )

    return {
        "times": sample_collection.times_ms,
        "freqs": sample_collection.freqs_hz,
        "labels": list(sample_collection.labels),
        "n_traces": len(sample_collection.labels),
        "cell_types": list(prefixes) if prefixes is not None else None,
    }
