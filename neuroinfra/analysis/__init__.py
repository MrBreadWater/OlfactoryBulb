"""Reusable analysis helpers extracted from notebook-facing workflows."""

from .overview import (
    ResultOverviewContext,
    build_result_overview,
    build_result_overview_context,
    first_result_file_metadata,
    metadata_value_or_result_length,
    result_file_metadata,
    result_value_length,
)
from .spectral import (
    DEFAULT_HFO_BANDS,
    butter_bandpass_filter,
    compute_band_power_summary,
    compute_spectrogram,
    compute_wavelet_band_power,
    compute_wavelet_map,
    fold_time_matrix_by_modulus,
    fold_time_series_by_modulus,
    normalize_time_modulus,
    trapezoid_integral,
    uniform_trace,
)
from .catalog import (
    CategoryCatalogHooks,
    group_rows_by_category,
    list_available_categories,
    list_unique_labels,
)
from .signals import (
    ResultSignalProvider,
    list_available_result_signals,
    resolve_result_signal,
)

__all__ = [
    "ResultOverviewContext",
    "build_result_overview",
    "build_result_overview_context",
    "butter_bandpass_filter",
    "CategoryCatalogHooks",
    "compute_band_power_summary",
    "compute_spectrogram",
    "compute_wavelet_band_power",
    "compute_wavelet_map",
    "DEFAULT_HFO_BANDS",
    "fold_time_matrix_by_modulus",
    "fold_time_series_by_modulus",
    "first_result_file_metadata",
    "normalize_time_modulus",
    "ResultSignalProvider",
    "group_rows_by_category",
    "list_available_categories",
    "list_unique_labels",
    "list_available_result_signals",
    "metadata_value_or_result_length",
    "result_file_metadata",
    "result_value_length",
    "resolve_result_signal",
    "trapezoid_integral",
    "uniform_trace",
]
