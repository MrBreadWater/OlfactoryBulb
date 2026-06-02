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
    "CategoryCatalogHooks",
    "first_result_file_metadata",
    "ResultSignalProvider",
    "group_rows_by_category",
    "list_available_categories",
    "list_unique_labels",
    "list_available_result_signals",
    "metadata_value_or_result_length",
    "result_file_metadata",
    "result_value_length",
    "resolve_result_signal",
]
