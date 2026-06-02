"""Reusable analysis helpers extracted from notebook-facing workflows."""

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
    "CategoryCatalogHooks",
    "ResultSignalProvider",
    "group_rows_by_category",
    "list_available_categories",
    "list_unique_labels",
    "list_available_result_signals",
    "resolve_result_signal",
]
