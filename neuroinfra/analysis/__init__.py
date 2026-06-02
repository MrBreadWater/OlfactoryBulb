"""Reusable analysis helpers extracted from notebook-facing workflows."""

from .signals import (
    ResultSignalProvider,
    list_available_result_signals,
    resolve_result_signal,
)

__all__ = [
    "ResultSignalProvider",
    "list_available_result_signals",
    "resolve_result_signal",
]
