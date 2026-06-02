"""Reusable named-signal registry helpers for result analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class ResultSignalProvider:
    """One provider that can enumerate and resolve named analysis signals."""

    list_names_fn: Callable[[dict[str, Any], dict[str, Any]], Iterable[str]]
    matches_fn: Callable[[str], bool]
    resolve_fn: Callable[[dict[str, Any], str, dict[str, Any]], Any]


def list_available_result_signals(
    result: dict[str, Any],
    providers: Sequence[ResultSignalProvider],
    **context: Any,
) -> list[str]:
    """List currently available named analysis signals from ordered providers."""
    resolved_context = dict(context)
    signals: list[str] = []
    seen: set[str] = set()
    for provider in providers:
        for name in provider.list_names_fn(result, resolved_context):
            label = str(name)
            if label in seen:
                continue
            seen.add(label)
            signals.append(label)
    return signals


def resolve_result_signal(
    result: dict[str, Any],
    signal: str,
    providers: Sequence[ResultSignalProvider],
    **context: Any,
) -> Any:
    """Resolve one named signal by consulting ordered providers."""
    resolved_context = dict(context)
    for provider in providers:
        if not provider.matches_fn(signal):
            continue
        try:
            return provider.resolve_fn(result, signal, resolved_context)
        except KeyError:
            continue
    raise KeyError(f"Unsupported signal {signal!r}")
