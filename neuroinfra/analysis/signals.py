"""Reusable named-signal registry helpers for result analysis."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ResultSignalProvider:
    """One provider that can enumerate and resolve named analysis signals."""

    list_names_fn: Callable[[dict[str, Any], dict[str, Any]], Iterable[str]]
    matches_fn: Callable[[str], bool]
    resolve_fn: Callable[[dict[str, Any], str, dict[str, Any]], Any]


@dataclass(frozen=True)
class ResultSignalRegistry:
    """Ordered registry of named analysis signal providers."""

    providers: Sequence[ResultSignalProvider]

    def list_available(
        self,
        result: dict[str, Any],
        **context: Any,
    ) -> list[str]:
        """List currently available named analysis signals in registry order."""
        resolved_context = dict(context)
        signals: list[str] = []
        seen: set[str] = set()
        for provider in self.providers:
            for name in provider.list_names_fn(result, resolved_context):
                label = str(name)
                if label in seen:
                    continue
                seen.add(label)
                signals.append(label)
        return signals

    def resolve(
        self,
        result: dict[str, Any],
        signal: str,
        **context: Any,
    ) -> Any:
        """Resolve one named signal by consulting ordered providers."""
        resolved_context = dict(context)
        for provider in self.providers:
            if not provider.matches_fn(signal):
                continue
            try:
                return provider.resolve_fn(result, signal, resolved_context)
            except KeyError:
                continue
        raise KeyError(f"Unsupported signal {signal!r}")


def keyed_trace_signal_provider(
    signal_name: str,
    *,
    time_key: str,
    value_key: str,
    uniform_trace_fn: Callable[[Any, Any], Any],
) -> ResultSignalProvider:
    """Build a provider for one direct time/value trace stored under result keys."""

    def _list_names(result: dict[str, Any], _context: dict[str, Any]) -> list[str]:
        times = result.get(time_key)
        values = result.get(value_key)
        if times is None or values is None:
            return []
        try:
            if len(times) > 0 and len(values) > 0:
                return [signal_name]
        except TypeError:
            return []
        return []

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> Any:
        if signal != signal_name:
            raise KeyError(signal)
        return uniform_trace_fn(result[time_key], result[value_key], dt_ms=context.get("dt_ms"))

    return ResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda signal: signal == signal_name,
        resolve_fn=_resolve,
    )


def suffix_variant_signal_provider(
    *,
    base_name: str,
    suffix_payloads: Mapping[str, Any],
    availability_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
    resolve_variant_fn: Callable[[dict[str, Any], Any, dict[str, Any]], Any],
) -> ResultSignalProvider:
    """Build a provider for one base signal name plus ordered suffix variants."""
    ordered_suffixes = tuple(str(suffix) for suffix in suffix_payloads.keys())
    ordered_signal_names = tuple(f"{base_name}{suffix}" for suffix in ordered_suffixes)
    signal_name_set = set(ordered_signal_names)

    def _list_names(result: dict[str, Any], context: dict[str, Any]) -> list[str]:
        return list(ordered_signal_names) if availability_fn(result, context) else []

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> Any:
        if signal not in signal_name_set:
            raise KeyError(signal)
        suffix = signal[len(base_name) :]
        return resolve_variant_fn(result, suffix_payloads[suffix], context)

    return ResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda signal: signal in signal_name_set,
        resolve_fn=_resolve,
    )


def pattern_result_signal_provider(
    pattern: str | re.Pattern[str],
    *,
    list_names_fn: Callable[[dict[str, Any], dict[str, Any]], Iterable[str]],
    resolve_match_fn: Callable[[dict[str, Any], re.Match[str], dict[str, Any]], Any],
) -> ResultSignalProvider:
    """Build a provider whose supported names are defined by a full-match pattern."""
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> Any:
        match = compiled.fullmatch(signal)
        if match is None:
            raise KeyError(signal)
        return resolve_match_fn(result, match, context)

    return ResultSignalProvider(
        list_names_fn=list_names_fn,
        matches_fn=lambda signal: compiled.fullmatch(signal) is not None,
        resolve_fn=_resolve,
    )


def labeled_trace_signal_provider(
    *,
    include_context_key: str,
    list_labels_fn: Callable[[dict[str, Any]], Iterable[str]],
    iter_rows_fn: Callable[[dict[str, Any]], Iterable[Any]],
    label_fn: Callable[[Any], str],
    time_fn: Callable[[Any], Any],
    value_fn: Callable[[Any], Any],
    uniform_trace_fn: Callable[[Any, Any], Any],
) -> ResultSignalProvider:
    """Build a provider for direct labeled traces gated by a context flag."""

    def _list_names(result: dict[str, Any], context: dict[str, Any]) -> list[str]:
        if not bool(context.get(include_context_key, False)):
            return []
        return [str(label) for label in list_labels_fn(result)]

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> Any:
        for row in iter_rows_fn(result):
            if str(label_fn(row)) == signal:
                return uniform_trace_fn(time_fn(row), value_fn(row), dt_ms=context.get("dt_ms"))
        raise KeyError(signal)

    return ResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda _signal: True,
        resolve_fn=_resolve,
    )


def mean_aligned_row_trace(
    rows: Sequence[Any],
    *,
    time_fn: Callable[[Any], Any],
    value_fn: Callable[[Any], Any],
    uniform_trace_fn: Callable[[Any, Any], Any],
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align multiple time/value rows onto one grid and return their mean trace."""
    if not rows:
        raise KeyError("No rows available for aligned mean trace")

    first_t, _first_v = uniform_trace_fn(time_fn(rows[0]), value_fn(rows[0]), dt_ms=dt_ms)
    aligned = []
    for row in rows:
        interp_t, interp_v = uniform_trace_fn(
            time_fn(row),
            value_fn(row),
            dt_ms=float(np.median(np.diff(first_t))) if len(first_t) > 1 else dt_ms,
        )
        n = min(len(first_t), len(interp_t))
        aligned.append(np.asarray(interp_v[:n], dtype=float))
    n = min(len(values) for values in aligned)
    return np.asarray(first_t[:n], dtype=float), np.mean(np.vstack([values[:n] for values in aligned]), axis=0)


def list_available_result_signals(
    result: dict[str, Any],
    providers: Sequence[ResultSignalProvider],
    **context: Any,
) -> list[str]:
    """List currently available named analysis signals from ordered providers."""
    return ResultSignalRegistry(tuple(providers)).list_available(result, **context)


def resolve_result_signal(
    result: dict[str, Any],
    signal: str,
    providers: Sequence[ResultSignalProvider],
    **context: Any,
) -> Any:
    """Resolve one named signal by consulting ordered providers."""
    return ResultSignalRegistry(tuple(providers)).resolve(
        result,
        signal,
        **context,
    )
