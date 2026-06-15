"""Typed protocol contract, registry, and execution cache for validation protocols."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from olfactorybulb.audit.protocol_evidence import ProtocolEvidenceBundle, coerce_protocol_evidence_bundle
from olfactorybulb.neuronunit.frozen_payloads import FrozenMappingPayload, coerce_mapping_payload
from olfactorybulb.neuronunit.metric_tables import MetricTable, coerce_metric_table


class ProtocolExecutionArgMap(FrozenMappingPayload):
    """Typed frozen wrapper for protocol-cache argument values."""


class ProtocolExecutionConfigMap(FrozenMappingPayload):
    """Typed frozen wrapper for protocol-cache protocol-config values."""


def _normalize_cache_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_cache_value(nested_value)
            for key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_cache_value(item) for item in value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _freeze_cache_value(value: Any) -> Any:
    normalized = _normalize_cache_value(value)
    if isinstance(normalized, dict):
        return tuple((key, _freeze_cache_value(nested_value)) for key, nested_value in normalized.items())
    if isinstance(normalized, list):
        return tuple(_freeze_cache_value(item) for item in normalized)
    return normalized


@dataclass(frozen=True)
class ProtocolExecutionCacheInfo:
    status: str
    scope: str
    protocol_id: str
    cache_key: str
    arg_values: ProtocolExecutionArgMap | Mapping[str, object]
    protocol_config: ProtocolExecutionConfigMap | Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arg_values",
            coerce_mapping_payload(self.arg_values, payload_type=ProtocolExecutionArgMap)
            or ProtocolExecutionArgMap(entries=()),
        )
        object.__setattr__(
            self,
            "protocol_config",
            coerce_mapping_payload(self.protocol_config, payload_type=ProtocolExecutionConfigMap)
            or ProtocolExecutionConfigMap(entries=()),
        )

    def to_evidence(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scope": self.scope,
            "protocol_id": self.protocol_id,
            "cache_key": self.cache_key,
            "arg_values": self.arg_values.to_dict(),
            "protocol_config": self.protocol_config.to_dict(),
        }


@dataclass(frozen=True)
class ProtocolRunResult:
    metrics: MetricTable | list[dict[str, Any]]
    protocol_evidence: ProtocolEvidenceBundle
    group_field: str = "cell_type"
    cache_info: ProtocolExecutionCacheInfo | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", coerce_metric_table(self.metrics, group_field=self.group_field))
        object.__setattr__(self, "protocol_evidence", coerce_protocol_evidence_bundle(self.protocol_evidence))


@dataclass(frozen=True)
class ValidationProtocolSpec:
    protocol_id: str
    title: str
    description: str
    add_cli_args: Callable[[argparse.ArgumentParser], None] | None
    run: Callable[[argparse.Namespace, dict[str, Any]], ProtocolRunResult]
    cache_enabled: bool = False
    cache_arg_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProtocolExecutionCacheKey:
    protocol_id: str
    arg_items: tuple[tuple[str, Any], ...]
    protocol_config_items: tuple[tuple[str, Any], ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr((self.protocol_id, self.arg_items, self.protocol_config_items)).encode("utf-8")).hexdigest()[:16]


PROTOCOL_SPECS: dict[str, ValidationProtocolSpec] = {}
_PROTOCOL_RESULT_CACHE: dict[ProtocolExecutionCacheKey, ProtocolRunResult] = {}


def register_validation_protocol(spec: ValidationProtocolSpec) -> ValidationProtocolSpec:
    PROTOCOL_SPECS[spec.protocol_id] = spec
    return spec


def get_validation_protocol_spec(protocol_id: str) -> ValidationProtocolSpec:
    try:
        return PROTOCOL_SPECS[protocol_id]
    except KeyError as exc:
        known = ", ".join(sorted(PROTOCOL_SPECS))
        raise KeyError(f"Unknown reference validation protocol {protocol_id!r}. Known protocols: {known}") from exc


def iter_validation_protocol_specs() -> list[ValidationProtocolSpec]:
    return [PROTOCOL_SPECS[key] for key in sorted(PROTOCOL_SPECS)]


def clear_protocol_execution_cache() -> None:
    _PROTOCOL_RESULT_CACHE.clear()


def protocol_execution_cache_size() -> int:
    return len(_PROTOCOL_RESULT_CACHE)


def _protocol_cache_arg_values(
    spec: ValidationProtocolSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        arg_name: _normalize_cache_value(getattr(args, arg_name, None))
        for arg_name in spec.cache_arg_names
    }


def _protocol_cache_config_values(
    protocol_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): _normalize_cache_value(value)
        for key, value in sorted(protocol_config.items(), key=lambda item: str(item[0]))
    }


def build_protocol_execution_cache_key(
    spec: ValidationProtocolSpec,
    *,
    args: argparse.Namespace,
    protocol_config: Mapping[str, Any],
) -> ProtocolExecutionCacheKey:
    arg_values = _protocol_cache_arg_values(spec, args)
    config_values = _protocol_cache_config_values(protocol_config)
    return ProtocolExecutionCacheKey(
        protocol_id=spec.protocol_id,
        arg_items=tuple((key, _freeze_cache_value(value)) for key, value in arg_values.items()),
        protocol_config_items=tuple((key, _freeze_cache_value(value)) for key, value in config_values.items()),
    )


def _cache_info(
    *,
    spec: ValidationProtocolSpec,
    key: ProtocolExecutionCacheKey,
    args: argparse.Namespace,
    protocol_config: Mapping[str, Any],
    status: str,
) -> ProtocolExecutionCacheInfo:
    return ProtocolExecutionCacheInfo(
        status=status,
        scope="process",
        protocol_id=spec.protocol_id,
        cache_key=key.digest,
        arg_values=_protocol_cache_arg_values(spec, args),
        protocol_config=_protocol_cache_config_values(protocol_config),
    )


def _annotate_result_with_cache_info(
    result: ProtocolRunResult,
    cache_info: ProtocolExecutionCacheInfo,
) -> ProtocolRunResult:
    return ProtocolRunResult(
        metrics=result.metrics,
        protocol_evidence=result.protocol_evidence.with_value("protocol_cache", cache_info.to_evidence()),
        group_field=result.group_field,
        cache_info=cache_info,
    )


def execute_validation_protocol(
    spec: ValidationProtocolSpec,
    *,
    args: argparse.Namespace,
    protocol_config: dict[str, Any],
) -> ProtocolRunResult:
    if not spec.cache_enabled:
        return spec.run(args, dict(protocol_config))
    key = build_protocol_execution_cache_key(spec, args=args, protocol_config=protocol_config)
    cached = _PROTOCOL_RESULT_CACHE.get(key)
    if cached is not None:
        return _annotate_result_with_cache_info(
            cached,
            _cache_info(spec=spec, key=key, args=args, protocol_config=protocol_config, status="hit"),
        )
    fresh = spec.run(args, dict(protocol_config))
    _PROTOCOL_RESULT_CACHE[key] = ProtocolRunResult(
        metrics=fresh.metrics,
        protocol_evidence=fresh.protocol_evidence,
        group_field=fresh.group_field,
        cache_info=None,
    )
    return _annotate_result_with_cache_info(
        fresh,
        _cache_info(spec=spec, key=key, args=args, protocol_config=protocol_config, status="miss"),
    )


__all__ = [
    "ProtocolExecutionArgMap",
    "ProtocolExecutionCacheInfo",
    "ProtocolExecutionConfigMap",
    "ProtocolExecutionCacheKey",
    "ProtocolRunResult",
    "ValidationProtocolSpec",
    "build_protocol_execution_cache_key",
    "clear_protocol_execution_cache",
    "execute_validation_protocol",
    "get_validation_protocol_spec",
    "iter_validation_protocol_specs",
    "protocol_execution_cache_size",
    "register_validation_protocol",
]
