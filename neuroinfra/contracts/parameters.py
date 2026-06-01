"""Generic parameter-space registry and contract helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable scalar parameter in an optimizer or runtime contract."""

    path: str
    low: float
    high: float
    scale: str = "log"
    dtype: str = "float"
    description: str = ""
    default: float | None = None

    def clamp(self, value: float) -> float:
        return min(max(float(value), float(self.low)), float(self.high))

    def encode(self, value: float) -> float:
        value = self.clamp(value)
        if self.scale == "log":
            return math.log10(max(value, 1e-12))
        if self.scale != "linear":
            raise ValueError(f"Unsupported scale {self.scale!r}")
        return float(value)

    def decode(self, value: float) -> float:
        if self.scale == "log":
            decoded = 10.0 ** float(value)
        elif self.scale == "linear":
            decoded = float(value)
        else:
            raise ValueError(f"Unsupported scale {self.scale!r}")
        decoded = self.clamp(decoded)
        if self.dtype == "int":
            return int(round(decoded))
        return float(decoded)

    def low_encoded(self) -> float:
        return self.encode(self.low)

    def high_encoded(self) -> float:
        return self.encode(self.high)

    def default_value(self) -> float:
        if self.default is not None:
            return self.clamp(float(self.default))
        if self.scale == "log":
            return self.decode(0.5 * (self.low_encoded() + self.high_encoded()))
        return self.decode(0.5 * (float(self.low) + float(self.high)))


def search_space_rows(search_space: Sequence[ParameterSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in search_space]


def search_space_paths(search_space: Sequence[ParameterSpec]) -> list[str]:
    return [spec.path for spec in search_space]


def parameter_contract_snapshot(
    *,
    version: int,
    search_space_paths: Sequence[str],
    runtime_parameter_keys: Sequence[str],
) -> dict[str, Any]:
    return {
        "version": int(version),
        "search_space_paths": list(search_space_paths),
        "runtime_parameter_keys": list(runtime_parameter_keys),
    }


def parameter_display_order(
    parameters: dict[str, Any] | None = None,
    *,
    preferred_paths: Sequence[str],
    runtime_parameter_keys: Sequence[str],
) -> list[str]:
    params = parameters if isinstance(parameters, dict) else {}
    preferred = list(preferred_paths)
    known_tail = [key for key in runtime_parameter_keys if key not in preferred]
    extras = sorted(
        key
        for key in params
        if key not in preferred and key not in known_tail and not str(key).startswith("optimizer_")
    )
    return [*preferred, *known_tail, *extras]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def _cached_campaign_search_space_paths(
    campaign_config_path: str,
    *,
    search_space_key: str,
) -> tuple[str, ...]:
    payload = _read_json(Path(campaign_config_path))
    rows = payload.get(search_space_key) or []
    paths = [
        str(row.get("path"))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("path")
    ]
    return tuple(paths)


def campaign_search_space_paths(
    campaign_dir: str | Path | None,
    *,
    fallback: Sequence[str] | None = None,
    config_filename: str = "campaign_config.json",
    search_space_key: str = "search_space",
) -> list[str]:
    fallback_paths = list(fallback or [])
    if campaign_dir is None:
        return fallback_paths
    campaign_config_path = Path(campaign_dir) / config_filename
    if not campaign_config_path.exists():
        return fallback_paths
    paths = list(
        _cached_campaign_search_space_paths(
            str(campaign_config_path.resolve()),
            search_space_key=search_space_key,
        )
    )
    return paths or fallback_paths


__all__ = [
    "ParameterSpec",
    "campaign_search_space_paths",
    "parameter_contract_snapshot",
    "parameter_display_order",
    "search_space_paths",
    "search_space_rows",
]
