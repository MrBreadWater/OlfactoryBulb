"""Configuration loader for declarative literature-validation audits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import importlib
from pathlib import Path
from typing import Any
import tomllib

from .reference_data import REPO_ROOT


REFERENCE_VALIDATION_CONFIG_DIR = REPO_ROOT / "research_context" / "reference_validations"
DEFAULT_REFERENCE_VALIDATION_ID = "burton_urban_fi"
_LOADED_EXTENSION_SPECS: set[str] = set()


def _validation_config_path(validation_id: str) -> Path:
    return REFERENCE_VALIDATION_CONFIG_DIR / f"{validation_id}.validation.toml"


def load_reference_validation_config(
    *,
    validation_id: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    if path is None:
        validation_id = str(validation_id or DEFAULT_REFERENCE_VALIDATION_ID)
        path = _validation_config_path(validation_id)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference validation config not found: {path}")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if "validation_id" not in config:
        config["validation_id"] = path.stem.replace(".validation", "")
    config["__path__"] = str(path)
    return config


def list_reference_validation_ids() -> list[str]:
    ids: list[str] = []
    if not REFERENCE_VALIDATION_CONFIG_DIR.exists():
        return ids
    for path in sorted(REFERENCE_VALIDATION_CONFIG_DIR.glob("*.validation.toml")):
        if path.name.startswith("TEMPLATE."):
            continue
        ids.append(path.name[: -len(".validation.toml")])
    return ids


def _normalize_extension_specs(
    source: Mapping[str, Any] | Iterable[str] | None,
) -> list[str]:
    if source is None:
        return []
    raw: Any = source.get("extensions", []) if isinstance(source, Mapping) else source
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    if not isinstance(raw, Iterable):
        raise ValueError("Reference validation extensions must be an iterable of module specs")
    return [str(spec).strip() for spec in raw if str(spec).strip()]


def load_validation_extensions(source: Mapping[str, Any] | Iterable[str] | None) -> list[str]:
    loaded: list[str] = []
    for spec in _normalize_extension_specs(source):
        if spec in _LOADED_EXTENSION_SPECS:
            loaded.append(spec)
            continue
        module_name, _, callable_name = spec.partition(":")
        module = importlib.import_module(module_name)
        if callable_name:
            registrar = getattr(module, callable_name)
            registrar()
        elif hasattr(module, "register_reference_validation_extensions"):
            getattr(module, "register_reference_validation_extensions")()
        _LOADED_EXTENSION_SPECS.add(spec)
        loaded.append(spec)
    return loaded


__all__ = [
    "DEFAULT_REFERENCE_VALIDATION_ID",
    "REFERENCE_VALIDATION_CONFIG_DIR",
    "list_reference_validation_ids",
    "load_reference_validation_config",
    "load_validation_extensions",
]
