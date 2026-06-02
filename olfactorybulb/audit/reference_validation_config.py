"""Configuration loader for declarative literature-validation audits."""

from __future__ import annotations

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
        ids.append(path.name[: -len(".validation.toml")])
    return ids


def validation_title(config: dict[str, Any]) -> str:
    return str(config.get("title") or config.get("validation_id") or "Reference validation")


def validation_protocol_runner_id(config: dict[str, Any]) -> str:
    return str(config.get("protocol_runner") or "").strip()


def validation_rule_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rules = config.get("checks", [])
    if not isinstance(rules, list):
        raise ValueError("Reference validation config 'checks' must be an array of tables")
    return [dict(rule) for rule in rules]


def validation_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Reference validation config 'defaults' must be a table")
    return dict(defaults)


def validation_protocol_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("protocol", {})
    if not isinstance(defaults, dict):
        raise ValueError("Reference validation config 'protocol' must be a table")
    return dict(defaults)


def validation_extension_specs(config: dict[str, Any]) -> list[str]:
    raw = config.get("extensions", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Reference validation config 'extensions' must be an array of module specs")
    return [str(spec).strip() for spec in raw if str(spec).strip()]


def load_validation_extensions(config: dict[str, Any]) -> list[str]:
    loaded: list[str] = []
    for spec in validation_extension_specs(config):
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


def validation_skip_item(config: dict[str, Any]) -> dict[str, Any] | None:
    skip_item = config.get("skip_item")
    if skip_item is None:
        return None
    if not isinstance(skip_item, dict):
        raise ValueError("Reference validation config 'skip_item' must be a table")
    return dict(skip_item)


__all__ = [
    "DEFAULT_REFERENCE_VALIDATION_ID",
    "REFERENCE_VALIDATION_CONFIG_DIR",
    "list_reference_validation_ids",
    "load_reference_validation_config",
    "load_validation_extensions",
    "validation_defaults",
    "validation_extension_specs",
    "validation_protocol_defaults",
    "validation_protocol_runner_id",
    "validation_rule_specs",
    "validation_skip_item",
    "validation_title",
]
