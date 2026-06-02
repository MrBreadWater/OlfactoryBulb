"""Concrete olfactory-bulb notebook config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from neuroinfra.notebooks.config_store import (
    list_json_configs,
    load_json_config,
    save_json_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class NotebookConfigHooks:
    """Domain hooks for notebook config normalization and comparison."""

    normalize_input_odors_fn: Callable[[Any], Any]
    resolve_effective_params_fn: Callable[[dict[str, Any]], dict[str, Any]]
    diff_values_fn: Callable[[Any, Any], list[dict[str, Any]]]


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Save one olfactory-bulb notebook config to JSON."""
    return save_json_config(dict(config), path)


def load_config(hooks: NotebookConfigHooks, path: str | Path) -> dict[str, Any]:
    """Load and normalize one olfactory-bulb notebook config from JSON."""
    data = load_json_config(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict config payload in {path}, got {type(data).__name__}")
    if data.get("input_odors") is not None:
        data["input_odors"] = hooks.normalize_input_odors_fn(data["input_odors"])
    return data


def list_saved_configs(directory: str | Path | None = None) -> list[Path]:
    """Return a sorted list of saved notebook config JSON files."""
    return list_json_configs(directory, default_directory=DEFAULT_CONFIGS_DIR)


def list_paramsets(
    include_saved: bool = False,
    configs_dir: str | Path | None = None,
) -> list[str] | dict[str, list[Any]]:
    """Return available built-in paramsets, optionally plus saved JSON configs."""
    import olfactorybulb.model as obmodel
    from olfactorybulb.paramsets.base import SilentNetwork

    names = sorted(
        name
        for name, obj in vars(obmodel).items()
        if isinstance(obj, type)
        and issubclass(obj, SilentNetwork)
        and obj is not SilentNetwork
    )

    if not include_saved:
        return names

    return {
        "builtin": names,
        "saved": list_saved_configs(configs_dir),
    }


def config_diff(
    hooks: NotebookConfigHooks,
    config1: dict[str, Any],
    config2: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare two notebook configs at the effective-params level."""
    snap1 = hooks.resolve_effective_params_fn(config1)["full_param_snapshot"]
    snap2 = hooks.resolve_effective_params_fn(config2)["full_param_snapshot"]
    return hooks.diff_values_fn(snap1, snap2)
