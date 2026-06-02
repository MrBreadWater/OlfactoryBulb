"""Generic notebook configuration persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    """Convert arrays, scalars, and paths into JSON-serializable equivalents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json_config(config: Any, path: str | Path) -> Path:
    """Save one config-like payload as pretty JSON and return the written path."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(config), indent=2, sort_keys=True))
    return path


def load_json_config(path: str | Path) -> Any:
    """Load one JSON config-like payload from disk."""
    path = Path(path).expanduser().resolve()
    with open(path) as handle:
        return json.load(handle)


def list_json_configs(
    directory: str | Path | None = None,
    *,
    default_directory: str | Path | None = None,
    suffix: str = ".json",
) -> list[Path]:
    """Return a sorted list of config files in one directory."""
    if directory is None and default_directory is None:
        return []
    base = Path(directory).expanduser().resolve() if directory else Path(default_directory).expanduser().resolve()
    if not base.is_dir():
        return []
    return sorted(base.glob(f"*{suffix}"))
