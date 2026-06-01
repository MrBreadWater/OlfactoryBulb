"""Declarative dataset configuration for reusable reference-data extraction."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from .reference_data import REFERENCE_DATA_DIR


REFERENCE_DATASET_CONFIG_DIR = REFERENCE_DATA_DIR / "reference_datasets"
DEFAULT_REFERENCE_DATASET_ID = "pv_crh_epl_fsi"


def dataset_config_path(dataset_id: str) -> Path:
    return REFERENCE_DATASET_CONFIG_DIR / f"{dataset_id}.dataset.toml"


def load_dataset_config(*, dataset_id: str | None = None, path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = dataset_config_path(dataset_id or DEFAULT_REFERENCE_DATASET_ID)
    data = tomllib.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset config at {path}")
    config = dict(data)
    config["_config_path"] = str(path)
    config.setdefault("dataset_id", dataset_id or path.stem.replace(".dataset", ""))
    return config


def dataset_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = config.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("Dataset config 'sources' must be a list")
    return [dict(source) for source in sources]


def dataset_source_data_dir(config: dict[str, Any]) -> Path:
    subdir = str(config.get("source_data_subdir", config.get("dataset_id", DEFAULT_REFERENCE_DATASET_ID))).strip()
    if not subdir:
        raise ValueError("Dataset config must define source_data_subdir or dataset_id")
    return REFERENCE_DATA_DIR / "source_data" / subdir


def dataset_output_filenames(config: dict[str, Any]) -> dict[str, str]:
    outputs = config.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Dataset config 'outputs' must be a table")
    return {str(key): str(value) for key, value in outputs.items()}


def dataset_output_path(config: dict[str, Any], output_key: str) -> Path:
    filenames = dataset_output_filenames(config)
    try:
        filename = filenames[output_key]
    except KeyError as exc:
        raise KeyError(f"Dataset config missing output filename for {output_key!r}") from exc
    return REFERENCE_DATA_DIR / filename


def dataset_section(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = config.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"Dataset config section {key!r} must be a list")
    return [dict(item) for item in value]
