"""Generic source acquisition for declarative reference-data datasets."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import requests

from .reference_dataset_config import (
    DEFAULT_REFERENCE_DATASET_ID,
    dataset_source_data_dir,
    dataset_sources,
    load_dataset_config,
)


BURTON2024_ARTICLE_PDF_SOURCE_ID = "burton2024_article_pdf"
BURTON2024_S1_TABLE_SOURCE_ID = "burton2024_s1_table"
BURTON2024_S2_TABLE_SOURCE_ID = "burton2024_s2_table"
BURTON2024_S8_DATA_SOURCE_ID = "burton2024_s8_data"
BURTON2024_S15_DATA_SOURCE_ID = "burton2024_s15_data"
BURTON2024_S16_DATA_SOURCE_ID = "burton2024_s16_data"

REQUIRED_BURTON2024_SOURCE_IDS = (
    BURTON2024_ARTICLE_PDF_SOURCE_ID,
    BURTON2024_S1_TABLE_SOURCE_ID,
    BURTON2024_S2_TABLE_SOURCE_ID,
    BURTON2024_S8_DATA_SOURCE_ID,
    BURTON2024_S15_DATA_SOURCE_ID,
    BURTON2024_S16_DATA_SOURCE_ID,
)


def _resolved_config(
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    return dict(config) if config is not None else load_dataset_config(dataset_id=dataset_id, path=config_path)


def manifest_sources(
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    return dataset_sources(_resolved_config(config=config, dataset_id=dataset_id, config_path=config_path))


def source_entries_by_id(
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        str(entry["source_id"]): entry
        for entry in manifest_sources(config=config, dataset_id=dataset_id, config_path=config_path)
    }


def source_entry(
    source_id: str,
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    try:
        return source_entries_by_id(config=config, dataset_id=dataset_id, config_path=config_path)[source_id]
    except KeyError as exc:
        raise KeyError(f"Unknown reference-data source id: {source_id}") from exc


def stable_source_url(
    source_id: str,
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
) -> str:
    return str(
        source_entry(source_id, config=config, dataset_id=dataset_id, config_path=config_path).get("source_url", "")
    )


def local_source_path(
    source_id: str,
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
) -> Path:
    resolved = _resolved_config(config=config, dataset_id=dataset_id, config_path=config_path)
    entry = source_entry(source_id, config=resolved)
    return dataset_source_data_dir(resolved) / str(entry["filename"])


def downloadable_entries(
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
    include_optional: bool = False,
    source_ids: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    resolved = _resolved_config(config=config, dataset_id=dataset_id, config_path=config_path)
    requested_ids = set(source_ids or [])
    entries: list[dict[str, Any]] = []
    for entry in manifest_sources(config=resolved):
        if not bool(entry.get("downloadable", True)):
            continue
        if requested_ids and str(entry["source_id"]) not in requested_ids:
            continue
        if not include_optional and not bool(entry.get("required", False)):
            continue
        entries.append(entry)
    return entries


def download_source_entry(
    entry: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
    force: bool = False,
    timeout_s: float = 120.0,
) -> Path:
    resolved = _resolved_config(config=config, dataset_id=dataset_id, config_path=config_path)
    source_dir = dataset_source_data_dir(resolved)
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / str(entry["filename"])
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    tmp_path = destination.with_name(destination.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    download_url = str(entry.get("download_url") or entry["source_url"])
    response = requests.get(download_url, allow_redirects=True, timeout=timeout_s, stream=True)
    try:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)
    finally:
        response.close()

    if _looks_like_block_page(tmp_path):
        tmp_path.unlink(missing_ok=True)
        _download_with_curl(download_url, tmp_path, timeout_s=timeout_s)

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded reference-data source is empty: {entry['source_id']}")

    expected_extension = str(entry.get("expected_extension", "") or "").strip().lower()
    if expected_extension and destination.suffix.lower() != expected_extension:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded reference-data source extension mismatch for {entry['source_id']}: "
            f"expected {expected_extension}, destination is {destination.suffix.lower()}"
        )

    tmp_path.replace(destination)
    return destination


def _looks_like_block_page(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        head = path.read_text(errors="ignore")[:4096].lower()
    except Exception:
        return False
    return "recaptcha/challengepage" in head or "google.com/recaptcha" in head


def _download_with_curl(download_url: str, destination: Path, *, timeout_s: float) -> None:
    command = [
        "curl",
        "-L",
        "--fail",
        "-A",
        "Mozilla/5.0",
        "--max-time",
        str(int(max(timeout_s, 1.0))),
        "-o",
        str(destination),
        download_url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        stderr = completed.stderr.strip() or completed.stdout.strip() or "curl download failed"
        raise RuntimeError(stderr)


def ensure_reference_sources(
    *,
    config: dict[str, Any] | None = None,
    dataset_id: str | None = None,
    config_path: Path | None = None,
    include_optional: bool = False,
    source_ids: list[str] | tuple[str, ...] | None = None,
    force: bool = False,
    strict: bool = True,
    timeout_s: float = 120.0,
) -> tuple[dict[str, Path], dict[str, str]]:
    resolved = _resolved_config(config=config, dataset_id=dataset_id, config_path=config_path)
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for entry in downloadable_entries(config=resolved, include_optional=include_optional, source_ids=source_ids):
        source_id = str(entry["source_id"])
        try:
            paths[source_id] = download_source_entry(entry, config=resolved, force=force, timeout_s=timeout_s)
        except Exception as exc:
            errors[source_id] = str(exc)
            if strict:
                raise
    return paths, errors


def default_dataset_id() -> str:
    return DEFAULT_REFERENCE_DATASET_ID
