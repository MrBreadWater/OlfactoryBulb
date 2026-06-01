"""Manifest-backed source acquisition for EPL fast-spiking interneuron reference data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from .reference_data import REFERENCE_DATA_DIR, SOURCE_MANIFEST_FILENAME


SOURCE_DATA_DIR = REFERENCE_DATA_DIR / "source_data" / "epl_fsi"
SOURCE_MANIFEST_PATH = REFERENCE_DATA_DIR / SOURCE_MANIFEST_FILENAME

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


def load_source_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or SOURCE_MANIFEST_PATH
    return json.loads(manifest_path.read_text())


def manifest_sources(path: Path | None = None) -> list[dict[str, Any]]:
    manifest = load_source_manifest(path)
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"Invalid source manifest at {path or SOURCE_MANIFEST_PATH}: missing 'sources' list")
    return [dict(source) for source in sources]


def source_entries_by_id(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return {str(entry["source_id"]): entry for entry in manifest_sources(path)}


def source_entry(source_id: str, path: Path | None = None) -> dict[str, Any]:
    try:
        return source_entries_by_id(path)[source_id]
    except KeyError as exc:
        raise KeyError(f"Unknown EPL-FSI source id: {source_id}") from exc


def stable_source_url(source_id: str, path: Path | None = None) -> str:
    return str(source_entry(source_id, path).get("source_url", ""))


def local_source_path(source_id: str, path: Path | None = None) -> Path:
    entry = source_entry(source_id, path)
    return SOURCE_DATA_DIR / str(entry["filename"])


def downloadable_entries(
    *,
    include_optional: bool = False,
    source_ids: list[str] | tuple[str, ...] | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    requested_ids = set(source_ids or [])
    entries: list[dict[str, Any]] = []
    for entry in manifest_sources(path):
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
    force: bool = False,
    timeout_s: float = 120.0,
) -> Path:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = SOURCE_DATA_DIR / str(entry["filename"])
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    tmp_path = destination.with_name(destination.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    response = requests.get(str(entry["source_url"]), allow_redirects=True, timeout=timeout_s, stream=True)
    try:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)
    finally:
        response.close()

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded EPL-FSI source is empty: {entry['source_id']}")

    expected_extension = str(entry.get("expected_extension", "") or "").strip().lower()
    if expected_extension and destination.suffix.lower() != expected_extension:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded EPL-FSI source extension mismatch for {entry['source_id']}: "
            f"expected {expected_extension}, destination is {destination.suffix.lower()}"
        )

    tmp_path.replace(destination)
    return destination


def ensure_reference_sources(
    *,
    include_optional: bool = False,
    source_ids: list[str] | tuple[str, ...] | None = None,
    force: bool = False,
    strict: bool = True,
    timeout_s: float = 120.0,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for entry in downloadable_entries(include_optional=include_optional, source_ids=source_ids):
        source_id = str(entry["source_id"])
        try:
            paths[source_id] = download_source_entry(entry, force=force, timeout_s=timeout_s)
        except Exception as exc:
            errors[source_id] = str(exc)
            if strict:
                raise
    return paths, errors
