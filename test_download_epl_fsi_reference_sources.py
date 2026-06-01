"""Downloader checks for EPL-FSI reference-source acquisition."""

from __future__ import annotations

from olfactorybulb.audit.reference_sources import (
    REQUIRED_BURTON2024_SOURCE_IDS,
    ensure_reference_sources,
    local_source_path,
    source_entry,
)


downloaded, errors = ensure_reference_sources(source_ids=list(REQUIRED_BURTON2024_SOURCE_IDS), strict=False)
assert not errors, errors

for source_id in REQUIRED_BURTON2024_SOURCE_IDS:
    path = local_source_path(source_id)
    entry = source_entry(source_id)
    assert source_id in downloaded or path.exists(), source_id
    assert path.exists(), path
    assert path.stat().st_size > 0, path
    expected_extension = str(entry.get("expected_extension", "") or "")
    if expected_extension:
        assert path.suffix.lower() == expected_extension.lower(), (path, expected_extension)

for source_id in ("burton2024_s8_data", "burton2024_s15_data"):
    path = local_source_path(source_id)
    assert path.stat().st_size > 0, path

print("download_epl_fsi_reference_sources: OK")
