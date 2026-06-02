"""Downloader checks for the granule-cell reference-data source bundle."""

from __future__ import annotations

from pathlib import Path

from olfactorybulb.audit.reference_dataset_config import load_dataset_config
from olfactorybulb.audit.reference_sources import ensure_reference_sources, local_source_path


DATASET_ID = "granule_cells"
REQUIRED_SOURCE_IDS = (
    "burton2015_article_html",
    "geramita2016_article_html",
    "geramita2016_fig4_source_data",
    "hu2016_article_html",
    "egger2005_article_html",
    "labarrera2013_article_html",
    "giridhar2012_article_html",
)


config = load_dataset_config(dataset_id=DATASET_ID)
downloaded, errors = ensure_reference_sources(config=config, source_ids=REQUIRED_SOURCE_IDS, force=False, strict=False)
assert not errors, errors

for source_id in REQUIRED_SOURCE_IDS:
    path = local_source_path(source_id, config=config)
    assert path.exists(), source_id
    assert path.stat().st_size > 0, source_id
    assert path.suffix in {".html", ".docx"}, source_id
    assert downloaded.get(source_id, path).exists(), source_id

s8_like = local_source_path("geramita2016_fig4_source_data", config=config)
assert s8_like.suffix == ".docx"
assert s8_like.stat().st_size > 0

print("download_gc_reference_sources: OK")
