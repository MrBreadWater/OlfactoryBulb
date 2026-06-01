"""Download sources for a declarative reference-data dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_dataset_config import DEFAULT_REFERENCE_DATASET_ID, dataset_source_data_dir, load_dataset_config  # noqa: E402
from olfactorybulb.audit.reference_sources import ensure_reference_sources, source_entry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_REFERENCE_DATASET_ID, help="Dataset id to download.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional explicit dataset config path.")
    parser.add_argument("--include-optional", action="store_true", help="Download optional sources as well.")
    parser.add_argument("--force", action="store_true", help="Re-download even when local files exist.")
    parser.add_argument("--source-id", action="append", default=[], help="Restrict to one or more source ids.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_dataset_config(dataset_id=args.dataset_id, path=args.config_path)
    source_ids = args.source_id or None
    downloaded, errors = ensure_reference_sources(
        config=config,
        include_optional=args.include_optional,
        source_ids=source_ids,
        force=args.force,
        strict=False,
    )

    print(f"Source data directory: {dataset_source_data_dir(config)}")
    for source_id, path in sorted(downloaded.items()):
        entry = source_entry(source_id, config=config)
        print(f"[ok] {source_id}: {path} ({path.stat().st_size} bytes) <- {entry['source_url']}")

    if errors:
        for source_id, message in sorted(errors.items()):
            entry = source_entry(source_id, config=config)
            print(f"[error] {source_id}: {message} <- {entry['source_url']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
