"""Download stable EPL-FSI reference sources from the manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_sources import (  # noqa: E402
    SOURCE_DATA_DIR,
    ensure_reference_sources,
    source_entry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Download optional sources from the manifest in addition to the required Burton 2024 files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if a non-empty local copy already exists.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Restrict downloads to one or more manifest source ids.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_ids = args.source_id or None
    downloaded, errors = ensure_reference_sources(
        include_optional=args.include_optional,
        source_ids=source_ids,
        force=args.force,
        strict=False,
    )

    print(f"Source data directory: {SOURCE_DATA_DIR}")
    for source_id, path in sorted(downloaded.items()):
        entry = source_entry(source_id)
        print(f"[ok] {source_id}: {path} ({path.stat().st_size} bytes) <- {entry['source_url']}")

    if errors:
        for source_id, message in sorted(errors.items()):
            entry = source_entry(source_id)
            print(f"[error] {source_id}: {message} <- {entry['source_url']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
