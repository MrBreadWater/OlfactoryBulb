"""Backward-compatible wrapper for the granule-cell reference-data source downloader."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.download_reference_dataset_sources import main as generic_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = ["--dataset-id", "granule_cells"]
    if argv:
        args.extend(argv)
    return generic_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
