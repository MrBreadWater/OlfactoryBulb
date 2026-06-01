"""Extract a declarative reference-data dataset into normalized CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_dataset_config import DEFAULT_REFERENCE_DATASET_ID  # noqa: E402
from olfactorybulb.audit.reference_dataset_engine import write_reference_dataset_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_REFERENCE_DATASET_ID, help="Dataset id to extract.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional explicit dataset config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_reference_dataset_outputs(dataset_id=args.dataset_id, config_path=args.config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
