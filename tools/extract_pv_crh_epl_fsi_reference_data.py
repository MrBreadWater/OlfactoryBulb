"""Backward-compatible wrapper for the PV/CRH-overlap EPL-FSI reference dataset."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.reference_dataset_engine import write_reference_dataset_outputs  # noqa: E402


DATASET_ID = "pv_crh_epl_fsi"


def main() -> int:
    write_reference_dataset_outputs(dataset_id=DATASET_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
