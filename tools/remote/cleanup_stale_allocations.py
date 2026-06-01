"""Cancel stale notebook-managed Slurm allocations and emit a JSON summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _install_import_root() -> None:
    """Prepend the nearest parent directory that exposes ``neuroinfra``."""
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents]
    for candidate in candidates:
        if (candidate / "neuroinfra" / "__init__.py").exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return


_install_import_root()

from neuroinfra.remote_script_allocations import (  # noqa: E402
    cancel_job,
    determine_stale_reason,
    load_allocation_payload,
    stale_allocation_actions,
)


def main() -> None:
    """Scan one allocation root and print JSON cancellation actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--default-timeout-s", type=int, default=120)
    args = parser.parse_args()
    actions = stale_allocation_actions(
        args.root,
        default_timeout_s=max(int(args.default_timeout_s), 0),
    )
    print(json.dumps(actions, sort_keys=True))


if __name__ == "__main__":
    main()
