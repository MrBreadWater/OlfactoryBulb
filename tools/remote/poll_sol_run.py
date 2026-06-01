"""Poll one remote Slurm-backed OBGPU run and emit JSON status."""

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

from neuroinfra.remote_script_polling import (  # noqa: E402
    TERMINAL_FAIL,
    TERMINAL_OK,
    cleanup_worktree,
    normalize_state,
    poll_result_payload,
    query_state,
    read_json_file,
    read_tail,
    run_command,
)


def main() -> None:
    """Emit JSON job state and result readiness for one remote run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--wrapper-dir", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--worktree-path", default=None)
    parser.add_argument("--skip-sacct", action="store_true")
    parser.add_argument("--skip-tails", action="store_true")
    args = parser.parse_args()

    payload = poll_result_payload(
        job_id=str(args.job_id),
        result_dir=args.result_dir,
        wrapper_dir=args.wrapper_dir,
        repo_root=args.repo_root,
        worktree_path=args.worktree_path,
        include_sacct=not bool(args.skip_sacct),
        include_tails=not bool(args.skip_tails),
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
