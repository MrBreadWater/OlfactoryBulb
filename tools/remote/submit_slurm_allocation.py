"""Submit one reusable Slurm allocation for notebook-managed follow-on job steps."""

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
    allocation_payload,
    submit_batch,
    write_holder_script,
)


def main() -> None:
    """Parse CLI args, write the holder script, submit it, and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--heartbeat-timeout-s", type=int, default=120)
    parser.add_argument("--sbatch-arg", action="append", default=[])
    args = parser.parse_args()

    alloc_root = Path(args.alloc_root).expanduser().resolve()
    script_path, slurm_log_path, heartbeat_path = write_holder_script(args, alloc_root)
    job_id = submit_batch(script_path)
    payload = allocation_payload(
        job_id=str(job_id),
        name=str(args.name),
        allocation_root=alloc_root,
        batch_script=script_path,
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_s=max(int(args.heartbeat_timeout_s), 0),
        slurm_log_pattern=slurm_log_path,
    )
    (alloc_root / "allocation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
