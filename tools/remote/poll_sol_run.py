"""Poll one remote Slurm-backed OBGPU run and emit JSON status."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


TERMINAL_OK = {"COMPLETED"}
TERMINAL_FAIL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED_WITH_ERRORS",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def run_command(command: list[str]) -> str:
    """Return stripped stdout from a subprocess or an empty string on failure."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def normalize_state(raw_state: str) -> str:
    """Normalize Slurm state tokens by removing suffixes such as '+'."""
    return raw_state.split()[0].split("+", 1)[0].strip().upper()


def query_state(job_id: str) -> str:
    """Query Slurm for one job state using sacct first, then squeue."""
    sacct_output = run_command(["sacct", "-j", str(job_id), "--format=State", "--noheader"])
    if sacct_output:
        for line in sacct_output.splitlines():
            state = normalize_state(line)
            if state:
                return state

    squeue_output = run_command(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    if squeue_output:
        return normalize_state(squeue_output)

    return "UNKNOWN"


def main() -> None:
    """Emit JSON job state and result readiness for one remote run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    summary_exists = (result_dir / "summary.json").exists()
    stdout_exists = (result_dir / "stdout.txt").exists()
    stderr_exists = (result_dir / "stderr.txt").exists()

    state = query_state(args.job_id)
    done = False
    ok = False

    if summary_exists:
        done = True
        ok = True
    elif state in TERMINAL_FAIL:
        done = True
        ok = False
    elif state in TERMINAL_OK:
        done = True
        ok = False
    elif stdout_exists or stderr_exists:
        # The wrapper started writing logs, but the final summary is not present yet.
        done = False
        ok = False

    payload = {
        "job_id": str(args.job_id),
        "state": state,
        "done": done,
        "ok": ok,
        "result_dir": str(result_dir),
        "summary_exists": summary_exists,
        "stdout_exists": stdout_exists,
        "stderr_exists": stderr_exists,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
