"""Reusable helpers for remote Slurm state queries and preflight caching."""

from __future__ import annotations

from hashlib import sha1
import json
from pathlib import PurePosixPath
import shlex
import subprocess
import time
from typing import Any, Callable, Mapping, MutableMapping


REMOTE_SLURM_TERMINAL_OK = {"COMPLETED"}
REMOTE_SLURM_TERMINAL_FAIL = {
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


def normalize_slurm_state(raw_state: str) -> str:
    """Normalize Slurm state tokens by removing suffixes such as '+'."""
    return str(raw_state).split()[0].split("+", 1)[0].strip().upper()


def build_remote_preflight_command(*, remote_repo_root: PurePosixPath) -> str:
    """Build one remote shell command that validates remote-side prerequisites."""
    checks = [
        "test -d {}".format(shlex.quote(remote_repo_root.as_posix())),
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"',
        'test -n "$REMOTE_PYTHON"',
        "command -v bash >/dev/null",
        "command -v git >/dev/null",
        "command -v sbatch >/dev/null",
        "command -v sacct >/dev/null",
        "command -v scancel >/dev/null",
        "command -v squeue >/dev/null",
        "command -v srun >/dev/null",
    ]
    return " && ".join(checks)


def remote_preflight_cache_key(
    *,
    connection_key: str,
    remote_repo_root: PurePosixPath,
    remote_conda_activate_cmd: str = "",
    helper_signature: str = "",
) -> str:
    """Return the runtime cache key for one successful remote preflight."""
    payload = json.dumps(
        {
            "endpoint": str(connection_key),
            "remote_repo_root": remote_repo_root.as_posix(),
            "remote_conda_activate_cmd": str(remote_conda_activate_cmd or ""),
            "helper_signature": str(helper_signature or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def run_remote_preflight_cached(
    *,
    cache: MutableMapping[str, Any],
    cache_key: str,
    remote_repo_root: PurePosixPath,
    command: str,
    run_command: Callable[[str], subprocess.CompletedProcess[str]],
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run one remote preflight only once per notebook session."""
    cached = cache.get(cache_key)
    if cached is not None:
        return (
            subprocess.CompletedProcess(
                args=["remote-preflight-cache", remote_repo_root.as_posix()],
                returncode=0,
                stdout=str(cached.get("stdout") or ""),
                stderr="",
            ),
            True,
        )

    completed = run_command(command)
    if completed.returncode == 0:
        cache[cache_key] = {
            "timestamp": time.time(),
            "stdout": completed.stdout or "",
        }
    return completed, False


def build_remote_result_listing_command(*, remote_result_dir: PurePosixPath) -> str:
    """Build one remote shell command that lists synced result-directory files."""
    quoted_dir = shlex.quote(remote_result_dir.as_posix())
    return (
        "if test -d {quoted}; then "
        "find {quoted} -maxdepth 1 -type f -printf '%f\\t%s\\n' | sort; "
        "fi"
    ).format(quoted=quoted_dir)


def build_remote_cancel_command(*, job_id: str) -> str:
    """Build one remote shell command that cancels a submitted Slurm job."""
    return "scancel {}".format(shlex.quote(str(job_id)))


def query_remote_slurm_job_state(
    *,
    job_id: str,
    run_command: Callable[[str], subprocess.CompletedProcess[str]],
) -> dict[str, str]:
    """Query one remote Slurm job state without requiring a result directory."""
    query_command = (
        "squeue -j {job_id} -h -o '%T|%R' || true; "
        "printf '%s\\n' '__SACCT__'; "
        "sacct -j {job_id} --format=JobIDRaw,State --parsable2 --noheader || true"
    ).format(job_id=shlex.quote(str(job_id)))
    completed = run_command(query_command)
    if completed.returncode != 0:
        raise RuntimeError(
            "Remote Slurm job-state query failed.\n"
            "Job id: {job_id}\n"
            "Stdout:\n{stdout}\n\nStderr:\n{stderr}".format(
                job_id=job_id,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )

    squeue_text, _marker, sacct_text = (completed.stdout or "").partition("__SACCT__\n")
    squeue_output = squeue_text.strip()
    sacct_output = sacct_text.strip()
    squeue_reason = ""
    squeue_location = ""

    if squeue_output:
        first_line = squeue_output.splitlines()[0]
        parts = first_line.split("|", 1)
        if len(parts) == 2:
            squeue_state = normalize_slurm_state(parts[0])
            squeue_detail = parts[1].strip()
            if squeue_state == "PENDING":
                squeue_reason = squeue_detail
            else:
                squeue_location = squeue_detail
        else:
            squeue_state = normalize_slurm_state(first_line)
        if squeue_state == "PENDING":
            return {"state": squeue_state, "reason": squeue_reason, "location": squeue_location}

    if sacct_output:
        for line in sacct_output.splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            raw_job_id, raw_state = parts
            if raw_job_id.strip() == str(job_id):
                state = normalize_slurm_state(raw_state)
                if state:
                    return {"state": state, "reason": squeue_reason, "location": squeue_location}
        for line in sacct_output.splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            state = normalize_slurm_state(parts[1])
            if state:
                return {"state": state, "reason": squeue_reason, "location": squeue_location}

    if squeue_output:
        return {
            "state": normalize_slurm_state(squeue_output.split("|", 1)[0]),
            "reason": squeue_reason,
            "location": squeue_location,
        }
    return {"state": "UNKNOWN", "reason": "", "location": ""}


def remote_status_has_artifacts(status: Mapping[str, Any] | None) -> bool:
    """Return whether the remote poll status saw any useful output artifacts."""
    if not status:
        return False
    return any(
        bool(status.get(key))
        for key in (
            "summary_exists",
            "stdout_exists",
            "stderr_exists",
            "bootstrap_exists",
            "command_exists",
            "slurm_log_exists",
        )
    )
