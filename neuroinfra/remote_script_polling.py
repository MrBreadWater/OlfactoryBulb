"""Remote-safe polling helpers for uploaded Slurm wrapper scripts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


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

TAIL_BYTES = 4096


def run_command(command: list[str] | tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Return one completed subprocess without raising on non-zero exit."""
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )


def normalize_state(raw_state: str) -> str:
    """Normalize Slurm state tokens by removing suffixes such as '+'."""
    return str(raw_state).split()[0].split("+", 1)[0].strip().upper()


def query_state(
    job_id: str,
    *,
    include_sacct: bool = True,
    run_command_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
) -> dict[str, str]:
    """Query Slurm for the top-level job state using squeue, then sacct when allowed."""
    squeue_reason = ""
    squeue_location = ""
    squeue_completed = run_command_fn(["squeue", "-j", str(job_id), "-h", "-o", "%T|%R"])
    squeue_output = (squeue_completed.stdout or "").strip()
    if squeue_completed.returncode == 0 and squeue_output:
        first_line = squeue_output.splitlines()[0]
        parts = first_line.split("|", 1)
        if len(parts) == 2:
            squeue_state = normalize_state(parts[0])
            squeue_detail = parts[1].strip()
            if squeue_state == "PENDING":
                squeue_reason = squeue_detail
            else:
                squeue_location = squeue_detail
        else:
            squeue_state = normalize_state(first_line)
        if squeue_state == "PENDING":
            return {"state": squeue_state, "reason": squeue_reason, "location": squeue_location}

    if include_sacct:
        sacct_completed = run_command_fn(
            [
                "sacct",
                "-j",
                str(job_id),
                "--format=JobIDRaw,State",
                "--parsable2",
                "--noheader",
            ]
        )
        sacct_output = (sacct_completed.stdout or "").strip()
        if sacct_completed.returncode == 0 and sacct_output:
            for line in sacct_output.splitlines():
                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue
                raw_job_id, raw_state = parts
                if raw_job_id.strip() == str(job_id):
                    state = normalize_state(raw_state)
                    if state:
                        return {"state": state, "reason": squeue_reason, "location": squeue_location}
            for line in sacct_output.splitlines():
                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue
                state = normalize_state(parts[1])
                if state:
                    return {"state": state, "reason": squeue_reason, "location": squeue_location}

    if squeue_completed.returncode == 0 and squeue_output:
        return {
            "state": normalize_state(squeue_output.split("|", 1)[0]),
            "reason": squeue_reason,
            "location": squeue_location,
        }

    return {"state": "UNKNOWN", "reason": "", "location": ""}


def cleanup_worktree(
    repo_root: str,
    worktree_path: str,
    *,
    run_command_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
    remove_tree_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Best-effort cleanup for a per-run detached git worktree."""
    repo_root_path = Path(repo_root).expanduser().resolve()
    worktree = Path(worktree_path).expanduser().resolve()
    remove_tree = remove_tree_fn or (lambda path_text: shutil.rmtree(path_text, ignore_errors=True))

    remove_completed = run_command_fn(
        ["git", "-C", str(repo_root_path), "worktree", "remove", "--force", str(worktree)]
    )
    remove_ok = remove_completed.returncode == 0 or not worktree.exists()
    if worktree.exists():
        remove_tree(str(worktree))
    prune_completed = run_command_fn(["git", "-C", str(repo_root_path), "worktree", "prune"])
    return {
        "attempted": True,
        "ok": remove_ok and prune_completed.returncode == 0,
        "remove_stderr": (remove_completed.stderr or "").strip(),
        "prune_stderr": (prune_completed.stderr or "").strip(),
    }


def read_tail(path: str | Path, *, tail_bytes: int = TAIL_BYTES) -> str:
    """Return the trailing text from one file when it exists."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    with open(file_path, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(size - max(int(tail_bytes), 0), 0))
        return handle.read().decode("utf-8", errors="replace")


def read_json_file(path: str | Path, *, default: Any) -> Any:
    """Return one decoded JSON file or ``default`` when missing/invalid."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return default
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return default


def poll_result_payload(
    *,
    job_id: str,
    result_dir: str | Path,
    wrapper_dir: str | Path | None = None,
    repo_root: str | None = None,
    worktree_path: str | None = None,
    include_sacct: bool = True,
    include_tails: bool = True,
    run_command_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
    remove_tree_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return one JSON-ready job-status payload for a remote Slurm run."""
    result_dir_path = Path(result_dir).expanduser().resolve()
    wrapper_dir_path = (
        Path(wrapper_dir).expanduser().resolve()
        if wrapper_dir not in (None, "")
        else result_dir_path.parent / ".obgpu-wrapper" / result_dir_path.name
    )
    summary_exists = (result_dir_path / "summary.json").exists()
    stdout_path = (
        wrapper_dir_path / "stdout.txt"
        if (wrapper_dir_path / "stdout.txt").exists()
        else result_dir_path / "stdout.txt"
    )
    stderr_path = (
        wrapper_dir_path / "stderr.txt"
        if (wrapper_dir_path / "stderr.txt").exists()
        else result_dir_path / "stderr.txt"
    )
    bootstrap_path = (
        wrapper_dir_path / "bootstrap.log"
        if (wrapper_dir_path / "bootstrap.log").exists()
        else result_dir_path / "bootstrap.log"
    )
    command_path = (
        wrapper_dir_path / "command.txt"
        if (wrapper_dir_path / "command.txt").exists()
        else result_dir_path / "command.txt"
    )
    progress_path = result_dir_path / "sim_progress.json"
    wrapper_slurm_logs = sorted(wrapper_dir_path.glob("slurm-*.out")) if wrapper_dir_path.exists() else []
    result_slurm_logs = sorted(result_dir_path.glob("slurm-*.out"))
    slurm_logs = wrapper_slurm_logs or result_slurm_logs
    stdout_exists = stdout_path.exists()
    stderr_exists = stderr_path.exists()
    bootstrap_exists = bootstrap_path.exists()
    command_exists = command_path.exists()
    progress_exists = progress_path.exists()
    slurm_log_exists = bool(slurm_logs)
    progress_payload = read_json_file(progress_path, default={})

    state_payload = query_state(job_id, include_sacct=include_sacct, run_command_fn=run_command_fn)
    state = state_payload["state"]
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
        done = False
        ok = False

    cleanup_payload = {"attempted": False, "ok": True, "remove_stderr": "", "prune_stderr": ""}
    if done and repo_root and worktree_path:
        cleanup_payload = cleanup_worktree(
            repo_root,
            worktree_path,
            run_command_fn=run_command_fn,
            remove_tree_fn=remove_tree_fn,
        )

    return {
        "job_id": str(job_id),
        "state": state,
        "reason": state_payload.get("reason", ""),
        "location": state_payload.get("location", ""),
        "done": done,
        "ok": ok,
        "result_dir": str(result_dir_path),
        "summary_exists": summary_exists,
        "stdout_exists": stdout_exists,
        "stderr_exists": stderr_exists,
        "bootstrap_exists": bootstrap_exists,
        "bootstrap_tail": "" if not include_tails else read_tail(bootstrap_path),
        "command_exists": command_exists,
        "progress_exists": progress_exists,
        "progress_current_ms": progress_payload.get("current_ms"),
        "progress_total_ms": progress_payload.get("total_ms"),
        "progress_percent": progress_payload.get("percent"),
        "progress_payload": progress_payload,
        "stdout_tail": "" if not include_tails else read_tail(stdout_path),
        "stderr_tail": "" if not include_tails else read_tail(stderr_path),
        "slurm_log_exists": slurm_log_exists,
        "slurm_logs": [str(path) for path in slurm_logs],
        "slurm_tail": "" if not include_tails else (read_tail(slurm_logs[-1]) if slurm_logs else ""),
        "wrapper_dir": str(wrapper_dir_path),
        "cleanup": cleanup_payload,
    }
