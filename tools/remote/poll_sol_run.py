"""Poll one remote Slurm-backed OBGPU run and emit JSON status."""

import argparse
import json
import shutil
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

TAIL_BYTES = 4096


def run_command(command):
    """Return one completed subprocess without raising on non-zero exit."""
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )


def normalize_state(raw_state):
    """Normalize Slurm state tokens by removing suffixes such as '+'."""
    return raw_state.split()[0].split("+", 1)[0].strip().upper()


def query_state(job_id):
    """Query Slurm for the top-level job state using sacct first, then squeue."""
    squeue_reason = ""
    squeue_location = ""
    squeue_completed = run_command(["squeue", "-j", str(job_id), "-h", "-o", "%T|%R"])
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

    sacct_completed = run_command(
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


def cleanup_worktree(repo_root, worktree_path):
    """Best-effort cleanup for a per-run detached git worktree."""
    repo_root_path = Path(repo_root).expanduser().resolve()
    worktree = Path(worktree_path).expanduser().resolve()

    remove_completed = subprocess.run(
        ["git", "-C", str(repo_root_path), "worktree", "remove", "--force", str(worktree)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    remove_ok = remove_completed.returncode == 0 or not worktree.exists()
    if worktree.exists():
        shutil.rmtree(str(worktree), ignore_errors=True)
    prune_completed = subprocess.run(
        ["git", "-C", str(repo_root_path), "worktree", "prune"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    return {
        "attempted": True,
        "ok": remove_ok and prune_completed.returncode == 0,
        "remove_stderr": (remove_completed.stderr or "").strip(),
        "prune_stderr": (prune_completed.stderr or "").strip(),
    }


def read_tail(path):
    """Return the trailing text from one file when it exists."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    with open(file_path, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(size - TAIL_BYTES, 0))
        return handle.read().decode("utf-8", errors="replace")


def main():
    """Emit JSON job state and result readiness for one remote run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--wrapper-dir", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--worktree-path", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    wrapper_dir = (
        Path(args.wrapper_dir).expanduser().resolve()
        if args.wrapper_dir not in (None, "")
        else result_dir.parent / ".obgpu-wrapper" / result_dir.name
    )
    summary_exists = (result_dir / "summary.json").exists()
    stdout_path = wrapper_dir / "stdout.txt" if (wrapper_dir / "stdout.txt").exists() else result_dir / "stdout.txt"
    stderr_path = wrapper_dir / "stderr.txt" if (wrapper_dir / "stderr.txt").exists() else result_dir / "stderr.txt"
    bootstrap_path = (
        wrapper_dir / "bootstrap.log"
        if (wrapper_dir / "bootstrap.log").exists()
        else result_dir / "bootstrap.log"
    )
    command_path = wrapper_dir / "command.txt" if (wrapper_dir / "command.txt").exists() else result_dir / "command.txt"
    wrapper_slurm_logs = sorted(wrapper_dir.glob("slurm-*.out")) if wrapper_dir.exists() else []
    result_slurm_logs = sorted(result_dir.glob("slurm-*.out"))
    slurm_logs = wrapper_slurm_logs or result_slurm_logs
    stdout_exists = stdout_path.exists()
    stderr_exists = stderr_path.exists()
    bootstrap_exists = bootstrap_path.exists()
    command_exists = command_path.exists()
    slurm_log_exists = bool(slurm_logs)

    state_payload = query_state(args.job_id)
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
        # The wrapper started writing logs, but the final summary is not present yet.
        done = False
        ok = False

    cleanup_payload = {"attempted": False, "ok": True, "remove_stderr": "", "prune_stderr": ""}
    if done and args.repo_root and args.worktree_path:
        cleanup_payload = cleanup_worktree(args.repo_root, args.worktree_path)

    payload = {
        "job_id": str(args.job_id),
        "state": state,
        "reason": state_payload.get("reason", ""),
        "location": state_payload.get("location", ""),
        "done": done,
        "ok": ok,
        "result_dir": str(result_dir),
        "summary_exists": summary_exists,
        "stdout_exists": stdout_exists,
        "stderr_exists": stderr_exists,
        "bootstrap_exists": bootstrap_exists,
        "bootstrap_tail": read_tail(bootstrap_path),
        "command_exists": command_exists,
        "stdout_tail": read_tail(stdout_path),
        "stderr_tail": read_tail(stderr_path),
        "slurm_log_exists": slurm_log_exists,
        "slurm_logs": [str(path) for path in slurm_logs],
        "slurm_tail": read_tail(slurm_logs[-1]) if slurm_logs else "",
        "wrapper_dir": str(wrapper_dir),
        "cleanup": cleanup_payload,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
