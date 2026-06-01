"""Remote-safe allocation lifecycle helpers for uploaded Slurm wrapper scripts."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from neuroinfra.remote_script_common import slurm_directives


def holder_script_lines(args: Any, alloc_root: str | Path) -> list[str]:
    """Return the shell lines for one long-lived reusable allocation job."""
    alloc_root_path = Path(alloc_root).expanduser().resolve()
    slurm_log_path = alloc_root_path / "allocation-%j.out"
    heartbeat_path = alloc_root_path / "notebook-heartbeat.txt"
    lease_expired_path = alloc_root_path / "lease-expired.txt"
    heartbeat_timeout_s = max(int(args.heartbeat_timeout_s), 0)
    return [
        "#!/usr/bin/env bash",
        *slurm_directives(args, args.name),
        f"#SBATCH --output={slurm_log_path}",
        f"#SBATCH --error={slurm_log_path}",
        "set -Eeuo pipefail",
        "trap 'exit 0' TERM INT HUP",
        f"printf '%s\\n' \"$SLURM_JOB_ID\" > {alloc_root_path / 'job_id.txt'}",
        f"printf '%s\\n' \"${{SLURM_JOB_NODELIST:-}}\" > {alloc_root_path / 'nodelist.txt'}",
        f"heartbeat_path='{heartbeat_path}'",
        f"heartbeat_timeout_s={heartbeat_timeout_s}",
        f"lease_expired_path='{lease_expired_path}'",
        "touch \"$heartbeat_path\"",
        "while true; do",
        "  sleep 10",
        "  if [[ \"$heartbeat_timeout_s\" -le 0 ]]; then",
        "    continue",
        "  fi",
        "  now=$(date +%s)",
        "  if [[ -e \"$heartbeat_path\" ]]; then",
        "    last=$(stat -c %Y \"$heartbeat_path\" 2>/dev/null || echo 0)",
        "  else",
        "    last=0",
        "  fi",
        "  age=$((now - last))",
        "  if [[ \"$age\" -gt \"$heartbeat_timeout_s\" ]]; then",
        "    printf '[OBGPU allocation] notebook heartbeat expired after %ss at %s\\n' \"$age\" \"$(date -Is)\" > \"$lease_expired_path\"",
        "    exit 0",
        "  fi",
        "done",
    ]


def write_holder_script(args: Any, alloc_root: str | Path) -> tuple[Path, Path, Path]:
    """Write the long-lived batch script that keeps one allocation open."""
    alloc_root_path = Path(alloc_root).expanduser().resolve()
    script_path = alloc_root_path / "allocation_job.sh"
    slurm_log_path = alloc_root_path / "allocation-%j.out"
    heartbeat_path = alloc_root_path / "notebook-heartbeat.txt"
    alloc_root_path.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text("")
    script_path.write_text("\n".join(holder_script_lines(args, alloc_root_path)))
    script_path.chmod(0o755)
    return script_path, slurm_log_path, heartbeat_path


def submit_batch(
    script_path: str | Path,
    *,
    run_command_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> str:
    """Submit one generated holder script and return the parsed Slurm job id."""
    run = run_command_fn or _default_run_command
    completed = run(["sbatch", "--parsable", str(script_path)])
    if completed.returncode != 0:
        raise RuntimeError(
            "sbatch failed:\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                completed.stdout,
                completed.stderr,
            )
        )
    job_id = (completed.stdout or "").strip().split(";", 1)[0].strip()
    if not job_id:
        raise RuntimeError(f"Could not parse Slurm job id from sbatch output: {completed.stdout!r}")
    return job_id


def allocation_payload(
    *,
    job_id: str,
    name: str,
    allocation_root: str | Path,
    batch_script: str | Path,
    heartbeat_path: str | Path,
    heartbeat_timeout_s: int,
    slurm_log_pattern: str | Path,
) -> dict[str, Any]:
    """Return the JSON payload describing one reusable allocation."""
    return {
        "job_id": str(job_id),
        "name": str(name),
        "allocation_root": str(Path(allocation_root).expanduser().resolve()),
        "batch_script": str(Path(batch_script).expanduser().resolve()),
        "heartbeat_path": str(Path(heartbeat_path).expanduser().resolve()),
        "heartbeat_timeout_s": max(int(heartbeat_timeout_s), 0),
        "slurm_log_pattern": str(slurm_log_pattern),
    }


def load_allocation_payload(path: str | Path) -> dict[str, Any]:
    """Load one allocation metadata JSON file."""
    return json.loads(Path(path).read_text())


def determine_stale_reason(payload: dict[str, Any], *, default_timeout_s: int, now_s: float) -> str:
    """Return the stale-reason label for one allocation, or an empty string."""
    heartbeat_path = str(payload.get("heartbeat_path") or "").strip()
    try:
        timeout_s = int(payload.get("heartbeat_timeout_s") or default_timeout_s)
    except Exception:
        timeout_s = default_timeout_s

    if not heartbeat_path:
        return "legacy_no_heartbeat"

    heartbeat = Path(heartbeat_path)
    if not heartbeat.exists():
        return "missing_heartbeat"

    if timeout_s > 0 and now_s - heartbeat.stat().st_mtime > timeout_s:
        return "expired_heartbeat"
    return ""


def cancel_job(
    job_id: str,
    *,
    run_command_fn: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Request cancellation for one Slurm job id."""
    run = run_command_fn or _default_run_command
    return run(["scancel", str(job_id)])


def stale_allocation_actions(
    root: str | Path,
    *,
    default_timeout_s: int = 120,
    now_s: float | None = None,
    cancel_job_fn: Callable[[str], subprocess.CompletedProcess[str]] | None = None,
) -> list[dict[str, Any]]:
    """Scan one allocation root and return JSON-ready stale-allocation actions."""
    root_path = Path(root).expanduser()
    current_time = time.time() if now_s is None else float(now_s)
    actions: list[dict[str, Any]] = []
    cancel = cancel_job_fn or (lambda job_id: cancel_job(job_id))
    if not root_path.exists():
        return actions

    for allocation_json in sorted(root_path.glob("*/allocation.json")):
        try:
            payload = load_allocation_payload(allocation_json)
        except Exception as exc:
            actions.append(
                {
                    "allocation_json": str(allocation_json),
                    "action": "skip",
                    "reason": "invalid_json",
                    "error": str(exc),
                }
            )
            continue
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            continue
        reason = determine_stale_reason(
            payload,
            default_timeout_s=max(int(default_timeout_s), 0),
            now_s=current_time,
        )
        if not reason:
            continue
        completed = cancel(job_id)
        actions.append(
            {
                "job_id": job_id,
                "action": "cancel_requested",
                "reason": reason,
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "").strip(),
            }
        )
    return actions


def _default_run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
