"""Cancel stale notebook-managed Slurm allocations and emit a JSON summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_allocation_payload(path: Path) -> dict[str, Any]:
    """Load one allocation metadata JSON file."""
    return json.loads(path.read_text())


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


def cancel_job(job_id: str) -> subprocess.CompletedProcess[str]:
    """Request cancellation for one Slurm job id."""
    return subprocess.run(
        ["scancel", str(job_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )


def main() -> None:
    """Scan one allocation root and print JSON cancellation actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--default-timeout-s", type=int, default=120)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    now_s = time.time()
    actions: list[dict[str, Any]] = []
    if root.exists():
        for allocation_json in sorted(root.glob("*/allocation.json")):
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
                default_timeout_s=max(int(args.default_timeout_s), 0),
                now_s=now_s,
            )
            if not reason:
                continue
            completed = cancel_job(job_id)
            actions.append(
                {
                    "job_id": job_id,
                    "action": "cancel_requested",
                    "reason": reason,
                    "returncode": completed.returncode,
                    "stderr": (completed.stderr or "").strip(),
                }
            )

    print(json.dumps(actions, sort_keys=True))


if __name__ == "__main__":
    main()
