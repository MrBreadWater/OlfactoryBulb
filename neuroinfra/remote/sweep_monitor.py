"""Reusable live monitoring helpers for notebook-managed remote sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteSweepMonitorHooks:
    """Callbacks injected by the notebook-facing caller for sweep monitoring."""

    refresh_remote_leases_fn: Callable[..., None]
    poll_status_fn: Callable[..., dict[str, Any]]
    sync_finished_items_fn: Callable[[dict[str, Any]], None]
    cancel_job_fn: Callable[[], subprocess.CompletedProcess[str]]
    synced_count_fn: Callable[[], int]
    progress_write: Callable[[str], None]
    sleep_fn: Callable[[float], None] = time.sleep
    monotonic_fn: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class RemoteSweepMonitorResult:
    """Terminal status for one remote sweep job."""

    final_status: dict[str, Any] | None


def monitor_remote_sweep(
    *,
    job_id: str,
    poll_interval_s: float,
    log_poll_interval_s: float,
    live_status: bool,
    hooks: RemoteSweepMonitorHooks,
) -> RemoteSweepMonitorResult:
    """Monitor one remote sweep job until it reaches a terminal state."""
    final_status: dict[str, Any] | None = None
    last_signature: tuple[Any, ...] | None = None
    next_sacct_poll_at = hooks.monotonic_fn()

    try:
        while True:
            include_sacct = hooks.monotonic_fn() >= next_sacct_poll_at
            status = hooks.poll_status_fn(refresh_heartbeat=True, include_sacct=include_sacct)
            if include_sacct:
                next_sacct_poll_at = hooks.monotonic_fn() + log_poll_interval_s
            if not include_sacct and status.get("state") == "UNKNOWN":
                status = hooks.poll_status_fn(refresh_heartbeat=False, include_sacct=True)
                next_sacct_poll_at = hooks.monotonic_fn() + log_poll_interval_s

            status_signature = (
                status.get("state"),
                status.get("progress_current_ms"),
                status.get("progress_total_ms"),
                bool(status.get("summary_exists")),
            )
            progress_payload = status.get("progress_payload") or {}
            finished_items = progress_payload.get("finished_items") or []
            completed_count = len(progress_payload.get("completed_items") or [])
            failed_count = len(progress_payload.get("failed_items") or [])
            if finished_items:
                completed_count = sum(
                    1
                    for item in finished_items
                    if isinstance(item, dict) and bool(item.get("ok", False))
                )
                failed_count = sum(
                    1
                    for item in finished_items
                    if isinstance(item, dict) and not bool(item.get("ok", False))
                )
            running_count = len(progress_payload.get("running_items") or [])
            pending_count = len(progress_payload.get("pending_labels") or [])
            status_signature = (
                *status_signature,
                completed_count,
                failed_count,
                running_count,
                pending_count,
                int(hooks.synced_count_fn()),
            )

            if live_status and status_signature != last_signature:
                detail = f"{status.get('state', 'UNKNOWN')}"
                current = status.get("progress_current_ms")
                total = status.get("progress_total_ms")
                if current is not None and total not in (None, 0, ""):
                    detail += f" ({int(float(current))}/{int(float(total))} items)"
                elif completed_count or failed_count or running_count or pending_count:
                    detail += (
                        f" ({completed_count} done"
                        f", {failed_count} failed"
                        f", {running_count} running"
                        f", {pending_count} pending"
                        f", {int(hooks.synced_count_fn())} synced)"
                    )
                reason = str(status.get("reason") or "").strip()
                location = str(status.get("location") or "").strip()
                if detail.startswith("PENDING") and reason:
                    detail += f"; reason={reason}"
                elif location and not detail.startswith("PENDING"):
                    detail += f"; where={location}"
                hooks.progress_write(f"[Sol remote] Sweep job {job_id}: {detail}")
                last_signature = status_signature

            hooks.sync_finished_items_fn(status)
            if status.get("done"):
                final_status = status
                break
            hooks.sleep_fn(poll_interval_s)
    except KeyboardInterrupt:
        hooks.progress_write(f"[Sol remote] Interrupt received; cancelling sweep job {job_id}...")
        try:
            hooks.cancel_job_fn()
        finally:
            raise KeyboardInterrupt(
                f"Interrupted remote sweep and requested cancellation for job {job_id}."
            )

    return RemoteSweepMonitorResult(final_status=final_status)
