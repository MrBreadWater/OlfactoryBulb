"""Reusable live monitoring helpers for notebook-managed remote runs."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteRunMonitorHooks:
    """Callbacks injected by the notebook-facing caller for run monitoring."""

    refresh_remote_leases_fn: Callable[..., None]
    poll_status_fn: Callable[..., dict[str, Any]]
    cancel_job_fn: Callable[[], subprocess.CompletedProcess[str]]
    sync_partial_artifacts_fn: Callable[[], subprocess.CompletedProcess[str]]
    remote_status_has_artifacts_fn: Callable[[dict[str, Any]], bool]
    progress_bar_factory_fn: Callable[[int, str], Any]
    filter_live_log_line_fn: Callable[[str, str], str | None]
    progress_write: Callable[[str], None]
    sleep_fn: Callable[[float], None] = time.sleep
    monotonic_fn: Callable[[], float] = time.monotonic
    time_fn: Callable[[], float] = time.time


@dataclass(frozen=True)
class RemoteRunMonitorResult:
    """Terminal status and poll transcript for one remote run."""

    final_status: dict[str, Any] | None
    poll_transcript: list[dict[str, Any]]


def monitor_remote_run(
    *,
    job_id: str,
    poll_interval_s: float,
    log_poll_interval_s: float,
    live_status: bool,
    live_logs: bool,
    missing_artifact_retry_limit: int,
    hooks: RemoteRunMonitorHooks,
) -> RemoteRunMonitorResult:
    """Monitor one remote run until it reaches a terminal state."""
    poll_transcript: list[dict[str, Any]] = []
    final_status: dict[str, Any] | None = None
    missing_artifact_retries = 0
    last_status_signature: tuple[Any, ...] | None = None
    last_live_tails = {
        "bootstrap": "",
        "stdout": "",
        "stderr": "",
        "slurm": "",
    }
    last_live_lines = {
        "bootstrap": None,
        "stdout": None,
        "stderr": None,
        "slurm": None,
    }
    last_live_partials = {
        "bootstrap": "",
        "stdout": "",
        "stderr": "",
        "slurm": "",
    }
    sim_progress_bar = None
    sim_progress_total_ms: int | None = None
    sim_last_progress_ms: int | None = None
    sim_waiting_for_progress_logged = False
    sim_progress_complete = False
    sim_finalizing_logged = False
    next_full_poll_at = hooks.monotonic_fn()
    last_polled_state: tuple[Any, Any, Any] | None = None

    def poll_status_once(
        *,
        refresh_heartbeat: bool = True,
        include_logs: bool = True,
        include_sacct: bool = True,
    ) -> dict[str, Any]:
        if refresh_heartbeat:
            hooks.refresh_remote_leases_fn(warn=False)
        status = hooks.poll_status_fn(
            refresh_heartbeat=False,
            include_logs=include_logs,
            include_sacct=include_sacct,
        )
        poll_transcript.append(status)
        return status

    def emit_live_remote_updates(status: dict[str, Any]) -> None:
        nonlocal last_status_signature
        nonlocal sim_progress_bar
        nonlocal sim_progress_total_ms
        nonlocal sim_last_progress_ms
        nonlocal sim_waiting_for_progress_logged
        nonlocal sim_progress_complete
        nonlocal sim_finalizing_logged

        status_signature = (
            status.get("state"),
            bool(status.get("summary_exists")),
            bool(status.get("stdout_exists")),
            bool(status.get("stderr_exists")),
            bool(status.get("bootstrap_exists")),
            bool(status.get("command_exists")),
            bool(status.get("slurm_log_exists")),
        )
        if live_status and status_signature != last_status_signature:
            state = str(status.get("state", "UNKNOWN"))
            reason = str(status.get("reason") or "").strip()
            location = str(status.get("location") or "").strip()
            flags = []
            if status.get("bootstrap_exists"):
                flags.append("bootstrap")
            if status.get("command_exists"):
                flags.append("command")
            if status.get("stdout_exists"):
                flags.append("stdout")
            if status.get("stderr_exists"):
                flags.append("stderr")
            if status.get("slurm_log_exists"):
                flags.append("slurm")
            if status.get("summary_exists"):
                flags.append("summary")
            flag_text = ", ".join(flags) if flags else "no artifacts yet"
            if state == "PENDING" and reason:
                flag_text = f"{flag_text}; reason={reason}"
            elif location and state not in {"PENDING", "UNKNOWN"}:
                flag_text = f"{flag_text}; where={location}"
            hooks.progress_write(f"[Sol remote] Job {job_id}: {state} ({flag_text})")
            last_status_signature = status_signature

        progress_total_ms = status.get("progress_total_ms")
        progress_current_ms = status.get("progress_current_ms")
        if (
            not sim_progress_complete
            and progress_total_ms not in (None, "", 0)
            and progress_current_ms is not None
        ):
            total_ms = max(int(float(progress_total_ms)), 0)
            current_ms = max(0, min(int(float(progress_current_ms)), total_ms))
            if total_ms > 0:
                if sim_progress_bar is None or sim_progress_total_ms != total_ms:
                    if sim_progress_bar is not None:
                        sim_progress_bar.close()
                    sim_progress_total_ms = total_ms
                    sim_progress_bar = hooks.progress_bar_factory_fn(total_ms, "Sim")
                    sim_waiting_for_progress_logged = False
                sim_progress_bar.update_to(current_ms)
                sim_last_progress_ms = current_ms

        state = str(status.get("state", "UNKNOWN"))
        if (
            state == "RUNNING"
            and sim_progress_bar is None
            and not sim_waiting_for_progress_logged
            and not sim_progress_complete
            and not status.get("summary_exists")
        ):
            hooks.progress_write("[Sol remote] Simulation started; waiting for first progress update...")
            sim_waiting_for_progress_logged = True

        if (
            sim_progress_bar is not None
            and sim_progress_total_ms is not None
            and sim_last_progress_ms is not None
            and sim_last_progress_ms >= sim_progress_total_ms
        ) or status.get("summary_exists"):
            if sim_progress_bar is not None:
                sim_progress_bar.close()
                sim_progress_bar = None
                sim_progress_total_ms = None
            if status.get("summary_exists") and not sim_finalizing_logged:
                hooks.progress_write("[Sol remote] Remote simulation finished; finalizing artifacts...")
                sim_finalizing_logged = True
            sim_progress_complete = True

        if live_logs:
            for kind in ("bootstrap", "stdout", "stderr", "slurm"):
                tail_text = str(status.get(f"{kind}_tail") or "")
                if not tail_text or tail_text == last_live_tails[kind]:
                    continue
                previous = last_live_tails[kind]
                if previous and tail_text.startswith(previous):
                    delta_text = tail_text[len(previous):]
                else:
                    delta_text = tail_text
                delta_text = last_live_partials[kind] + delta_text.replace("\r", "\n")
                if delta_text:
                    segments = delta_text.split("\n")
                    if delta_text.endswith("\n"):
                        last_live_partials[kind] = ""
                    else:
                        last_live_partials[kind] = segments.pop() if segments else delta_text
                    for line in segments:
                        filtered = hooks.filter_live_log_line_fn(kind, line)
                        if filtered is None:
                            continue
                        if filtered == last_live_lines[kind]:
                            continue
                        hooks.progress_write(f"[Sol remote][{kind}] {filtered}")
                        last_live_lines[kind] = filtered
                else:
                    last_live_partials[kind] = ""
                last_live_tails[kind] = tail_text
        if status.get("done") and sim_progress_bar is not None:
            sim_progress_bar.close()
            sim_progress_bar = None
            sim_progress_total_ms = None

    def close_live_progress_bars() -> None:
        nonlocal sim_progress_bar, sim_progress_total_ms
        if sim_progress_bar is not None:
            sim_progress_bar.close()
            sim_progress_bar = None
            sim_progress_total_ms = None

    def cancel_remote_job_and_sync(reason_text: str) -> None:
        nonlocal final_status
        close_live_progress_bars()
        hooks.progress_write(f"[Sol remote] {reason_text}; beginning shutdown for job {job_id}...")
        try:
            cancel_completed = hooks.cancel_job_fn()
            if cancel_completed.returncode != 0 and (cancel_completed.stderr or "").strip():
                hooks.progress_write(f"[Sol remote] scancel stderr: {(cancel_completed.stderr or '').strip()}")
            else:
                hooks.progress_write("[Sol remote] Cancellation requested; waiting for remote cleanup...")
        except Exception as exc:
            hooks.progress_write(f"[Sol remote] Failed to request cancellation: {exc}")

        cancel_deadline = hooks.time_fn() + 30.0
        cancel_confirmed = False
        while hooks.time_fn() < cancel_deadline:
            try:
                status = poll_status_once(
                    refresh_heartbeat=False,
                    include_logs=True,
                    include_sacct=True,
                )
            except Exception as exc:
                hooks.progress_write(f"[Sol remote] Remote shutdown poll failed: {exc}")
                break
            try:
                emit_live_remote_updates(status)
            except Exception as exc:
                hooks.progress_write(f"[Sol remote] Remote shutdown status rendering failed: {exc}")
            if status.get("done"):
                final_status = status
                cancel_confirmed = True
                hooks.progress_write(
                    f"[Sol remote] Job {job_id} reached terminal state {status.get('state', 'UNKNOWN')}."
                )
                break
            hooks.sleep_fn(1.0)

        if not cancel_confirmed:
            hooks.progress_write("[Sol remote] Remote shutdown not yet confirmed; syncing partial artifacts anyway...")
        else:
            hooks.progress_write("[Sol remote] Syncing partial remote artifacts...")
        try:
            sync_completed = hooks.sync_partial_artifacts_fn()
            if sync_completed.returncode == 0:
                hooks.progress_write("[Sol remote] Partial artifacts synced successfully.")
            else:
                hooks.progress_write(
                    f"[Sol remote] Partial artifact sync failed (rc={sync_completed.returncode})."
                )
        except Exception as exc:
            hooks.progress_write(f"[Sol remote] Partial artifact sync failed: {exc}")

    try:
        while True:
            hooks.refresh_remote_leases_fn(warn=True)
            include_logs = hooks.monotonic_fn() >= next_full_poll_at
            include_sacct = include_logs
            status = poll_status_once(
                refresh_heartbeat=False,
                include_logs=include_logs,
                include_sacct=include_sacct,
            )
            state_signature = (
                status.get("state"),
                status.get("reason"),
                status.get("location"),
            )
            if (
                not include_logs
                and (
                    state_signature != last_polled_state
                    or status.get("done")
                    or status.get("summary_exists")
                    or status.get("state") == "UNKNOWN"
                )
            ):
                status = poll_status_once(
                    refresh_heartbeat=False,
                    include_logs=live_logs,
                    include_sacct=True,
                )
                include_logs = True
                state_signature = (
                    status.get("state"),
                    status.get("reason"),
                    status.get("location"),
                )
            if include_logs:
                next_full_poll_at = hooks.monotonic_fn() + log_poll_interval_s
            last_polled_state = state_signature
            emit_live_remote_updates(status)
            if status.get("done"):
                if (
                    not status.get("ok")
                    and not hooks.remote_status_has_artifacts_fn(status)
                    and missing_artifact_retries < max(int(missing_artifact_retry_limit), 0)
                ):
                    missing_artifact_retries += 1
                    hooks.sleep_fn(3.0)
                    continue
                final_status = status
                break
            hooks.sleep_fn(poll_interval_s)
    except KeyboardInterrupt:
        cancel_remote_job_and_sync("Interrupt received")
        raise KeyboardInterrupt(
            f"Interrupted remote Sol run and requested cancellation for job {job_id}."
        )
    except Exception:
        cancel_remote_job_and_sync("Local notebook error while monitoring remote run")
        raise

    return RemoteRunMonitorResult(
        final_status=final_status,
        poll_transcript=poll_transcript,
    )
