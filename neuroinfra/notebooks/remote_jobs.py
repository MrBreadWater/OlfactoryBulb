"""Reusable notebook-facing remote job session and submission helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteJobSessionHooks:
    """Hook bundle for one remote notebook job session lifecycle."""

    ensure_remote_git_ref_available_fn: Callable[..., None]
    run_remote_preflight_fn: Callable[..., tuple[subprocess.CompletedProcess[str], bool]]
    ensure_remote_helper_cache_fn: Callable[[dict[str, Any]], PurePosixPath | None]
    helper_cache_hit_fn: Callable[[dict[str, Any]], bool]
    cleanup_stale_allocations_fn: Callable[..., list[Any]]
    ensure_cached_remote_allocation_fn: Callable[..., dict[str, Any]]
    record_timing_fn: Callable[[str, float], None]
    progress_write: Callable[[str], None]
    perf_counter_fn: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class RemoteJobSession:
    """Prepared state shared by remote run and remote sweep notebook workflows."""

    effective_config: dict[str, Any]
    remote_metadata: dict[str, Any]
    notebook_timings: dict[str, float]
    preflight_completed: subprocess.CompletedProcess[str]
    remote_helper_dir: PurePosixPath | None
    allocation_info: dict[str, Any]
    allocation_heartbeat_path: str | None


@dataclass(frozen=True)
class RemoteJobSubmitHooks:
    """Hook bundle for one remote notebook job submission protocol."""

    run_ssh_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]]
    heartbeat_timeout_s_fn: Callable[[dict[str, Any]], int | float]
    record_timing_fn: Callable[[str, float], None]
    perf_counter_fn: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class RemoteJobSubmission:
    """Captured submission artifacts plus parsed JSON payload when available."""

    completed: subprocess.CompletedProcess[str]
    submission: dict[str, Any] | None
    json_error: Exception | None
    job_heartbeat_path: str | None
    heartbeat_timeout_s: int | float | None


def prepare_remote_job_session(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    remote_git_ref: str | None,
    remote_metadata: dict[str, Any],
    preflight_message: str,
    hooks: RemoteJobSessionHooks,
    notebook_timings: dict[str, float] | None = None,
) -> RemoteJobSession:
    """Run the shared notebook-side remote session lifecycle for one job."""
    effective_config = dict(config)
    session_metadata = dict(remote_metadata)
    session_timings = notebook_timings if notebook_timings is not None else {}
    remote_helper_dir: PurePosixPath | None = None
    allocation_info: dict[str, Any] = {}
    allocation_heartbeat_path = None

    started = hooks.perf_counter_fn()
    hooks.ensure_remote_git_ref_available_fn(
        effective_config,
        remote_repo_root=remote_repo_root,
        remote_git_ref=remote_git_ref,
    )
    hooks.record_timing_fn("git_publish_s", started)

    hooks.progress_write(preflight_message)
    started = hooks.perf_counter_fn()
    preflight_completed, preflight_cached = hooks.run_remote_preflight_fn(
        effective_config,
        remote_repo_root=remote_repo_root,
    )
    hooks.record_timing_fn("preflight_s", started)
    session_metadata["preflight_cached"] = bool(preflight_cached)
    if preflight_completed.returncode != 0:
        return RemoteJobSession(
            effective_config=effective_config,
            remote_metadata=session_metadata,
            notebook_timings=session_timings,
            preflight_completed=preflight_completed,
            remote_helper_dir=None,
            allocation_info={},
            allocation_heartbeat_path=None,
        )

    started = hooks.perf_counter_fn()
    remote_helper_dir = hooks.ensure_remote_helper_cache_fn(effective_config)
    hooks.record_timing_fn("helper_cache_s", started)
    if remote_helper_dir is not None:
        session_metadata["remote_helper_dir"] = remote_helper_dir.as_posix()
        session_metadata["remote_helper_cache_hit"] = bool(hooks.helper_cache_hit_fn(effective_config))

    started = hooks.perf_counter_fn()
    cleanup_actions = hooks.cleanup_stale_allocations_fn(
        effective_config,
        remote_helper_dir=remote_helper_dir,
    )
    hooks.record_timing_fn("allocation_cleanup_s", started)
    session_metadata["stale_allocation_cleanup_count"] = len(cleanup_actions)

    started = hooks.perf_counter_fn()
    allocation_info = hooks.ensure_cached_remote_allocation_fn(
        effective_config,
        remote_helper_dir=remote_helper_dir,
    )
    hooks.record_timing_fn("allocation_wait_s", started)
    if allocation_info.get("job_id") not in (None, ""):
        effective_config["slurm_allocation_job_id"] = str(allocation_info["job_id"])
        allocation_heartbeat_path = allocation_info.get("heartbeat_path")
        session_metadata["auto_reused_allocation"] = bool(
            effective_config.get("slurm_reuse_allocation", False)
            and not allocation_info.get("manual", False)
        )
        session_metadata["allocation_state"] = allocation_info.get("state", "")
        session_metadata["allocation_reason"] = allocation_info.get("reason", "")
        session_metadata["allocation_location"] = allocation_info.get("location", "")
        session_metadata["allocation_heartbeat_path"] = allocation_heartbeat_path

    return RemoteJobSession(
        effective_config=effective_config,
        remote_metadata=session_metadata,
        notebook_timings=session_timings,
        preflight_completed=preflight_completed,
        remote_helper_dir=remote_helper_dir,
        allocation_info=allocation_info,
        allocation_heartbeat_path=allocation_heartbeat_path,
    )


def submit_remote_json_job(
    config: dict[str, Any],
    *,
    submit_shell: str,
    local_output_dir: str | Path,
    hooks: RemoteJobSubmitHooks,
) -> RemoteJobSubmission:
    """Execute one remote submission shell command and persist its captures."""
    started = hooks.perf_counter_fn()
    completed = hooks.run_ssh_shell_fn(config, submit_shell)
    hooks.record_timing_fn("submit_s", started)

    local_output_dir = Path(local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    (local_output_dir / "submit_stdout.txt").write_text(completed.stdout or "")
    (local_output_dir / "submit_stderr.txt").write_text(completed.stderr or "")

    if completed.returncode != 0:
        return RemoteJobSubmission(
            completed=completed,
            submission=None,
            json_error=None,
            job_heartbeat_path=None,
            heartbeat_timeout_s=None,
        )

    try:
        payload = json.loads((completed.stdout or "").strip())
        if not isinstance(payload, dict):
            raise TypeError(f"expected remote submission JSON object, got {type(payload).__name__}")
    except Exception as exc:  # pragma: no cover - exercised via focused tests
        return RemoteJobSubmission(
            completed=completed,
            submission=None,
            json_error=exc,
            job_heartbeat_path=None,
            heartbeat_timeout_s=None,
        )

    return RemoteJobSubmission(
        completed=completed,
        submission=payload,
        json_error=None,
        job_heartbeat_path=payload.get("heartbeat_path"),
        heartbeat_timeout_s=payload.get(
            "heartbeat_timeout_s",
            hooks.heartbeat_timeout_s_fn(config),
        ),
    )
