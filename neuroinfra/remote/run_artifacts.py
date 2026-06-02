"""Reusable remote single-run final sync and artifact collection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteRunArtifactHooks:
    """Callbacks injected by the notebook-facing caller for remote run finalization."""

    sync_remote_result_dir_resilient_fn: Callable[..., subprocess.CompletedProcess[str]]
    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]]
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]]
    build_remote_result_listing_command_fn: Callable[[PurePosixPath], str]
    local_result_dir_has_loadable_payload_fn: Callable[[Path], bool]
    local_result_dir_has_diagnostics_fn: Callable[[Path], bool]
    standard_result_artifact_sizes_fn: Callable[[Path], dict[str, int]]
    synthesize_partial_sync_summary_fn: Callable[..., dict[str, Any]]
    compact_remote_poll_events_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    read_json_if_present_fn: Callable[[str | Path], dict[str, Any] | None]
    progress_write: Callable[[str], None]
    record_timing_fn: Callable[[str, float], None]
    sleep_fn: Callable[[float], None] = time.sleep
    perf_counter_fn: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class RemoteRunArtifactResult:
    """Collected local artifacts and derived metadata for one remote run."""

    final_status: dict[str, Any] | None
    sync_completed: subprocess.CompletedProcess[str]
    sync_warning: str | None
    stdout_text: str
    stderr_text: str
    bootstrap_text: str
    slurm_text: str
    remote_listing_text: str
    remote_git_commit: str | None
    remote_git_ref: str | None
    returncode: int
    summary: dict[str, Any] | None
    artifact_sizes: dict[str, int]
    deferred_remote_artifacts: tuple[str, ...]
    compact_poll_events: list[dict[str, Any]]
    poll_events_path: Path | None


def _read_text_if_present(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def _latest_slurm_log_text(result_dir: Path) -> str:
    slurm_logs = sorted(result_dir.glob("slurm-*.out"))
    if slurm_logs:
        return slurm_logs[-1].read_text()
    return ""


def finalize_remote_run_artifacts(
    config: dict[str, Any],
    *,
    final_status: dict[str, Any] | None,
    local_result_dir: str | Path,
    remote_result_dir: PurePosixPath,
    wrapper_dir: str | PurePosixPath | None,
    label: str,
    timestamp: str,
    notebook_timings: dict[str, float],
    poll_transcript: list[dict[str, Any]],
    include_files: tuple[str, ...] | None,
    deferred_remote_artifacts: tuple[str, ...] | list[str],
    hooks: RemoteRunArtifactHooks,
) -> RemoteRunArtifactResult:
    """Sync the final payload for one remote run and collect local diagnostics."""
    local_result_dir = Path(local_result_dir)
    local_result_dir.mkdir(parents=True, exist_ok=True)
    deferred_remote_artifacts = tuple(str(name) for name in deferred_remote_artifacts)

    sync_started = hooks.perf_counter_fn()
    sync_completed = hooks.sync_remote_result_dir_resilient_fn(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
        expected_files=("summary.json",),
        include_files=include_files,
        wrapper_dir=wrapper_dir,
    )
    hooks.record_timing_fn("sync_s", sync_started)
    (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
    (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")

    sync_warning = None
    updated_final_status = dict(final_status) if isinstance(final_status, dict) else final_status
    if sync_completed.returncode != 0:
        if hooks.local_result_dir_has_loadable_payload_fn(local_result_dir):
            sync_warning = (
                "Remote Sol result sync reported an error, but standard payload files were already present locally. "
                "Proceeding with the partial local copy.\n"
                f"{sync_completed.stderr}"
            )
            hooks.progress_write(f"[OBGPU load] {sync_warning}")
        elif hooks.local_result_dir_has_diagnostics_fn(local_result_dir):
            sync_warning = (
                "Remote Sol result payload sync failed, but diagnostic logs were synced. "
                "Proceeding to build a failure report instead of aborting at sync.\n"
                f"{sync_completed.stderr}"
            )
            hooks.progress_write(f"[OBGPU load] {sync_warning}")
            if updated_final_status is not None:
                updated_final_status = dict(updated_final_status)
                updated_final_status["ok"] = False
                updated_final_status["sync_failed"] = True
                updated_final_status["sync_stderr"] = sync_completed.stderr or ""
        else:
            raise RuntimeError(
                "Remote Sol result sync failed.\n"
                f"Result dir: {local_result_dir}\n"
                f"sync stderr:\n{sync_completed.stderr}"
            )
    hooks.progress_write(f"[OBGPU load] Remote sync finished: {local_result_dir}")

    stdout_text = _read_text_if_present(local_result_dir / "stdout.txt")
    stderr_text = _read_text_if_present(local_result_dir / "stderr.txt")
    bootstrap_text = _read_text_if_present(local_result_dir / "bootstrap.log")
    slurm_text = _latest_slurm_log_text(local_result_dir)

    if updated_final_status and not updated_final_status.get("ok") and not any(
        (stdout_text, stderr_text, bootstrap_text, slurm_text)
    ):
        hooks.sleep_fn(3.0)
        retry_sync_started = hooks.perf_counter_fn()
        sync_completed = hooks.sync_remote_result_dir_fn(
            config,
            remote_result_dir=remote_result_dir,
            local_result_dir=local_result_dir,
        )
        hooks.record_timing_fn("retry_sync_s", retry_sync_started)
        (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
        (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
        if sync_completed.returncode == 0:
            stdout_text = _read_text_if_present(local_result_dir / "stdout.txt")
            stderr_text = _read_text_if_present(local_result_dir / "stderr.txt")
            bootstrap_text = _read_text_if_present(local_result_dir / "bootstrap.log")
            slurm_text = _latest_slurm_log_text(local_result_dir)

    remote_listing_text = ""
    if updated_final_status and not updated_final_status.get("ok") and not any(
        (stdout_text, stderr_text, bootstrap_text, slurm_text)
    ):
        listing_completed = hooks.run_paramiko_shell_fn(
            config,
            hooks.build_remote_result_listing_command_fn(remote_result_dir),
        )
        remote_listing_text = (listing_completed.stdout or "").strip()

    remote_git_commit = None
    git_commit_path = local_result_dir / "git_commit.txt"
    if git_commit_path.exists():
        remote_git_commit = git_commit_path.read_text().strip()

    remote_git_ref = None
    git_ref_path = local_result_dir / "git_ref.txt"
    if git_ref_path.exists():
        remote_git_ref = git_ref_path.read_text().strip()

    returncode = 0 if updated_final_status and updated_final_status.get("ok") else 1

    summary_path = local_result_dir / "summary.json"
    summary = hooks.read_json_if_present_fn(summary_path)
    if summary is None and sync_warning is not None and hooks.local_result_dir_has_loadable_payload_fn(local_result_dir):
        summary = hooks.synthesize_partial_sync_summary_fn(
            local_result_dir,
            label=label,
            timestamp=timestamp,
            config=config,
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    compact_poll_events = hooks.compact_remote_poll_events_fn(poll_transcript)
    poll_events_path = None
    if compact_poll_events:
        poll_events_path = local_result_dir / "remote_poll_events.json"
        poll_events_path.write_text(json.dumps(compact_poll_events, indent=2, sort_keys=True))

    artifact_sizes = hooks.standard_result_artifact_sizes_fn(local_result_dir)

    return RemoteRunArtifactResult(
        final_status=updated_final_status,
        sync_completed=sync_completed,
        sync_warning=sync_warning,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        bootstrap_text=bootstrap_text,
        slurm_text=slurm_text,
        remote_listing_text=remote_listing_text,
        remote_git_commit=remote_git_commit,
        remote_git_ref=remote_git_ref,
        returncode=returncode,
        summary=summary,
        artifact_sizes=artifact_sizes,
        deferred_remote_artifacts=deferred_remote_artifacts,
        compact_poll_events=compact_poll_events,
        poll_events_path=poll_events_path,
    )
