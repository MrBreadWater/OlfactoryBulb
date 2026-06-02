"""Reusable notebook-facing remote single-run workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteRunWorkflowHooks:
    """Hook bundle for one notebook-facing remote single-run workflow."""

    prepare_remote_session_fn: Callable[..., Any]
    refresh_submission_payload_fn: Callable[..., tuple[list[str], dict[str, Any], str]]
    upload_runtime_payload_fn: Callable[[dict[str, Any]], dict[str, Any] | None]
    submit_remote_job_fn: Callable[..., Any]
    monitor_remote_job_fn: Callable[..., Any]
    build_final_sync_plan_fn: Callable[[dict[str, Any], dict[str, Any] | None], tuple[tuple[str, ...] | None, tuple[str, ...]]]
    finalize_remote_artifacts_fn: Callable[..., Any]
    write_run_info_fn: Callable[..., Any]
    summarize_submit_response_fn: Callable[[dict[str, Any]], Any]
    summarize_status_fn: Callable[[dict[str, Any] | None], Any]
    timing_summary_text_fn: Callable[[dict[str, float]], str]
    build_return_value_fn: Callable[..., Any]
    shell_join_fn: Callable[[list[str]], str]
    progress_write: Callable[[str], None]


def execute_remote_run_workflow(
    config: dict[str, Any],
    *,
    record_config: dict[str, Any] | None = None,
    label: str,
    timestamp: str,
    local_result_dir: str | Path,
    remote_repo_root: PurePosixPath,
    remote_git_ref: str | None,
    remote_command: list[str],
    remote_metadata: dict[str, Any],
    submit_shell: str,
    hooks: RemoteRunWorkflowHooks,
    ) -> Any:
    """Execute one remote single-run notebook workflow and return its final run record."""
    local_result_dir = Path(local_result_dir)
    record_config = dict(config) if record_config is None else dict(record_config)
    session = hooks.prepare_remote_session_fn(
        config,
        remote_repo_root=remote_repo_root,
        remote_git_ref=remote_git_ref,
        remote_metadata=remote_metadata,
    )
    effective_config = session.effective_config
    remote_metadata = dict(session.remote_metadata)
    notebook_timings = session.notebook_timings
    preflight_completed = session.preflight_completed
    remote_helper_dir = session.remote_helper_dir
    allocation_heartbeat_path = session.allocation_heartbeat_path
    command = list(remote_command)
    submit_command = submit_shell

    if preflight_completed.returncode != 0:
        local_result_dir.mkdir(parents=True, exist_ok=True)
        completed = SimpleNamespace(
            returncode=preflight_completed.returncode,
            stdout=preflight_completed.stdout or "",
            stderr=preflight_completed.stderr or "",
        )
        hooks.write_run_info_fn(
            local_result_dir,
            config=config,
            label=label,
            timestamp=timestamp,
            command=command,
            env={},
            completed=completed,
            extra_payload={"remote": remote_metadata},
        )
        raise RuntimeError(
            "Remote Sol preflight failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Stdout:\n{preflight_completed.stdout}\n\n"
            f"Stderr:\n{preflight_completed.stderr}"
        )

    if remote_helper_dir is not None or effective_config.get("slurm_allocation_job_id"):
        command, refreshed_remote_metadata, submit_command = hooks.refresh_submission_payload_fn(
            effective_config,
            remote_helper_dir=remote_helper_dir,
        )
        refreshed_remote_metadata.update(remote_metadata)
        remote_metadata = refreshed_remote_metadata

    upload_metadata = hooks.upload_runtime_payload_fn(effective_config) or {}
    remote_metadata.update(upload_metadata)

    hooks.progress_write("[Sol remote] Submitting Slurm job...")
    submit_result = hooks.submit_remote_job_fn(
        effective_config,
        submit_shell=submit_command,
        local_output_dir=local_result_dir,
    )
    submit_completed = submit_result.completed
    if submit_completed.returncode != 0:
        completed = SimpleNamespace(
            returncode=submit_completed.returncode,
            stdout=submit_completed.stdout or "",
            stderr=submit_completed.stderr or "",
        )
        hooks.write_run_info_fn(
            local_result_dir,
            config=effective_config,
            label=label,
            timestamp=timestamp,
            command=command,
            env={},
            completed=completed,
            extra_payload={"remote": remote_metadata},
        )
        raise RuntimeError(
            "Remote Sol submission failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Submit stderr:\n{submit_completed.stderr}"
        )

    if submit_result.submission is None:
        raise RuntimeError(
            "Remote Sol submission did not return valid JSON.\n"
            f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
        ) from submit_result.json_error
    submission = submit_result.submission

    remote_result_dir = PurePosixPath(str(submission["result_dir"]))
    remote_job_heartbeat_path = submit_result.job_heartbeat_path
    remote_metadata["job_heartbeat_path"] = remote_job_heartbeat_path
    remote_metadata["heartbeat_timeout_s"] = submit_result.heartbeat_timeout_s
    hooks.progress_write(f"[Sol remote] Submitted job {submission['job_id']}.")

    monitor_result = hooks.monitor_remote_job_fn(
        effective_config=effective_config,
        submission=submission,
        remote_job_heartbeat_path=remote_job_heartbeat_path,
        allocation_heartbeat_path=allocation_heartbeat_path,
        remote_repo_root=remote_repo_root,
        remote_result_dir=remote_result_dir,
        remote_helper_dir=remote_helper_dir,
        notebook_timings=notebook_timings,
        local_result_dir=local_result_dir,
    )
    final_status = monitor_result.final_status
    poll_transcript = monitor_result.poll_transcript

    include_files, deferred_remote_artifacts = hooks.build_final_sync_plan_fn(
        effective_config,
        final_status,
    )
    artifact_result = hooks.finalize_remote_artifacts_fn(
        effective_config,
        final_status=final_status,
        local_result_dir=local_result_dir,
        remote_result_dir=remote_result_dir,
        wrapper_dir=submission.get("wrapper_dir"),
        label=label,
        timestamp=timestamp,
        notebook_timings=notebook_timings,
        poll_transcript=poll_transcript,
        include_files=include_files,
        deferred_remote_artifacts=deferred_remote_artifacts,
    )
    final_status = artifact_result.final_status
    completed = SimpleNamespace(
        returncode=artifact_result.returncode,
        stdout=artifact_result.stdout_text,
        stderr=artifact_result.stderr_text,
    )
    summary = artifact_result.summary
    compact_poll_events = artifact_result.compact_poll_events
    poll_events_path = artifact_result.poll_events_path
    artifact_sizes = artifact_result.artifact_sizes
    remote_metadata["deferred_remote_artifacts"] = list(artifact_result.deferred_remote_artifacts)
    remote_metadata["artifact_sizes"] = artifact_sizes
    remote_metadata["notebook_timing_seconds"] = notebook_timings

    hooks.write_run_info_fn(
        local_result_dir,
        config=effective_config,
        label=label,
        timestamp=timestamp,
        command=command,
        env={},
        completed=completed,
        summary=summary,
        extra_payload={
            "remote": {
                **remote_metadata,
                "job_id": submission.get("job_id"),
                "remote_result_dir": str(remote_result_dir),
                "submit_response": hooks.summarize_submit_response_fn(submission),
                "final_status": hooks.summarize_status_fn(final_status),
                "sync_warning": artifact_result.sync_warning,
                "poll_sample_count": len(poll_transcript),
                "poll_event_count": len(compact_poll_events),
                "poll_events_file": poll_events_path.name if poll_events_path is not None else None,
                "resolved_git_ref": artifact_result.remote_git_ref or remote_metadata.get("remote_git_ref"),
                "resolved_git_commit": artifact_result.remote_git_commit,
                "artifact_sizes": artifact_sizes,
                "notebook_timing_seconds": notebook_timings,
            }
        },
    )

    timing_summary = hooks.timing_summary_text_fn(notebook_timings)
    if timing_summary:
        hooks.progress_write(f"[OBGPU load] Notebook pipeline timings: {timing_summary}")

    if artifact_result.returncode != 0:
        stderr_tail = artifact_result.stderr_text.strip()[-4000:]
        stdout_tail = artifact_result.stdout_text.strip()[-2000:]
        bootstrap_tail = artifact_result.bootstrap_text.strip()[-4000:]
        slurm_tail = artifact_result.slurm_text.strip()[-4000:]
        remote_listing_tail = artifact_result.remote_listing_text.strip()[-4000:]
        raise RuntimeError(
            "Remote Sol simulation failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Command: {hooks.shell_join_fn(command)}\n"
            f"Stdout tail:\n{stdout_tail}\n\n"
            f"Stderr tail:\n{stderr_tail}\n\n"
            f"Bootstrap tail:\n{bootstrap_tail}\n\n"
            f"Slurm tail:\n{slurm_tail}\n\n"
            f"Remote files:\n{remote_listing_tail}"
        )

    if summary is None:
        raise FileNotFoundError(f"Expected synced benchmark summary at {local_result_dir / 'summary.json'}")

    return hooks.build_return_value_fn(
        label=label,
        timestamp=timestamp,
        result_dir=local_result_dir,
        summary=summary,
        config=record_config,
        effective_config=effective_config,
        command=command,
        completed=completed,
    )
