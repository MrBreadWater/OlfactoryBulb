"""Concrete olfactory-bulb remote single-run workflow adapters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from neuroinfra.notebooks.remote_jobs import (
    prepare_remote_job_session,
    submit_remote_json_job,
)
from neuroinfra.notebooks.remote_runs import RemoteRunWorkflowHooks
from neuroinfra.remote import (
    finalize_remote_run_artifacts,
    monitor_remote_run,
)


@dataclass(frozen=True)
class RemoteRunPayloadHooks:
    """Hooks for building one concrete olfactory-bulb remote run payload."""

    build_run_command_fn: Callable[..., list[str]]
    build_remote_submit_command_fn: Callable[..., str]
    require_remote_host_fn: Callable[[dict[str, Any]], str]
    default_remote_mpi_exec_fn: Callable[[], str]


@dataclass(frozen=True)
class NotebookRemoteRunWorkflowBuilderHooks:
    """Hooks for constructing the concrete olfactory-bulb remote run workflow."""

    remote_job_session_hooks_fn: Callable[[dict[str, float]], Any]
    remote_job_submit_hooks_fn: Callable[[dict[str, float]], Any]
    remote_run_monitor_hooks_fn: Callable[..., Any]
    remote_run_artifact_hooks_fn: Callable[[dict[str, float]], Any]
    build_remote_run_payload_fn: Callable[..., tuple[list[str], dict[str, Any], str]]
    upload_remote_text_file_fn: Callable[..., Any]
    json_ready_fn: Callable[[Any], Any]
    remote_fast_sync_files_fn: Callable[[dict[str, Any]], tuple[str, ...]]
    preferred_soma_trace_artifact_name_fn: Callable[[], str]
    write_run_info_fn: Callable[..., Any]
    summarize_submit_response_fn: Callable[[dict[str, Any]], Any]
    summarize_status_fn: Callable[[dict[str, Any] | None], Any]
    timing_summary_text_fn: Callable[[dict[str, float]], str]
    build_return_value_fn: Callable[..., Any]
    shell_join_fn: Callable[[list[str]], str]
    progress_write: Callable[[str], None]
    record_timing_fn: Callable[[dict[str, float], str, float], Any]
    perf_counter_fn: Callable[[], float]


def build_remote_run_payload(
    hooks: RemoteRunPayloadHooks,
    config: dict[str, Any],
    *,
    label: str,
    remote_repo_root: PurePosixPath,
    remote_results_root: PurePosixPath,
    remote_git_ref: str | None,
    remote_helper_dir: PurePosixPath | None = None,
    overrides_file: str | PurePosixPath | None = None,
    param_overrides: dict[str, Any] | None = None,
    input_spec_file: str | Path | None = None,
) -> tuple[list[str], dict[str, Any], str]:
    """Prepare the concrete olfactory-bulb remote command and metadata payload."""
    remote_mpi_exec = config.get("remote_mpi_exec") or hooks.default_remote_mpi_exec_fn()
    allocation_job_id = config.get("slurm_allocation_job_id")
    include_mpi_launcher = True
    if allocation_job_id not in (None, ""):
        include_mpi_launcher = int(config.get("nranks", 1) or 1) != 1
    remote_command = hooks.build_run_command_fn(
        config,
        label,
        repo_root=remote_repo_root,
        results_base=remote_results_root,
        mpi_exec=str(remote_mpi_exec),
        include_mpi_launcher=include_mpi_launcher,
        overrides_file=overrides_file,
        param_overrides=param_overrides,
        input_spec_file=input_spec_file,
    )
    submit_command = hooks.build_remote_submit_command_fn(
        config,
        label=label,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        benchmark_command=remote_command,
        remote_mpi_exec=str(remote_mpi_exec),
        remote_git_ref=remote_git_ref,
        step_ntasks=max(
            int(config.get("slurm_step_ntasks", 1) or 1),
            int(config.get("nranks", 1) or 1),
        ),
        remote_helper_dir=remote_helper_dir,
    )
    return (
        remote_command,
        {
            "runner_backend": str(config.get("runner_backend", "slurm_remote")),
            "remote_host": hooks.require_remote_host_fn(config),
            "remote_repo_root": remote_repo_root.as_posix(),
            "remote_results_root": remote_results_root.as_posix(),
            "remote_mpi_exec": str(remote_mpi_exec),
            "remote_repo_mode": str(config.get("remote_repo_mode", "shared")),
            "remote_git_ref": remote_git_ref,
            "remote_git_fetch": bool(config.get("remote_git_fetch", False)),
            "remote_git_remote": str(config.get("remote_git_remote", "origin")),
            "slurm_allocation_job_id": None if allocation_job_id in (None, "") else str(allocation_job_id),
        },
        submit_command,
    )


def build_remote_run_workflow_hooks(
    hooks: NotebookRemoteRunWorkflowBuilderHooks,
    *,
    label: str,
    timestamp: str,
    remote_repo_root: PurePosixPath,
    remote_results_root: PurePosixPath,
    remote_git_ref: str | None,
    remote_overrides_path: PurePosixPath,
    param_overrides: dict[str, Any],
    input_spec_file: str | None,
) -> RemoteRunWorkflowHooks:
    """Build the notebook-facing olfactory-bulb remote single-run workflow hooks."""
    workflow_state: dict[str, dict[str, float] | None] = {"notebook_timings": None}

    def prepare_remote_session(
        config: dict[str, Any],
        *,
        remote_repo_root: PurePosixPath,
        remote_git_ref: str | None,
        remote_metadata: dict[str, Any],
    ) -> Any:
        notebook_timings: dict[str, float] = {}
        workflow_state["notebook_timings"] = notebook_timings
        return prepare_remote_job_session(
            config,
            remote_repo_root=remote_repo_root,
            remote_git_ref=remote_git_ref,
            remote_metadata=remote_metadata,
            preflight_message="[Sol remote] Running remote preflight checks...",
            hooks=hooks.remote_job_session_hooks_fn(notebook_timings),
            notebook_timings=notebook_timings,
        )

    def refresh_submission_payload(
        config: dict[str, Any],
        *,
        remote_helper_dir: PurePosixPath | None,
    ) -> tuple[list[str], dict[str, Any], str]:
        return hooks.build_remote_run_payload_fn(
            config,
            label=label,
            remote_repo_root=remote_repo_root,
            remote_results_root=remote_results_root,
            remote_git_ref=remote_git_ref,
            remote_helper_dir=remote_helper_dir,
            overrides_file=remote_overrides_path,
            param_overrides=param_overrides,
            input_spec_file=input_spec_file,
        )

    def upload_runtime_payload(config: dict[str, Any]) -> dict[str, Any]:
        hooks.progress_write("[Sol remote] Uploading benchmark overrides file...")
        notebook_timings = workflow_state["notebook_timings"]
        if notebook_timings is None:
            raise RuntimeError("remote run workflow timings were not initialized before runtime upload")
        started = hooks.perf_counter_fn()
        hooks.upload_remote_text_file_fn(
            config,
            remote_path=remote_overrides_path,
            text=json.dumps(hooks.json_ready_fn(param_overrides), indent=2, sort_keys=True),
        )
        hooks.record_timing_fn(notebook_timings, "overrides_upload_s", started)
        return {"benchmark_overrides_file": remote_overrides_path.as_posix()}

    def submit_remote_job(
        config: dict[str, Any],
        *,
        submit_shell: str,
        local_output_dir: str | Path,
    ) -> Any:
        notebook_timings = workflow_state["notebook_timings"]
        if notebook_timings is None:
            raise RuntimeError("remote run workflow timings were not initialized before submit")
        return submit_remote_json_job(
            config,
            submit_shell=submit_shell,
            local_output_dir=local_output_dir,
            hooks=hooks.remote_job_submit_hooks_fn(notebook_timings),
        )

    def monitor_remote_job(
        *,
        effective_config: dict[str, Any],
        submission: dict[str, Any],
        remote_job_heartbeat_path: str | None,
        allocation_heartbeat_path: str | None,
        remote_repo_root: PurePosixPath,
        remote_result_dir: PurePosixPath,
        remote_helper_dir: PurePosixPath | None,
        notebook_timings: dict[str, float],
        local_result_dir: Path,
    ) -> Any:
        poll_interval_s = max(float(effective_config.get("remote_poll_interval_s", 1.0)), 1.0)
        log_poll_interval_s = max(
            float(effective_config.get("remote_log_poll_interval_s", max(poll_interval_s, 5.0))),
            poll_interval_s,
        )
        live_status = bool(effective_config.get("remote_live_status", True))
        live_logs = bool(effective_config.get("remote_live_logs", True))
        return monitor_remote_run(
            job_id=str(submission["job_id"]),
            poll_interval_s=poll_interval_s,
            log_poll_interval_s=log_poll_interval_s,
            live_status=live_status,
            live_logs=live_logs,
            missing_artifact_retry_limit=3,
            hooks=hooks.remote_run_monitor_hooks_fn(
                effective_config=effective_config,
                remote_job_heartbeat_path=remote_job_heartbeat_path,
                allocation_heartbeat_path=allocation_heartbeat_path,
                remote_repo_root=remote_repo_root,
                remote_result_dir=remote_result_dir,
                remote_helper_dir=remote_helper_dir,
                notebook_timings=notebook_timings,
                submission=submission,
                local_result_dir=local_result_dir,
            ),
        )

    def build_final_sync_plan(
        effective_config: dict[str, Any],
        final_status: dict[str, Any] | None,
    ) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
        include_files: tuple[str, ...] | None = None
        deferred_remote_artifacts: list[str] = []
        if final_status and final_status.get("ok") and bool(effective_config.get("remote_defer_soma_vs_sync", False)):
            include_files = hooks.remote_fast_sync_files_fn(effective_config)
            deferred_remote_artifacts.append(hooks.preferred_soma_trace_artifact_name_fn())
        return include_files, tuple(deferred_remote_artifacts)

    def finalize_remote_artifacts(
        effective_config: dict[str, Any],
        *,
        final_status: dict[str, Any] | None,
        local_result_dir: Path,
        remote_result_dir: PurePosixPath,
        wrapper_dir: str | PurePosixPath | None,
        label: str,
        timestamp: str,
        notebook_timings: dict[str, float],
        poll_transcript: list[dict[str, Any]],
        include_files: tuple[str, ...] | None,
        deferred_remote_artifacts: tuple[str, ...],
    ) -> Any:
        return finalize_remote_run_artifacts(
            effective_config,
            final_status=final_status,
            local_result_dir=local_result_dir,
            remote_result_dir=remote_result_dir,
            wrapper_dir=wrapper_dir,
            label=label,
            timestamp=timestamp,
            notebook_timings=notebook_timings,
            poll_transcript=poll_transcript,
            include_files=include_files,
            deferred_remote_artifacts=deferred_remote_artifacts,
            hooks=hooks.remote_run_artifact_hooks_fn(notebook_timings),
        )

    return RemoteRunWorkflowHooks(
        prepare_remote_session_fn=prepare_remote_session,
        refresh_submission_payload_fn=refresh_submission_payload,
        upload_runtime_payload_fn=upload_runtime_payload,
        submit_remote_job_fn=submit_remote_job,
        monitor_remote_job_fn=monitor_remote_job,
        build_final_sync_plan_fn=build_final_sync_plan,
        finalize_remote_artifacts_fn=finalize_remote_artifacts,
        write_run_info_fn=hooks.write_run_info_fn,
        summarize_submit_response_fn=hooks.summarize_submit_response_fn,
        summarize_status_fn=hooks.summarize_status_fn,
        timing_summary_text_fn=hooks.timing_summary_text_fn,
        build_return_value_fn=hooks.build_return_value_fn,
        shell_join_fn=hooks.shell_join_fn,
        progress_write=hooks.progress_write,
    )
