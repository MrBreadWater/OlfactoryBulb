"""Reusable notebook-facing remote sweep workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteSweepWorkflowHooks:
    """Hook bundle for one notebook-facing remote sweep workflow."""

    prepare_remote_session_fn: Callable[..., Any]
    upload_manifest_fn: Callable[[dict[str, Any]], dict[str, Any] | None]
    build_submit_shell_fn: Callable[[dict[str, Any], PurePosixPath | None], str]
    submit_remote_job_fn: Callable[..., Any]
    monitor_remote_job_fn: Callable[..., Any]
    finalize_remote_artifacts_fn: Callable[..., Any]
    finalize_local_items_fn: Callable[..., tuple[list[dict[str, Any]], list[str], dict[str, str]]]
    persist_sweep_fn: Callable[..., Any]
    merge_sweep_info_payload_fn: Callable[[str | Path, dict[str, Any]], Any]
    summarize_status_fn: Callable[[dict[str, Any] | None], Any]
    timing_summary_text_fn: Callable[[dict[str, float]], str]
    progress_write: Callable[[str], None]


def execute_remote_sweep_workflow(
    config: dict[str, Any],
    *,
    sweep_plan: dict[str, Any],
    sweep_label: str,
    timestamp: str,
    local_sweep_dir: str | Path,
    local_runs_dir: str | Path,
    remote_repo_root: PurePosixPath,
    remote_git_ref: str | None,
    remote_sweeps_root: PurePosixPath,
    remote_sweep_root: PurePosixPath,
    manifest_items: list[dict[str, Any]],
    manifest_json: str,
    max_concurrent: int,
    remote_metadata: dict[str, Any],
    hooks: RemoteSweepWorkflowHooks,
) -> dict[str, Any]:
    """Execute one remote notebook sweep workflow and return the saved sweep payload."""
    local_sweep_dir = Path(local_sweep_dir)
    local_runs_dir = Path(local_runs_dir)
    local_sweep_dir.mkdir(parents=True, exist_ok=True)
    local_runs_dir.mkdir(parents=True, exist_ok=True)
    (local_sweep_dir / "sweep_manifest.submit.json").write_text(manifest_json)

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
    if preflight_completed.returncode != 0:
        raise RuntimeError(
            "Remote sweep preflight failed.\n"
            f"Stdout:\n{preflight_completed.stdout}\n\n"
            f"Stderr:\n{preflight_completed.stderr}"
        )

    upload_metadata = hooks.upload_manifest_fn(effective_config) or {}
    remote_metadata.update(upload_metadata)

    submit_shell = hooks.build_submit_shell_fn(effective_config, remote_helper_dir)
    hooks.progress_write("[Sol remote] Submitting remote sweep batch job...")
    submit_result = hooks.submit_remote_job_fn(
        effective_config,
        submit_shell=submit_shell,
        local_output_dir=local_sweep_dir,
    )
    submit_completed = submit_result.completed
    if submit_completed.returncode != 0:
        raise RuntimeError(
            "Remote sweep submission failed.\n"
            f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
        )

    if submit_result.submission is None:
        raise RuntimeError(
            "Remote sweep submission did not return valid JSON.\n"
            f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
        ) from submit_result.json_error
    submission = submit_result.submission

    remote_job_heartbeat_path = submit_result.job_heartbeat_path
    remote_metadata["job_heartbeat_path"] = remote_job_heartbeat_path
    remote_metadata["heartbeat_timeout_s"] = submit_result.heartbeat_timeout_s

    synced_labels: set[str] = set()
    item_status_by_label: dict[str, dict[str, Any]] = {}
    hooks.progress_write(
        f"[Sol remote] Submitted sweep job {submission['job_id']} "
        f"for {len(manifest_items)} items (parallelism={max_concurrent})."
    )
    monitor_result = hooks.monitor_remote_job_fn(
        effective_config=effective_config,
        submission=submission,
        remote_job_heartbeat_path=remote_job_heartbeat_path,
        allocation_heartbeat_path=allocation_heartbeat_path,
        remote_repo_root=remote_repo_root,
        remote_sweep_root=remote_sweep_root,
        remote_helper_dir=remote_helper_dir,
        notebook_timings=notebook_timings,
        synced_labels=synced_labels,
        item_status_by_label=item_status_by_label,
    )
    final_status = monitor_result.final_status

    sweep_artifacts = hooks.finalize_remote_artifacts_fn(
        effective_config,
        final_status=final_status,
        local_sweep_dir=local_sweep_dir,
        local_runs_dir=local_runs_dir,
        remote_sweep_root=remote_sweep_root,
        sweep_label=sweep_label,
        manifest_items=manifest_items,
        item_status_by_label=item_status_by_label,
        notebook_timings=notebook_timings,
    )
    final_sync = sweep_artifacts.final_sync
    sweep_summary = sweep_artifacts.sweep_summary
    item_status_by_label = sweep_artifacts.item_status_by_label
    if final_sync.returncode != 0:
        raise RuntimeError(
            "Remote sweep result sync failed.\n"
            f"Sweep dir: {local_sweep_dir}\n"
            f"Stderr:\n{final_sync.stderr}"
        )

    remote_metadata["notebook_timing_seconds"] = notebook_timings
    sweep_items, missing_labels, load_errors = hooks.finalize_local_items_fn(
        manifest_items=manifest_items,
        sweep_plan=sweep_plan,
        local_runs_dir=local_runs_dir,
        timestamp=timestamp,
        remote_metadata=remote_metadata,
        notebook_timings=notebook_timings,
        item_status_by_label=item_status_by_label,
    )

    sweep = {
        "path": sweep_plan["path"],
        "values": list(sweep_plan["values"]),
        "items": sweep_items,
        "paramset": sweep_plan["paramset"],
    }
    if sweep_plan.get("grid") is not None:
        sweep["grid"] = sweep_plan["grid"]
    hooks.persist_sweep_fn(sweep, sweep_dir=local_sweep_dir, timestamp=timestamp)
    hooks.merge_sweep_info_payload_fn(
        local_sweep_dir,
        {
            "remote": {
                **remote_metadata,
                "job_id": submission.get("job_id"),
                "final_status": hooks.summarize_status_fn(final_status),
                "notebook_timing_seconds": notebook_timings,
            }
        },
    )

    timing_summary = hooks.timing_summary_text_fn(notebook_timings)
    if timing_summary:
        hooks.progress_write(f"[OBGPU load] Sweep notebook pipeline timings: {timing_summary}")

    failed_labels = []
    for failed in sweep_summary.get("failed_items", []):
        if isinstance(failed, dict) and failed.get("label"):
            failed_labels.append(str(failed["label"]))
    result_labels = {str(item.get("label")) for item in sweep_items if item.get("result") is not None}
    failed_without_result = [label for label in failed_labels if label not in result_labels]
    recovered_failed_labels = [label for label in failed_labels if label in result_labels]
    loaded_count = sum(1 for item in sweep_items if item.get("result") is not None)
    partial_reasons = []
    if failed_without_result:
        partial_reasons.append(f"{len(failed_without_result)} failed")
    if missing_labels:
        partial_reasons.append(f"{len(missing_labels)} missing")
    if load_errors:
        partial_reasons.append(f"{len(load_errors)} load errors")
    sweep["partial"] = bool(partial_reasons)
    sweep["failed_labels"] = failed_labels
    sweep["failed_without_result"] = failed_without_result
    sweep["recovered_failed_labels"] = recovered_failed_labels
    sweep["missing_labels"] = missing_labels
    sweep["load_errors"] = load_errors
    if partial_reasons:
        hooks.persist_sweep_fn(sweep, sweep_dir=local_sweep_dir, timestamp=timestamp)
        hooks.progress_write(
            "[OBGPU load] Remote sweep returned partial results: "
            f"{loaded_count}/{len(manifest_items)} usable items "
            f"({', '.join(partial_reasons)})."
        )
    if final_status is not None and not final_status.get("ok", True) and not sweep_summary and loaded_count == 0:
        raise RuntimeError(
            "Remote sweep failed before writing a summary.\n"
            f"Sweep dir: {local_sweep_dir}\n"
            f"State: {final_status.get('state')}"
        )
    return sweep
