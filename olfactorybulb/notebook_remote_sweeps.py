"""Concrete olfactory-bulb remote sweep workflow adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable

from neuroinfra.notebooks.remote_jobs import (
    prepare_remote_job_session,
    submit_remote_json_job,
)
from neuroinfra.notebooks.remote_sweeps import RemoteSweepWorkflowHooks
from neuroinfra.remote import (
    finalize_remote_sweep_artifacts,
    monitor_remote_sweep,
)


@dataclass(frozen=True)
class RemoteSweepPayloadHooks:
    """Hooks for building one concrete olfactory-bulb remote sweep payload."""

    json_ready_fn: Callable[[Any], Any]
    benchmark_param_overrides_payload_fn: Callable[[dict[str, Any]], tuple[dict[str, Any], str | None]]
    build_run_command_fn: Callable[..., list[str]]
    remote_sweep_parallelism_fn: Callable[[dict[str, Any], int], int]
    require_remote_host_fn: Callable[[dict[str, Any]], str]
    default_remote_mpi_exec_fn: Callable[[], str]


@dataclass(frozen=True)
class FinalizeSyncedSweepItemHooks:
    """Hooks for finalizing one synced remote sweep item into notebook artifacts."""

    read_json_if_present_fn: Callable[[str | Path], Any]
    json_ready_fn: Callable[[Any], Any]
    write_run_info_fn: Callable[..., Any]
    load_run_record_fn: Callable[[str | Path], Any]
    load_result_fn: Callable[[Any], Any]


@dataclass(frozen=True)
class NotebookRemoteSweepWorkflowBuilderHooks:
    """Hooks for constructing the concrete olfactory-bulb remote sweep workflow."""

    remote_job_session_hooks_fn: Callable[[dict[str, float]], Any]
    remote_job_submit_hooks_fn: Callable[[dict[str, float]], Any]
    remote_sweep_monitor_hooks_fn: Callable[..., Any]
    remote_sweep_artifact_hooks_fn: Callable[..., Any]
    build_remote_submit_command_fn: Callable[..., str]
    upload_remote_text_file_fn: Callable[..., Any]
    refresh_remote_heartbeat_fn: Callable[..., Any]
    should_sync_remote_sweep_finished_items_fn: Callable[..., bool]
    sync_remote_result_dir_fn: Callable[..., Any]
    remote_sweep_item_sync_files_fn: Callable[[dict[str, Any]], tuple[str, ...]]
    local_sync_artifact_is_usable_fn: Callable[[str | Path], bool]
    synthesize_partial_sync_summary_fn: Callable[..., dict[str, Any]]
    persist_sweep_fn: Callable[..., Any]
    merge_sweep_info_payload_fn: Callable[..., Any]
    summarize_status_fn: Callable[[dict[str, Any] | None], Any]
    timing_summary_text_fn: Callable[[dict[str, float]], str]
    write_run_info_fn: Callable[..., Any]
    load_run_record_fn: Callable[[str | Path], Any]
    load_result_fn: Callable[[Any], Any]
    resolve_local_sweep_item_dir_fn: Callable[[str | Path, str], Path | None]
    json_ready_fn: Callable[[Any], Any]
    read_json_if_present_fn: Callable[[str | Path], Any]
    progress_write: Callable[[str], None]
    record_timing_fn: Callable[[dict[str, float], str, float], Any]
    perf_counter_fn: Callable[[], float]
    default_remote_mpi_exec_fn: Callable[[], str]


def build_remote_sweep_payload(
    hooks: RemoteSweepPayloadHooks,
    config: dict[str, Any],
    *,
    sweep_plan: dict[str, Any],
    sweep_label: str,
    remote_repo_root: PurePosixPath,
    remote_sweeps_root: PurePosixPath,
    remote_sweep_root: PurePosixPath,
    remote_git_ref: str | None,
) -> tuple[list[str], list[dict[str, Any]], str, PurePosixPath, int, dict[str, Any]]:
    """Build the concrete remote sweep command, manifest, and metadata payload."""
    remote_driver = Path(remote_repo_root) / "tools" / "remote" / "remote_sweep_driver.py"
    remote_runs_root = remote_sweep_root / "item_runs"
    remote_manifest_path = remote_sweep_root / "sweep_manifest.submit.json"
    remote_mpi_exec = str(config.get("remote_mpi_exec") or hooks.default_remote_mpi_exec_fn())
    tasks_per_item = max(int(config.get("nranks", 1) or 1), 1)
    max_concurrent = hooks.remote_sweep_parallelism_fn(
        config,
        tasks_per_item=tasks_per_item,
    )

    manifest_items: list[dict[str, Any]] = []
    for item in sweep_plan["items"]:
        remote_result_dir = remote_runs_root / item["label"]
        item_param_overrides, item_input_spec_file = hooks.benchmark_param_overrides_payload_fn(item["config"])
        item_overrides_file = remote_sweep_root / "overrides" / f"{item['label']}.json"
        benchmark_command = hooks.build_run_command_fn(
            item["config"],
            item["label"],
            repo_root=remote_repo_root,
            results_base=remote_runs_root,
            mpi_exec=remote_mpi_exec,
            include_mpi_launcher=True,
            overrides_file=item_overrides_file,
            param_overrides=item_param_overrides,
            input_spec_file=item_input_spec_file,
        )
        manifest_items.append(
            {
                "index": int(item["index"]),
                "label": str(item["label"]),
                "value": hooks.json_ready_fn(item["value"]),
                "result_dir": remote_result_dir.as_posix(),
                "command": benchmark_command,
                "overrides_file": item_overrides_file.as_posix(),
                "overrides": hooks.json_ready_fn(item_param_overrides),
            }
        )

    manifest_json = json.dumps(manifest_items, indent=2, sort_keys=True)
    driver_command = [
        "python3",
        str(remote_driver),
        "--repo-root",
        remote_repo_root.as_posix(),
        "--sweep-root",
        remote_sweep_root.as_posix(),
        "--items-json",
        remote_manifest_path.as_posix(),
        "--max-concurrent",
        str(max_concurrent),
    ]
    remote_metadata = {
        "runner_backend": str(config.get("runner_backend", "slurm_remote")),
        "remote_host": hooks.require_remote_host_fn(config),
        "remote_repo_root": remote_repo_root.as_posix(),
        "remote_results_root": remote_sweeps_root.as_posix(),
        "remote_mpi_exec": remote_mpi_exec,
        "remote_repo_mode": str(config.get("remote_repo_mode", "shared")),
        "remote_git_ref": remote_git_ref,
        "remote_git_fetch": bool(config.get("remote_git_fetch", False)),
        "remote_git_remote": str(config.get("remote_git_remote", "origin")),
        "sweep_label": sweep_label,
        "sweep_parallelism": int(max_concurrent),
        "sweep_items": len(manifest_items),
    }
    return (
        driver_command,
        manifest_items,
        manifest_json,
        remote_manifest_path,
        max_concurrent,
        remote_metadata,
    )


def finalize_synced_sweep_item(
    hooks: FinalizeSyncedSweepItemHooks,
    *,
    item: dict[str, Any],
    local_result_dir: str | Path,
    timestamp: str,
    remote_payload: dict[str, Any],
    returncode: int,
):
    """Write run-info for one synced sweep item, then load its run/result pair."""
    local_result_dir = Path(local_result_dir)
    stdout_path = local_result_dir / "stdout.txt"
    stderr_path = local_result_dir / "stderr.txt"
    stdout = stdout_path.read_text() if stdout_path.exists() else ""
    stderr = stderr_path.read_text() if stderr_path.exists() else ""
    summary = hooks.read_json_if_present_fn(local_result_dir / "summary.json")
    completed = SimpleNamespace(returncode=int(returncode), stdout=stdout, stderr=stderr)
    hooks.write_run_info_fn(
        local_result_dir,
        config=item["config"],
        label=item["label"],
        timestamp=timestamp,
        command=item["command"],
        env={},
        completed=completed,
        summary=summary,
        extra_payload={
            "remote": remote_payload,
            "sweep_item": {
                "index": int(item["index"]),
                "value": hooks.json_ready_fn(item["value"]),
            },
        },
    )
    run = hooks.load_run_record_fn(local_result_dir)
    result = hooks.load_result_fn(run)
    return run, result


def build_remote_sweep_workflow_hooks(
    hooks: NotebookRemoteSweepWorkflowBuilderHooks,
    *,
    sweep_label: str,
    timestamp: str,
    remote_repo_root: PurePosixPath,
    remote_sweeps_root: PurePosixPath,
    remote_sweep_root: PurePosixPath,
    remote_driver_command: list[str],
    remote_git_ref: str | None,
    manifest_json: str,
    manifest_items: list[dict[str, Any]],
    remote_manifest_path: PurePosixPath,
    max_concurrent: int,
    local_runs_dir: str | Path,
) -> RemoteSweepWorkflowHooks:
    """Build the notebook-facing olfactory-bulb remote sweep workflow hooks."""
    local_runs_dir = Path(local_runs_dir)
    workflow_state: dict[str, Any] = {
        "notebook_timings": None,
        "manifest_items": manifest_items,
        "local_runs_dir": local_runs_dir,
        "remote_job_heartbeat_path": None,
        "allocation_heartbeat_path": None,
    }
    finalize_item_hooks = FinalizeSyncedSweepItemHooks(
        read_json_if_present_fn=hooks.read_json_if_present_fn,
        json_ready_fn=hooks.json_ready_fn,
        write_run_info_fn=hooks.write_run_info_fn,
        load_run_record_fn=hooks.load_run_record_fn,
        load_result_fn=hooks.load_result_fn,
    )

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
            preflight_message="[Sol remote] Running remote preflight checks for sweep...",
            hooks=hooks.remote_job_session_hooks_fn(notebook_timings),
            notebook_timings=notebook_timings,
        )

    def upload_manifest(config: dict[str, Any]) -> dict[str, Any]:
        notebook_timings = workflow_state["notebook_timings"]
        if notebook_timings is None:
            raise RuntimeError("remote sweep workflow timings were not initialized before manifest upload")
        hooks.progress_write("[Sol remote] Uploading remote sweep manifest...")
        started = hooks.perf_counter_fn()
        hooks.upload_remote_text_file_fn(
            config,
            remote_path=remote_manifest_path,
            text=manifest_json,
        )
        hooks.record_timing_fn(notebook_timings, "manifest_upload_s", started)
        return {"sweep_manifest_path": remote_manifest_path.as_posix()}

    def build_submit_shell(config: dict[str, Any], remote_helper_dir: PurePosixPath | None) -> str:
        return hooks.build_remote_submit_command_fn(
            config,
            label=sweep_label,
            remote_repo_root=remote_repo_root,
            remote_results_root=remote_sweeps_root,
            benchmark_command=remote_driver_command,
            remote_mpi_exec=str(config.get("remote_mpi_exec") or hooks.default_remote_mpi_exec_fn()),
            remote_git_ref=remote_git_ref,
            step_ntasks=1,
            remote_helper_dir=remote_helper_dir,
        )

    def submit_remote_job(
        config: dict[str, Any],
        *,
        submit_shell: str,
        local_output_dir: str | Path,
    ) -> Any:
        notebook_timings = workflow_state["notebook_timings"]
        if notebook_timings is None:
            raise RuntimeError("remote sweep workflow timings were not initialized before submit")
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
        remote_sweep_root: PurePosixPath,
        remote_helper_dir: PurePosixPath | None,
        notebook_timings: dict[str, float],
        synced_labels: set[str],
        item_status_by_label: dict[str, dict[str, Any]],
    ) -> Any:
        workflow_state["remote_job_heartbeat_path"] = remote_job_heartbeat_path
        workflow_state["allocation_heartbeat_path"] = allocation_heartbeat_path
        manifest_by_label = {str(item["label"]): item for item in workflow_state["manifest_items"]}
        live_sync_max_items_per_poll = max(
            int(effective_config.get("sweep_live_sync_max_items_per_poll", 8) or 0),
            0,
        )

        def refresh_remote_leases(*, warn: bool = False) -> None:
            hooks.refresh_remote_heartbeat_fn(effective_config, remote_job_heartbeat_path, warn=warn)
            hooks.refresh_remote_heartbeat_fn(effective_config, allocation_heartbeat_path, warn=warn)

        def sync_finished_items(status: dict[str, Any]) -> None:
            progress_payload = status.get("progress_payload") or {}
            pending_labels = progress_payload.get("pending_labels") or []
            running_items = progress_payload.get("running_items") or []
            if not hooks.should_sync_remote_sweep_finished_items_fn(
                effective_config,
                pending_count=len(pending_labels),
                running_count=len(running_items),
            ):
                return
            finished_items = progress_payload.get("finished_items") or []
            synced_this_poll = 0
            for finished in finished_items:
                if not isinstance(finished, dict):
                    continue
                label = str(finished.get("label") or "").strip()
                if not label or label in synced_labels or label not in manifest_by_label:
                    continue
                if live_sync_max_items_per_poll and synced_this_poll >= live_sync_max_items_per_poll:
                    break
                manifest_item = manifest_by_label[label]
                remote_result_dir = PurePosixPath(str(finished.get("result_dir") or manifest_item["result_dir"]))
                local_result_dir = workflow_state["local_runs_dir"] / label
                refresh_remote_leases()
                sync_completed = hooks.sync_remote_result_dir_fn(
                    effective_config,
                    remote_result_dir=remote_result_dir,
                    local_result_dir=local_result_dir,
                    expected_files=("summary.json",),
                    include_files=hooks.remote_sweep_item_sync_files_fn(effective_config),
                )
                refresh_remote_leases()
                if sync_completed.returncode != 0:
                    continue
                item_status_by_label[label] = dict(finished)
                synced_labels.add(label)
                synced_this_poll += 1

        poll_interval_s = max(float(effective_config.get("remote_poll_interval_s", 1.0)), 1.0)
        log_poll_interval_s = max(
            float(effective_config.get("remote_log_poll_interval_s", max(poll_interval_s, 5.0))),
            poll_interval_s,
        )
        live_status = bool(effective_config.get("remote_live_status", True))
        return monitor_remote_sweep(
            job_id=str(submission["job_id"]),
            poll_interval_s=poll_interval_s,
            log_poll_interval_s=log_poll_interval_s,
            live_status=live_status,
            hooks=hooks.remote_sweep_monitor_hooks_fn(
                effective_config=effective_config,
                remote_job_heartbeat_path=remote_job_heartbeat_path,
                allocation_heartbeat_path=allocation_heartbeat_path,
                remote_repo_root=remote_repo_root,
                remote_sweep_root=remote_sweep_root,
                remote_helper_dir=remote_helper_dir,
                notebook_timings=notebook_timings,
                submission=submission,
                synced_labels=synced_labels,
                sync_finished_items_fn=sync_finished_items,
            ),
        )

    def finalize_remote_artifacts(
        effective_config: dict[str, Any],
        *,
        final_status: dict[str, Any] | None,
        local_sweep_dir: Path,
        local_runs_dir: Path,
        remote_sweep_root: PurePosixPath,
        sweep_label: str,
        manifest_items: list[dict[str, Any]],
        item_status_by_label: dict[str, dict[str, Any]],
        notebook_timings: dict[str, float],
    ) -> Any:
        def refresh_remote_leases(*, warn: bool = False) -> None:
            hooks.refresh_remote_heartbeat_fn(
                effective_config,
                workflow_state["remote_job_heartbeat_path"],
                warn=warn,
            )
            hooks.refresh_remote_heartbeat_fn(
                effective_config,
                workflow_state["allocation_heartbeat_path"],
                warn=warn,
            )

        return finalize_remote_sweep_artifacts(
            effective_config,
            final_status=final_status,
            local_sweep_dir=local_sweep_dir,
            local_runs_dir=local_runs_dir,
            remote_sweep_root=remote_sweep_root,
            sweep_label=sweep_label,
            manifest_items=manifest_items,
            item_status_by_label=item_status_by_label,
            hooks=hooks.remote_sweep_artifact_hooks_fn(
                refresh_remote_leases_fn=refresh_remote_leases,
                notebook_timings=notebook_timings,
            ),
        )

    def finalize_local_items(
        *,
        manifest_items: list[dict[str, Any]],
        sweep_plan: dict[str, Any],
        local_runs_dir: Path,
        timestamp: str,
        remote_metadata: dict[str, Any],
        notebook_timings: dict[str, float],
        item_status_by_label: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
        sweep_items = []
        load_errors: dict[str, str] = {}
        for item in manifest_items:
            plan_item = sweep_plan["items"][int(item["index"])]
            finalize_item = {**item, "config": plan_item["config"], "value": plan_item["value"]}
            local_result_dir = hooks.resolve_local_sweep_item_dir_fn(local_runs_dir, str(item["label"]))
            status_payload = item_status_by_label.get(item["label"], {})
            item_entry = {
                "index": int(item["index"]),
                "label": str(item["label"]),
                "value": plan_item["value"],
                "config": plan_item["config"],
                "run": None,
                "result": None,
                "status": status_payload,
            }
            if local_result_dir is None:
                sweep_items.append(item_entry)
                continue
            if not hooks.local_sync_artifact_is_usable_fn(local_result_dir / "summary.json"):
                summary = hooks.synthesize_partial_sync_summary_fn(
                    local_result_dir,
                    label=str(item["label"]),
                    timestamp=timestamp,
                    config=plan_item["config"],
                )
                (local_result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
            remote_metadata["notebook_timing_seconds"] = notebook_timings
            try:
                run, result = finalize_synced_sweep_item(
                    finalize_item_hooks,
                    item=finalize_item,
                    local_result_dir=local_result_dir,
                    timestamp=timestamp,
                    remote_payload=remote_metadata,
                    returncode=int(status_payload.get("returncode", 0) or 0),
                )
            except Exception as exc:
                load_errors[str(item["label"])] = str(exc)
                item_entry["status"] = {**status_payload, "load_error": str(exc)}
            else:
                item_entry["run"] = run
                item_entry["result"] = result
            sweep_items.append(item_entry)

        missing_labels = [
            item["label"]
            for item in manifest_items
            if hooks.resolve_local_sweep_item_dir_fn(local_runs_dir, str(item["label"])) is None
        ]
        return sweep_items, missing_labels, load_errors

    return RemoteSweepWorkflowHooks(
        prepare_remote_session_fn=prepare_remote_session,
        upload_manifest_fn=upload_manifest,
        build_submit_shell_fn=build_submit_shell,
        submit_remote_job_fn=submit_remote_job,
        monitor_remote_job_fn=monitor_remote_job,
        finalize_remote_artifacts_fn=finalize_remote_artifacts,
        finalize_local_items_fn=finalize_local_items,
        persist_sweep_fn=hooks.persist_sweep_fn,
        merge_sweep_info_payload_fn=hooks.merge_sweep_info_payload_fn,
        summarize_status_fn=hooks.summarize_status_fn,
        timing_summary_text_fn=hooks.timing_summary_text_fn,
        progress_write=hooks.progress_write,
    )
