"""Concrete olfactory-bulb runtime hook builders for extracted neuroinfra layers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import Any, Callable, MutableMapping

from neuroinfra.artifacts.loading import ArtifactLoadingHooks
from neuroinfra.artifacts.result_view import ResultViewHooks
from neuroinfra.notebooks.remote_jobs import (
    RemoteJobSessionHooks,
    RemoteJobSubmitHooks,
)
from neuroinfra.remote.deferred_artifacts import DeferredArtifactSyncHooks
from neuroinfra.remote.result_sync import RemoteResultSyncHooks
from neuroinfra.remote.run_artifacts import RemoteRunArtifactHooks
from neuroinfra.remote.run_monitor import RemoteRunMonitorHooks
from neuroinfra.remote.status_poll import RemoteJSONPollHooks
from neuroinfra.remote.stream_sync import ParamikoStreamSyncHooks
from neuroinfra.remote.sweep_artifacts import RemoteSweepArtifactHooks
from neuroinfra.remote.sweep_monitor import RemoteSweepMonitorHooks


def build_paramiko_stream_sync_hooks(
    *,
    transport_for_config_fn: Callable[[dict[str, Any]], Any],
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]],
    build_remote_stream_archive_command_fn: Callable[[PurePosixPath, str], str],
    build_remote_selected_archive_probe_command_fn: Callable[[PurePosixPath, tuple[str, ...]], str],
    local_archive_decompress_command_fn: Callable[[str], list[str]],
    channel_stream_finished_fn: Callable[[Any], bool],
    progress_factory_fn: Callable[[int | None, str], Any],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ParamikoStreamSyncHooks:
    """Build notebook-facing Paramiko stream-sync hooks."""
    return ParamikoStreamSyncHooks(
        transport_for_config_fn=transport_for_config_fn,
        run_paramiko_shell_fn=run_paramiko_shell_fn,
        build_remote_stream_archive_command_fn=build_remote_stream_archive_command_fn,
        build_remote_selected_archive_probe_command_fn=build_remote_selected_archive_probe_command_fn,
        local_archive_decompress_command_fn=local_archive_decompress_command_fn,
        channel_stream_finished_fn=channel_stream_finished_fn,
        progress_factory_fn=progress_factory_fn,
        sleep_fn=sleep_fn,
    )


def build_remote_result_sync_hooks(
    *,
    remote_transport_fn: Callable[[dict[str, Any]], str],
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]],
    build_remote_archive_probe_command_fn: Callable[[PurePosixPath], str],
    probe_selected_sync_files_fn: Callable[[dict[str, Any], PurePosixPath, tuple[str, ...]], Any],
    build_remote_selected_stream_archive_command_fn: Callable[[PurePosixPath, tuple[str, ...], str], str],
    stream_archive_to_local_dir_fn: Callable[..., subprocess.CompletedProcess[str]],
    get_paramiko_sftp_fn: Callable[[dict[str, Any]], Any],
    close_paramiko_sftp_fn: Callable[[dict[str, Any]], None],
    sftp_copy_files_fn: Callable[[Any, str, Path, tuple[str, ...] | list[str]], None],
    sftp_copy_tree_fn: Callable[[Any, str, Path], None],
    cached_transport_fn: Callable[[dict[str, Any]], Any],
    transport_is_usable_fn: Callable[[Any], bool],
    preserve_reauth_blocked_fn: Callable[[dict[str, Any]], bool],
    drop_paramiko_connection_fn: Callable[[dict[str, Any]], None],
    midrun_reauth_error_fn: Callable[[dict[str, Any]], str],
    progress_write: Callable[[str], None],
    missing_local_sync_artifacts_fn: Callable[[Path, tuple[str, ...] | None], list[str]],
    local_sync_artifact_is_usable_fn: Callable[[Path], bool],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RemoteResultSyncHooks:
    """Build notebook-facing remote result-sync hooks."""
    return RemoteResultSyncHooks(
        remote_transport_fn=remote_transport_fn,
        run_paramiko_shell_fn=run_paramiko_shell_fn,
        build_remote_archive_probe_command_fn=build_remote_archive_probe_command_fn,
        probe_selected_sync_files_fn=probe_selected_sync_files_fn,
        build_remote_selected_stream_archive_command_fn=build_remote_selected_stream_archive_command_fn,
        stream_archive_to_local_dir_fn=stream_archive_to_local_dir_fn,
        get_paramiko_sftp_fn=get_paramiko_sftp_fn,
        close_paramiko_sftp_fn=close_paramiko_sftp_fn,
        sftp_copy_files_fn=sftp_copy_files_fn,
        sftp_copy_tree_fn=sftp_copy_tree_fn,
        cached_transport_fn=cached_transport_fn,
        transport_is_usable_fn=transport_is_usable_fn,
        preserve_reauth_blocked_fn=preserve_reauth_blocked_fn,
        drop_paramiko_connection_fn=drop_paramiko_connection_fn,
        midrun_reauth_error_fn=midrun_reauth_error_fn,
        progress_write=progress_write,
        missing_local_sync_artifacts_fn=missing_local_sync_artifacts_fn,
        local_sync_artifact_is_usable_fn=local_sync_artifact_is_usable_fn,
        sleep_fn=sleep_fn,
    )


def build_deferred_artifact_sync_hooks(
    *,
    local_sync_artifact_is_usable_fn: Callable[[Path], bool],
    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]],
    progress_write: Callable[[str], None],
    format_bytes_fn: Callable[[int | float], str],
    direct_stream_supported_fn: Callable[[str], bool],
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]],
    stream_file_to_local_path_fn: Callable[..., subprocess.CompletedProcess[str]],
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> DeferredArtifactSyncHooks:
    """Build notebook-facing deferred-artifact sync hooks."""
    return DeferredArtifactSyncHooks(
        local_sync_artifact_is_usable_fn=local_sync_artifact_is_usable_fn,
        sync_remote_result_dir_fn=sync_remote_result_dir_fn,
        progress_write=progress_write,
        format_bytes_fn=format_bytes_fn,
        direct_stream_supported_fn=direct_stream_supported_fn,
        run_paramiko_shell_fn=run_paramiko_shell_fn,
        stream_file_to_local_path_fn=stream_file_to_local_path_fn,
        perf_counter_fn=perf_counter_fn,
    )


def build_artifact_loading_hooks(
    *,
    load_pickle_fn: Callable[[str | Path], Any],
    apply_loaded_fn: Callable[[MutableMapping[str, Any], str, Any], None],
    progress_factory_fn: Callable[[int, str], Any | None],
    progress_write: Callable[[str], None],
    format_bytes_fn: Callable[[int | float], str],
    render_progress_bar_fn: Callable[[int, int], str],
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> ArtifactLoadingHooks:
    """Build notebook-facing local artifact-loading hooks."""
    return ArtifactLoadingHooks(
        load_pickle_fn=load_pickle_fn,
        apply_loaded_fn=apply_loaded_fn,
        progress_factory_fn=progress_factory_fn,
        progress_write=progress_write,
        format_bytes_fn=format_bytes_fn,
        render_progress_bar_fn=render_progress_bar_fn,
        perf_counter_fn=perf_counter_fn,
    )


def build_result_view_hooks(
    *,
    read_json_if_present_fn: Callable[[str | Path], dict[str, Any] | None],
    standard_result_artifact_sizes_fn: Callable[[str | Path], dict[str, int]],
    local_sync_artifact_is_usable_fn: Callable[[str | Path], bool],
    sync_deferred_artifact_fn: Callable[..., Path],
    load_pickle_fn: Callable[[str | Path], Any],
    set_lazy_artifact_path_fn: Callable[[MutableMapping[str, Any], str, Path], None],
    format_bytes_fn: Callable[[int | float], str],
    progress_write: Callable[[str], None],
) -> ResultViewHooks:
    """Build notebook-facing result-view hooks."""
    return ResultViewHooks(
        read_json_if_present_fn=read_json_if_present_fn,
        standard_result_artifact_sizes_fn=standard_result_artifact_sizes_fn,
        local_sync_artifact_is_usable_fn=local_sync_artifact_is_usable_fn,
        sync_deferred_artifact_fn=sync_deferred_artifact_fn,
        load_pickle_fn=load_pickle_fn,
        set_lazy_artifact_path_fn=set_lazy_artifact_path_fn,
        local_lazy_notice_fn=lambda key, path: (
            f"[OBGPU load] Deferred {key} ({format_bytes_fn(path.stat().st_size)}) until result['{key}'] is accessed."
        ),
        remote_lazy_notice_fn=lambda key, _path: (
            f"[OBGPU load] Deferred {key} stays remote until result['{key}'] is accessed."
        ),
        progress_write=progress_write,
    )


def build_remote_json_poll_hooks(
    config: dict[str, Any],
    notebook_timings: dict[str, float],
    *,
    run_ssh_shell_fn: Callable[..., subprocess.CompletedProcess[str]],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    sleep_fn: Callable[[float], None] = time.sleep,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> RemoteJSONPollHooks:
    """Build notebook-facing remote JSON poll hooks."""
    return RemoteJSONPollHooks(
        run_command_fn=lambda command, timeout_s=None: run_ssh_shell_fn(
            config,
            command,
            timeout_s=timeout_s,
        ),
        record_timing_fn=lambda key, started: record_timing_fn(notebook_timings, key, started),
        sleep_fn=sleep_fn,
        perf_counter_fn=perf_counter_fn,
    )


def build_remote_job_session_hooks(
    notebook_timings: dict[str, float],
    *,
    ensure_remote_git_ref_available_fn: Callable[..., None],
    run_remote_preflight_fn: Callable[..., tuple[subprocess.CompletedProcess[str], bool]],
    ensure_remote_helper_cache_fn: Callable[[dict[str, Any]], PurePosixPath | None],
    helper_cache_lookup_fn: Callable[[dict[str, Any]], bool],
    cleanup_stale_allocations_fn: Callable[..., list[Any]],
    ensure_cached_remote_allocation_fn: Callable[..., dict[str, Any]],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    progress_write: Callable[[str], None],
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> RemoteJobSessionHooks:
    """Build notebook-facing remote job session hooks."""
    return RemoteJobSessionHooks(
        ensure_remote_git_ref_available_fn=ensure_remote_git_ref_available_fn,
        run_remote_preflight_fn=run_remote_preflight_fn,
        ensure_remote_helper_cache_fn=ensure_remote_helper_cache_fn,
        helper_cache_hit_fn=helper_cache_lookup_fn,
        cleanup_stale_allocations_fn=cleanup_stale_allocations_fn,
        ensure_cached_remote_allocation_fn=ensure_cached_remote_allocation_fn,
        record_timing_fn=lambda key, started: record_timing_fn(notebook_timings, key, started),
        progress_write=progress_write,
        perf_counter_fn=perf_counter_fn,
    )


def build_remote_job_submit_hooks(
    notebook_timings: dict[str, float],
    *,
    run_ssh_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]],
    heartbeat_timeout_s_fn: Callable[[dict[str, Any]], int | float],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> RemoteJobSubmitHooks:
    """Build notebook-facing remote job submit hooks."""
    return RemoteJobSubmitHooks(
        run_ssh_shell_fn=run_ssh_shell_fn,
        heartbeat_timeout_s_fn=heartbeat_timeout_s_fn,
        record_timing_fn=lambda key, started: record_timing_fn(notebook_timings, key, started),
        perf_counter_fn=perf_counter_fn,
    )


def build_remote_run_artifact_hooks(
    notebook_timings: dict[str, float],
    *,
    sync_remote_result_dir_resilient_fn: Callable[..., subprocess.CompletedProcess[str]],
    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]],
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]],
    build_remote_result_listing_command_fn: Callable[[PurePosixPath], str],
    local_result_dir_has_loadable_payload_fn: Callable[[Path], bool],
    local_result_dir_has_diagnostics_fn: Callable[[Path], bool],
    standard_result_artifact_sizes_fn: Callable[[Path], dict[str, int]],
    synthesize_partial_sync_summary_fn: Callable[..., dict[str, Any]],
    compact_remote_poll_events_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    read_json_if_present_fn: Callable[[str | Path], dict[str, Any] | None],
    progress_write: Callable[[str], None],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    sleep_fn: Callable[[float], None] = time.sleep,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> RemoteRunArtifactHooks:
    """Build notebook-facing remote run artifact hooks."""
    return RemoteRunArtifactHooks(
        sync_remote_result_dir_resilient_fn=sync_remote_result_dir_resilient_fn,
        sync_remote_result_dir_fn=sync_remote_result_dir_fn,
        run_paramiko_shell_fn=run_paramiko_shell_fn,
        build_remote_result_listing_command_fn=build_remote_result_listing_command_fn,
        local_result_dir_has_loadable_payload_fn=local_result_dir_has_loadable_payload_fn,
        local_result_dir_has_diagnostics_fn=local_result_dir_has_diagnostics_fn,
        standard_result_artifact_sizes_fn=standard_result_artifact_sizes_fn,
        synthesize_partial_sync_summary_fn=synthesize_partial_sync_summary_fn,
        compact_remote_poll_events_fn=compact_remote_poll_events_fn,
        read_json_if_present_fn=read_json_if_present_fn,
        progress_write=progress_write,
        record_timing_fn=lambda key, started: record_timing_fn(notebook_timings, key, started),
        sleep_fn=sleep_fn,
        perf_counter_fn=perf_counter_fn,
    )


def build_remote_run_monitor_hooks(
    *,
    effective_config: dict[str, Any],
    remote_job_heartbeat_path: str | None,
    allocation_heartbeat_path: str | None,
    remote_repo_root: PurePosixPath,
    remote_result_dir: PurePosixPath,
    remote_helper_dir: PurePosixPath | None,
    notebook_timings: dict[str, float],
    submission: dict[str, Any],
    local_result_dir: Path,
    refresh_remote_heartbeat_fn: Callable[[dict[str, Any], str | PurePosixPath | None], bool],
    build_remote_poll_command_fn: Callable[..., str],
    poll_remote_json_status_fn: Callable[..., dict[str, Any]],
    remote_json_poll_hooks_fn: Callable[[dict[str, Any], dict[str, float]], RemoteJSONPollHooks],
    remote_poll_command_timeout_s_fn: Callable[[dict[str, Any]], float | None],
    run_ssh_shell_fn: Callable[..., subprocess.CompletedProcess[str]],
    build_remote_cancel_command_fn: Callable[..., str],
    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    remote_status_has_artifacts_fn: Callable[[dict[str, Any] | None], bool],
    progress_bar_factory_fn: Callable[[int, str], Any],
    filter_live_log_line_fn: Callable[[str, str], str | None],
    progress_write: Callable[[str], None],
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    time_fn: Callable[[], float] = time.time,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> RemoteRunMonitorHooks:
    """Build notebook-facing remote single-run monitor hooks."""

    def refresh_remote_leases(*, warn: bool = False) -> None:
        refresh_remote_heartbeat_fn(effective_config, remote_job_heartbeat_path, warn=warn)
        refresh_remote_heartbeat_fn(effective_config, allocation_heartbeat_path, warn=warn)

    def poll_status_once(
        *,
        refresh_heartbeat: bool = True,
        include_logs: bool = True,
        include_sacct: bool = True,
    ) -> dict[str, Any]:
        if refresh_heartbeat:
            refresh_remote_leases()
        poll_shell = build_remote_poll_command_fn(
            effective_config,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_result_dir,
            job_id=str(submission["job_id"]),
            wrapper_dir=str(submission.get("wrapper_dir") or ""),
            worktree_path=str(submission.get("worktree_path") or ""),
            remote_helper_dir=remote_helper_dir,
            include_sacct=include_sacct,
            include_tails=include_logs,
        )
        return poll_remote_json_status_fn(
            poll_shell,
            poll_json_retries=max(int(effective_config.get("remote_poll_json_retries", 3) or 1), 1),
            error_prefix="Remote Sol status poll",
            hooks=remote_json_poll_hooks_fn(effective_config, notebook_timings),
        )

    def cancel_job() -> subprocess.CompletedProcess[str]:
        return run_ssh_shell_fn(
            effective_config,
            build_remote_cancel_command_fn(job_id=str(submission["job_id"])),
        )

    def sync_partial_artifacts() -> subprocess.CompletedProcess[str]:
        sync_started = perf_counter_fn()
        sync_completed = sync_remote_result_dir_fn(
            effective_config,
            remote_result_dir=remote_result_dir,
            local_result_dir=local_result_dir,
        )
        record_timing_fn(notebook_timings, "partial_sync_s", sync_started)
        (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
        (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
        return sync_completed

    return RemoteRunMonitorHooks(
        refresh_remote_leases_fn=refresh_remote_leases,
        poll_status_fn=poll_status_once,
        cancel_job_fn=cancel_job,
        sync_partial_artifacts_fn=sync_partial_artifacts,
        remote_status_has_artifacts_fn=remote_status_has_artifacts_fn,
        progress_bar_factory_fn=progress_bar_factory_fn,
        filter_live_log_line_fn=filter_live_log_line_fn,
        progress_write=progress_write,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        time_fn=time_fn,
    )


def build_remote_sweep_monitor_hooks(
    *,
    effective_config: dict[str, Any],
    remote_job_heartbeat_path: str | None,
    allocation_heartbeat_path: str | None,
    remote_repo_root: PurePosixPath,
    remote_sweep_root: PurePosixPath,
    remote_helper_dir: PurePosixPath | None,
    notebook_timings: dict[str, float],
    submission: dict[str, Any],
    synced_labels: set[str],
    sync_finished_items_fn: Callable[[dict[str, Any]], None],
    refresh_remote_heartbeat_fn: Callable[[dict[str, Any], str | PurePosixPath | None], bool],
    build_remote_poll_command_fn: Callable[..., str],
    poll_remote_json_status_fn: Callable[..., dict[str, Any]],
    remote_json_poll_hooks_fn: Callable[[dict[str, Any], dict[str, float]], RemoteJSONPollHooks],
    remote_poll_command_timeout_s_fn: Callable[[dict[str, Any]], float | None],
    run_ssh_shell_fn: Callable[..., subprocess.CompletedProcess[str]],
    build_remote_cancel_command_fn: Callable[..., str],
    progress_write: Callable[[str], None],
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> RemoteSweepMonitorHooks:
    """Build notebook-facing remote sweep monitor hooks."""

    def refresh_remote_leases(*, warn: bool = False) -> None:
        refresh_remote_heartbeat_fn(effective_config, remote_job_heartbeat_path, warn=warn)
        refresh_remote_heartbeat_fn(effective_config, allocation_heartbeat_path, warn=warn)

    def poll_status_once(*, refresh_heartbeat: bool = True, include_sacct: bool = True) -> dict[str, Any]:
        if refresh_heartbeat:
            refresh_remote_leases()
        poll_shell = build_remote_poll_command_fn(
            effective_config,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_sweep_root,
            job_id=str(submission["job_id"]),
            wrapper_dir=str(submission.get("wrapper_dir") or ""),
            worktree_path=str(submission.get("worktree_path") or ""),
            remote_helper_dir=remote_helper_dir,
            include_sacct=include_sacct,
            include_tails=False,
        )
        return poll_remote_json_status_fn(
            poll_shell,
            poll_json_retries=max(int(effective_config.get("remote_poll_json_retries", 3) or 1), 1),
            error_prefix="Remote sweep status poll",
            hooks=remote_json_poll_hooks_fn(effective_config, notebook_timings),
            timeout_s=remote_poll_command_timeout_s_fn(effective_config),
        )

    def cancel_job() -> subprocess.CompletedProcess[str]:
        return run_ssh_shell_fn(
            effective_config,
            build_remote_cancel_command_fn(job_id=str(submission["job_id"])),
        )

    return RemoteSweepMonitorHooks(
        refresh_remote_leases_fn=refresh_remote_leases,
        poll_status_fn=poll_status_once,
        sync_finished_items_fn=sync_finished_items_fn,
        cancel_job_fn=cancel_job,
        synced_count_fn=lambda: len(synced_labels),
        progress_write=progress_write,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )


def build_remote_sweep_artifact_hooks(
    *,
    refresh_remote_leases_fn: Callable[..., None],
    notebook_timings: dict[str, float],
    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]],
    sync_remote_sweep_compact_items_fn: Callable[..., subprocess.CompletedProcess[str]],
    read_json_if_present_fn: Callable[[str | Path], dict[str, Any] | None],
    recover_local_sweep_summary_fn: Callable[..., dict[str, Any]],
    remote_sweep_metadata_files_fn: Callable[[], tuple[str, ...]],
    remote_sweep_item_sync_files_fn: Callable[[dict[str, Any]], tuple[str, ...]],
    remote_sweep_item_diagnostic_files_fn: Callable[[], tuple[str, ...]],
    local_sweep_item_sync_complete_fn: Callable[[str | Path], bool],
    local_result_dir_has_diagnostics_fn: Callable[[str | Path], bool],
    progress_write: Callable[[str], None],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> RemoteSweepArtifactHooks:
    """Build notebook-facing remote sweep artifact hooks."""
    return RemoteSweepArtifactHooks(
        sync_remote_result_dir_fn=sync_remote_result_dir_fn,
        sync_remote_sweep_compact_items_fn=sync_remote_sweep_compact_items_fn,
        read_json_if_present_fn=read_json_if_present_fn,
        recover_local_sweep_summary_fn=recover_local_sweep_summary_fn,
        remote_sweep_metadata_files_fn=remote_sweep_metadata_files_fn,
        remote_sweep_item_sync_files_fn=remote_sweep_item_sync_files_fn,
        remote_sweep_item_diagnostic_files_fn=remote_sweep_item_diagnostic_files_fn,
        local_sweep_item_sync_complete_fn=local_sweep_item_sync_complete_fn,
        local_result_dir_has_diagnostics_fn=local_result_dir_has_diagnostics_fn,
        progress_write=progress_write,
        refresh_remote_leases_fn=refresh_remote_leases_fn,
        record_timing_fn=lambda key, started: record_timing_fn(notebook_timings, key, started),
        perf_counter_fn=perf_counter_fn,
    )
