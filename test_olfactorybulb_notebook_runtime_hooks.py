from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from neuroinfra.artifacts.loading import ArtifactLoadingHooks
from neuroinfra.artifacts.result_view import ResultViewHooks
from neuroinfra.notebooks.remote_jobs import RemoteJobSessionHooks, RemoteJobSubmitHooks
from neuroinfra.remote.deferred_artifacts import DeferredArtifactSyncHooks
from neuroinfra.remote.result_sync import RemoteResultSyncHooks
from neuroinfra.remote.run_artifacts import RemoteRunArtifactHooks
from neuroinfra.remote.run_monitor import RemoteRunMonitorHooks
from neuroinfra.remote.status_poll import RemoteJSONPollHooks
from neuroinfra.remote.stream_sync import ParamikoStreamSyncHooks
from neuroinfra.remote.sweep_artifacts import RemoteSweepArtifactHooks
from neuroinfra.remote.sweep_monitor import RemoteSweepMonitorHooks
from olfactorybulb.notebook_runtime_hooks import (
    build_artifact_loading_hooks,
    build_deferred_artifact_sync_hooks,
    build_paramiko_stream_sync_hooks,
    build_remote_job_session_hooks,
    build_remote_job_submit_hooks,
    build_remote_json_poll_hooks,
    build_remote_result_sync_hooks,
    build_remote_run_artifact_hooks,
    build_remote_run_monitor_hooks,
    build_remote_sweep_artifact_hooks,
    build_remote_sweep_monitor_hooks,
    build_result_view_hooks,
)


def _sentinel(name: str):
    def _fn(*args, **kwargs):
        return (name, args, kwargs)

    _fn.__name__ = name
    return _fn


cfg = {"remote_preserve_paramiko_session": True}
command_runner = _sentinel("run")
timing_calls = []
refresh_calls = []
cancel_calls = []
poll_calls = []
sync_calls = []


def _record_timing(notebook_timings, key, started):
    timing_calls.append((notebook_timings, key, started))


paramiko_hooks = build_paramiko_stream_sync_hooks(
    transport_for_config_fn=_sentinel("transport"),
    run_paramiko_shell_fn=_sentinel("paramiko_run"),
    build_remote_stream_archive_command_fn=_sentinel("archive_cmd"),
    build_remote_selected_archive_probe_command_fn=_sentinel("probe_cmd"),
    local_archive_decompress_command_fn=_sentinel("decompress"),
    channel_stream_finished_fn=_sentinel("finished"),
    progress_factory_fn=_sentinel("progress"),
)
assert isinstance(paramiko_hooks, ParamikoStreamSyncHooks)

result_sync_hooks = build_remote_result_sync_hooks(
    remote_transport_fn=_sentinel("remote_transport"),
    run_paramiko_shell_fn=_sentinel("paramiko_run"),
    build_remote_archive_probe_command_fn=_sentinel("archive_probe"),
    probe_selected_sync_files_fn=_sentinel("probe_selected"),
    build_remote_selected_stream_archive_command_fn=_sentinel("selected_stream"),
    stream_archive_to_local_dir_fn=_sentinel("stream_dir"),
    get_paramiko_sftp_fn=_sentinel("get_sftp"),
    close_paramiko_sftp_fn=_sentinel("close_sftp"),
    sftp_copy_files_fn=_sentinel("copy_files"),
    sftp_copy_tree_fn=_sentinel("copy_tree"),
    cached_transport_fn=_sentinel("cached_transport"),
    transport_is_usable_fn=_sentinel("transport_ok"),
    preserve_reauth_blocked_fn=_sentinel("reauth_blocked"),
    drop_paramiko_connection_fn=_sentinel("drop_transport"),
    midrun_reauth_error_fn=_sentinel("reauth_error"),
    progress_write=_sentinel("progress_write"),
    missing_local_sync_artifacts_fn=_sentinel("missing"),
    local_sync_artifact_is_usable_fn=_sentinel("usable"),
)
assert isinstance(result_sync_hooks, RemoteResultSyncHooks)

deferred_hooks = build_deferred_artifact_sync_hooks(
    local_sync_artifact_is_usable_fn=_sentinel("usable"),
    sync_remote_result_dir_fn=_sentinel("sync_dir"),
    progress_write=_sentinel("progress_write"),
    format_bytes_fn=lambda value: f"{value}B",
    direct_stream_supported_fn=lambda filename: filename.endswith(".pkl"),
    run_paramiko_shell_fn=_sentinel("paramiko_run"),
    stream_file_to_local_path_fn=_sentinel("stream_file"),
)
assert isinstance(deferred_hooks, DeferredArtifactSyncHooks)
assert deferred_hooks.direct_stream_supported_fn("soma_vs.pkl") is True

artifact_loading_hooks = build_artifact_loading_hooks(
    load_pickle_fn=_sentinel("load_pickle"),
    apply_loaded_fn=_sentinel("apply_loaded"),
    progress_factory_fn=_sentinel("progress"),
    progress_write=_sentinel("progress_write"),
    format_bytes_fn=lambda value: f"{value}B",
    render_progress_bar_fn=lambda a, b: f"{a}/{b}",
)
assert isinstance(artifact_loading_hooks, ArtifactLoadingHooks)

with tempfile.TemporaryDirectory() as tmp:
    lazy_path = Path(tmp) / "soma_vs.pkl"
    lazy_path.write_bytes(b"payload")
    result_view_hooks = build_result_view_hooks(
        read_json_if_present_fn=_sentinel("read_json"),
        standard_result_artifact_sizes_fn=_sentinel("artifact_sizes"),
        local_sync_artifact_is_usable_fn=_sentinel("usable"),
        sync_deferred_artifact_fn=_sentinel("sync_deferred"),
        load_pickle_fn=_sentinel("load_pickle"),
        set_lazy_artifact_path_fn=_sentinel("set_path"),
        format_bytes_fn=lambda value: f"{value}B",
        progress_write=_sentinel("progress_write"),
    )
    assert isinstance(result_view_hooks, ResultViewHooks)
    assert "Deferred soma_vs" in result_view_hooks.local_lazy_notice_fn("soma_vs", lazy_path)
    assert "stays remote" in result_view_hooks.remote_lazy_notice_fn("soma_vs", lazy_path)

timings = {}
json_poll_hooks = build_remote_json_poll_hooks(
    cfg,
    timings,
    run_ssh_shell_fn=lambda config, command, timeout_s=None: command_runner(config, command, timeout_s=timeout_s),
    record_timing_fn=_record_timing,
)
assert isinstance(json_poll_hooks, RemoteJSONPollHooks)
json_poll_hooks.run_command_fn("poll-json", 12.0)
json_poll_hooks.record_timing_fn("poll_s", 1.25)
assert timing_calls[-1][0] is timings
assert timing_calls[-1][1] == "poll_s"

remote_session_hooks = build_remote_job_session_hooks(
    timings,
    ensure_remote_git_ref_available_fn=_sentinel("git_ref"),
    run_remote_preflight_fn=_sentinel("preflight"),
    ensure_remote_helper_cache_fn=_sentinel("helper_cache"),
    helper_cache_lookup_fn=lambda config: config is cfg,
    cleanup_stale_allocations_fn=_sentinel("cleanup_alloc"),
    ensure_cached_remote_allocation_fn=_sentinel("cached_alloc"),
    record_timing_fn=_record_timing,
    progress_write=_sentinel("progress_write"),
)
assert isinstance(remote_session_hooks, RemoteJobSessionHooks)
assert remote_session_hooks.helper_cache_hit_fn(cfg) is True

remote_submit_hooks = build_remote_job_submit_hooks(
    timings,
    run_ssh_shell_fn=lambda config, command: command_runner(config, command),
    heartbeat_timeout_s_fn=lambda config: 120,
    record_timing_fn=_record_timing,
)
assert isinstance(remote_submit_hooks, RemoteJobSubmitHooks)
assert remote_submit_hooks.heartbeat_timeout_s_fn(cfg) == 120

run_artifact_hooks = build_remote_run_artifact_hooks(
    timings,
    sync_remote_result_dir_resilient_fn=_sentinel("sync_resilient"),
    sync_remote_result_dir_fn=_sentinel("sync"),
    run_paramiko_shell_fn=_sentinel("run_paramiko"),
    build_remote_result_listing_command_fn=_sentinel("listing"),
    local_result_dir_has_loadable_payload_fn=_sentinel("has_payload"),
    local_result_dir_has_diagnostics_fn=_sentinel("has_diag"),
    standard_result_artifact_sizes_fn=_sentinel("sizes"),
    synthesize_partial_sync_summary_fn=_sentinel("partial_summary"),
    compact_remote_poll_events_fn=_sentinel("compact"),
    read_json_if_present_fn=_sentinel("read_json"),
    progress_write=_sentinel("progress_write"),
    record_timing_fn=_record_timing,
)
assert isinstance(run_artifact_hooks, RemoteRunArtifactHooks)
run_artifact_hooks.record_timing_fn("sync_s", 2.0)
assert timing_calls[-1][1] == "sync_s"

with tempfile.TemporaryDirectory() as tmp:
    local_result_dir = Path(tmp)
    poll_helper_calls = []

    def _refresh_remote_heartbeat(config, heartbeat_path, *, warn=False):
        refresh_calls.append((heartbeat_path, warn))
        return True

    def _build_remote_poll_command(config, **kwargs):
        poll_calls.append(("build", kwargs))
        return "poll-shell"

    def _poll_remote_json_status(command, **kwargs):
        poll_helper_calls.append((command, kwargs))
        return {"state": "RUNNING", "done": False}

    def _run_ssh_shell(config, command, timeout_s=None):
        cancel_calls.append((command, timeout_s))
        return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="ok", stderr="")

    def _sync_remote_result_dir(config, **kwargs):
        sync_calls.append(kwargs)
        return subprocess.CompletedProcess(args=["sync"], returncode=0, stdout="sync out", stderr="sync err")

    run_monitor_hooks = build_remote_run_monitor_hooks(
        effective_config={"remote_poll_json_retries": 4},
        remote_job_heartbeat_path="job-heartbeat",
        allocation_heartbeat_path="alloc-heartbeat",
        remote_repo_root=PurePosixPath("/remote/repo"),
        remote_result_dir=PurePosixPath("/remote/result"),
        remote_helper_dir=PurePosixPath("/remote/helper"),
        notebook_timings=timings,
        submission={"job_id": "123", "wrapper_dir": "/tmp/w", "worktree_path": "/tmp/tree"},
        local_result_dir=local_result_dir,
        refresh_remote_heartbeat_fn=_refresh_remote_heartbeat,
        build_remote_poll_command_fn=_build_remote_poll_command,
        poll_remote_json_status_fn=_poll_remote_json_status,
        remote_json_poll_hooks_fn=lambda config, notebook_timings: "JSON-HOOKS",
        remote_poll_command_timeout_s_fn=lambda config: 45.0,
        run_ssh_shell_fn=_run_ssh_shell,
        build_remote_cancel_command_fn=lambda *, job_id: f"cancel {job_id}",
        sync_remote_result_dir_fn=_sync_remote_result_dir,
        record_timing_fn=_record_timing,
        remote_status_has_artifacts_fn=lambda status: True,
        progress_bar_factory_fn=_sentinel("progress_bar"),
        filter_live_log_line_fn=lambda kind, line: line,
        progress_write=_sentinel("progress_write"),
    )
    assert isinstance(run_monitor_hooks, RemoteRunMonitorHooks)
    run_monitor_hooks.refresh_remote_leases_fn(warn=True)
    status = run_monitor_hooks.poll_status_fn(refresh_heartbeat=True, include_logs=False, include_sacct=True)
    assert status["state"] == "RUNNING"
    assert refresh_calls[0] == ("job-heartbeat", True)
    assert refresh_calls[1] == ("alloc-heartbeat", True)
    assert poll_helper_calls[-1][0] == "poll-shell"
    assert poll_helper_calls[-1][1]["hooks"] == "JSON-HOOKS"
    run_monitor_hooks.cancel_job_fn()
    assert cancel_calls[-1][0] == "cancel 123"
    run_monitor_hooks.sync_partial_artifacts_fn()
    assert (local_result_dir / "sync_stdout.txt").read_text() == "sync out"
    assert sync_calls[-1]["remote_result_dir"] == PurePosixPath("/remote/result")

    sweep_monitor_hooks = build_remote_sweep_monitor_hooks(
        effective_config={"remote_poll_json_retries": 2},
        remote_job_heartbeat_path="job-heartbeat",
        allocation_heartbeat_path="alloc-heartbeat",
        remote_repo_root=PurePosixPath("/remote/repo"),
        remote_sweep_root=PurePosixPath("/remote/sweep"),
        remote_helper_dir=PurePosixPath("/remote/helper"),
        notebook_timings=timings,
        submission={"job_id": "999", "wrapper_dir": "/tmp/w", "worktree_path": "/tmp/tree"},
        synced_labels={"a", "b"},
        sync_finished_items_fn=lambda status: poll_calls.append(("sync_finished", status)),
        refresh_remote_heartbeat_fn=_refresh_remote_heartbeat,
        build_remote_poll_command_fn=_build_remote_poll_command,
        poll_remote_json_status_fn=_poll_remote_json_status,
        remote_json_poll_hooks_fn=lambda config, notebook_timings: "JSON-HOOKS-SWEEP",
        remote_poll_command_timeout_s_fn=lambda config: 30.0,
        run_ssh_shell_fn=_run_ssh_shell,
        build_remote_cancel_command_fn=lambda *, job_id: f"cancel {job_id}",
        progress_write=_sentinel("progress_write"),
    )
    assert isinstance(sweep_monitor_hooks, RemoteSweepMonitorHooks)
    sweep_status = sweep_monitor_hooks.poll_status_fn(refresh_heartbeat=False, include_sacct=False)
    assert sweep_status["state"] == "RUNNING"
    assert sweep_monitor_hooks.synced_count_fn() == 2
    sweep_monitor_hooks.cancel_job_fn()
    assert cancel_calls[-1][0] == "cancel 999"

    sweep_artifact_hooks = build_remote_sweep_artifact_hooks(
        refresh_remote_leases_fn=lambda: refresh_calls.append(("refresh", False)),
        notebook_timings=timings,
        sync_remote_result_dir_fn=_sentinel("sync_dir"),
        sync_remote_sweep_compact_items_fn=_sentinel("sync_compact"),
        read_json_if_present_fn=_sentinel("read_json"),
        recover_local_sweep_summary_fn=_sentinel("recover_summary"),
        remote_sweep_metadata_files_fn=_sentinel("metadata_files"),
        remote_sweep_item_sync_files_fn=_sentinel("sync_files"),
        remote_sweep_item_diagnostic_files_fn=_sentinel("diag_files"),
        local_sweep_item_sync_complete_fn=_sentinel("item_complete"),
        local_result_dir_has_diagnostics_fn=_sentinel("has_diag"),
        progress_write=_sentinel("progress_write"),
        record_timing_fn=_record_timing,
    )
    assert isinstance(sweep_artifact_hooks, RemoteSweepArtifactHooks)
    sweep_artifact_hooks.refresh_remote_leases_fn()
    sweep_artifact_hooks.record_timing_fn("bulk_sync_s", 3.0)
    assert refresh_calls[-1][0] == "refresh"
    assert timing_calls[-1][1] == "bulk_sync_s"

print("olfactorybulb notebook runtime hooks: OK")
