"""Smoke tests for the extracted notebook-managed remote allocation runtime."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import PurePosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-test-neuroinfra-remote-allocation-runtime")

import obgpu_experiment_helpers as hlp
from neuroinfra.remote.allocation_runtime import RemoteAllocationRuntimeContext


def _completed(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh", "bash", "-lc", "remote-command"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _build_context(
    config: dict[str, object],
    *,
    run_ssh_shell_fn,
    query_remote_slurm_job_state_fn,
    live_slurm_allocations: dict[str, object] | None = None,
    live_remote_stale_cleanups: dict[str, object] | None = None,
    progress_log: list[str] | None = None,
) -> RemoteAllocationRuntimeContext:
    if live_slurm_allocations is None:
        live_slurm_allocations = {}
    if live_remote_stale_cleanups is None:
        live_remote_stale_cleanups = {}
    if progress_log is None:
        progress_log = []
    return RemoteAllocationRuntimeContext(
        config=config,
        live_slurm_allocations=live_slurm_allocations,
        live_remote_stale_cleanups=live_remote_stale_cleanups,
        progress_write=progress_log.append,
        connection_key_fn=lambda _cfg: "user@host:22",
        remote_results_root_fn=lambda _cfg: PurePosixPath("/remote/results"),
        poll_command_timeout_s_fn=lambda _cfg: 60.0,
        heartbeat_timeout_s_fn=lambda _cfg: 120,
        allocation_cache_key_fn=lambda _cfg: "alloc-key",
        allocation_runtime_config_fn=lambda cfg: {"remote_host": cfg.get("remote_host", "user@host")},
        build_remote_touch_command_fn=lambda heartbeat_path: f"touch::{heartbeat_path}",
        build_remote_cleanup_allocations_command_fn=lambda _cfg, _remote_helper_dir: "cleanup",
        build_remote_allocation_discovery_command_fn=lambda _cfg, _remote_helper_dir: (
            "discover",
            PurePosixPath("/remote/results/.obgpu-allocations/alloc-key"),
            "alloc-name",
        ),
        build_remote_allocation_submit_command_fn=lambda _cfg, _remote_helper_dir: (
            "submit",
            PurePosixPath("/remote/results/.obgpu-allocations/alloc-key"),
            "alloc-name",
        ),
        build_remote_cancel_command_fn=lambda job_id: f"cancel::{job_id}",
        run_ssh_shell_fn=run_ssh_shell_fn,
        query_remote_slurm_job_state_fn=query_remote_slurm_job_state_fn,
        time_fn=lambda: 123.0,
        sleep_fn=lambda _seconds: None,
    )


class _WrapperRuntimeProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def refresh_heartbeat(self, heartbeat_path, *, warn=False):
        self.calls.append(("refresh", heartbeat_path, warn))
        return "refresh-ok"

    def cleanup_stale_allocations(self, *, remote_helper_dir=None):
        self.calls.append(("cleanup", remote_helper_dir, None))
        return ["cleanup-ok"]

    def maybe_cleanup_stale_allocations(self, *, remote_helper_dir=None):
        self.calls.append(("maybe_cleanup", remote_helper_dir, None))
        return ["maybe-cleanup-ok"]

    def ensure_cached_allocation(self, *, remote_helper_dir=None):
        self.calls.append(("ensure", remote_helper_dir, None))
        return {"job_id": "123"}

    def release_allocation(self):
        self.calls.append(("release", None, None))
        return True


def main() -> None:
    manual_cfg = {"slurm_allocation_job_id": "14537854", "slurm_reuse_allocation": True}
    manual_ctx = _build_context(
        manual_cfg,
        run_ssh_shell_fn=lambda *_args, **_kwargs: _completed(),
        query_remote_slurm_job_state_fn=lambda *_args, **_kwargs: {"state": "UNKNOWN"},
    )
    manual = manual_ctx.ensure_cached_allocation()
    assert manual["manual"] is True
    assert manual["job_id"] == "14537854"
    print("allocation runtime manual reuse: OK")

    disabled_cfg = {"slurm_allocation_job_id": None, "slurm_reuse_allocation": False}
    disabled_ctx = _build_context(
        disabled_cfg,
        run_ssh_shell_fn=lambda *_args, **_kwargs: _completed(),
        query_remote_slurm_job_state_fn=lambda *_args, **_kwargs: {"state": "UNKNOWN"},
    )
    disabled = disabled_ctx.ensure_cached_allocation()
    assert disabled["job_id"] is None
    assert disabled["manual"] is False
    print("allocation runtime disabled path: OK")

    cleanup_calls: list[str] = []
    cleanup_progress: list[str] = []

    def _cleanup_run(_cfg, command, _check, _timeout_s):
        cleanup_calls.append(command)
        assert command == "cleanup"
        return _completed(stdout=json.dumps([{"job_id": "5", "action": "cancel_requested", "reason": "stale"}]))

    cleanup_ctx = _build_context(
        {"slurm_allocation_job_id": None, "slurm_reuse_allocation": True, "remote_cleanup_stale_allocations": True},
        run_ssh_shell_fn=_cleanup_run,
        query_remote_slurm_job_state_fn=lambda *_args, **_kwargs: {"state": "UNKNOWN"},
        progress_log=cleanup_progress,
    )
    cleanup_1 = cleanup_ctx.maybe_cleanup_stale_allocations()
    cleanup_2 = cleanup_ctx.maybe_cleanup_stale_allocations()
    assert cleanup_1 == cleanup_2 == [{"job_id": "5", "action": "cancel_requested", "reason": "stale"}]
    assert cleanup_calls == ["cleanup"]
    assert any("Cancelled stale reusable allocation 5" in line for line in cleanup_progress)
    print("allocation runtime stale cleanup cache: OK")

    allocation_states = iter(
        [
            {"state": "PENDING", "reason": "Resources", "location": ""},
            {"state": "RUNNING", "reason": "", "location": "pcc080"},
        ]
    )
    runtime_commands: list[str] = []
    live_allocations: dict[str, object] = {}

    def _runtime_run(_cfg, command, _check, _timeout_s):
        runtime_commands.append(command)
        if command == "discover":
            return _completed(
                stdout=json.dumps(
                    {
                        "job_id": "24680",
                        "allocation_root": "/remote/results/.obgpu-allocations/alloc-key",
                        "heartbeat_path": "/remote/results/.obgpu-allocations/alloc-key/notebook-heartbeat.txt",
                        "name": "alloc-name",
                    }
                )
            )
        if command.startswith("touch::"):
            return _completed()
        raise AssertionError(f"unexpected remote command: {command}")

    def _runtime_query(_cfg, job_id):
        assert job_id == "24680"
        return next(allocation_states)

    runtime_ctx = _build_context(
        {"slurm_allocation_job_id": None, "slurm_reuse_allocation": True, "remote_host": "user@host"},
        run_ssh_shell_fn=_runtime_run,
        query_remote_slurm_job_state_fn=_runtime_query,
        live_slurm_allocations=live_allocations,
    )
    discovered = runtime_ctx.ensure_cached_allocation()
    assert discovered["job_id"] == "24680"
    assert discovered["state"] == "RUNNING"
    assert discovered["location"] == "pcc080"
    assert live_allocations["alloc-key"]["job_id"] == "24680"
    assert runtime_commands.count("discover") == 1
    assert any(command.startswith("touch::") for command in runtime_commands)
    print("allocation runtime discovered allocation path: OK")

    release_commands: list[str] = []

    def _release_run(_cfg, command, _check, _timeout_s):
        release_commands.append(command)
        if command == "discover":
            return _completed(stdout=json.dumps({"job_id": "98765"}))
        if command == "cancel::98765":
            return _completed()
        raise AssertionError(f"unexpected release command: {command}")

    release_ctx = _build_context(
        {"slurm_allocation_job_id": None, "slurm_reuse_allocation": True},
        run_ssh_shell_fn=_release_run,
        query_remote_slurm_job_state_fn=lambda *_args, **_kwargs: {"state": "UNKNOWN"},
    )
    assert release_ctx.release_allocation() is True
    assert release_commands == ["discover", "cancel::98765"]
    print("allocation runtime release path: OK")

    original_runtime_context = hlp._remote_allocation_runtime_context
    try:
        probe = _WrapperRuntimeProbe()
        hlp._remote_allocation_runtime_context = lambda _config: probe
        assert hlp._refresh_remote_heartbeat({}, "/remote/heartbeat", warn=True) == "refresh-ok"
        assert hlp._cleanup_stale_remote_slurm_allocations({}, remote_helper_dir=PurePosixPath("/helper")) == [
            "cleanup-ok"
        ]
        assert hlp._maybe_cleanup_stale_remote_slurm_allocations(
            {},
            remote_helper_dir=PurePosixPath("/helper"),
        ) == ["maybe-cleanup-ok"]
        assert hlp._ensure_cached_remote_slurm_allocation(
            {},
            remote_helper_dir=PurePosixPath("/helper"),
        ) == {"job_id": "123"}
        assert hlp.release_remote_slurm_allocation({}) is True
        assert probe.calls == [
            ("refresh", "/remote/heartbeat", True),
            ("cleanup", PurePosixPath("/helper"), None),
            ("maybe_cleanup", PurePosixPath("/helper"), None),
            ("ensure", PurePosixPath("/helper"), None),
            ("release", None, None),
        ]
        print("allocation runtime notebook wrapper delegation: OK")
    finally:
        hlp._remote_allocation_runtime_context = original_runtime_context


if __name__ == "__main__":
    main()
