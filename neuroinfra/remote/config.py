"""Reusable remote-config normalization helpers for notebook-managed Slurm runs."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Any


def require_remote_host(config: dict[str, Any]) -> str:
    """Return the configured remote SSH target."""
    remote_host = str(config.get("remote_host") or "").strip()
    if not remote_host:
        raise ValueError("remote Slurm runner requires remote_host")
    return remote_host


def resolve_remote_endpoint(
    config: dict[str, Any],
    *,
    default_username: str | None = None,
) -> tuple[str, int, str]:
    """Resolve hostname, port, and username from one remote config."""
    host = require_remote_host(config)
    if "@" in host:
        username, hostname = host.split("@", 1)
    else:
        username = default_username or os.environ.get("USER") or os.environ.get("USERNAME") or ""
        hostname = host
    if not username:
        raise ValueError(f"Could not infer SSH username from remote_host={host!r}")

    port = 22
    options = [str(option) for option in config.get("ssh_options", [])]
    index = 0
    while index < len(options):
        option = options[index]
        if option == "-p" and index + 1 < len(options):
            port = int(options[index + 1])
            index += 2
            continue
        if option.startswith("-p") and option != "-p":
            port = int(option[2:])
            index += 1
            continue
        if option == "-o" and index + 1 < len(options):
            key_value = options[index + 1]
            if key_value.lower().startswith("port="):
                port = int(key_value.split("=", 1)[1])
            index += 2
            continue
        index += 1
    return hostname, port, username


def remote_connection_key(
    config: dict[str, Any],
    *,
    default_username: str | None = None,
) -> str:
    """Return the stable cache key for one SSH endpoint."""
    hostname, port, username = resolve_remote_endpoint(config, default_username=default_username)
    return f"{username}@{hostname}:{port}"


def connect_retry_count(config: dict[str, Any]) -> int:
    """Return how many times one fresh Paramiko connect may be retried."""
    try:
        return max(int(config.get("ssh_connect_retries", 4) or 1), 1)
    except Exception:
        return 4


def connect_retry_backoff_s(config: dict[str, Any]) -> float:
    """Return the base sleep between fresh Paramiko connect retries."""
    try:
        return max(float(config.get("ssh_connect_retry_backoff_s", 1.0) or 0.0), 0.0)
    except Exception:
        return 1.0


def heartbeat_timeout_s(config: dict[str, Any]) -> int:
    """Return the notebook heartbeat timeout used by remote watchdogs."""
    value = config.get("remote_heartbeat_timeout_s", 120)
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return 120


def ssh_command_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the per-command Paramiko shell timeout, or None when disabled."""
    value = config.get("remote_ssh_command_timeout_s", 300)
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 300.0
    if timeout <= 0:
        return None
    return timeout


def ssh_exec_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the Paramiko exec request acknowledgement timeout."""
    value = config.get("remote_ssh_exec_timeout_s", 30)
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 30.0
    if timeout <= 0:
        return None
    return timeout


def ssh_upload_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the Paramiko shell upload timeout, or None when disabled."""
    value = config.get("remote_ssh_upload_timeout_s", 120)
    if value is None:
        return ssh_command_timeout_s(config)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 120.0
    if timeout <= 0:
        return None
    return timeout


def poll_command_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the tighter timeout for lightweight remote polling commands."""
    value = config.get("remote_poll_command_timeout_s", 60)
    if value is None:
        inherited = ssh_command_timeout_s(config)
        return 60.0 if inherited is None else inherited
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 60.0
    if timeout <= 0:
        return None
    return timeout


def build_remote_slurm_config(
    *,
    remote_host: str,
    remote_repo_root: str | PurePosixPath,
    remote_results_root: str | PurePosixPath | None = None,
    remote_conda_activate_cmd: str = "source tools/setup/activate_obgpu.sh",
    remote_runtime_profiles: list[dict[str, Any]] | None = None,
    remote_fallback_conda_activate_cmd: str | None = None,
    remote_fast_node_feature: str | None = None,
    remote_mechanism_profile: str = "default",
    remote_fallback_mechanism_profile: str = "portable",
    remote_mpi_exec: str | None = None,
    default_remote_mpi_exec: str,
    slurm_partition: str | None = None,
    slurm_account: str | None = None,
    slurm_time: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_step_ntasks: int | None = None,
    slurm_mem: str | None = None,
    sweep_sync_live: bool = False,
    remote_poll_interval_s: float = 1.0,
    remote_log_poll_interval_s: float = 5.0,
    remote_live_status: bool = True,
    remote_live_logs: bool = True,
    remote_heartbeat_timeout_s: int = 120,
    remote_ssh_command_timeout_s: float | None = 300,
    remote_ssh_exec_timeout_s: float | None = 30,
    remote_ssh_upload_timeout_s: float | None = 120,
    remote_poll_command_timeout_s: float | None = 60,
    remote_cleanup_stale_allocations: bool = True,
    remote_defer_soma_vs_sync: bool = False,
    sweep_live_sync_max_items_per_poll: int = 8,
    sweep_sync_soma_vs: bool = False,
    sweep_sync_voltage_summary: bool = False,
    remote_preserve_paramiko_session: bool = True,
    remote_allow_paramiko_reauth: bool = False,
    remote_repo_mode: str = "shared",
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = False,
    remote_git_remote: str = "origin",
    slurm_allocation_job_id: str | None = None,
    slurm_reuse_allocation: bool = False,
    slurm_allocation_time: str | None = None,
    slurm_allocation_name: str | None = None,
    ssh_options: list[str] | None = None,
    slurm_extra_args: list[str] | None = None,
    ssh_connect_retries: int = 4,
    ssh_connect_retry_backoff_s: float = 1.0,
    runner_backend: str = "slurm_remote",
) -> dict[str, Any]:
    """Return a generic Paramiko-backed remote Slurm config."""
    repo_root = str(remote_repo_root)
    if remote_results_root is None:
        remote_results_root = str(PurePosixPath(repo_root) / "results" / "notebook_runs")

    return {
        "runner_backend": str(runner_backend),
        "use_corenrn": None,
        "use_gpu": None,
        "remote_host": str(remote_host),
        "remote_repo_root": repo_root,
        "remote_results_root": str(remote_results_root),
        "remote_conda_activate_cmd": str(remote_conda_activate_cmd),
        "remote_runtime_profiles": list(remote_runtime_profiles or []),
        "remote_fallback_conda_activate_cmd": None
        if remote_fallback_conda_activate_cmd in (None, "")
        else str(remote_fallback_conda_activate_cmd),
        "remote_fast_node_feature": None if remote_fast_node_feature in (None, "") else str(remote_fast_node_feature),
        "remote_mechanism_profile": str(remote_mechanism_profile or "default"),
        "remote_fallback_mechanism_profile": str(remote_fallback_mechanism_profile or "portable"),
        "remote_mpi_exec": str(remote_mpi_exec or default_remote_mpi_exec),
        "sweep_sync_live": bool(sweep_sync_live),
        "remote_poll_interval_s": float(remote_poll_interval_s),
        "remote_log_poll_interval_s": float(remote_log_poll_interval_s),
        "remote_live_status": bool(remote_live_status),
        "remote_live_logs": bool(remote_live_logs),
        "remote_heartbeat_timeout_s": int(remote_heartbeat_timeout_s),
        "remote_ssh_command_timeout_s": (
            None if remote_ssh_command_timeout_s in (None, "") else float(remote_ssh_command_timeout_s)
        ),
        "remote_ssh_exec_timeout_s": (
            None if remote_ssh_exec_timeout_s in (None, "") else float(remote_ssh_exec_timeout_s)
        ),
        "remote_ssh_upload_timeout_s": (
            None if remote_ssh_upload_timeout_s in (None, "") else float(remote_ssh_upload_timeout_s)
        ),
        "remote_poll_command_timeout_s": (
            None if remote_poll_command_timeout_s in (None, "") else float(remote_poll_command_timeout_s)
        ),
        "remote_cleanup_stale_allocations": bool(remote_cleanup_stale_allocations),
        "remote_sync_compress": True,
        "remote_defer_soma_vs_sync": bool(remote_defer_soma_vs_sync),
        "sweep_live_sync_max_items_per_poll": int(sweep_live_sync_max_items_per_poll),
        "sweep_sync_soma_vs": bool(sweep_sync_soma_vs),
        "sweep_sync_voltage_summary": bool(sweep_sync_voltage_summary),
        "remote_preserve_paramiko_session": bool(remote_preserve_paramiko_session),
        "remote_allow_paramiko_reauth": bool(remote_allow_paramiko_reauth),
        "disable_status_report": False,
        "remote_repo_mode": str(remote_repo_mode),
        "remote_git_ref": remote_git_ref,
        "remote_git_fetch": bool(remote_git_fetch),
        "remote_git_remote": str(remote_git_remote),
        "slurm_allocation_job_id": None if slurm_allocation_job_id in (None, "") else str(slurm_allocation_job_id),
        "slurm_reuse_allocation": bool(slurm_reuse_allocation),
        "slurm_allocation_time": None if slurm_allocation_time in (None, "") else str(slurm_allocation_time),
        "slurm_allocation_name": None if slurm_allocation_name in (None, "") else str(slurm_allocation_name),
        "slurm_partition": None if slurm_partition in (None, "") else str(slurm_partition),
        "slurm_account": None if slurm_account in (None, "") else str(slurm_account),
        "slurm_time": None if slurm_time in (None, "") else str(slurm_time),
        "slurm_gpus": None if slurm_gpus in (None, "") else int(slurm_gpus),
        "slurm_cpus_per_task": None if slurm_cpus_per_task in (None, "") else int(slurm_cpus_per_task),
        "slurm_step_ntasks": None if slurm_step_ntasks in (None, "") else int(max(int(slurm_step_ntasks), 1)),
        "slurm_mem": None if slurm_mem in (None, "") else str(slurm_mem),
        "slurm_extra_args": list(slurm_extra_args or []),
        "ssh_options": list(ssh_options or []),
        "ssh_transport": "paramiko",
        "ssh_keepalive_s": 30,
        "ssh_connect_retries": max(int(ssh_connect_retries), 1),
        "ssh_connect_retry_backoff_s": max(float(ssh_connect_retry_backoff_s), 0.0),
    }

