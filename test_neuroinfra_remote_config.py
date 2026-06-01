"""Smoke tests for standardized remote config normalization helpers."""

from __future__ import annotations

import neuroinfra.remote.config as remote_config
import obgpu_experiment_helpers as hlp


def main() -> None:
    cfg = {
        "remote_host": "user@cluster.example",
        "ssh_options": ["-o", "Port=2223"],
        "remote_ssh_command_timeout_s": 42,
        "remote_ssh_exec_timeout_s": 17,
        "remote_ssh_upload_timeout_s": None,
        "remote_poll_command_timeout_s": None,
        "ssh_connect_retries": 7,
        "ssh_connect_retry_backoff_s": 2.5,
        "remote_heartbeat_timeout_s": 155,
    }

    assert remote_config.require_remote_host(cfg) == "user@cluster.example"
    assert remote_config.resolve_remote_endpoint(cfg) == ("cluster.example", 2223, "user")
    assert remote_config.remote_connection_key(cfg) == "user@cluster.example:2223"
    assert remote_config.connect_retry_count(cfg) == 7
    assert remote_config.connect_retry_backoff_s(cfg) == 2.5
    assert remote_config.heartbeat_timeout_s(cfg) == 155
    assert remote_config.ssh_command_timeout_s(cfg) == 42.0
    assert remote_config.ssh_exec_timeout_s(cfg) == 17.0
    assert remote_config.ssh_upload_timeout_s(cfg) == 42.0
    assert remote_config.poll_command_timeout_s(cfg) == 42.0

    zero_cfg = {
        "remote_host": "user@cluster.example",
        "ssh_options": [],
        "remote_ssh_command_timeout_s": 0,
        "remote_ssh_exec_timeout_s": 0,
        "remote_ssh_upload_timeout_s": 0,
        "remote_poll_command_timeout_s": None,
    }
    assert remote_config.ssh_command_timeout_s(zero_cfg) is None
    assert remote_config.ssh_exec_timeout_s(zero_cfg) is None
    assert remote_config.ssh_upload_timeout_s(zero_cfg) is None
    assert remote_config.poll_command_timeout_s(zero_cfg) == 60.0

    built = remote_config.build_remote_slurm_config(
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        default_remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        sweep_sync_live=True,
        sweep_sync_voltage_summary=True,
        sweep_live_sync_max_items_per_poll=3,
        remote_allow_paramiko_reauth=True,
        runner_backend="sol_slurm",
    )
    assert built["runner_backend"] == "sol_slurm"
    assert built["remote_results_root"] == "/remote/OlfactoryBulb/results/notebook_runs"
    assert built["remote_mpi_exec"] == "srun --mpi=pmix_v4 --cpu-bind=none"
    assert built["sweep_sync_live"] is True
    assert built["sweep_sync_voltage_summary"] is True
    assert built["sweep_live_sync_max_items_per_poll"] == 3
    assert built["remote_allow_paramiko_reauth"] is True
    assert built["ssh_transport"] == "paramiko"

    wrapper_cfg = hlp.build_slurm_remote_config(
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        sweep_sync_live=True,
        sweep_sync_voltage_summary=True,
        sweep_live_sync_max_items_per_poll=3,
        remote_allow_paramiko_reauth=True,
    )
    expected_wrapper_cfg = remote_config.build_remote_slurm_config(
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        default_remote_mpi_exec=hlp.default_remote_mpi_exec(),
        sweep_sync_live=True,
        sweep_sync_voltage_summary=True,
        sweep_live_sync_max_items_per_poll=3,
        remote_allow_paramiko_reauth=True,
    )
    assert wrapper_cfg == expected_wrapper_cfg
    assert hlp._require_remote_host(cfg) == remote_config.require_remote_host(cfg)
    assert hlp._remote_endpoint(cfg) == remote_config.resolve_remote_endpoint(cfg)
    assert hlp._paramiko_connection_key(cfg) == remote_config.remote_connection_key(cfg)
    assert hlp._paramiko_connect_retry_count(cfg) == remote_config.connect_retry_count(cfg)
    assert hlp._paramiko_connect_retry_backoff_s(cfg) == remote_config.connect_retry_backoff_s(cfg)
    assert hlp._remote_heartbeat_timeout_s(cfg) == remote_config.heartbeat_timeout_s(cfg)
    assert hlp._remote_ssh_command_timeout_s(cfg) == remote_config.ssh_command_timeout_s(cfg)
    assert hlp._remote_ssh_exec_timeout_s(cfg) == remote_config.ssh_exec_timeout_s(cfg)
    assert hlp._remote_ssh_upload_timeout_s(cfg) == remote_config.ssh_upload_timeout_s(cfg)
    assert hlp._remote_poll_command_timeout_s(cfg) == remote_config.poll_command_timeout_s(cfg)

    print("neuroinfra remote config smoke test: OK")


if __name__ == "__main__":
    main()
