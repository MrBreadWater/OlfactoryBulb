"""Smoke tests for standardized remote allocation-cache helpers."""

from __future__ import annotations

from pathlib import PurePosixPath

import neuroinfra.remote.allocation_cache as allocation_cache
import obgpu_experiment_helpers as hlp


def main() -> None:
    cfg = hlp.build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        remote_heartbeat_timeout_s=180,
        slurm_partition="debug",
        slurm_account="lab",
        slurm_time="01:00:00",
        slurm_gpus=1,
        slurm_cpus_per_task=8,
        slurm_mem="64G",
        slurm_extra_args=["--constraint=cascadelake"],
        remote_conda_activate_cmd="source activate OBGPU",
        remote_runtime_profiles=[{"name": "portable"}],
        remote_fallback_conda_activate_cmd="source activate OBGPU-portable",
        remote_fast_node_feature="cascadelake",
        remote_mechanism_profile="fast",
        remote_fallback_mechanism_profile="portable",
        slurm_allocation_name="obgpu_notebook_alloc",
    )

    runtime_cfg = allocation_cache.allocation_runtime_config(cfg)
    assert runtime_cfg == hlp._slurm_allocation_runtime_config(cfg)
    assert runtime_cfg["remote_host"] == "user@host"
    assert runtime_cfg["runner_backend"] == "sol_slurm"

    signature = allocation_cache.allocation_signature(
        connection_key=hlp._paramiko_connection_key(cfg),
        results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        partition="debug",
        account="lab",
        time_limit="01:00:00",
        gpus=1,
        cpus_per_task=8,
        mem="64G",
        extra_args=["--constraint=cascadelake"],
        remote_conda_activate_cmd="source activate OBGPU",
        remote_runtime_profiles=[{"name": "portable"}],
        remote_fallback_conda_activate_cmd="source activate OBGPU-portable",
        remote_fast_node_feature="cascadelake",
        remote_mechanism_profile="fast",
        remote_fallback_mechanism_profile="portable",
        name="obgpu_notebook_alloc",
    )
    assert signature == hlp._slurm_allocation_signature(cfg)

    cache_key = allocation_cache.allocation_cache_key(signature)
    assert cache_key == hlp._slurm_allocation_cache_key(cfg)
    assert len(cache_key) == 16

    manual = allocation_cache.manual_allocation_record("12345")
    assert manual == {
        "job_id": "12345",
        "cached": False,
        "manual": True,
        "state": "",
        "reason": "",
        "location": "",
    }
    assert manual == hlp._ensure_cached_remote_slurm_allocation(
        {"slurm_allocation_job_id": "12345"},
        remote_helper_dir=None,
    )

    disabled = allocation_cache.disabled_allocation_record()
    assert disabled == {
        "job_id": None,
        "cached": False,
        "manual": False,
        "state": "",
        "reason": "",
        "location": "",
    }
    assert disabled == hlp._ensure_cached_remote_slurm_allocation(
        {"slurm_reuse_allocation": False},
        remote_helper_dir=None,
    )

    record = allocation_cache.allocation_record(
        job_id="12345",
        cache_key=cache_key,
        allocation_root="/remote/OlfactoryBulb/results/notebook_runs/.obgpu-allocations/abc",
        batch_script="/remote/OlfactoryBulb/results/notebook_runs/.obgpu-allocations/abc/allocation_job.sh",
        heartbeat_path="/remote/OlfactoryBulb/results/notebook_runs/.obgpu-allocations/abc/notebook-heartbeat.txt",
        heartbeat_timeout_s=180,
        slurm_log_pattern="/remote/OlfactoryBulb/results/notebook_runs/.obgpu-allocations/abc/allocation-%j.out",
        name="obgpu_notebook_alloc_abc",
        cached=True,
        manual=False,
        config=runtime_cfg,
        state="RUNNING",
        reason="",
        location="pcc080",
    )
    assert record["job_id"] == "12345"
    assert record["cache_key"] == cache_key
    assert record["manual"] is False
    assert record["cached"] is True
    assert record["location"] == "pcc080"
    assert record["config"] == runtime_cfg

    print("neuroinfra remote allocation cache smoke test: OK")


if __name__ == "__main__":
    main()
