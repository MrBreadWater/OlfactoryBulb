"""Smoke tests for standardized remote Slurm helper argv and launch builders."""

from __future__ import annotations

from pathlib import PurePosixPath

import neuroinfra.remote.slurm_launch as slurm_launch
import obgpu_experiment_helpers as hlp


def main() -> None:
    helper_dir = PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test")
    submit_run_script = hlp.REPO_ROOT / "tools" / "remote" / "submit_sol_run.py"
    submit_alloc_script = hlp.REPO_ROOT / "tools" / "remote" / "submit_slurm_allocation.py"
    poll_script = hlp.REPO_ROOT / "tools" / "remote" / "poll_sol_run.py"
    cleanup_script = hlp.REPO_ROOT / "tools" / "remote" / "cleanup_stale_allocations.py"

    remote_cfg = hlp.build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        remote_conda_activate_cmd="source activate OBGPU",
        remote_git_ref="abcdef1234567890",
        slurm_partition="debug",
        slurm_account="lab",
        slurm_mem="64G",
        slurm_time="00:30:00",
        slurm_gpus=1,
        slurm_cpus_per_task=8,
        slurm_extra_args=["--constraint=cascadelake"],
    )

    submit_command, allocation_root, allocation_name = hlp._build_remote_allocation_submit_command(
        remote_cfg,
        remote_helper_dir=helper_dir,
    )
    expected_alloc_argv = slurm_launch.build_submit_slurm_allocation_argv(
        allocation_root=allocation_root,
        allocation_name=allocation_name,
        heartbeat_timeout_s=hlp._remote_heartbeat_timeout_s(remote_cfg),
        partition="debug",
        account="lab",
        time_limit="00:30:00",
        mem="64G",
        gpus=1,
        cpus_per_task=8,
        sbatch_args=["--constraint=cascadelake"],
    )
    expected_submit_command = slurm_launch.build_remote_helper_launch_command(
        submit_alloc_script,
        expected_alloc_argv,
        remote_helper_dir=helper_dir,
    )
    assert submit_command == expected_submit_command

    discover_command, discover_root, discover_name = hlp._build_remote_allocation_discovery_command(
        remote_cfg,
        remote_helper_dir=helper_dir,
    )
    assert discover_root == allocation_root
    assert discover_name == allocation_name
    assert discover_command == slurm_launch.build_allocation_discovery_command(
        allocation_root / "allocation.json"
    )

    cleanup_command = hlp._build_remote_cleanup_allocations_command(
        remote_cfg,
        remote_helper_dir=helper_dir,
    )
    expected_cleanup_argv = slurm_launch.build_cleanup_stale_allocations_argv(
        cleanup_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-allocations"),
        default_timeout_s=hlp._remote_heartbeat_timeout_s(remote_cfg),
    )
    assert cleanup_command == slurm_launch.build_remote_helper_launch_command(
        cleanup_script,
        expected_cleanup_argv,
        remote_helper_dir=helper_dir,
    )

    allocation_cfg = hlp.build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        remote_conda_activate_cmd="source activate OBGPU",
        remote_git_ref="abcdef1234567890",
        remote_runtime_profiles=[{"name": "portable"}],
        remote_fallback_conda_activate_cmd="source activate OBGPU-portable",
        remote_fast_node_feature="cascadelake",
        remote_mechanism_profile="fast",
        remote_fallback_mechanism_profile="portable",
        remote_git_fetch=True,
        remote_git_remote="origin",
        slurm_partition="debug",
        slurm_account="lab",
        slurm_time="00:30:00",
        slurm_mem="64G",
        slurm_gpus=1,
        slurm_cpus_per_task=8,
        slurm_extra_args=["--constraint=cascadelake"],
        slurm_allocation_job_id="12345",
        slurm_step_ntasks=15,
        nranks=15,
    )
    remote_submit = hlp._build_remote_submit_command(
        allocation_cfg,
        label="test_label",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        benchmark_command=["nrniv", "-mpi", "-python", "bench.py"],
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        remote_git_ref="abcdef1234567890",
        remote_helper_dir=helper_dir,
    )
    expected_submit_argv = slurm_launch.build_submit_sol_run_argv(
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        label="test_label",
        benchmark_command=["nrniv", "-mpi", "-python", "bench.py"],
        repo_mode="shared",
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        conda_activate_cmd="source activate OBGPU",
        heartbeat_timeout_s=hlp._remote_heartbeat_timeout_s(allocation_cfg),
        runtime_profiles=[{"name": "portable"}],
        fallback_conda_activate_cmd="source activate OBGPU-portable",
        fast_node_feature="cascadelake",
        mechanism_profile="fast",
        fallback_mechanism_profile="portable",
        remote_git_ref="abcdef1234567890",
        remote_git_fetch=True,
        remote_git_remote="origin",
        allocation_job_id="12345",
        step_ntasks=15,
        partition="debug",
        account="lab",
        time_limit="00:30:00",
        mem="64G",
        gpus=1,
        cpus_per_task=8,
        sbatch_args=["--constraint=cascadelake"],
    )
    assert remote_submit == slurm_launch.build_remote_helper_launch_command(
        submit_run_script,
        expected_submit_argv,
        remote_helper_dir=helper_dir,
    )

    remote_poll = hlp._build_remote_poll_command(
        allocation_cfg,
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_result_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label"),
        job_id="12345",
        wrapper_dir="/remote/OlfactoryBulb/results/notebook_runs/.obgpu-wrapper/test_label",
        worktree_path="/remote/OlfactoryBulb-worktree/test_label",
        remote_helper_dir=helper_dir,
        include_sacct=False,
        include_tails=False,
    )
    expected_poll_argv = slurm_launch.build_poll_sol_run_argv(
        job_id="12345",
        remote_result_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label"),
        wrapper_dir="/remote/OlfactoryBulb/results/notebook_runs/.obgpu-wrapper/test_label",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        worktree_path="/remote/OlfactoryBulb-worktree/test_label",
        include_sacct=False,
        include_tails=False,
    )
    assert remote_poll == slurm_launch.build_remote_helper_launch_command(
        poll_script,
        expected_poll_argv,
        remote_helper_dir=helper_dir,
    )
    assert "--skip-sacct" in remote_poll
    assert "--skip-tails" in remote_poll

    print("neuroinfra remote slurm launch smoke test: OK")


if __name__ == "__main__":
    main()
