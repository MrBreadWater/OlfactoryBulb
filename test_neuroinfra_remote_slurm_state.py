"""Smoke tests for standardized remote Slurm state and preflight helpers."""

from __future__ import annotations

import subprocess
from pathlib import PurePosixPath

import neuroinfra.remote.slurm_state as slurm_state
import obgpu_experiment_helpers as hlp


def _completed(stdout: str, *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh", "bash", "-lc", "test"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def main() -> None:
    cfg = hlp.build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        remote_conda_activate_cmd="source activate OBGPU",
    )
    remote_repo_root = PurePosixPath("/remote/OlfactoryBulb")

    assert slurm_state.REMOTE_SLURM_TERMINAL_OK == hlp._REMOTE_SLURM_TERMINAL_OK
    assert slurm_state.REMOTE_SLURM_TERMINAL_FAIL == hlp._REMOTE_SLURM_TERMINAL_FAIL
    assert slurm_state.normalize_slurm_state("running+") == "RUNNING"
    assert slurm_state.normalize_slurm_state(" pending ") == "PENDING"
    assert hlp._normalize_slurm_state("completed+") == "COMPLETED"

    preflight_command = slurm_state.build_remote_preflight_command(
        remote_repo_root=remote_repo_root
    )
    assert preflight_command == hlp._build_remote_preflight_command(
        remote_repo_root=remote_repo_root
    )
    assert "command -v sbatch" in preflight_command
    assert "command -v srun" in preflight_command
    assert "test -d /remote/OlfactoryBulb" in preflight_command

    cache_key = slurm_state.remote_preflight_cache_key(
        connection_key=hlp._paramiko_connection_key(cfg),
        remote_repo_root=remote_repo_root,
        remote_conda_activate_cmd="source activate OBGPU",
        helper_signature=hlp._remote_helper_signature(),
    )
    assert cache_key == hlp._remote_preflight_cache_key(cfg, remote_repo_root)
    assert len(cache_key) == 16

    cache: dict[str, object] = {}
    preflight_calls: list[str] = []

    def _fake_preflight_run(command: str) -> subprocess.CompletedProcess[str]:
        preflight_calls.append(command)
        return _completed("ok\n")

    completed_1, cached_1 = slurm_state.run_remote_preflight_cached(
        cache=cache,
        cache_key="cache-key",
        remote_repo_root=remote_repo_root,
        command=preflight_command,
        run_command=_fake_preflight_run,
    )
    completed_2, cached_2 = slurm_state.run_remote_preflight_cached(
        cache=cache,
        cache_key="cache-key",
        remote_repo_root=remote_repo_root,
        command=preflight_command,
        run_command=_fake_preflight_run,
    )
    assert completed_1.returncode == 0 and completed_2.returncode == 0
    assert cached_1 is False and cached_2 is True
    assert len(preflight_calls) == 1

    original_run_ssh_shell = hlp._run_ssh_shell
    try:
        wrapper_calls: list[str] = []
        hlp._LIVE_REMOTE_PREFLIGHTS.clear()

        def _fake_wrapper_run(_config, command, check=False):
            wrapper_calls.append(command)
            assert check is False
            return _completed("ok\n")

        hlp._run_ssh_shell = _fake_wrapper_run
        wrapper_1, wrapper_cached_1 = hlp._run_remote_preflight_cached(
            cfg,
            remote_repo_root=remote_repo_root,
        )
        wrapper_2, wrapper_cached_2 = hlp._run_remote_preflight_cached(
            cfg,
            remote_repo_root=remote_repo_root,
        )
        assert wrapper_1.returncode == 0 and wrapper_2.returncode == 0
        assert wrapper_cached_1 is False and wrapper_cached_2 is True
        assert wrapper_calls == [preflight_command]
    finally:
        hlp._run_ssh_shell = original_run_ssh_shell
        hlp._LIVE_REMOTE_PREFLIGHTS.clear()

    listing_command = slurm_state.build_remote_result_listing_command(
        remote_result_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label")
    )
    assert listing_command == hlp._build_remote_result_listing_command(
        remote_result_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label")
    )
    assert "find" in listing_command
    assert "test_label" in listing_command

    cancel_command = slurm_state.build_remote_cancel_command(job_id="12345")
    assert cancel_command == "scancel 12345"
    assert cancel_command == hlp._build_remote_cancel_command(job_id="12345")

    running = slurm_state.query_remote_slurm_job_state(
        job_id="12345",
        run_command=lambda command: (
            _completed("RUNNING|pcc080\n__SACCT__\n12345|RUNNING\n")
            if "squeue -j 12345" in command
            else _completed("")
        ),
    )
    assert running == {"state": "RUNNING", "reason": "", "location": "pcc080"}

    pending = slurm_state.query_remote_slurm_job_state(
        job_id="12345",
        run_command=lambda _command: _completed("PENDING|Priority\n__SACCT__\n"),
    )
    assert pending == {"state": "PENDING", "reason": "Priority", "location": ""}

    matched_sacct = slurm_state.query_remote_slurm_job_state(
        job_id="12345",
        run_command=lambda _command: _completed("__SACCT__\n12345.batch|RUNNING\n12345|FAILED+\n"),
    )
    assert matched_sacct == {"state": "FAILED", "reason": "", "location": ""}

    fallback_sacct = slurm_state.query_remote_slurm_job_state(
        job_id="12345",
        run_command=lambda _command: _completed("__SACCT__\n12345.batch|COMPLETED\n"),
    )
    assert fallback_sacct == {"state": "COMPLETED", "reason": "", "location": ""}

    unknown = slurm_state.query_remote_slurm_job_state(
        job_id="12345",
        run_command=lambda _command: _completed(""),
    )
    assert unknown == {"state": "UNKNOWN", "reason": "", "location": ""}

    original_run_ssh_shell = hlp._run_ssh_shell
    try:
        query_calls: list[str] = []

        def _fake_query_run(_config, command, check=False):
            query_calls.append(command)
            assert check is False
            return _completed("RUNNING|pcc081\n__SACCT__\n12345|RUNNING\n")

        hlp._run_ssh_shell = _fake_query_run
        wrapper_state = hlp._query_remote_slurm_job_state(cfg, "12345")
        assert wrapper_state == {"state": "RUNNING", "reason": "", "location": "pcc081"}
        assert len(query_calls) == 1
        assert "sacct -j 12345" in query_calls[0]
    finally:
        hlp._run_ssh_shell = original_run_ssh_shell

    assert slurm_state.remote_status_has_artifacts({"stdout_exists": True}) is True
    assert slurm_state.remote_status_has_artifacts({"bootstrap_exists": True}) is True
    assert slurm_state.remote_status_has_artifacts({"stdout_exists": False}) is False
    assert slurm_state.remote_status_has_artifacts(None) is False
    assert hlp._remote_status_has_artifacts({"stderr_exists": True}) is True

    print("neuroinfra remote slurm state smoke test: OK")


if __name__ == "__main__":
    main()
