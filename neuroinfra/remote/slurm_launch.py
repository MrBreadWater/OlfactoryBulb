"""Reusable local-side argv and command builders for remote Slurm helpers."""

from __future__ import annotations

from base64 import b64encode
import json
from pathlib import Path, PurePosixPath
import shlex
from typing import Any, Iterable, Sequence

from .command_launch import (
    build_remote_python_file_command,
    build_remote_python_inline_command,
    remote_helper_script_path,
)


def _append_optional_arg(argv: list[str], flag: str, value: object | None) -> None:
    """Append one flag/value pair when the value is present."""
    if value not in (None, ""):
        argv.extend([flag, str(value)])


def _append_optional_int_arg(argv: list[str], flag: str, value: object | None) -> None:
    """Append one integer flag/value pair when the value is present."""
    if value not in (None, ""):
        argv.extend([flag, str(int(value))])


def _append_sbatch_args(argv: list[str], values: Iterable[object]) -> None:
    """Append repeated ``--sbatch-arg=...`` values."""
    for extra in values:
        argv.append("--sbatch-arg={}".format(str(extra)))


def build_remote_helper_launch_command(
    helper_script_source: Path,
    argv: Sequence[str],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> str:
    """Return the remote shell command that launches one helper script."""
    helper_path = remote_helper_script_path(remote_helper_dir, helper_script_source.name)
    if helper_path is not None:
        return build_remote_python_file_command(helper_path, list(argv))
    return build_remote_python_inline_command(helper_script_source, list(argv))


def build_submit_slurm_allocation_argv(
    *,
    allocation_root: PurePosixPath,
    allocation_name: str,
    heartbeat_timeout_s: int,
    partition: str | None = None,
    account: str | None = None,
    time_limit: str | None = None,
    mem: str | None = None,
    gpus: int | None = None,
    cpus_per_task: int | None = None,
    sbatch_args: Iterable[object] = (),
) -> list[str]:
    """Build argv for ``submit_slurm_allocation.py``."""
    argv = [
        "--alloc-root",
        allocation_root.as_posix(),
        "--name",
        str(allocation_name),
        "--heartbeat-timeout-s",
        str(max(int(heartbeat_timeout_s), 0)),
    ]
    _append_optional_arg(argv, "--partition", partition)
    _append_optional_arg(argv, "--account", account)
    _append_optional_arg(argv, "--mem", mem)
    _append_optional_arg(argv, "--time", time_limit)
    _append_optional_int_arg(argv, "--gpus", gpus)
    _append_optional_int_arg(argv, "--cpus-per-task", cpus_per_task)
    _append_sbatch_args(argv, sbatch_args)
    return argv


def build_cleanup_stale_allocations_argv(
    *,
    cleanup_root: PurePosixPath,
    default_timeout_s: int,
) -> list[str]:
    """Build argv for ``cleanup_stale_allocations.py``."""
    return [
        "--root",
        cleanup_root.as_posix(),
        "--default-timeout-s",
        str(max(int(default_timeout_s), 0)),
    ]


def build_allocation_discovery_command(allocation_json_path: PurePosixPath) -> str:
    """Build the shell command that prints one allocation JSON when it exists."""
    quoted = shlex.quote(allocation_json_path.as_posix())
    return f"if test -f {quoted}; then cat {quoted}; fi"


def build_submit_sol_run_argv(
    *,
    remote_repo_root: PurePosixPath,
    remote_results_root: PurePosixPath,
    label: str,
    benchmark_command: Sequence[str],
    repo_mode: str,
    remote_mpi_exec: str,
    conda_activate_cmd: str,
    heartbeat_timeout_s: int,
    runtime_profiles: Sequence[dict[str, Any]] = (),
    fallback_conda_activate_cmd: str | None = None,
    fast_node_feature: str | None = None,
    mechanism_profile: str | None = None,
    fallback_mechanism_profile: str | None = None,
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = False,
    remote_git_remote: str = "origin",
    allocation_job_id: str | None = None,
    step_ntasks: int | None = None,
    partition: str | None = None,
    account: str | None = None,
    time_limit: str | None = None,
    mem: str | None = None,
    gpus: int | None = None,
    cpus_per_task: int | None = None,
    sbatch_args: Iterable[object] = (),
) -> list[str]:
    """Build argv for ``submit_sol_run.py``."""
    benchmark_b64 = b64encode(json.dumps(list(benchmark_command)).encode("utf-8")).decode("ascii")
    argv = [
        "--repo-root",
        remote_repo_root.as_posix(),
        "--results-base",
        remote_results_root.as_posix(),
        "--label",
        str(label),
        "--benchmark-command-b64",
        benchmark_b64,
        "--repo-mode",
        str(repo_mode),
        "--mpi-exec",
        str(remote_mpi_exec),
        "--conda-activate-cmd",
        str(conda_activate_cmd),
        "--heartbeat-timeout-s",
        str(max(int(heartbeat_timeout_s), 0)),
    ]
    if runtime_profiles:
        profiles_b64 = b64encode(json.dumps(list(runtime_profiles), sort_keys=True).encode("utf-8")).decode("ascii")
        argv.extend(["--runtime-profiles-b64", profiles_b64])
    _append_optional_arg(argv, "--fallback-conda-activate-cmd", fallback_conda_activate_cmd)
    _append_optional_arg(argv, "--fast-node-feature", fast_node_feature)
    _append_optional_arg(argv, "--mechanism-profile", mechanism_profile)
    _append_optional_arg(argv, "--fallback-mechanism-profile", fallback_mechanism_profile)
    _append_optional_arg(argv, "--git-ref", remote_git_ref)
    if remote_git_fetch:
        argv.append("--git-fetch")
        argv.extend(["--git-remote", str(remote_git_remote)])
    _append_optional_arg(argv, "--allocation-job-id", allocation_job_id)
    if allocation_job_id not in (None, "") and step_ntasks not in (None, ""):
        argv.extend(["--step-ntasks", str(max(int(step_ntasks), 1))])
    _append_optional_arg(argv, "--partition", partition)
    _append_optional_arg(argv, "--account", account)
    _append_optional_arg(argv, "--time", time_limit)
    _append_optional_arg(argv, "--mem", mem)
    _append_optional_int_arg(argv, "--gpus", gpus)
    _append_optional_int_arg(argv, "--cpus-per-task", cpus_per_task)
    _append_sbatch_args(argv, sbatch_args)
    return argv


def build_poll_sol_run_argv(
    *,
    job_id: str,
    remote_result_dir: PurePosixPath,
    wrapper_dir: str | None = None,
    remote_repo_root: PurePosixPath | None = None,
    worktree_path: str | None = None,
    include_sacct: bool = True,
    include_tails: bool = True,
) -> list[str]:
    """Build argv for ``poll_sol_run.py``."""
    argv = [
        "--job-id",
        str(job_id),
        "--result-dir",
        remote_result_dir.as_posix(),
    ]
    _append_optional_arg(argv, "--wrapper-dir", wrapper_dir)
    if worktree_path not in (None, ""):
        if remote_repo_root is None:
            raise ValueError("remote_repo_root is required when worktree_path is set")
        argv.extend(
            [
                "--repo-root",
                remote_repo_root.as_posix(),
                "--worktree-path",
                str(worktree_path),
            ]
        )
    if not include_sacct:
        argv.append("--skip-sacct")
    if not include_tails:
        argv.append("--skip-tails")
    return argv
