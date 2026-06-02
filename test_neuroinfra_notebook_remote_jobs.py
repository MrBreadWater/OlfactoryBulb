"""Focused tests for generic notebook remote-session and submit helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
import subprocess

from neuroinfra.notebooks.remote_jobs import (
    RemoteJobSessionHooks,
    RemoteJobSubmitHooks,
    prepare_remote_job_session,
    submit_remote_json_job,
)


def main() -> None:
    timing_keys: list[str] = []
    timings: dict[str, float] = {}
    helper_cache_calls = []
    cleanup_calls = []
    allocation_calls = []

    def _record_timing(key: str, _started: float) -> None:
        timing_keys.append(key)
        timings[key] = timings.get(key, 0.0) + 1.0

    session = prepare_remote_job_session(
        {"slurm_reuse_allocation": True},
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_git_ref="abcdef1234567890",
        remote_metadata={"runner_backend": "slurm_remote"},
        preflight_message="[Sol remote] Running remote preflight checks...",
        notebook_timings=timings,
        hooks=RemoteJobSessionHooks(
            ensure_remote_git_ref_available_fn=lambda config, **kwargs: config.setdefault("git_checked", True),
            run_remote_preflight_fn=lambda config, **kwargs: (
                subprocess.CompletedProcess(["preflight"], 0, stdout="ok\n", stderr=""),
                True,
            ),
            ensure_remote_helper_cache_fn=lambda config: helper_cache_calls.append(dict(config)) or PurePosixPath("/remote/cache"),
            helper_cache_hit_fn=lambda config: config.get("git_checked", False),
            cleanup_stale_allocations_fn=lambda config, *, remote_helper_dir=None: cleanup_calls.append(
                (dict(config), remote_helper_dir)
            ) or [{"job_id": "old-1"}],
            ensure_cached_remote_allocation_fn=lambda config, *, remote_helper_dir=None: allocation_calls.append(
                (dict(config), remote_helper_dir)
            ) or {
                "job_id": "12345",
                "heartbeat_path": "/remote/allocation.touch",
                "state": "RUNNING",
                "reason": "",
                "location": "cn001",
                "manual": False,
            },
            record_timing_fn=_record_timing,
            progress_write=lambda _message: None,
            perf_counter_fn=lambda: 1.0,
        ),
    )
    assert session.preflight_completed.returncode == 0
    assert session.remote_helper_dir == PurePosixPath("/remote/cache")
    assert session.allocation_heartbeat_path == "/remote/allocation.touch"
    assert session.effective_config["slurm_allocation_job_id"] == "12345"
    assert session.remote_metadata["preflight_cached"] is True
    assert session.remote_metadata["remote_helper_cache_hit"] is True
    assert session.remote_metadata["stale_allocation_cleanup_count"] == 1
    assert session.remote_metadata["allocation_state"] == "RUNNING"
    assert session.remote_metadata["allocation_location"] == "cn001"
    assert session.notebook_timings is timings
    assert timing_keys == [
        "git_publish_s",
        "preflight_s",
        "helper_cache_s",
        "allocation_cleanup_s",
        "allocation_wait_s",
    ]
    assert helper_cache_calls and cleanup_calls and allocation_calls

    helper_cache_calls.clear()
    cleanup_calls.clear()
    allocation_calls.clear()
    timing_keys.clear()
    timings = {}
    failed_session = prepare_remote_job_session(
        {},
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_git_ref="abcdef1234567890",
        remote_metadata={},
        preflight_message="[Sol remote] Running remote preflight checks...",
        notebook_timings=timings,
        hooks=RemoteJobSessionHooks(
            ensure_remote_git_ref_available_fn=lambda *_args, **_kwargs: None,
            run_remote_preflight_fn=lambda *_args, **_kwargs: (
                subprocess.CompletedProcess(["preflight"], 2, stdout="", stderr="bad preflight\n"),
                False,
            ),
            ensure_remote_helper_cache_fn=lambda _config: (_ for _ in ()).throw(AssertionError("should not run helper cache")),
            helper_cache_hit_fn=lambda _config: False,
            cleanup_stale_allocations_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("should not run cleanup")
            ),
            ensure_cached_remote_allocation_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("should not run allocation wait")
            ),
            record_timing_fn=_record_timing,
            progress_write=lambda _message: None,
            perf_counter_fn=lambda: 1.0,
        ),
    )
    assert failed_session.preflight_completed.returncode == 2
    assert failed_session.remote_helper_dir is None
    assert failed_session.allocation_info == {}
    assert failed_session.remote_metadata["preflight_cached"] is False
    assert "helper_cache_s" not in failed_session.notebook_timings

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        submit_timing_keys: list[str] = []

        def _submit_timing(key: str, _started: float) -> None:
            submit_timing_keys.append(key)

        submit_result = submit_remote_json_job(
            {"remote_poll_interval_s": 1.0},
            submit_shell="submit shell",
            local_output_dir=output_dir,
            hooks=RemoteJobSubmitHooks(
                run_ssh_shell_fn=lambda _config, _shell: subprocess.CompletedProcess(
                    ["ssh"], 0, stdout='{"job_id":"123","heartbeat_path":"/remote/job.touch"}\n', stderr=""
                ),
                heartbeat_timeout_s_fn=lambda _config: 30,
                record_timing_fn=_submit_timing,
                perf_counter_fn=lambda: 1.0,
            ),
        )
        assert submit_result.completed.returncode == 0
        assert submit_result.submission == {"job_id": "123", "heartbeat_path": "/remote/job.touch"}
        assert submit_result.job_heartbeat_path == "/remote/job.touch"
        assert submit_result.heartbeat_timeout_s == 30
        assert (output_dir / "submit_stdout.txt").read_text().strip() == '{"job_id":"123","heartbeat_path":"/remote/job.touch"}'
        assert (output_dir / "submit_stderr.txt").read_text() == ""
        assert submit_timing_keys == ["submit_s"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        invalid = submit_remote_json_job(
            {},
            submit_shell="submit shell",
            local_output_dir=tmp_dir,
            hooks=RemoteJobSubmitHooks(
                run_ssh_shell_fn=lambda _config, _shell: subprocess.CompletedProcess(
                    ["ssh"], 0, stdout="not json\n", stderr=""
                ),
                heartbeat_timeout_s_fn=lambda _config: 45,
                record_timing_fn=lambda *_args, **_kwargs: None,
                perf_counter_fn=lambda: 1.0,
            ),
        )
        assert invalid.completed.returncode == 0
        assert invalid.submission is None
        assert invalid.json_error is not None
        assert invalid.job_heartbeat_path is None
        assert invalid.heartbeat_timeout_s is None

    print("neuroinfra notebook remote jobs: OK")


if __name__ == "__main__":
    main()
