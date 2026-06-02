"""Focused tests for generic notebook remote single-run workflow helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import subprocess

from neuroinfra.notebooks.remote_runs import RemoteRunWorkflowHooks, execute_remote_run_workflow


def main() -> None:
    write_calls = []
    progress_messages = []

    result = execute_remote_run_workflow(
        {"runner_backend": "slurm_remote", "remote_defer_soma_vs_sync": False},
        label="demo",
        timestamp="2026-06-02T12-00-00",
        local_result_dir=Path("/tmp/demo-remote-run"),
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_git_ref="abcdef1234567890",
        remote_command=["python3", "remote_driver.py"],
        remote_metadata={"runner_backend": "slurm_remote", "remote_git_ref": "abcdef1234567890"},
        submit_shell="submit-initial",
        hooks=RemoteRunWorkflowHooks(
            prepare_remote_session_fn=lambda config, **kwargs: SimpleNamespace(
                effective_config={**config, "slurm_allocation_job_id": "12345"},
                remote_metadata={**kwargs["remote_metadata"], "preflight_cached": True},
                notebook_timings={},
                preflight_completed=subprocess.CompletedProcess(["preflight"], 0, stdout="ok\n", stderr=""),
                remote_helper_dir=PurePosixPath("/remote/cache"),
                allocation_info={"job_id": "12345"},
                allocation_heartbeat_path="/remote/allocation.touch",
            ),
            refresh_submission_payload_fn=lambda config, *, remote_helper_dir=None: (
                ["python3", "remote_driver.py", "--cached"],
                {"refreshed": True},
                "submit-cached",
            ),
            upload_runtime_payload_fn=lambda config: {"benchmark_overrides_file": "/remote/overrides/demo.json"},
            submit_remote_job_fn=lambda config, *, submit_shell, local_output_dir: SimpleNamespace(
                completed=subprocess.CompletedProcess(["ssh"], 0, stdout='{"job_id":"42","result_dir":"/remote/result"}\n', stderr=""),
                submission={"job_id": "42", "result_dir": "/remote/result", "wrapper_dir": "/remote/wrapper"},
                json_error=None,
                job_heartbeat_path="/remote/job.touch",
                heartbeat_timeout_s=60,
            ),
            monitor_remote_job_fn=lambda **kwargs: SimpleNamespace(
                final_status={"ok": True, "state": "COMPLETED"},
                poll_transcript=[{"state": "RUNNING"}, {"state": "COMPLETED"}],
            ),
            build_final_sync_plan_fn=lambda effective_config, final_status: (("summary.json",), ("soma_vs.npz",)),
            finalize_remote_artifacts_fn=lambda config, **kwargs: SimpleNamespace(
                final_status={"ok": True, "state": "COMPLETED"},
                sync_warning=None,
                stdout_text="stdout\n",
                stderr_text="",
                bootstrap_text="",
                slurm_text="",
                remote_listing_text="",
                remote_git_commit="abc123",
                remote_git_ref="refs/heads/main",
                returncode=0,
                summary={"ok": True},
                compact_poll_events=[{"state": "COMPLETED"}],
                poll_events_path=Path("/tmp/demo-remote-run/remote_poll_events.json"),
                artifact_sizes={"summary.json": 12},
                deferred_remote_artifacts=("soma_vs.npz",),
            ),
            write_run_info_fn=lambda *args, **kwargs: write_calls.append((args, kwargs)),
            summarize_submit_response_fn=lambda submission: {"job_id": submission["job_id"]},
            summarize_status_fn=lambda status: {"state": status.get("state")},
            timing_summary_text_fn=lambda timings: "submit_s=1.00s",
            build_return_value_fn=lambda **kwargs: kwargs,
            shell_join_fn=lambda command: " ".join(command),
            progress_write=lambda message: progress_messages.append(message),
        ),
    )
    assert result["summary"] == {"ok": True}
    assert result["command"] == ["python3", "remote_driver.py", "--cached"]
    assert write_calls
    remote_payload = write_calls[-1][1]["extra_payload"]["remote"]
    assert remote_payload["job_id"] == "42"
    assert remote_payload["submit_response"] == {"job_id": "42"}
    assert remote_payload["final_status"] == {"state": "COMPLETED"}
    assert remote_payload["deferred_remote_artifacts"] == ["soma_vs.npz"]
    assert any("Submitted job 42." in message for message in progress_messages)
    assert any("Notebook pipeline timings" in message for message in progress_messages)

    with tempfile.TemporaryDirectory() as tmp_dir:
        write_calls = []
        try:
            execute_remote_run_workflow(
                {"runner_backend": "slurm_remote"},
                label="failed",
                timestamp="2026-06-02T12-00-00",
                local_result_dir=Path(tmp_dir) / "failed-run",
                remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
                remote_git_ref="abcdef1234567890",
                remote_command=["python3", "remote_driver.py"],
                remote_metadata={"runner_backend": "slurm_remote"},
                submit_shell="submit-initial",
                hooks=RemoteRunWorkflowHooks(
                    prepare_remote_session_fn=lambda config, **kwargs: SimpleNamespace(
                        effective_config=dict(config),
                        remote_metadata=dict(kwargs["remote_metadata"]),
                        notebook_timings={},
                        preflight_completed=subprocess.CompletedProcess(["preflight"], 2, stdout="", stderr="bad preflight\n"),
                        remote_helper_dir=None,
                        allocation_info={},
                        allocation_heartbeat_path=None,
                    ),
                    refresh_submission_payload_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("should not refresh payload after failed preflight")
                    ),
                    upload_runtime_payload_fn=lambda _config: (_ for _ in ()).throw(
                        AssertionError("should not upload after failed preflight")
                    ),
                    submit_remote_job_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("should not submit after failed preflight")
                    ),
                    monitor_remote_job_fn=lambda **kwargs: None,
                    build_final_sync_plan_fn=lambda effective_config, final_status: (None, ()),
                    finalize_remote_artifacts_fn=lambda config, **kwargs: None,
                    write_run_info_fn=lambda *args, **kwargs: write_calls.append((args, kwargs)),
                    summarize_submit_response_fn=lambda submission: submission,
                    summarize_status_fn=lambda status: status,
                    timing_summary_text_fn=lambda timings: "",
                    build_return_value_fn=lambda **kwargs: kwargs,
                    shell_join_fn=lambda command: " ".join(command),
                    progress_write=lambda _message: None,
                ),
            )
            raise AssertionError("expected remote preflight failure to raise")
        except RuntimeError as exc:
            assert "Remote Sol preflight failed." in str(exc)
        assert write_calls
        assert write_calls[0][1]["completed"].returncode == 2

    print("neuroinfra notebook remote runs: OK")


if __name__ == "__main__":
    main()
