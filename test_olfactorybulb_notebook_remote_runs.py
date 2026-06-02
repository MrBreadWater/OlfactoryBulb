"""Focused tests for olfactory-bulb remote single-run notebook adapters."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import olfactorybulb.notebook_remote_runs as run_mod
from neuroinfra.notebooks.remote_jobs import RemoteJobSessionHooks, RemoteJobSubmitHooks
from neuroinfra.notebooks.remote_runs import execute_remote_run_workflow


def main() -> None:
    command_calls: list[dict[str, object]] = []

    payload_hooks = run_mod.RemoteRunPayloadHooks(
        build_run_command_fn=lambda config, label, **kwargs: (
            command_calls.append(
                {
                    "config": dict(config),
                    "label": label,
                    **kwargs,
                }
            )
            or ["python3", f"{label}.py", str(kwargs["overrides_file"])]
        ),
        build_remote_submit_command_fn=lambda config, **kwargs: (
            f"submit::{kwargs['label']}::{kwargs['step_ntasks']}::{kwargs['remote_helper_dir']}"
        ),
        require_remote_host_fn=lambda config: "user@sol",
        default_remote_mpi_exec_fn=lambda: "mpiexec",
    )

    remote_command, remote_metadata, submit_shell = run_mod.build_remote_run_payload(
        payload_hooks,
        {
            "runner_backend": "slurm_remote",
            "nranks": 1,
            "slurm_allocation_job_id": "12345",
            "remote_git_fetch": True,
            "remote_git_remote": "upstream",
            "remote_repo_mode": "bundle",
        },
        label="demo_run",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        remote_git_ref="abcdef1234567890",
        remote_helper_dir=PurePosixPath("/remote/cache"),
        overrides_file=PurePosixPath("/remote/overrides/demo_run.json"),
        param_overrides={"gaba_tau2_ms": 36.0},
        input_spec_file=None,
    )
    assert remote_command == ["python3", "demo_run.py", "/remote/overrides/demo_run.json"]
    assert command_calls[0]["include_mpi_launcher"] is False
    assert remote_metadata["remote_host"] == "user@sol"
    assert remote_metadata["slurm_allocation_job_id"] == "12345"
    assert remote_metadata["remote_git_fetch"] is True
    assert submit_shell == "submit::demo_run::1::/remote/cache"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        upload_calls = []
        write_run_info_calls = []
        progress_messages = []

        original_monitor_remote_run = run_mod.monitor_remote_run
        original_finalize_remote_run_artifacts = run_mod.finalize_remote_run_artifacts
        try:
            run_mod.monitor_remote_run = lambda **kwargs: SimpleNamespace(
                final_status={"ok": True, "state": "COMPLETED"},
                poll_transcript=[{"state": "COMPLETED"}],
            )
            run_mod.finalize_remote_run_artifacts = lambda config, **kwargs: SimpleNamespace(
                final_status=kwargs["final_status"],
                sync_completed=subprocess.CompletedProcess(["sync"], 0, stdout="", stderr=""),
                sync_warning=None,
                stdout_text="stdout\n",
                stderr_text="",
                bootstrap_text="",
                slurm_text="",
                remote_listing_text="",
                remote_git_commit="deadbeef",
                remote_git_ref="abcdef1234567890",
                returncode=0,
                summary={"label": "demo_run", "ok": True},
                artifact_sizes={"summary.json": 64},
                deferred_remote_artifacts=tuple(kwargs["deferred_remote_artifacts"]),
                compact_poll_events=[{"state": "COMPLETED"}],
                poll_events_path=Path(kwargs["local_result_dir"]) / "remote_poll_events.json",
            )

            workflow_hooks = run_mod.build_remote_run_workflow_hooks(
                run_mod.NotebookRemoteRunWorkflowBuilderHooks(
                    remote_job_session_hooks_fn=lambda notebook_timings: RemoteJobSessionHooks(
                        ensure_remote_git_ref_available_fn=lambda config, **kwargs: None,
                        run_remote_preflight_fn=lambda config, **kwargs: (
                            subprocess.CompletedProcess(["preflight"], 0, stdout="ok\n", stderr=""),
                            True,
                        ),
                        ensure_remote_helper_cache_fn=lambda config: PurePosixPath("/remote/cache"),
                        helper_cache_hit_fn=lambda config: True,
                        cleanup_stale_allocations_fn=lambda config, **kwargs: [],
                        ensure_cached_remote_allocation_fn=lambda config, **kwargs: {},
                        record_timing_fn=lambda key, started: notebook_timings.__setitem__(key, 1.0),
                        progress_write=lambda message: progress_messages.append(message),
                    ),
                    remote_job_submit_hooks_fn=lambda notebook_timings: RemoteJobSubmitHooks(
                        run_ssh_shell_fn=lambda config, submit_shell: subprocess.CompletedProcess(
                            ["ssh"],
                            0,
                            stdout=json.dumps({"job_id": "77", "result_dir": "/remote/results/demo_run"}) + "\n",
                            stderr="",
                        ),
                        heartbeat_timeout_s_fn=lambda config: 60,
                        record_timing_fn=lambda key, started: notebook_timings.__setitem__(key, 2.0),
                    ),
                    remote_run_monitor_hooks_fn=lambda **kwargs: SimpleNamespace(**kwargs),
                    remote_run_artifact_hooks_fn=lambda notebook_timings: SimpleNamespace(notebook_timings=notebook_timings),
                    build_remote_run_payload_fn=lambda config, **kwargs: (
                        ["python3", "remote_driver.py"],
                        {"runner_backend": "slurm_remote", "remote_host": "user@sol"},
                        "submit-shell",
                    ),
                    upload_remote_text_file_fn=lambda config, **kwargs: upload_calls.append(kwargs),
                    json_ready_fn=lambda value: value,
                    remote_fast_sync_files_fn=lambda config: ("summary.json",),
                    preferred_soma_trace_artifact_name_fn=lambda: "soma_vs.npz",
                    write_run_info_fn=lambda *args, **kwargs: write_run_info_calls.append((args, kwargs)),
                    summarize_submit_response_fn=lambda submission: {"job_id": submission.get("job_id")},
                    summarize_status_fn=lambda status: {"state": status.get("state")} if status is not None else None,
                    timing_summary_text_fn=lambda timings: "",
                    build_return_value_fn=lambda **kwargs: kwargs,
                    shell_join_fn=lambda command: " ".join(command),
                    progress_write=lambda message: progress_messages.append(message),
                    record_timing_fn=lambda notebook_timings, key, started: notebook_timings.__setitem__(key, 3.0),
                    perf_counter_fn=lambda: 100.0,
                ),
                label="demo_run",
                timestamp="2026-06-02T12-00-00",
                remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
                remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
                remote_git_ref="abcdef1234567890",
                remote_overrides_path=PurePosixPath("/remote/overrides/demo_run.json"),
                param_overrides={"gaba_tau2_ms": 36.0},
                input_spec_file=None,
            )

            returned = execute_remote_run_workflow(
                {
                    "runner_backend": "slurm_remote",
                    "remote_defer_soma_vs_sync": True,
                    "remote_poll_interval_s": 1.0,
                    "remote_log_poll_interval_s": 5.0,
                    "remote_live_status": True,
                    "remote_live_logs": True,
                },
                record_config={"paramset": "GammaSignature"},
                label="demo_run",
                timestamp="2026-06-02T12-00-00",
                local_result_dir=tmp / "demo_run",
                remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
                remote_git_ref="abcdef1234567890",
                remote_command=["python3", "remote_driver.py"],
                remote_metadata={"runner_backend": "slurm_remote"},
                submit_shell="submit-shell",
                hooks=workflow_hooks,
            )
        finally:
            run_mod.monitor_remote_run = original_monitor_remote_run
            run_mod.finalize_remote_run_artifacts = original_finalize_remote_run_artifacts

        assert upload_calls and upload_calls[0]["remote_path"] == PurePosixPath("/remote/overrides/demo_run.json")
        assert upload_calls[0]["text"] == json.dumps({"gaba_tau2_ms": 36.0}, indent=2, sort_keys=True)
        assert write_run_info_calls and write_run_info_calls[0][1]["extra_payload"]["remote"]["submit_response"] == {"job_id": "77"}
        assert write_run_info_calls[0][1]["extra_payload"]["remote"]["deferred_remote_artifacts"] == ["soma_vs.npz"]
        assert returned["summary"] == {"label": "demo_run", "ok": True}
        assert returned["command"] == ["python3", "remote_driver.py"]
        assert any("Submitting Slurm job" in message for message in progress_messages)

    print("olfactorybulb notebook remote runs: OK")


if __name__ == "__main__":
    main()
