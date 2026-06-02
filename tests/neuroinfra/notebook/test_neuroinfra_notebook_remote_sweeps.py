"""Focused tests for generic notebook remote sweep workflow helpers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import subprocess
import tempfile

from neuroinfra.notebooks.remote_sweeps import RemoteSweepWorkflowHooks, execute_remote_sweep_workflow


def main() -> None:
    progress_messages = []
    merge_calls = []
    persist_calls = []

    sweep_plan = {
        "path": "gaba_tau2_ms",
        "values": [36.0, 50.0],
        "items": [
            {"index": 0, "label": "item0", "value": 36.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 36.0}},
            {"index": 1, "label": "item1", "value": 50.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 50.0}},
        ],
        "paramset": "GammaSignature",
        "grid": None,
    }
    result = execute_remote_sweep_workflow(
        {"runner_backend": "slurm_remote"},
        sweep_plan=sweep_plan,
        sweep_label="demo_sweep",
        timestamp="2026-06-02T12-00-00",
        local_sweep_dir=Path("/tmp/demo-sweep"),
        local_runs_dir=Path("/tmp/demo-sweep/item_runs"),
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_git_ref="abcdef1234567890",
        remote_sweeps_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
        remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/demo_sweep"),
        manifest_items=[
            {"index": 0, "label": "item0", "value": 36.0, "result_dir": "/remote/item0", "command": ["python3", "run0.py"]},
            {"index": 1, "label": "item1", "value": 50.0, "result_dir": "/remote/item1", "command": ["python3", "run1.py"]},
        ],
        manifest_json='[{"label":"item0"},{"label":"item1"}]',
        max_concurrent=2,
        remote_metadata={"runner_backend": "slurm_remote"},
        hooks=RemoteSweepWorkflowHooks(
            prepare_remote_session_fn=lambda config, **kwargs: SimpleNamespace(
                effective_config=dict(config),
                remote_metadata={**kwargs["remote_metadata"], "preflight_cached": True},
                notebook_timings={},
                preflight_completed=subprocess.CompletedProcess(["preflight"], 0, stdout="ok\n", stderr=""),
                remote_helper_dir=PurePosixPath("/remote/cache"),
                allocation_info={"job_id": "12345"},
                allocation_heartbeat_path="/remote/allocation.touch",
            ),
            upload_manifest_fn=lambda config: {"sweep_manifest_path": "/remote/manifest.json"},
            build_submit_shell_fn=lambda config, remote_helper_dir: "submit-shell",
            submit_remote_job_fn=lambda config, *, submit_shell, local_output_dir: SimpleNamespace(
                completed=subprocess.CompletedProcess(["ssh"], 0, stdout='{"job_id":"77"}\n', stderr=""),
                submission={"job_id": "77"},
                json_error=None,
                job_heartbeat_path="/remote/job.touch",
                heartbeat_timeout_s=60,
            ),
            monitor_remote_job_fn=lambda **kwargs: (
                kwargs["synced_labels"].add("item0")
                or kwargs["item_status_by_label"].update({"item0": {"returncode": 0}, "item1": {"returncode": 1}})
                or SimpleNamespace(final_status={"ok": True, "state": "COMPLETED"})
            ),
            finalize_remote_artifacts_fn=lambda config, **kwargs: SimpleNamespace(
                final_sync=subprocess.CompletedProcess(["sync"], 0, stdout="", stderr=""),
                sweep_summary={"failed_items": [{"label": "item1"}]},
                item_status_by_label={"item0": {"returncode": 0}, "item1": {"returncode": 1}},
            ),
            finalize_local_items_fn=lambda **kwargs: (
                [
                    {"label": "item0", "result": {"ok": True}, "run": "run0"},
                    {"label": "item1", "result": None, "run": None},
                ],
                ["item1"],
                {"item1": "failed to load"},
            ),
            persist_sweep_fn=lambda sweep, **kwargs: persist_calls.append((sweep, kwargs)) or Path("/tmp/demo-sweep"),
            merge_sweep_info_payload_fn=lambda sweep_dir, payload: merge_calls.append((Path(sweep_dir), payload)),
            summarize_status_fn=lambda status: {"state": status.get("state")},
            timing_summary_text_fn=lambda timings: "sync_s=1.00s",
            progress_write=lambda message: progress_messages.append(message),
        ),
    )
    assert result["path"] == "gaba_tau2_ms"
    assert result["partial"] is True
    assert result["failed_labels"] == ["item1"]
    assert result["failed_without_result"] == ["item1"]
    assert result["missing_labels"] == ["item1"]
    assert result["load_errors"] == {"item1": "failed to load"}
    assert merge_calls
    assert merge_calls[0][1]["remote"]["job_id"] == "77"
    assert merge_calls[0][1]["remote"]["final_status"] == {"state": "COMPLETED"}
    assert len(persist_calls) == 2
    assert any("Submitted sweep job 77" in message for message in progress_messages)
    assert any("Remote sweep returned partial results" in message for message in progress_messages)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            execute_remote_sweep_workflow(
                {"runner_backend": "slurm_remote"},
                sweep_plan=sweep_plan,
                sweep_label="failed_sweep",
                timestamp="2026-06-02T12-00-00",
                local_sweep_dir=Path(tmp_dir) / "sweep",
                local_runs_dir=Path(tmp_dir) / "sweep" / "item_runs",
                remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
                remote_git_ref="abcdef1234567890",
                remote_sweeps_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
                remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/failed_sweep"),
                manifest_items=[],
                manifest_json="[]",
                max_concurrent=1,
                remote_metadata={"runner_backend": "slurm_remote"},
                hooks=RemoteSweepWorkflowHooks(
                    prepare_remote_session_fn=lambda config, **kwargs: SimpleNamespace(
                        effective_config=dict(config),
                        remote_metadata=dict(kwargs["remote_metadata"]),
                        notebook_timings={},
                        preflight_completed=subprocess.CompletedProcess(["preflight"], 2, stdout="", stderr="bad preflight\n"),
                        remote_helper_dir=None,
                        allocation_info={},
                        allocation_heartbeat_path=None,
                    ),
                    upload_manifest_fn=lambda config: {},
                    build_submit_shell_fn=lambda config, remote_helper_dir: "submit-shell",
                    submit_remote_job_fn=lambda **kwargs: None,
                    monitor_remote_job_fn=lambda **kwargs: None,
                    finalize_remote_artifacts_fn=lambda **kwargs: None,
                    finalize_local_items_fn=lambda **kwargs: ([], [], {}),
                    persist_sweep_fn=lambda *args, **kwargs: None,
                    merge_sweep_info_payload_fn=lambda *args, **kwargs: None,
                    summarize_status_fn=lambda status: status,
                    timing_summary_text_fn=lambda timings: "",
                    progress_write=lambda _message: None,
                ),
            )
            raise AssertionError("expected remote sweep preflight failure to raise")
        except RuntimeError as exc:
            assert "Remote sweep preflight failed." in str(exc)

    print("neuroinfra notebook remote sweeps: OK")


if __name__ == "__main__":
    main()
