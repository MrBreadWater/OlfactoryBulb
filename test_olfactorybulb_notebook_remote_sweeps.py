"""Focused tests for olfactory-bulb remote sweep notebook adapters."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import olfactorybulb.notebook_remote_sweeps as sweep_mod
from neuroinfra.notebooks.remote_jobs import RemoteJobSessionHooks, RemoteJobSubmitHooks
from neuroinfra.notebooks.remote_sweeps import execute_remote_sweep_workflow


def _read_json_if_present(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    payload_hooks = sweep_mod.RemoteSweepPayloadHooks(
        json_ready_fn=lambda value: value,
        benchmark_param_overrides_payload_fn=lambda config: (
            {"gaba_tau2_ms": config["gaba_tau2_ms"]},
            None,
        ),
        build_run_command_fn=lambda config, label, **kwargs: [
            "python3",
            f"{label}.py",
            kwargs["overrides_file"].as_posix(),
        ],
        remote_sweep_parallelism_fn=lambda config, tasks_per_item: 4,
        require_remote_host_fn=lambda config: "user@sol",
        default_remote_mpi_exec_fn=lambda: "mpiexec",
    )
    sweep_plan = {
        "path": "gaba_tau2_ms",
        "values": [36.0, 50.0],
        "items": [
            {"index": 0, "label": "item0", "value": 36.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 36.0}},
            {"index": 1, "label": "item1", "value": 50.0, "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 50.0}},
        ],
        "paramset": "GammaSignature",
    }
    (
        driver_command,
        manifest_items,
        manifest_json,
        remote_manifest_path,
        max_concurrent,
        remote_metadata,
    ) = sweep_mod.build_remote_sweep_payload(
        payload_hooks,
        {
            "runner_backend": "slurm_remote",
            "nranks": 2,
            "remote_mpi_exec": "srun",
            "remote_git_fetch": True,
            "remote_git_remote": "upstream",
            "remote_repo_mode": "bundle",
        },
        sweep_plan=sweep_plan,
        sweep_label="demo_sweep",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_sweeps_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
        remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/demo_sweep"),
        remote_git_ref="abcdef1234567890",
    )
    assert driver_command[:2] == ["python3", "/remote/OlfactoryBulb/tools/remote/remote_sweep_driver.py"]
    assert remote_manifest_path.as_posix().endswith("/demo_sweep/sweep_manifest.submit.json")
    assert max_concurrent == 4
    assert len(manifest_items) == 2
    assert manifest_items[0]["overrides"] == {"gaba_tau2_ms": 36.0}
    assert manifest_items[1]["command"][1] == "item1.py"
    assert '"label": "item0"' in manifest_json
    assert remote_metadata["remote_host"] == "user@sol"
    assert remote_metadata["sweep_parallelism"] == 4
    assert remote_metadata["sweep_items"] == 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        upload_calls = []
        progress_messages = []
        merge_calls = []
        write_run_info_calls = []

        def _sync_remote_result_dir(
            config,
            *,
            remote_result_dir,
            local_result_dir,
            expected_files,
            include_files,
        ):
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            (local_result_dir / "summary.json").write_text(
                json.dumps({"label": local_result_dir.name, "ok": True})
            )
            (local_result_dir / "stdout.txt").write_text("stdout\n")
            (local_result_dir / "stderr.txt").write_text("")
            return subprocess.CompletedProcess(["sync"], 0, stdout="", stderr="")

        original_monitor_remote_sweep = sweep_mod.monitor_remote_sweep
        original_finalize_remote_sweep_artifacts = sweep_mod.finalize_remote_sweep_artifacts
        try:
            def _fake_monitor_remote_sweep(*, job_id, poll_interval_s, log_poll_interval_s, live_status, hooks):
                hooks.sync_finished_items_fn(
                    {
                        "progress_payload": {
                            "pending_labels": [],
                            "running_items": [],
                            "finished_items": [
                                {"label": "item0", "result_dir": "/remote/item0", "returncode": 0},
                            ],
                        }
                    }
                )
                return SimpleNamespace(final_status={"ok": True, "state": "COMPLETED"})

            def _fake_finalize_remote_sweep_artifacts(
                config,
                *,
                final_status,
                local_sweep_dir,
                local_runs_dir,
                remote_sweep_root,
                sweep_label,
                manifest_items,
                item_status_by_label,
                hooks,
            ):
                (Path(local_runs_dir) / "item1").mkdir(parents=True, exist_ok=True)
                (Path(local_runs_dir) / "item1" / "stdout.txt").write_text("item1 stdout\n")
                (Path(local_runs_dir) / "item1" / "stderr.txt").write_text("item1 stderr\n")
                return SimpleNamespace(
                    final_sync=subprocess.CompletedProcess(["sync"], 0, stdout="", stderr=""),
                    sweep_summary={"failed_items": [{"label": "item1"}]},
                    item_status_by_label={
                        **item_status_by_label,
                        "item0": {"returncode": 0},
                        "item1": {"returncode": 1},
                    },
                )

            sweep_mod.monitor_remote_sweep = _fake_monitor_remote_sweep
            sweep_mod.finalize_remote_sweep_artifacts = _fake_finalize_remote_sweep_artifacts

            workflow_hooks = sweep_mod.build_remote_sweep_workflow_hooks(
                sweep_mod.NotebookRemoteSweepWorkflowBuilderHooks(
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
                            stdout=json.dumps({"job_id": "77"}) + "\n",
                            stderr="",
                        ),
                        heartbeat_timeout_s_fn=lambda config: 60,
                        record_timing_fn=lambda key, started: notebook_timings.__setitem__(key, 2.0),
                    ),
                    remote_sweep_monitor_hooks_fn=lambda **kwargs: SimpleNamespace(
                        sync_finished_items_fn=kwargs["sync_finished_items_fn"],
                    ),
                    remote_sweep_artifact_hooks_fn=lambda **kwargs: SimpleNamespace(),
                    build_remote_submit_command_fn=lambda config, **kwargs: (
                        f"submit::{kwargs['label']}::{kwargs['remote_mpi_exec']}::{kwargs['remote_helper_dir']}"
                    ),
                    upload_remote_text_file_fn=lambda config, **kwargs: upload_calls.append(kwargs),
                    refresh_remote_heartbeat_fn=lambda config, heartbeat_path, warn=False: None,
                    should_sync_remote_sweep_finished_items_fn=lambda config, pending_count, running_count: True,
                    sync_remote_result_dir_fn=_sync_remote_result_dir,
                    remote_sweep_item_sync_files_fn=lambda config: ("summary.json",),
                    local_sync_artifact_is_usable_fn=lambda path: Path(path).exists(),
                    synthesize_partial_sync_summary_fn=lambda result_dir, **kwargs: {
                        "label": kwargs["label"],
                        "partial": True,
                    },
                    persist_sweep_fn=lambda sweep, **kwargs: Path(kwargs["sweep_dir"]),
                    merge_sweep_info_payload_fn=lambda sweep_dir, payload: merge_calls.append((Path(sweep_dir), payload)),
                    summarize_status_fn=lambda status: {"state": status.get("state")},
                    timing_summary_text_fn=lambda timings: "",
                    write_run_info_fn=lambda *args, **kwargs: write_run_info_calls.append((args, kwargs)),
                    load_run_record_fn=lambda path: SimpleNamespace(label=Path(path).name, result_dir=Path(path)),
                    load_result_fn=lambda run: {"result_dir": str(run.result_dir), "loaded": True},
                    resolve_local_sweep_item_dir_fn=lambda runs_dir, label: (
                        Path(runs_dir) / label if (Path(runs_dir) / label).exists() else None
                    ),
                    json_ready_fn=lambda value: value,
                    read_json_if_present_fn=_read_json_if_present,
                    progress_write=lambda message: progress_messages.append(message),
                    record_timing_fn=lambda notebook_timings, key, started: notebook_timings.__setitem__(key, 3.0),
                    perf_counter_fn=lambda: 100.0,
                    default_remote_mpi_exec_fn=lambda: "mpiexec",
                ),
                sweep_label="demo_sweep",
                timestamp="2026-06-02T12-00-00",
                remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
                remote_sweeps_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
                remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/demo_sweep"),
                remote_driver_command=driver_command,
                remote_git_ref="abcdef1234567890",
                manifest_json=manifest_json,
                manifest_items=manifest_items,
                remote_manifest_path=remote_manifest_path,
                max_concurrent=max_concurrent,
                local_runs_dir=tmp / "item_runs",
            )
            sweep = execute_remote_sweep_workflow(
                {
                    "runner_backend": "slurm_remote",
                    "remote_poll_interval_s": 1.0,
                    "remote_log_poll_interval_s": 5.0,
                    "remote_live_status": True,
                },
                sweep_plan=sweep_plan,
                sweep_label="demo_sweep",
                timestamp="2026-06-02T12-00-00",
                local_sweep_dir=tmp / "sweep",
                local_runs_dir=tmp / "item_runs",
                remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
                remote_git_ref="abcdef1234567890",
                remote_sweeps_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
                remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/demo_sweep"),
                manifest_items=manifest_items,
                manifest_json=manifest_json,
                max_concurrent=max_concurrent,
                remote_metadata=remote_metadata,
                hooks=workflow_hooks,
            )
        finally:
            sweep_mod.monitor_remote_sweep = original_monitor_remote_sweep
            sweep_mod.finalize_remote_sweep_artifacts = original_finalize_remote_sweep_artifacts

        assert upload_calls and upload_calls[0]["remote_path"] == remote_manifest_path
        assert write_run_info_calls and len(write_run_info_calls) == 2
        assert merge_calls and merge_calls[0][1]["remote"]["job_id"] == "77"
        assert sweep["partial"] is False
        assert sweep["failed_labels"] == ["item1"]
        assert sweep["recovered_failed_labels"] == ["item1"]
        assert sweep["missing_labels"] == []
        assert sweep["items"][0]["result"]["loaded"] is True
        assert sweep["items"][1]["result"]["loaded"] is True
        assert (tmp / "item_runs" / "item1" / "summary.json").exists()
        assert any("Submitting remote sweep batch job" in message for message in progress_messages)

    print("olfactorybulb notebook remote sweeps: OK")


if __name__ == "__main__":
    main()
