"""Smoke tests for extracted remote single-run final artifact handling."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from neuroinfra.remote.run_artifacts import (
    RemoteRunArtifactHooks,
    finalize_remote_run_artifacts,
)


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh", "bash", "-lc", "test"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _read_json_if_present(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _artifact_sizes(result_dir: Path) -> dict[str, int]:
    return {
        path.name: int(path.stat().st_size)
        for path in sorted(Path(result_dir).iterdir())
        if path.is_file()
    }


def _default_hooks(
    *,
    sync_remote_result_dir_resilient_fn,
    sync_remote_result_dir_fn=None,
    run_paramiko_shell_fn=None,
    local_result_dir_has_loadable_payload_fn=None,
    local_result_dir_has_diagnostics_fn=None,
    compact_remote_poll_events_fn=None,
    synthesize_partial_sync_summary_fn=None,
    progress_messages=None,
    timing_calls=None,
    sleep_calls=None,
) -> RemoteRunArtifactHooks:
    progress_messages = progress_messages if progress_messages is not None else []
    timing_calls = timing_calls if timing_calls is not None else []
    sleep_calls = sleep_calls if sleep_calls is not None else []
    return RemoteRunArtifactHooks(
        sync_remote_result_dir_resilient_fn=sync_remote_result_dir_resilient_fn,
        sync_remote_result_dir_fn=sync_remote_result_dir_fn
        or (lambda *_args, **_kwargs: _completed(returncode=0)),
        run_paramiko_shell_fn=run_paramiko_shell_fn
        or (lambda _config, _command: _completed(returncode=0, stdout="remote-listing")),
        build_remote_result_listing_command_fn=lambda remote_result_dir: f"ls::{remote_result_dir.as_posix()}",
        local_result_dir_has_loadable_payload_fn=local_result_dir_has_loadable_payload_fn
        or (lambda result_dir: (Path(result_dir) / "input_times.pkl").exists()),
        local_result_dir_has_diagnostics_fn=local_result_dir_has_diagnostics_fn
        or (
            lambda result_dir: any(
                (Path(result_dir) / name).exists()
                for name in ("stdout.txt", "stderr.txt", "bootstrap.log")
            )
            or bool(list(Path(result_dir).glob("slurm-*.out")))
        ),
        standard_result_artifact_sizes_fn=_artifact_sizes,
        synthesize_partial_sync_summary_fn=synthesize_partial_sync_summary_fn
        or (
            lambda result_dir, *, label, timestamp, config: {
                "label": label,
                "timestamp": timestamp,
                "partial_sync": True,
                "paramset": config.get("paramset"),
            }
        ),
        compact_remote_poll_events_fn=compact_remote_poll_events_fn
        or (lambda poll_transcript: list(poll_transcript)),
        read_json_if_present_fn=_read_json_if_present,
        progress_write=progress_messages.append,
        record_timing_fn=lambda key, started: timing_calls.append((key, started)),
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
        perf_counter_fn=lambda: 1.0,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        success_dir = tmpdir_path / "success"

        def _success_sync(_config, *, local_result_dir, **_kwargs):
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            (local_result_dir / "summary.json").write_text(json.dumps({"label": "run-1"}))
            (local_result_dir / "stdout.txt").write_text("stdout")
            (local_result_dir / "stderr.txt").write_text("stderr")
            (local_result_dir / "bootstrap.log").write_text("bootstrap")
            (local_result_dir / "slurm-1.out").write_text("slurm")
            (local_result_dir / "git_ref.txt").write_text("refs/heads/main\n")
            (local_result_dir / "git_commit.txt").write_text("deadbeef\n")
            return _completed(returncode=0)

        success_progress: list[str] = []
        success_timings: list[tuple[str, float]] = []
        success_result = finalize_remote_run_artifacts(
            {"paramset": "GammaSignature"},
            final_status={"ok": True, "state": "COMPLETED"},
            local_result_dir=success_dir,
            remote_result_dir=PurePosixPath("/remote/run-1"),
            wrapper_dir="/remote/wrapper",
            label="run-1",
            timestamp="20260601_120000",
            notebook_timings={},
            poll_transcript=[{"state": "RUNNING"}],
            include_files=("summary.json",),
            deferred_remote_artifacts=("soma_vs.npz",),
            hooks=_default_hooks(
                sync_remote_result_dir_resilient_fn=_success_sync,
                progress_messages=success_progress,
                timing_calls=success_timings,
            ),
        )
        assert success_result.returncode == 0
        assert success_result.summary == {"label": "run-1"}
        assert success_result.remote_git_ref == "refs/heads/main"
        assert success_result.remote_git_commit == "deadbeef"
        assert success_result.poll_events_path is not None
        assert success_result.poll_events_path.exists()
        assert success_result.deferred_remote_artifacts == ("soma_vs.npz",)
        assert success_timings == [("sync_s", 1.0)]
        assert any("Remote sync finished" in message for message in success_progress)
        print("remote run artifact success path: OK")

        partial_dir = tmpdir_path / "partial"
        (partial_dir / "input_times.pkl").parent.mkdir(parents=True, exist_ok=True)
        (partial_dir / "input_times.pkl").write_text("placeholder")
        partial_result = finalize_remote_run_artifacts(
            {"paramset": "GammaSignature"},
            final_status={"ok": True, "state": "COMPLETED"},
            local_result_dir=partial_dir,
            remote_result_dir=PurePosixPath("/remote/run-2"),
            wrapper_dir=None,
            label="run-2",
            timestamp="20260601_120001",
            notebook_timings={},
            poll_transcript=[],
            include_files=None,
            deferred_remote_artifacts=(),
            hooks=_default_hooks(
                sync_remote_result_dir_resilient_fn=lambda *_args, **_kwargs: _completed(
                    returncode=1,
                    stderr="sync boom",
                ),
            ),
        )
        assert partial_result.returncode == 0
        assert partial_result.sync_warning is not None
        assert partial_result.summary is not None
        assert partial_result.summary["partial_sync"] is True
        print("remote run artifact partial-payload warning path: OK")

        diag_dir = tmpdir_path / "diag"

        def _diag_sync(_config, *, local_result_dir, **_kwargs):
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            (local_result_dir / "bootstrap.log").write_text("diag-only")
            return _completed(returncode=1, stderr="diag sync failed")

        diag_result = finalize_remote_run_artifacts(
            {"paramset": "GammaSignature"},
            final_status={"ok": True, "state": "FAILED"},
            local_result_dir=diag_dir,
            remote_result_dir=PurePosixPath("/remote/run-3"),
            wrapper_dir="/remote/wrapper",
            label="run-3",
            timestamp="20260601_120002",
            notebook_timings={},
            poll_transcript=[],
            include_files=None,
            deferred_remote_artifacts=(),
            hooks=_default_hooks(sync_remote_result_dir_resilient_fn=_diag_sync),
        )
        assert diag_result.returncode == 1
        assert diag_result.final_status is not None
        assert diag_result.final_status["ok"] is False
        assert diag_result.final_status["sync_failed"] is True
        print("remote run artifact diagnostic-only failure path: OK")

        listing_dir = tmpdir_path / "listing"
        retry_calls: list[str] = []
        listing_calls: list[str] = []
        sleep_calls: list[float] = []

        listing_result = finalize_remote_run_artifacts(
            {"paramset": "GammaSignature"},
            final_status={"ok": False, "state": "FAILED"},
            local_result_dir=listing_dir,
            remote_result_dir=PurePosixPath("/remote/run-4"),
            wrapper_dir=None,
            label="run-4",
            timestamp="20260601_120003",
            notebook_timings={},
            poll_transcript=[],
            include_files=None,
            deferred_remote_artifacts=(),
            hooks=_default_hooks(
                sync_remote_result_dir_resilient_fn=lambda *_args, **_kwargs: _completed(returncode=0),
                sync_remote_result_dir_fn=lambda *_args, **_kwargs: retry_calls.append("retry") or _completed(returncode=0),
                run_paramiko_shell_fn=lambda _config, command: listing_calls.append(command) or _completed(
                    returncode=0,
                    stdout="remote files",
                ),
                local_result_dir_has_loadable_payload_fn=lambda _result_dir: False,
                local_result_dir_has_diagnostics_fn=lambda _result_dir: False,
                sleep_calls=sleep_calls,
            ),
        )
        assert listing_result.returncode == 1
        assert retry_calls == ["retry"]
        assert listing_calls == ["ls::/remote/run-4"]
        assert listing_result.remote_listing_text == "remote files"
        assert sleep_calls == [3.0]
        print("remote run artifact retry/listing path: OK")


if __name__ == "__main__":
    main()
