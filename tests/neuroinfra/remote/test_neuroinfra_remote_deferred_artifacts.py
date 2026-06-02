"""Smoke tests for extracted deferred remote artifact sync policy."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import pickle
import subprocess
import tempfile

from neuroinfra.remote.deferred_artifacts import (
    DeferredArtifactSyncHooks,
    sync_deferred_remote_artifact,
    sync_deferred_remote_artifact_direct,
)


def _local_sync_artifact_is_usable(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        run_info = {
            "config": {
                "runner_backend": "sol_slurm",
                "remote_host": "user@host",
                "ssh_transport": "paramiko",
            },
            "remote": {
                "remote_result_dir": "/remote/result",
            },
        }

        progress_messages: list[str] = []
        sync_calls: list[tuple[PurePosixPath, tuple[str, ...] | None, tuple[str, ...] | None]] = []
        direct_calls: list[tuple[PurePosixPath, str]] = []

        def _sync_remote_result_dir(_config, *, remote_result_dir, local_result_dir, expected_files=None, include_files=None):
            sync_calls.append((remote_result_dir, expected_files, include_files))
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(args=["sync"], returncode=1, stdout="", stderr="selected sync failed")

        def _stream_file_to_local_path(_config, *, remote_file_path, local_path, expected_bytes):
            direct_calls.append((remote_file_path.parent, remote_file_path.name))
            with open(local_path, "wb") as handle:
                pickle.dump([("MC0", [0.0, 0.1], [-65.0, -64.0])], handle)
            return subprocess.CompletedProcess(args=["direct"], returncode=0, stdout="", stderr="")

        direct_hooks = DeferredArtifactSyncHooks(
            local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
            sync_remote_result_dir_fn=_sync_remote_result_dir,
            progress_write=progress_messages.append,
            format_bytes_fn=lambda size: f"{size} B",
            direct_stream_supported_fn=lambda filename: filename == "soma_vs.pkl",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                args=["probe"], returncode=0, stdout="67\n", stderr=""
            ),
            stream_file_to_local_path_fn=_stream_file_to_local_path,
            perf_counter_fn=lambda: 1.0,
        )
        direct_result_dir = tmpdir_path / "direct-success"
        direct_result_dir.mkdir()
        direct_path = sync_deferred_remote_artifact(
            direct_result_dir,
            run_info=run_info,
            filename="soma_vs.pkl",
            hooks=direct_hooks,
        )
        assert direct_path == direct_result_dir / "soma_vs.pkl"
        assert direct_calls == [(PurePosixPath("/remote/result"), "soma_vs.pkl")]
        assert sync_calls == [
            (PurePosixPath("/remote/result"), ("soma_vs.pkl",), ("soma_vs.pkl",))
        ]
        assert any("Deferred remote artifact soma_vs.pkl synced" in message for message in progress_messages)
        print("deferred artifact selected-to-direct fallback: OK")

        progress_messages.clear()
        sync_calls.clear()
        direct_calls.clear()

        def _sync_remote_result_dir_fallback(_config, *, remote_result_dir, local_result_dir, expected_files=None, include_files=None):
            sync_calls.append((remote_result_dir, expected_files, include_files))
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            if include_files is None:
                with open(local_result_dir / "soma_vs.pkl", "wb") as handle:
                    pickle.dump([("MC0", [0.0, 0.1], [-65.0, -64.0])], handle)
                return subprocess.CompletedProcess(args=["sync-full"], returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=["sync-selected"], returncode=1, stdout="", stderr="selected failed")

        fallback_hooks = DeferredArtifactSyncHooks(
            local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
            sync_remote_result_dir_fn=_sync_remote_result_dir_fallback,
            progress_write=progress_messages.append,
            format_bytes_fn=lambda size: f"{size} B",
            direct_stream_supported_fn=lambda filename: filename == "soma_vs.pkl",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                args=["probe"], returncode=0, stdout="67\n", stderr=""
            ),
            stream_file_to_local_path_fn=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=["direct"], returncode=1, stdout="", stderr="direct failed"
            ),
            perf_counter_fn=lambda: 2.0,
        )
        fallback_result_dir = tmpdir_path / "full-fallback-success"
        fallback_result_dir.mkdir()
        fallback_path = sync_deferred_remote_artifact(
            fallback_result_dir,
            run_info=run_info,
            filename="soma_vs.pkl",
            hooks=fallback_hooks,
        )
        assert fallback_path == fallback_result_dir / "soma_vs.pkl"
        assert sync_calls == [
            (PurePosixPath("/remote/result"), ("soma_vs.pkl",), ("soma_vs.pkl",)),
            (PurePosixPath("/remote/result"), ("soma_vs.pkl",), None),
        ]
        print("deferred artifact full-dir fallback: OK")

        failure_result_dir = tmpdir_path / "failure"
        failure_result_dir.mkdir()
        try:
            _ = sync_deferred_remote_artifact(
                failure_result_dir,
                run_info=run_info,
                filename="soma_vs.pkl",
                hooks=DeferredArtifactSyncHooks(
                    local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
                    sync_remote_result_dir_fn=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                        args=["sync"], returncode=1, stdout="", stderr="selected failed"
                    ),
                    progress_write=lambda _message: None,
                    format_bytes_fn=lambda size: f"{size} B",
                    direct_stream_supported_fn=lambda filename: filename == "soma_vs.pkl",
                    run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                        args=["probe"], returncode=1, stdout="", stderr="probe failed"
                    ),
                    stream_file_to_local_path_fn=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                        args=["direct"], returncode=1, stdout="", stderr="direct failed"
                    ),
                    perf_counter_fn=lambda: 3.0,
                ),
            )
            raise AssertionError("Expected deferred artifact sync to fail")
        except RuntimeError as exc:
            text = str(exc)
            assert "[selected-file sync]" in text
            assert "[direct file stream]" in text
            assert "[full result-dir sync]" in text
        print("deferred artifact failure diagnostics: OK")

        direct_dir = tmpdir_path / "direct-probe"
        direct_dir.mkdir()
        direct_completed = sync_deferred_remote_artifact_direct(
            {"ssh_transport": "paramiko"},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=direct_dir,
            filename="artifact.bin",
            hooks=DeferredArtifactSyncHooks(
                local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
                sync_remote_result_dir_fn=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                    args=["sync"], returncode=0, stdout="", stderr=""
                ),
                progress_write=lambda _message: None,
                format_bytes_fn=lambda size: f"{size} B",
                direct_stream_supported_fn=lambda _filename: True,
                run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                    args=["probe"], returncode=0, stdout="4\n", stderr=""
                ),
                stream_file_to_local_path_fn=lambda _config, *, remote_file_path, local_path, expected_bytes: (
                    Path(local_path).write_bytes(b"data"),
                    subprocess.CompletedProcess(args=["direct"], returncode=0, stdout="", stderr=""),
                )[1],
                perf_counter_fn=lambda: 4.0,
            ),
        )
        assert direct_completed.returncode == 0
        assert (direct_dir / "artifact.bin").read_bytes() == b"data"
        print("deferred artifact direct stream helper: OK")

        existing_dir = tmpdir_path / "already-local"
        existing_dir.mkdir()
        existing_path = existing_dir / "soma_vs.pkl"
        existing_path.write_text(json.dumps({"ok": True}))
        returned_path = sync_deferred_remote_artifact(
            existing_dir,
            run_info=run_info,
            filename="soma_vs.pkl",
            hooks=direct_hooks,
        )
        assert returned_path == existing_path
        print("deferred artifact local short-circuit: OK")


if __name__ == "__main__":
    main()
