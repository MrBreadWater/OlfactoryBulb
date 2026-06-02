"""Smoke tests for extracted higher-level remote result sync policy."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from neuroinfra.remote.result_sync import (
    RemoteResultSyncHooks,
    combine_sync_attempt_stderr,
    sync_remote_result_dir,
    sync_remote_result_dir_resilient,
)


class _FakeTransport:
    pass


def _local_sync_artifact_is_usable(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _missing_local_sync_artifacts(result_dir: Path, expected_files: tuple[str, ...] | None) -> list[str]:
    result_dir = Path(result_dir)
    if expected_files:
        return [name for name in expected_files if not _local_sync_artifact_is_usable(result_dir / name)]
    return [] if any(result_dir.iterdir()) else ["remote result artifacts"]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        fallback_calls: list[str] = []
        close_calls: list[str] = []
        archive_probe_calls: list[str] = []

        def _archive_probe_run(_config, command):
            archive_probe_calls.append(command)
            return subprocess.CompletedProcess(
                args=["ssh", "bash", "-lc", command],
                returncode=0,
                stdout="gzip\n0\n.tar.gz\n",
                stderr="",
            )

        def _stream_to_dir(_config, *, remote_result_dir, local_result_dir, compressor, raw_bytes, stream_command=None):
            fallback_calls.append(f"stream:{compressor}:{raw_bytes}")
            Path(local_result_dir).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(args=["stream"], returncode=0, stdout="", stderr="")

        def _sftp_copy_tree(_sftp, remote_dir, local_dir):
            fallback_calls.append(f"sftp-tree:{remote_dir}")
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        fallback_hooks = RemoteResultSyncHooks(
            remote_transport_fn=lambda _config: "paramiko",
            run_paramiko_shell_fn=_archive_probe_run,
            build_remote_archive_probe_command_fn=lambda remote_result_dir: f"probe::{remote_result_dir.as_posix()}",
            probe_selected_sync_files_fn=lambda *_args, **_kwargs: ("gzip", 0, ("summary.json",)),
            build_remote_selected_stream_archive_command_fn=lambda remote_result_dir, include_files, compressor: (
                f"selected::{remote_result_dir.as_posix()}::{','.join(include_files)}::{compressor}"
            ),
            stream_archive_to_local_dir_fn=_stream_to_dir,
            get_paramiko_sftp_fn=lambda _config: object(),
            close_paramiko_sftp_fn=lambda _config: close_calls.append("close"),
            sftp_copy_files_fn=lambda *_args, **_kwargs: None,
            sftp_copy_tree_fn=_sftp_copy_tree,
            cached_transport_fn=lambda _config: _FakeTransport(),
            transport_is_usable_fn=lambda _transport: True,
            preserve_reauth_blocked_fn=lambda _config: False,
            drop_paramiko_connection_fn=lambda _config: fallback_calls.append("drop"),
            midrun_reauth_error_fn=lambda _config: "midrun reauth blocked",
            progress_write=lambda _message: None,
            missing_local_sync_artifacts_fn=_missing_local_sync_artifacts,
            local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
            sleep_fn=lambda _seconds: None,
        )
        sync_dir = tmpdir_path / "fallback-sync"
        completed = sync_remote_result_dir(
            {"remote_sync_compress": True},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("summary.json",),
            hooks=fallback_hooks,
        )
        assert completed.returncode == 0
        assert (sync_dir / "summary.json").exists()
        assert "fallback completed successfully" in (completed.stderr or "")
        assert archive_probe_calls == ["probe::/remote/result"]
        assert fallback_calls == ["stream:gzip:0", "sftp-tree:/remote/result"]
        assert close_calls
        print("result sync full-stream SFTP fallback: OK")

        retry_attempts: list[tuple[str, ...]] = []
        drop_calls: list[str] = []

        def _sftp_copy_files(_sftp, _remote_dir, local_dir, file_names):
            retry_attempts.append(tuple(file_names))
            if len(retry_attempts) == 1:
                raise OSError("transient sftp failure")
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        retry_hooks = RemoteResultSyncHooks(
            remote_transport_fn=lambda _config: "paramiko",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            build_remote_archive_probe_command_fn=lambda remote_result_dir: f"probe::{remote_result_dir.as_posix()}",
            probe_selected_sync_files_fn=lambda *_args, **_kwargs: ("gzip", 0, ("summary.json",)),
            build_remote_selected_stream_archive_command_fn=lambda remote_result_dir, include_files, compressor: (
                f"selected::{remote_result_dir.as_posix()}::{','.join(include_files)}::{compressor}"
            ),
            stream_archive_to_local_dir_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("plain selected-file sync should not call stream path")
            ),
            get_paramiko_sftp_fn=lambda _config: object(),
            close_paramiko_sftp_fn=lambda _config: None,
            sftp_copy_files_fn=_sftp_copy_files,
            sftp_copy_tree_fn=lambda *_args, **_kwargs: None,
            cached_transport_fn=lambda _config: _FakeTransport(),
            transport_is_usable_fn=lambda _transport: True,
            preserve_reauth_blocked_fn=lambda _config: False,
            drop_paramiko_connection_fn=lambda _config: drop_calls.append("drop"),
            midrun_reauth_error_fn=lambda _config: "midrun reauth blocked",
            progress_write=lambda _message: None,
            missing_local_sync_artifacts_fn=_missing_local_sync_artifacts,
            local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
            sleep_fn=lambda _seconds: None,
        )
        retry_dir = tmpdir_path / "selected-retry"
        retry_completed = sync_remote_result_dir(
            {"remote_sync_compress": False},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=retry_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
            hooks=retry_hooks,
        )
        assert retry_completed.returncode == 0
        assert retry_attempts == [("summary.json",), ("summary.json",)]
        assert drop_calls == []
        assert (retry_dir / "summary.json").exists()
        print("result sync selected-file retry on usable transport: OK")

        resilient_attempts: list[tuple[tuple[str, ...] | None, str]] = []

        def _resilient_sftp_copy_files(_sftp, _remote_dir, local_dir, file_names):
            resilient_attempts.append((tuple(file_names), "selected"))
            # Leave the directory empty so the resilient policy treats this as incomplete.
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        def _resilient_sftp_copy_tree(_sftp, _remote_dir, local_dir):
            resilient_attempts.append((None, "full"))
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        resilient_hooks = RemoteResultSyncHooks(
            remote_transport_fn=lambda _config: "paramiko",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            build_remote_archive_probe_command_fn=lambda remote_result_dir: f"probe::{remote_result_dir.as_posix()}",
            probe_selected_sync_files_fn=lambda *_args, **_kwargs: ("gzip", 0, ("summary.json",)),
            build_remote_selected_stream_archive_command_fn=lambda remote_result_dir, include_files, compressor: (
                f"selected::{remote_result_dir.as_posix()}::{','.join(include_files)}::{compressor}"
            ),
            stream_archive_to_local_dir_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("non-compressed resilient test should not call stream path")
            ),
            get_paramiko_sftp_fn=lambda _config: object(),
            close_paramiko_sftp_fn=lambda _config: None,
            sftp_copy_files_fn=_resilient_sftp_copy_files,
            sftp_copy_tree_fn=_resilient_sftp_copy_tree,
            cached_transport_fn=lambda _config: _FakeTransport(),
            transport_is_usable_fn=lambda _transport: True,
            preserve_reauth_blocked_fn=lambda _config: False,
            drop_paramiko_connection_fn=lambda _config: None,
            midrun_reauth_error_fn=lambda _config: "midrun reauth blocked",
            progress_write=lambda _message: None,
            missing_local_sync_artifacts_fn=_missing_local_sync_artifacts,
            local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
            sleep_fn=lambda _seconds: None,
        )
        resilient_dir = tmpdir_path / "resilient"
        resilient_completed = sync_remote_result_dir_resilient(
            {"remote_sync_compress": False},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=resilient_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
            retry_delay_s=0,
            hooks=resilient_hooks,
        )
        assert resilient_completed.returncode == 0
        assert resilient_attempts == [
            (("summary.json",), "selected"),
            (("summary.json",), "selected"),
            (None, "full"),
        ]
        assert (resilient_dir / "summary.json").exists()
        print("result sync resilient selected-to-full fallback: OK")

        stderr_text = combine_sync_attempt_stderr(
            [
                ("first", subprocess.CompletedProcess(args=["a"], returncode=1, stdout="", stderr="problem A")),
                ("second", subprocess.CompletedProcess(args=["b"], returncode=1, stdout="", stderr="problem B")),
            ]
        )
        assert "[first]" in stderr_text and "[second]" in stderr_text
        print("result sync stderr combination: OK")


if __name__ == "__main__":
    main()
