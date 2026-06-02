"""Smoke tests for extracted Paramiko stream-sync helpers."""

from __future__ import annotations

from io import BytesIO
import gzip
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from neuroinfra.remote.stream_sync import (
    ParamikoStreamSyncHooks,
    probe_selected_sync_files,
    stream_archive_to_local,
    stream_archive_to_local_dir,
    stream_file_to_local_path,
)


class _FakeProgress:
    def __init__(self) -> None:
        self.values: list[int | None] = []

    def update_to(self, value):
        self.values.append(value)

    def close(self) -> None:
        return None


class _FakeChannel:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", exit_status: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_status = exit_status
        self._started = False
        self.closed = False
        self.exec_calls: list[str] = []

    def exec_command(self, command: str) -> None:
        self.exec_calls.append(command)
        self._started = True

    def recv_ready(self) -> bool:
        return bool(self._stdout)

    def recv(self, _count: int) -> bytes:
        payload = self._stdout
        self._stdout = b""
        return payload

    def recv_stderr_ready(self) -> bool:
        return bool(self._stderr)

    def recv_stderr(self, _count: int) -> bytes:
        payload = self._stderr
        self._stderr = b""
        return payload

    def exit_status_ready(self) -> bool:
        return self._started

    def recv_exit_status(self) -> int:
        return self._exit_status

    def close(self) -> None:
        self.closed = True


class _FakeTransport:
    def __init__(self, channel: _FakeChannel) -> None:
        self.channel = channel
        self.open_calls = 0

    def open_session(self) -> _FakeChannel:
        self.open_calls += 1
        return self.channel


def _selected_probe_command(remote_result_dir: PurePosixPath, include_files: tuple[str, ...]) -> str:
    return f"probe::{remote_result_dir.as_posix()}::{','.join(include_files)}"


def _archive_command(remote_result_dir: PurePosixPath, compressor: str) -> str:
    return f"stream::{remote_result_dir.as_posix()}::{compressor}"


def _finished(channel: _FakeChannel) -> bool:
    return channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready()


def _progress_factory(_total, _desc):
    return _FakeProgress()


def _make_hooks(*, transport: _FakeTransport, shell_output: subprocess.CompletedProcess[str]) -> ParamikoStreamSyncHooks:
    return ParamikoStreamSyncHooks(
        transport_for_config_fn=lambda _config: transport,
        run_paramiko_shell_fn=lambda _config, _command: shell_output,
        build_remote_stream_archive_command_fn=_archive_command,
        build_remote_selected_archive_probe_command_fn=_selected_probe_command,
        local_archive_decompress_command_fn=lambda compressor: [str(shutil.which("gzip") or "gzip"), "-d"],
        channel_stream_finished_fn=_finished,
        progress_factory_fn=_progress_factory,
        sleep_fn=lambda _seconds: None,
    )


def _gzip_tar_bytes(filename: str, payload: bytes) -> bytes:
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as handle:
        info = tarfile.TarInfo(name=filename)
        info.size = len(payload)
        handle.addfile(info, BytesIO(payload))
    return gzip.compress(tar_buffer.getvalue())


def main() -> None:
    probe_hooks = _make_hooks(
        transport=_FakeTransport(_FakeChannel()),
        shell_output=subprocess.CompletedProcess(
            args=["ssh", "bash", "-lc", "probe"],
            returncode=0,
            stdout="zstd\n123\n.tar.zst\nsummary.json\nstdout.txt\n",
            stderr="",
        ),
    )
    probe_result = probe_selected_sync_files(
        {},
        remote_result_dir=PurePosixPath("/remote/result"),
        include_files=("summary.json", "stderr.txt", "stdout.txt"),
        hooks=probe_hooks,
    )
    assert probe_result == ("zstd", 123, ("summary.json", "stdout.txt"))
    print("stream sync selected-file probe parsing: OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        archive_bytes = _gzip_tar_bytes("payload.txt", b"hello stream archive\n")
        archive_channel = _FakeChannel(stdout=archive_bytes)
        archive_transport = _FakeTransport(archive_channel)
        archive_hooks = _make_hooks(
            transport=archive_transport,
            shell_output=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        )
        archive_path = tmpdir_path / "result.tar.gz"
        archive_completed = stream_archive_to_local(
            {},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_archive_path=archive_path,
            compressor="gzip",
            raw_bytes=len(archive_bytes),
            hooks=archive_hooks,
        )
        assert archive_completed.returncode == 0
        assert archive_path.read_bytes() == archive_bytes
        assert archive_transport.open_calls == 1
        print("stream sync archive-to-file path: OK")

        extract_bytes = _gzip_tar_bytes("summary.json", b"{\"ok\": true}\n")
        extract_channel = _FakeChannel(stdout=extract_bytes)
        extract_transport = _FakeTransport(extract_channel)
        extract_hooks = _make_hooks(
            transport=extract_transport,
            shell_output=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        )
        extract_dir = tmpdir_path / "extract"
        extract_completed = stream_archive_to_local_dir(
            {},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=extract_dir,
            compressor="gzip",
            raw_bytes=len(extract_bytes),
            hooks=extract_hooks,
        )
        assert extract_completed.returncode == 0
        assert (extract_dir / "summary.json").read_text() == "{\"ok\": true}\n"
        print("stream sync archive-to-dir extraction: OK")

        direct_channel = _FakeChannel(stdout=b"direct payload\n")
        direct_transport = _FakeTransport(direct_channel)
        direct_hooks = _make_hooks(
            transport=direct_transport,
            shell_output=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        )
        direct_path = tmpdir_path / "stdout.txt"
        direct_completed = stream_file_to_local_path(
            {},
            remote_file_path=PurePosixPath("/remote/result/stdout.txt"),
            local_path=direct_path,
            expected_bytes=len(b"direct payload\n"),
            hooks=direct_hooks,
        )
        assert direct_completed.returncode == 0
        assert direct_path.read_bytes() == b"direct payload\n"
        print("stream sync direct-file path: OK")


if __name__ == "__main__":
    main()
