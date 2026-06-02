"""Reusable Paramiko stream-sync helpers for notebook-managed remote runs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tempfile
import time
from typing import Any, Callable


@dataclass(frozen=True)
class ParamikoStreamSyncHooks:
    """Callbacks injected by the notebook-facing caller for stream syncs."""

    transport_for_config_fn: Callable[[dict[str, Any]], Any]
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]]
    build_remote_stream_archive_command_fn: Callable[[PurePosixPath, str], str]
    build_remote_selected_archive_probe_command_fn: Callable[[PurePosixPath, tuple[str, ...]], str]
    local_archive_decompress_command_fn: Callable[[str], list[str]]
    channel_stream_finished_fn: Callable[[Any], bool]
    progress_factory_fn: Callable[[int | None, str], Any]
    sleep_fn: Callable[[float], None] = time.sleep


def probe_selected_sync_files(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    include_files: tuple[str, ...],
    hooks: ParamikoStreamSyncHooks,
) -> tuple[str, int, tuple[str, ...]] | subprocess.CompletedProcess[str]:
    """Return a selected-file sync plan using only remote files that exist."""
    probe_completed = hooks.run_paramiko_shell_fn(
        config,
        hooks.build_remote_selected_archive_probe_command_fn(
            remote_result_dir,
            tuple(include_files),
        ),
    )
    if probe_completed.returncode != 0:
        return subprocess.CompletedProcess(
            args=["paramiko-probe-selected", remote_result_dir.as_posix(), str(include_files)],
            returncode=1,
            stdout=probe_completed.stdout or "",
            stderr=probe_completed.stderr or "",
        )
    probe_lines = [line.strip() for line in (probe_completed.stdout or "").splitlines() if line.strip()]
    if len(probe_lines) < 3:
        return subprocess.CompletedProcess(
            args=["paramiko-probe-selected", remote_result_dir.as_posix(), str(include_files)],
            returncode=1,
            stdout=probe_completed.stdout or "",
            stderr="Remote selected-file archive probe did not return the expected metadata",
        )
    compressor, raw_bytes_text, _archive_suffix = probe_lines[:3]
    raw_bytes = int(raw_bytes_text or "0")
    available_set = set(probe_lines[3:])
    available_files = tuple(name for name in include_files if name in available_set)
    return compressor, raw_bytes, available_files


def stream_archive_to_local(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_archive_path: Path,
    compressor: str,
    raw_bytes: int,
    hooks: ParamikoStreamSyncHooks,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote compressed tar archive over Paramiko into a local file."""
    transport = hooks.transport_for_config_fn(config)
    channel = None
    stderr_chunks: list[bytes] = []
    bytes_written = 0
    completed_ok = False
    progress = hooks.progress_factory_fn(None, "[OBGPU load] Download compressed stream")
    stream_command = hooks.build_remote_stream_archive_command_fn(
        remote_result_dir,
        compressor,
    )
    local_archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive_path = local_archive_path.with_suffix(local_archive_path.suffix + ".part")
    try:
        channel = transport.open_session()
        channel.exec_command(stream_command)
        with open(tmp_archive_path, "wb") as handle:
            while True:
                if channel.recv_ready():
                    data = channel.recv(1024 * 1024)
                    if data:
                        handle.write(data)
                        bytes_written += len(data)
                        progress.update_to(bytes_written)
                        continue
                if channel.recv_stderr_ready():
                    stderr_chunks.append(channel.recv_stderr(65536))
                    continue
                if hooks.channel_stream_finished_fn(channel):
                    break
                hooks.sleep_fn(0.05)
        returncode = channel.recv_exit_status()
        if bytes_written:
            progress.update_to(bytes_written)
        progress.close()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        if returncode == 0:
            tmp_archive_path.replace(local_archive_path)
            completed_ok = True
        else:
            try:
                tmp_archive_path.unlink(missing_ok=True)
            except Exception:
                pass
        return subprocess.CompletedProcess(
            args=["paramiko-stream", remote_result_dir.as_posix(), str(local_archive_path)],
            returncode=returncode,
            stdout="",
            stderr=stderr_text,
        )
    finally:
        progress.close()
        if not completed_ok:
            try:
                tmp_archive_path.unlink(missing_ok=True)
            except Exception:
                pass
        if channel is not None:
            channel.close()


def stream_archive_to_local_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    compressor: str,
    raw_bytes: int,
    hooks: ParamikoStreamSyncHooks,
    stream_command: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote compressed tar archive over Paramiko directly into local extraction."""
    transport = hooks.transport_for_config_fn(config)
    channel = None
    stderr_chunks: list[bytes] = []
    bytes_written = 0
    progress = hooks.progress_factory_fn(None, "[OBGPU load] Stream download/extract")
    local_result_dir.mkdir(parents=True, exist_ok=True)
    if stream_command is None:
        stream_command = hooks.build_remote_stream_archive_command_fn(
            remote_result_dir,
            compressor,
        )
    decompress_cmd = hooks.local_archive_decompress_command_fn(compressor)
    decompress_stderr = tempfile.NamedTemporaryFile(prefix="obgpu-decompress-", suffix=".log", delete=False)
    tar_stderr = tempfile.NamedTemporaryFile(prefix="obgpu-tar-", suffix=".log", delete=False)
    decompress_proc = None
    tar_proc = None
    decompress_stderr_handle = None
    tar_stderr_handle = None
    try:
        decompress_stderr.close()
        tar_stderr.close()
        decompress_stderr_handle = open(decompress_stderr.name, "wb")
        tar_stderr_handle = open(tar_stderr.name, "wb")
        decompress_proc = subprocess.Popen(
            decompress_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=decompress_stderr_handle,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-xf", "-", "-C", str(local_result_dir)],
            stdin=decompress_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=tar_stderr_handle,
        )
        if decompress_proc.stdout is not None:
            decompress_proc.stdout.close()
        if decompress_proc.stdin is None:
            raise RuntimeError("Could not open decompressor stdin for streaming extraction.")

        channel = transport.open_session()
        channel.exec_command(stream_command)
        while True:
            if channel.recv_ready():
                data = channel.recv(1024 * 1024)
                if data:
                    decompress_proc.stdin.write(data)
                    bytes_written += len(data)
                    progress.update_to(bytes_written)
                    continue
            if channel.recv_stderr_ready():
                stderr_chunks.append(channel.recv_stderr(65536))
                continue
            if hooks.channel_stream_finished_fn(channel):
                break
            hooks.sleep_fn(0.05)

        if decompress_proc.stdin is not None:
            decompress_proc.stdin.close()
        remote_returncode = channel.recv_exit_status()
        decompress_returncode = decompress_proc.wait()
        tar_returncode = tar_proc.wait()
        progress.close()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        if Path(decompress_stderr.name).exists():
            stderr_text += Path(decompress_stderr.name).read_text(errors="replace")
        if Path(tar_stderr.name).exists():
            stderr_text += Path(tar_stderr.name).read_text(errors="replace")
        returncode = 0 if remote_returncode == 0 and decompress_returncode == 0 and tar_returncode == 0 else 1
        return subprocess.CompletedProcess(
            args=["paramiko-stream-extract", remote_result_dir.as_posix(), str(local_result_dir)],
            returncode=returncode,
            stdout="",
            stderr=stderr_text,
        )
    finally:
        progress.close()
        if channel is not None:
            channel.close()
        for handle in (decompress_stderr_handle, tar_stderr_handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        if decompress_proc is not None and decompress_proc.poll() is None:
            try:
                decompress_proc.kill()
            except Exception:
                pass
        if tar_proc is not None and tar_proc.poll() is None:
            try:
                tar_proc.kill()
            except Exception:
                pass
        for path in (decompress_stderr.name, tar_stderr.name):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def stream_file_to_local_path(
    config: dict[str, Any],
    *,
    remote_file_path: PurePosixPath,
    local_path: Path,
    expected_bytes: int | None,
    hooks: ParamikoStreamSyncHooks,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote file over the existing Paramiko session without SFTP."""
    transport = hooks.transport_for_config_fn(config)
    channel = None
    stderr_chunks: list[bytes] = []
    bytes_written = 0
    temp_path = local_path.with_name(f".{local_path.name}.obgpu-direct-{os.getpid()}")
    progress = hooks.progress_factory_fn(expected_bytes, f"[OBGPU load] Direct sync {local_path.name}")
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.unlink(missing_ok=True)
        channel = transport.open_session()
        remote_command = (
            "set -euo pipefail && "
            f"remote_file={shlex.quote(remote_file_path.as_posix())} && "
            "if [ ! -f \"$remote_file\" ]; then "
            "  printf 'Remote artifact not found: %s\\n' \"$remote_file\" >&2; "
            "  exit 2; "
            "fi && "
            "cat -- \"$remote_file\""
        )
        channel.exec_command(f"bash -lc {shlex.quote(remote_command)}")
        with open(temp_path, "wb") as handle:
            while True:
                if channel.recv_ready():
                    data = channel.recv(1024 * 1024)
                    if data:
                        handle.write(data)
                        bytes_written += len(data)
                        progress.update_to(bytes_written)
                        continue
                if channel.recv_stderr_ready():
                    stderr_chunks.append(channel.recv_stderr(65536))
                    continue
                if hooks.channel_stream_finished_fn(channel):
                    break
                hooks.sleep_fn(0.05)
        while channel.recv_stderr_ready():
            stderr_chunks.append(channel.recv_stderr(65536))
        remote_returncode = channel.recv_exit_status()
        progress.close()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        if remote_returncode == 0 and expected_bytes is not None and bytes_written != int(expected_bytes):
            remote_returncode = 1
            stderr_text += (
                f"\n[OBGPU load] Direct file sync byte count mismatch for {remote_file_path}: "
                f"expected {expected_bytes}, received {bytes_written}\n"
            )
        if remote_returncode == 0:
            os.replace(temp_path, local_path)
        return subprocess.CompletedProcess(
            args=["paramiko-direct-file", remote_file_path.as_posix(), str(local_path)],
            returncode=0 if remote_returncode == 0 else 1,
            stdout="",
            stderr=stderr_text,
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["paramiko-direct-file", remote_file_path.as_posix(), str(local_path)],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )
    finally:
        progress.close()
        temp_path.unlink(missing_ok=True)
        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass
