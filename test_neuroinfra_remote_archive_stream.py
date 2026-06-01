"""Smoke tests for standardized remote archive-stream helpers."""

from __future__ import annotations

from pathlib import PurePosixPath

import neuroinfra.remote.archive_stream as archive_stream
import obgpu_experiment_helpers as hlp


class _DoneChannel:
    eof_received = True

    def exit_status_ready(self) -> bool:
        return True

    def recv_ready(self) -> bool:
        return False

    def recv_stderr_ready(self) -> bool:
        return False


class _OpenChannel:
    eof_received = False

    def exit_status_ready(self) -> bool:
        return False

    def recv_ready(self) -> bool:
        return True

    def recv_stderr_ready(self) -> bool:
        return False


def main() -> None:
    remote_dir = PurePosixPath("/remote/result")
    include_files = ("summary.json", "stderr.txt")

    archive_command = archive_stream.build_remote_archive_command(remote_dir)
    assert ".obgpu-transfer" in archive_command
    assert "archive_base" in archive_command
    assert hlp._build_remote_archive_command(remote_dir) == archive_command

    probe_command = archive_stream.build_remote_archive_probe_command(remote_dir)
    assert "du -sb" in probe_command
    assert "zstd" in probe_command
    assert hlp._build_remote_archive_probe_command(remote_dir) == probe_command

    selected_probe = archive_stream.build_remote_selected_archive_probe_command(
        remote_dir,
        include_files=include_files,
    )
    assert "existing_files" in selected_probe
    assert "summary.json" in selected_probe
    assert hlp._build_remote_selected_archive_probe_command(remote_dir, include_files=include_files) == selected_probe

    stream_command = archive_stream.build_remote_stream_archive_command(remote_dir, compressor="gzip")
    assert 'tar -C "$result_dir" -cf - . | gzip -6' in stream_command
    assert hlp._build_remote_stream_archive_command(remote_dir, compressor="gzip") == stream_command

    selected_stream = archive_stream.build_remote_selected_stream_archive_command(
        remote_dir,
        include_files=include_files,
        compressor="xz",
    )
    assert 'files=(' in selected_stream
    assert 'xz -6 -T0' in selected_stream
    assert (
        hlp._build_remote_selected_stream_archive_command(
            remote_dir,
            include_files=include_files,
            compressor="xz",
        )
        == selected_stream
    )

    compact_stream = archive_stream.build_remote_sweep_compact_stream_archive_command(
        entries=[{"label": "A", "result_dir": "/remote/result/A", "include_files": ["summary.json"]}],
        compressor="gzip",
    )
    assert "OBGPU_SELECTED_FILES" in compact_stream
    assert "python3 -c" in compact_stream
    assert "gzip -6" in compact_stream

    gzip_cmd = archive_stream.local_archive_decompress_command("gzip")
    assert gzip_cmd[-1] == "-d"
    assert hlp._local_archive_decompress_command("gzip") == gzip_cmd

    assert archive_stream.paramiko_channel_stream_finished(_DoneChannel()) is True
    assert archive_stream.paramiko_channel_stream_finished(_OpenChannel()) is False
    assert hlp._paramiko_channel_stream_finished(_DoneChannel()) is True
    assert hlp._paramiko_channel_stream_finished(_OpenChannel()) is False

    print("neuroinfra remote archive stream smoke test: OK")


if __name__ == "__main__":
    main()
