"""Reusable remote archive-stream command builders for notebook-managed sync."""

from __future__ import annotations

from base64 import b64encode
import json
from pathlib import PurePosixPath
import shlex
import shutil
from typing import Any, Callable


def build_remote_archive_command(remote_result_dir: PurePosixPath) -> str:
    """Build a remote shell command that packs a result dir into a compressed tar archive."""
    archive_dir = PurePosixPath(remote_result_dir.parent) / ".obgpu-transfer"
    archive_base = archive_dir / remote_result_dir.name
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        f"archive_dir={shlex.quote(archive_dir.as_posix())} && "
        f"archive_base={shlex.quote(archive_base.as_posix())} && "
        "mkdir -p \"$archive_dir\" && "
        "rm -f \"${archive_base}.tar.zst\" \"${archive_base}.tar.gz\" \"${archive_base}.tar.xz\" && "
        "raw_bytes=$(du -sb \"$result_dir\" 2>/dev/null | awk '{print $1}') && "
        "if command -v zstd >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.zst\"; "
        "  compressor=zstd; "
        "  tar -C \"$result_dir\" -cf - . | zstd -T0 -15 -q -o \"$archive_path\"; "
        "elif command -v pigz >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.gz\"; "
        "  compressor=pigz; "
        "  tar -C \"$result_dir\" -cf - . | pigz -6 > \"$archive_path\"; "
        "elif command -v gzip >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.gz\"; "
        "  compressor=gzip; "
        "  tar -C \"$result_dir\" -cf - . | gzip -6 > \"$archive_path\"; "
        "elif command -v xz >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.xz\"; "
        "  compressor=xz; "
        "  tar -C \"$result_dir\" -cf - . | xz -6 -T0 > \"$archive_path\"; "
        "else "
        "  printf '%s\\n' 'No supported compressor found on remote host' >&2; "
        "  exit 1; "
        "fi && "
        "archive_bytes=$(wc -c < \"$archive_path\") && "
        "printf '%s\\n%s\\n%s\\n%s\\n' \"$archive_path\" \"$compressor\" \"${raw_bytes:-0}\" \"$archive_bytes\""
    )


def build_remote_archive_probe_command(remote_result_dir: PurePosixPath) -> str:
    """Build a remote shell command that selects a compressor and reports stream metadata."""
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        "raw_bytes=$(du -sb \"$result_dir\" 2>/dev/null | awk '{print $1}') && "
        "if command -v zstd >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'zstd' \"${raw_bytes:-0}\" '.tar.zst'; "
        "elif command -v pigz >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'pigz' \"${raw_bytes:-0}\" '.tar.gz'; "
        "elif command -v gzip >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'gzip' \"${raw_bytes:-0}\" '.tar.gz'; "
        "elif command -v xz >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'xz' \"${raw_bytes:-0}\" '.tar.xz'; "
        "else "
        "  printf '%s\\n' 'No supported compressor found on remote host' >&2; "
        "  exit 1; "
        "fi"
    )


def build_remote_selected_archive_probe_command(
    remote_result_dir: PurePosixPath,
    *,
    include_files: tuple[str, ...],
) -> str:
    """Build a remote shell command that reports stream metadata for selected files."""
    quoted_files = " ".join(shlex.quote(str(name)) for name in include_files)
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        f"files=( {quoted_files} ) && "
        "existing_files=() && "
        "raw_bytes=0 && "
        "for rel in \"${files[@]}\"; do "
        "  path=\"$result_dir/$rel\"; "
        "  if [ -f \"$path\" ]; then "
        "    existing_files+=(\"$rel\"); "
        "    raw_bytes=$((raw_bytes + $(wc -c < \"$path\"))); "
        "  fi; "
        "done && "
        "if command -v zstd >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'zstd' \"${raw_bytes:-0}\" '.tar.zst'; "
        "elif command -v pigz >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'pigz' \"${raw_bytes:-0}\" '.tar.gz'; "
        "elif command -v gzip >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'gzip' \"${raw_bytes:-0}\" '.tar.gz'; "
        "elif command -v xz >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'xz' \"${raw_bytes:-0}\" '.tar.xz'; "
        "else "
        "  printf '%s\\n' 'No supported compressor found on remote host' >&2; "
        "  exit 1; "
        "fi && "
        "for rel in \"${existing_files[@]}\"; do "
        "  printf '%s\\n' \"$rel\"; "
        "done"
    )


def build_remote_stream_archive_command(
    remote_result_dir: PurePosixPath,
    *,
    compressor: str,
) -> str:
    """Build a remote shell command that streams a compressed tar archive to stdout."""
    compressor_commands = {
        "zstd": 'tar -C "$result_dir" -cf - . | zstd -T0 -15 -q -c',
        "pigz": 'tar -C "$result_dir" -cf - . | pigz -6',
        "gzip": 'tar -C "$result_dir" -cf - . | gzip -6',
        "xz": 'tar -C "$result_dir" -cf - . | xz -6 -T0',
    }
    if compressor not in compressor_commands:
        raise ValueError(f"Unsupported archive compressor {compressor!r}")
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        + compressor_commands[compressor]
    )


def build_remote_selected_stream_archive_command(
    remote_result_dir: PurePosixPath,
    *,
    include_files: tuple[str, ...],
    compressor: str,
) -> str:
    """Build a remote shell command that streams selected files as a compressed tar archive."""
    if not include_files:
        raise ValueError("Selected stream archive requires at least one file")
    quoted_files = " ".join(shlex.quote(str(name)) for name in include_files)
    compressor_commands = {
        "zstd": 'tar -C "$result_dir" -cf - -- "${files[@]}" | zstd -T0 -15 -q -c',
        "pigz": 'tar -C "$result_dir" -cf - -- "${files[@]}" | pigz -6',
        "gzip": 'tar -C "$result_dir" -cf - -- "${files[@]}" | gzip -6',
        "xz": 'tar -C "$result_dir" -cf - -- "${files[@]}" | xz -6 -T0',
    }
    if compressor not in compressor_commands:
        raise ValueError(f"Unsupported archive compressor {compressor!r}")
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        f"files=( {quoted_files} ) && "
        + compressor_commands[compressor]
    )


def build_remote_sweep_compact_stream_archive_command(
    *,
    entries: list[dict[str, Any]],
    compressor: str,
    json_ready: Callable[[Any], Any] | None = None,
) -> str:
    """Build a remote command that streams compact artifacts for many sweep items."""
    compressor_commands = {
        "zstd": "zstd -T0 -3 -q -c",
        "pigz": "pigz -6",
        "gzip": "gzip -6",
        "xz": "xz -3 -T0",
    }
    if compressor not in compressor_commands:
        raise ValueError(f"Unsupported archive compressor {compressor!r}")

    payload_obj = entries if json_ready is None else json_ready(entries)
    payload = b64encode(json.dumps({"entries": payload_obj}, separators=(",", ":")).encode()).decode()
    remote_python = r'''
import base64
import json
import sys
import tarfile
from pathlib import Path

payload = json.loads(base64.b64decode("__PAYLOAD__").decode())
added = 0
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
    for entry in payload.get("entries", []):
        label = str(entry.get("label") or "").strip()
        result_dir = Path(str(entry.get("result_dir") or ""))
        if not label or not result_dir.is_dir():
            continue
        for raw_name in entry.get("include_files", []):
            name = str(raw_name).strip()
            if not name or name.startswith("/") or ".." in Path(name).parts:
                continue
            path = result_dir / name
            if not path.is_file():
                continue
            tar.add(path, arcname=f"item_runs/{label}/{name}", recursive=False)
            added += 1
print(f"OBGPU_SELECTED_FILES={added}", file=sys.stderr, flush=True)
'''.replace("__PAYLOAD__", payload)
    return (
        "set -euo pipefail && "
        f"python3 -c {shlex.quote(remote_python)} | {compressor_commands[compressor]}"
    )


def local_archive_decompress_command(compressor: str) -> list[str]:
    """Return a local decompressor command for one archive stream."""
    compressor = str(compressor)
    if compressor == "zstd":
        return [str(shutil.which("zstd") or "zstd"), "-d", "-q"]
    if compressor in {"pigz", "gzip"}:
        return [str(shutil.which("gzip") or "gzip"), "-d"]
    if compressor == "xz":
        return [str(shutil.which("xz") or "xz"), "-d"]
    raise ValueError(f"Unsupported archive compressor {compressor!r}")


def paramiko_channel_stream_finished(channel: Any) -> bool:
    """Return whether one Paramiko exec channel has reached a fully drained EOF."""
    return bool(
        channel.exit_status_ready()
        and getattr(channel, "eof_received", False)
        and not channel.recv_ready()
        and not channel.recv_stderr_ready()
    )
