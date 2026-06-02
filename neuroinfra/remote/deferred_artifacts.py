"""Reusable deferred remote artifact sync helpers for notebook-managed runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class DeferredArtifactSyncHooks:
    """Callbacks injected by the notebook-facing caller for deferred artifacts."""

    local_sync_artifact_is_usable_fn: Callable[[Path], bool]
    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]]
    progress_write: Callable[[str], None]
    format_bytes_fn: Callable[[int | float], str]
    direct_stream_supported_fn: Callable[[str], bool]
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]]
    stream_file_to_local_path_fn: Callable[..., subprocess.CompletedProcess[str]]
    perf_counter_fn: Callable[[], float] = time.perf_counter


def sync_deferred_remote_artifact(
    result_dir: str | Path,
    *,
    run_info: dict[str, Any] | None,
    filename: str,
    hooks: DeferredArtifactSyncHooks,
) -> Path:
    """Fetch one deferred remote artifact into the local result directory."""
    result_dir = Path(result_dir)
    local_path = result_dir / filename
    if hooks.local_sync_artifact_is_usable_fn(local_path):
        return local_path
    if not isinstance(run_info, dict):
        raise FileNotFoundError(f"Deferred remote artifact {filename} is not available locally.")
    remote_payload = run_info.get("remote") or {}
    remote_result_dir_value = remote_payload.get("remote_result_dir")
    config = deepcopy(run_info.get("config") or {})
    if not remote_result_dir_value or not isinstance(config, dict):
        raise FileNotFoundError(f"Deferred remote artifact {filename} is not available locally.")

    hooks.progress_write(f"[OBGPU load] Fetching deferred remote artifact {filename}...")
    started = hooks.perf_counter_fn()
    remote_result_dir = PurePosixPath(str(remote_result_dir_value))
    attempt_errors: list[tuple[str, str]] = []

    completed = hooks.sync_remote_result_dir_fn(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=result_dir,
        expected_files=(filename,),
        include_files=(filename,),
    )
    if not hooks.local_sync_artifact_is_usable_fn(local_path):
        attempt_errors.append(("selected-file sync", completed.stderr or ""))

    if not hooks.local_sync_artifact_is_usable_fn(local_path) and hooks.direct_stream_supported_fn(filename):
        hooks.progress_write(
            "[OBGPU load] Deferred soma selected-file sync failed; "
            "retrying with direct SSH-channel file streaming..."
        )
        direct_completed = sync_deferred_remote_artifact_direct(
            config,
            remote_result_dir=remote_result_dir,
            local_result_dir=result_dir,
            filename=filename,
            hooks=hooks,
        )
        if direct_completed.returncode == 0 and hooks.local_sync_artifact_is_usable_fn(local_path):
            completed = direct_completed
        else:
            attempt_errors.append(("direct file stream", direct_completed.stderr or ""))

    if not hooks.local_sync_artifact_is_usable_fn(local_path) and hooks.direct_stream_supported_fn(filename):
        hooks.progress_write(
            "[OBGPU load] Deferred soma trace sync fell back from direct-file mode; "
            "retrying by syncing the full remote result directory..."
        )
        completed = hooks.sync_remote_result_dir_fn(
            config,
            remote_result_dir=remote_result_dir,
            local_result_dir=result_dir,
            expected_files=(filename,),
        )
        if not hooks.local_sync_artifact_is_usable_fn(local_path):
            attempt_errors.append(("full result-dir sync", completed.stderr or ""))

    if not hooks.local_sync_artifact_is_usable_fn(local_path):
        stderr = completed.stderr or ""
        if attempt_errors:
            stderr = "\n".join(
                f"[{label}]\n{detail.strip() or '<no stderr>'}"
                for label, detail in attempt_errors
            )
        raise RuntimeError(
            "Deferred remote artifact sync failed.\n"
            f"Result dir: {result_dir}\n"
            f"Artifact: {filename}\n"
            f"Stderr:\n{stderr}"
        )

    elapsed_s = hooks.perf_counter_fn() - started
    hooks.progress_write(
        f"[OBGPU load] Deferred remote artifact {filename} synced in {elapsed_s:.1f}s "
        f"({hooks.format_bytes_fn(local_path.stat().st_size)})."
    )
    return local_path


def sync_deferred_remote_artifact_direct(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    filename: str,
    hooks: DeferredArtifactSyncHooks,
) -> subprocess.CompletedProcess[str]:
    """Fetch one deferred artifact via a direct SSH-channel byte stream."""
    remote_file_path = remote_result_dir / filename
    local_path = Path(local_result_dir) / filename
    probe_command = (
        "set -euo pipefail && "
        f"remote_file={shlex.quote(remote_file_path.as_posix())} && "
        "test -f \"$remote_file\" && wc -c < \"$remote_file\""
    )
    try:
        probe_completed = hooks.run_paramiko_shell_fn(config, probe_command)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["paramiko-direct-file-probe", remote_file_path.as_posix(), str(local_path)],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )
    if probe_completed.returncode != 0:
        return subprocess.CompletedProcess(
            args=["paramiko-direct-file-probe", remote_file_path.as_posix(), str(local_path)],
            returncode=1,
            stdout=probe_completed.stdout or "",
            stderr=probe_completed.stderr or "Remote deferred artifact probe failed.",
        )
    try:
        expected_bytes = int((probe_completed.stdout or "").strip().splitlines()[-1])
    except (IndexError, ValueError):
        expected_bytes = None
    completed = hooks.stream_file_to_local_path_fn(
        config,
        remote_file_path=remote_file_path,
        local_path=local_path,
        expected_bytes=expected_bytes,
    )
    if completed.returncode == 0 and not hooks.local_sync_artifact_is_usable_fn(local_path):
        return subprocess.CompletedProcess(
            args=completed.args,
            returncode=1,
            stdout=completed.stdout or "",
            stderr=(completed.stderr or "")
            + f"\n[OBGPU load] Direct file stream did not produce usable local artifact: {filename}\n",
        )
    return completed
