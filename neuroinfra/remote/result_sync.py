"""Reusable higher-level remote result sync policy for notebook-managed runs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteResultSyncHooks:
    """Callbacks injected by the notebook-facing caller for result sync policy."""

    remote_transport_fn: Callable[[dict[str, Any]], str]
    run_paramiko_shell_fn: Callable[[dict[str, Any], str], subprocess.CompletedProcess[str]]
    build_remote_archive_probe_command_fn: Callable[[PurePosixPath], str]
    probe_selected_sync_files_fn: Callable[[dict[str, Any], PurePosixPath, tuple[str, ...]], tuple[str, int, tuple[str, ...]] | subprocess.CompletedProcess[str]]
    build_remote_selected_stream_archive_command_fn: Callable[[PurePosixPath, tuple[str, ...], str], str]
    stream_archive_to_local_dir_fn: Callable[..., subprocess.CompletedProcess[str]]
    get_paramiko_sftp_fn: Callable[[dict[str, Any]], Any]
    close_paramiko_sftp_fn: Callable[[dict[str, Any]], None]
    sftp_copy_files_fn: Callable[[Any, str, Path, tuple[str, ...] | list[str]], None]
    sftp_copy_tree_fn: Callable[[Any, str, Path], None]
    cached_transport_fn: Callable[[dict[str, Any]], Any]
    transport_is_usable_fn: Callable[[Any], bool]
    preserve_reauth_blocked_fn: Callable[[dict[str, Any]], bool]
    drop_paramiko_connection_fn: Callable[[dict[str, Any]], None]
    midrun_reauth_error_fn: Callable[[dict[str, Any]], str]
    progress_write: Callable[[str], None]
    missing_local_sync_artifacts_fn: Callable[[Path, tuple[str, ...] | None], list[str]]
    local_sync_artifact_is_usable_fn: Callable[[Path], bool]
    sleep_fn: Callable[[float], None] = time.sleep


def combine_sync_attempt_stderr(attempts: list[tuple[str, subprocess.CompletedProcess[str]]]) -> str:
    """Render sync-attempt stderr with stage labels for actionable diagnostics."""
    chunks = []
    for label, completed in attempts:
        stderr = (completed.stderr or "").strip() or "<no stderr>"
        chunks.append(f"[{label}]\n{stderr}")
    return "\n".join(chunks)


def sync_remote_result_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    expected_files: tuple[str, ...] | None = None,
    include_files: tuple[str, ...] | None = None,
    hooks: RemoteResultSyncHooks,
) -> subprocess.CompletedProcess[str]:
    """Sync one remote result directory back into the local notebook results tree."""
    local_result_dir = Path(local_result_dir)
    local_result_dir.mkdir(parents=True, exist_ok=True)
    if hooks.remote_transport_fn(config) != "paramiko":
        return subprocess.CompletedProcess(
            args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
            returncode=1,
            stdout="",
            stderr="Remote result sync reached an unreachable non-Paramiko path.",
        )

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            if include_files:
                stream_selected = bool(config.get("remote_sync_compress", True)) and bool(include_files)
                if stream_selected:
                    selected_probe = hooks.probe_selected_sync_files_fn(
                        config,
                        remote_result_dir,
                        tuple(include_files),
                    )
                    if isinstance(selected_probe, subprocess.CompletedProcess):
                        return selected_probe
                    compressor, raw_bytes, available_files = selected_probe
                    if not available_files:
                        missing_selected_files = hooks.missing_local_sync_artifacts_fn(
                            local_result_dir,
                            expected_files,
                        )
                        expected_text = ", ".join(missing_selected_files or include_files)
                        return subprocess.CompletedProcess(
                            args=["paramiko-probe-selected", remote_result_dir.as_posix(), str(local_result_dir)],
                            returncode=1,
                            stdout="",
                            stderr=(
                                "[OBGPU load] None of the requested fast-sync files currently exist on the remote result dir. "
                                f"Missing: {expected_text}"
                            ),
                        )
                    selected_stage_dir = Path(
                        tempfile.mkdtemp(prefix="obgpu-selected-sync-", dir=str(local_result_dir.parent))
                    )
                    try:
                        stream_completed = hooks.stream_archive_to_local_dir_fn(
                            config,
                            remote_result_dir=remote_result_dir,
                            local_result_dir=selected_stage_dir,
                            compressor=compressor,
                            raw_bytes=raw_bytes,
                            stream_command=hooks.build_remote_selected_stream_archive_command_fn(
                                remote_result_dir,
                                available_files,
                                compressor,
                            ),
                        )
                        if stream_completed.returncode == 0:
                            for selected_name in available_files:
                                staged_path = selected_stage_dir / selected_name
                                if hooks.local_sync_artifact_is_usable_fn(staged_path):
                                    target_path = local_result_dir / selected_name
                                    target_path.parent.mkdir(parents=True, exist_ok=True)
                                    os.replace(staged_path, target_path)
                    finally:
                        shutil.rmtree(selected_stage_dir, ignore_errors=True)

                    missing_stream_files = hooks.missing_local_sync_artifacts_fn(local_result_dir, expected_files)
                    if stream_completed.returncode == 0 and missing_stream_files:
                        expected_text = ", ".join(missing_stream_files)
                        stream_completed = subprocess.CompletedProcess(
                            args=stream_completed.args,
                            returncode=1,
                            stdout=stream_completed.stdout or "",
                            stderr=(stream_completed.stderr or "")
                            + (
                                "\n[OBGPU load] Streamed selected-file sync produced no usable local artifacts. "
                                f"Missing: {expected_text}\n"
                            ),
                        )
                    if stream_completed.returncode != 0:
                        hooks.progress_write(
                            "[OBGPU load] Streamed selected-file sync failed; retrying the same files over SFTP..."
                        )
                        hooks.close_paramiko_sftp_fn(config)
                        hooks.sftp_copy_files_fn(
                            hooks.get_paramiko_sftp_fn(config),
                            remote_result_dir.as_posix(),
                            local_result_dir,
                            available_files,
                        )
                        missing_fallback_files = hooks.missing_local_sync_artifacts_fn(local_result_dir, expected_files)
                        if missing_fallback_files:
                            expected_text = ", ".join(missing_fallback_files)
                            return subprocess.CompletedProcess(
                                args=["paramiko-stream-selected-fallback", remote_result_dir.as_posix(), str(local_result_dir)],
                                returncode=1,
                                stdout=stream_completed.stdout or "",
                                stderr=(stream_completed.stderr or "")
                                + (
                                    "\n[OBGPU load] Streamed selected-file sync failed, SFTP fallback ran, "
                                    f"but required local artifacts are still missing: {expected_text}\n"
                                ),
                            )
                        return subprocess.CompletedProcess(
                            args=["paramiko-stream-selected-fallback", remote_result_dir.as_posix(), str(local_result_dir)],
                            returncode=0,
                            stdout=stream_completed.stdout or "",
                            stderr=(stream_completed.stderr or "")
                            + "\n[OBGPU load] Streamed selected-file sync failed, but SFTP fallback completed successfully.\n",
                        )
                else:
                    hooks.close_paramiko_sftp_fn(config)
                    hooks.sftp_copy_files_fn(
                        hooks.get_paramiko_sftp_fn(config),
                        remote_result_dir.as_posix(),
                        local_result_dir,
                        include_files,
                    )
            elif bool(config.get("remote_sync_compress", True)):
                probe_completed = hooks.run_paramiko_shell_fn(
                    config,
                    hooks.build_remote_archive_probe_command_fn(remote_result_dir),
                )
                if probe_completed.returncode != 0:
                    return subprocess.CompletedProcess(
                        args=["paramiko-probe", remote_result_dir.as_posix(), str(local_result_dir)],
                        returncode=1,
                        stdout=probe_completed.stdout or "",
                        stderr=probe_completed.stderr or "",
                    )
                probe_lines = [line.strip() for line in (probe_completed.stdout or "").splitlines() if line.strip()]
                if len(probe_lines) < 3:
                    return subprocess.CompletedProcess(
                        args=["paramiko-probe", remote_result_dir.as_posix(), str(local_result_dir)],
                        returncode=1,
                        stdout=probe_completed.stdout or "",
                        stderr="Remote archive probe did not return the expected metadata",
                    )
                compressor, raw_bytes_text, _archive_suffix = probe_lines[:3]
                raw_bytes = int(raw_bytes_text or "0")
                stream_completed = hooks.stream_archive_to_local_dir_fn(
                    config,
                    remote_result_dir=remote_result_dir,
                    local_result_dir=local_result_dir,
                    compressor=compressor,
                    raw_bytes=raw_bytes,
                )
                missing_stream_files = hooks.missing_local_sync_artifacts_fn(local_result_dir, expected_files)
                if stream_completed.returncode == 0 and missing_stream_files:
                    expected_text = ", ".join(missing_stream_files)
                    stream_completed = subprocess.CompletedProcess(
                        args=stream_completed.args,
                        returncode=1,
                        stdout=stream_completed.stdout or "",
                        stderr=(stream_completed.stderr or "")
                        + (
                            "\n[OBGPU load] Streamed archive sync produced no usable local artifacts. "
                            f"Missing: {expected_text}\n"
                        ),
                    )
                if stream_completed.returncode != 0:
                    hooks.progress_write(
                        "[OBGPU load] Streamed archive sync failed; retrying the same result dir over SFTP..."
                    )
                    hooks.close_paramiko_sftp_fn(config)
                    hooks.sftp_copy_tree_fn(
                        hooks.get_paramiko_sftp_fn(config),
                        remote_result_dir.as_posix(),
                        local_result_dir,
                    )
                    missing_fallback_files = hooks.missing_local_sync_artifacts_fn(local_result_dir, expected_files)
                    if missing_fallback_files:
                        expected_text = ", ".join(missing_fallback_files)
                        return subprocess.CompletedProcess(
                            args=["paramiko-stream-extract-fallback", remote_result_dir.as_posix(), str(local_result_dir)],
                            returncode=1,
                            stdout=stream_completed.stdout or "",
                            stderr=(stream_completed.stderr or "")
                            + (
                                "\n[OBGPU load] Stream archive sync failed, SFTP fallback ran, "
                                f"but required local artifacts are still missing: {expected_text}\n"
                            ),
                        )
                    return subprocess.CompletedProcess(
                        args=["paramiko-stream-extract-fallback", remote_result_dir.as_posix(), str(local_result_dir)],
                        returncode=0,
                        stdout=stream_completed.stdout or "",
                        stderr=(stream_completed.stderr or "")
                        + "\n[OBGPU load] Stream archive sync failed, but SFTP fallback completed successfully.\n",
                    )
            else:
                hooks.sftp_copy_tree_fn(
                    hooks.get_paramiko_sftp_fn(config),
                    remote_result_dir.as_posix(),
                    local_result_dir,
                )
        except Exception as exc:
            last_exc = exc
            hooks.close_paramiko_sftp_fn(config)
            transport_usable = hooks.transport_is_usable_fn(hooks.cached_transport_fn(config))
            if not transport_usable:
                if hooks.preserve_reauth_blocked_fn(config):
                    return subprocess.CompletedProcess(
                        args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
                        returncode=1,
                        stdout="",
                        stderr=hooks.midrun_reauth_error_fn(config) + f"\nOriginal error: {exc}",
                    )
                hooks.drop_paramiko_connection_fn(config)
            if attempt == 0 and transport_usable:
                continue
            return subprocess.CompletedProcess(
                args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
                returncode=1,
                stdout="",
                stderr=str(exc),
            )
        finally:
            hooks.close_paramiko_sftp_fn(config)

        missing_direct_files = hooks.missing_local_sync_artifacts_fn(local_result_dir, expected_files)
        if missing_direct_files:
            expected_text = ", ".join(missing_direct_files)
            return subprocess.CompletedProcess(
                args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
                returncode=1,
                stdout="",
                stderr=(
                    "[OBGPU load] Paramiko sync completed without producing the expected local artifacts: "
                    f"{expected_text}"
                ),
            )
        return subprocess.CompletedProcess(
            args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
            returncode=0,
            stdout="",
            stderr="",
        )

    return subprocess.CompletedProcess(
        args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
        returncode=1,
        stdout="",
        stderr=str(last_exc) if last_exc is not None else "unknown paramiko sftp failure",
    )


def sync_remote_result_dir_resilient(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    expected_files: tuple[str, ...] | None = None,
    include_files: tuple[str, ...] | None = None,
    wrapper_dir: str | PurePosixPath | None = None,
    retry_delay_s: float = 2.0,
    hooks: RemoteResultSyncHooks,
) -> subprocess.CompletedProcess[str]:
    """Sync remote results while treating selected-file sync as an optimization."""

    def complete_enough(completed: subprocess.CompletedProcess[str]) -> bool:
        if completed.returncode == 0:
            return True
        if not hooks.missing_local_sync_artifacts_fn(Path(local_result_dir), expected_files):
            return True
        return False

    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    completed = sync_remote_result_dir(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=Path(local_result_dir),
        expected_files=expected_files,
        include_files=include_files,
        hooks=hooks,
    )
    attempts.append(("selected fast sync" if include_files else "full result sync", completed))
    if complete_enough(completed):
        return completed

    if include_files:
        hooks.progress_write("[OBGPU load] Fast remote artifact sync was incomplete; retrying once...")
        if retry_delay_s > 0:
            hooks.sleep_fn(float(retry_delay_s))
        completed = sync_remote_result_dir(
            config,
            remote_result_dir=remote_result_dir,
            local_result_dir=Path(local_result_dir),
            expected_files=expected_files,
            include_files=include_files,
            hooks=hooks,
        )
        attempts.append(("selected fast sync retry", completed))
        if complete_enough(completed):
            return completed

        hooks.progress_write("[OBGPU load] Fast remote artifact sync still missing files; falling back to full result sync...")
        completed = sync_remote_result_dir(
            config,
            remote_result_dir=remote_result_dir,
            local_result_dir=Path(local_result_dir),
            expected_files=expected_files,
            hooks=hooks,
        )
        attempts.append(("full result sync fallback", completed))
        if complete_enough(completed):
            return completed

    if wrapper_dir not in (None, ""):
        hooks.progress_write("[OBGPU load] Result payload sync failed; syncing wrapper diagnostics...")
        wrapper_completed = sync_remote_result_dir(
            config,
            remote_result_dir=PurePosixPath(str(wrapper_dir)),
            local_result_dir=Path(local_result_dir),
            hooks=hooks,
        )
        attempts.append(("wrapper diagnostic sync", wrapper_completed))

    return subprocess.CompletedProcess(
        args=["remote-result-sync-resilient", remote_result_dir.as_posix(), str(local_result_dir)],
        returncode=1,
        stdout="\n".join((completed.stdout or "") for _label, completed in attempts if completed.stdout),
        stderr=combine_sync_attempt_stderr(attempts),
    )
