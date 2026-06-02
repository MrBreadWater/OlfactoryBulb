"""Reusable remote sweep final sync and artifact-finalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteSweepArtifactHooks:
    """Callbacks injected by the notebook-facing caller for sweep finalization."""

    sync_remote_result_dir_fn: Callable[..., subprocess.CompletedProcess[str]]
    sync_remote_sweep_compact_items_fn: Callable[..., subprocess.CompletedProcess[str]]
    read_json_if_present_fn: Callable[[str | Path], dict[str, Any] | None]
    recover_local_sweep_summary_fn: Callable[..., dict[str, Any]]
    remote_sweep_metadata_files_fn: Callable[[], tuple[str, ...]]
    remote_sweep_item_sync_files_fn: Callable[[dict[str, Any]], tuple[str, ...]]
    remote_sweep_item_diagnostic_files_fn: Callable[[], tuple[str, ...]]
    local_sweep_item_sync_complete_fn: Callable[[str | Path], bool]
    local_result_dir_has_diagnostics_fn: Callable[[str | Path], bool]
    progress_write: Callable[[str], None]
    refresh_remote_leases_fn: Callable[..., None]
    record_timing_fn: Callable[[str, float], None]
    perf_counter_fn: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class RemoteSweepArtifactResult:
    """Collected local sweep metadata and compact item artifacts."""

    final_sync: subprocess.CompletedProcess[str]
    sweep_summary: dict[str, Any]
    item_status_by_label: dict[str, dict[str, Any]]


def _record_completed(
    completed: subprocess.CompletedProcess[str],
    *,
    prefix: str,
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> None:
    if completed.stdout:
        stdout_parts.append(f"[{prefix}]\n{completed.stdout}")
    if completed.stderr:
        stderr_parts.append(f"[{prefix}]\n{completed.stderr}")


def _final_sync_completed_process(
    *,
    remote_sweep_root: PurePosixPath,
    local_sweep_dir: Path,
    returncode: int,
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["remote-sweep-compact-sync", remote_sweep_root.as_posix(), str(local_sweep_dir)],
        returncode=returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
    )


def _persist_sync_logs(local_sweep_dir: Path, completed: subprocess.CompletedProcess[str]) -> None:
    (local_sweep_dir / "sync_stdout.txt").write_text(completed.stdout or "")
    (local_sweep_dir / "sync_stderr.txt").write_text(completed.stderr or "")


def finalize_remote_sweep_artifacts(
    config: dict[str, Any],
    *,
    final_status: dict[str, Any] | None,
    local_sweep_dir: str | Path,
    local_runs_dir: str | Path,
    remote_sweep_root: PurePosixPath,
    sweep_label: str,
    manifest_items: list[dict[str, Any]],
    item_status_by_label: dict[str, dict[str, Any]],
    hooks: RemoteSweepArtifactHooks,
) -> RemoteSweepArtifactResult:
    """Sync compact sweep artifacts and recover or load the sweep summary."""
    local_sweep_dir = Path(local_sweep_dir)
    local_runs_dir = Path(local_runs_dir)
    local_sweep_dir.mkdir(parents=True, exist_ok=True)
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    sync_started = hooks.perf_counter_fn()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    final_progress_payload = (final_status or {}).get("progress_payload") if final_status else None
    if isinstance(final_progress_payload, dict) and not (local_sweep_dir / "sim_progress.json").exists():
        (local_sweep_dir / "sim_progress.json").write_text(
            json.dumps(final_progress_payload, indent=2, sort_keys=True)
        )

    hooks.refresh_remote_leases_fn()
    metadata_sync = hooks.sync_remote_result_dir_fn(
        config,
        remote_result_dir=remote_sweep_root,
        local_result_dir=local_sweep_dir,
        expected_files=("summary.json",),
        include_files=hooks.remote_sweep_metadata_files_fn(),
    )
    hooks.refresh_remote_leases_fn()
    _record_completed(
        metadata_sync,
        prefix="sweep-metadata",
        stdout_parts=stdout_parts,
        stderr_parts=stderr_parts,
    )

    sweep_summary = hooks.read_json_if_present_fn(local_sweep_dir / "summary.json") or hooks.recover_local_sweep_summary_fn(
        local_sweep_dir,
        sweep_label=sweep_label,
        total_items=len(manifest_items),
    )
    if metadata_sync.returncode != 0 and sweep_summary:
        stderr_parts.append(
            "[OBGPU load] Incremental sweep metadata sync reported an error, "
            "but local progress metadata was sufficient to recover a sweep summary.\n"
        )
    if not sweep_summary:
        stderr_parts.append(
            "[OBGPU load] Incremental sweep final sync could not fetch summary metadata; "
            "not attempting a bulk sweep-root sync because that would pull raw soma traces.\n"
        )
        final_sync = _final_sync_completed_process(
            remote_sweep_root=remote_sweep_root,
            local_sweep_dir=local_sweep_dir,
            returncode=metadata_sync.returncode or 1,
            stdout_parts=stdout_parts,
            stderr_parts=stderr_parts,
        )
        _persist_sync_logs(local_sweep_dir, final_sync)
        hooks.record_timing_fn("sync_s", sync_started)
        return RemoteSweepArtifactResult(
            final_sync=final_sync,
            sweep_summary={},
            item_status_by_label={
                str(label): dict(payload)
                for label, payload in item_status_by_label.items()
                if isinstance(payload, dict)
            },
        )

    summary_by_label: dict[str, dict[str, Any]] = {}
    for bucket in ("completed_items", "failed_items", "items"):
        for payload in sweep_summary.get(bucket, []) or []:
            if isinstance(payload, dict) and payload.get("label"):
                summary_by_label[str(payload["label"])] = dict(payload)

    bulk_sync_entries: list[dict[str, Any]] = []
    for item in manifest_items:
        label = str(item["label"])
        payload = summary_by_label.get(label)
        if payload is None:
            continue
        local_result_dir = local_runs_dir / label
        ok = bool(payload.get("ok", False))
        if ok and hooks.local_sweep_item_sync_complete_fn(local_result_dir):
            continue
        if not ok and hooks.local_result_dir_has_diagnostics_fn(local_result_dir):
            continue
        remote_result_dir = PurePosixPath(str(payload.get("result_dir") or item["result_dir"]))
        include_files = (
            hooks.remote_sweep_item_sync_files_fn(config)
            if ok
            else hooks.remote_sweep_item_diagnostic_files_fn()
        )
        bulk_sync_entries.append(
            {
                "label": label,
                "result_dir": remote_result_dir.as_posix(),
                "include_files": list(include_files),
                "ok": ok,
            }
        )

    if bulk_sync_entries:
        hooks.progress_write(
            f"[OBGPU load] Syncing compact artifacts for {len(bulk_sync_entries)} sweep items in one stream..."
        )
        hooks.refresh_remote_leases_fn()
        bulk_sync = hooks.sync_remote_sweep_compact_items_fn(
            config,
            local_sweep_dir=local_sweep_dir,
            entries=bulk_sync_entries,
        )
        hooks.refresh_remote_leases_fn()
        _record_completed(
            bulk_sync,
            prefix="sweep-items-bulk",
            stdout_parts=stdout_parts,
            stderr_parts=stderr_parts,
        )
        if bulk_sync.returncode != 0:
            stderr_parts.append(
                "[OBGPU load] Bulk compact sweep item sync reported an error; "
                "continuing with any local artifacts already available.\n"
            )
        for entry in bulk_sync_entries:
            label = str(entry["label"])
            local_result_dir = local_runs_dir / label
            if bool(entry.get("ok", False)):
                if not hooks.local_sweep_item_sync_complete_fn(local_result_dir):
                    stderr_parts.append(
                        f"[OBGPU load] Compact artifacts for {label} are still incomplete after bulk sync.\n"
                    )
            elif not hooks.local_result_dir_has_diagnostics_fn(local_result_dir):
                stderr_parts.append(
                    f"[OBGPU load] Diagnostics for failed sweep item {label} are still incomplete after bulk sync.\n"
                )

    merged_item_status_by_label = {
        str(label): dict(payload)
        for label, payload in item_status_by_label.items()
        if isinstance(payload, dict)
    }
    for bucket in ("completed_items", "failed_items", "items"):
        for payload in sweep_summary.get(bucket, []) or []:
            if isinstance(payload, dict) and payload.get("label"):
                merged_item_status_by_label[str(payload["label"])] = dict(payload)

    final_sync = _final_sync_completed_process(
        remote_sweep_root=remote_sweep_root,
        local_sweep_dir=local_sweep_dir,
        returncode=0,
        stdout_parts=stdout_parts,
        stderr_parts=stderr_parts,
    )
    _persist_sync_logs(local_sweep_dir, final_sync)
    hooks.record_timing_fn("sync_s", sync_started)
    return RemoteSweepArtifactResult(
        final_sync=final_sync,
        sweep_summary=sweep_summary,
        item_status_by_label=merged_item_status_by_label,
    )
