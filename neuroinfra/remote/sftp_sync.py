"""Reusable SFTP sync planning and transfer loops for notebook-managed runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat as _stat
from typing import Any, Callable


@dataclass(frozen=True)
class TransferItem:
    """One remote file slated for local synchronization."""

    remote_path: str
    local_path: Path
    size_bytes: int


@dataclass(frozen=True)
class SFTPSyncHooks:
    """UI and write hooks injected by the notebook-facing caller."""

    progress_factory: Callable[[int], Any]
    progress_write: Callable[[str], None]
    format_bytes: Callable[[int | float], str]
    render_progress_bar: Callable[[int | float, int | float], str]
    replace_file_via_temp_copy: Callable[[Callable[[Path], None], Path], None]


def collect_tree_transfer_items(sftp: Any, remote_dir: str, local_dir: Path) -> list[TransferItem]:
    """Recursively collect one remote directory tree into a transfer plan."""

    def collect(current_remote_dir: str, current_local_dir: Path) -> list[TransferItem]:
        current_local_dir.mkdir(parents=True, exist_ok=True)
        files: list[TransferItem] = []
        for entry in sftp.listdir_attr(current_remote_dir):
            remote_path = f"{current_remote_dir.rstrip('/')}/{entry.filename}"
            local_path = current_local_dir / entry.filename
            if _stat.S_ISDIR(entry.st_mode):
                files.extend(collect(remote_path, local_path))
                continue
            files.append(
                TransferItem(
                    remote_path=remote_path,
                    local_path=local_path,
                    size_bytes=int(getattr(entry, "st_size", 0)),
                )
            )
        return files

    return collect(str(remote_dir), Path(local_dir))


def collect_selected_transfer_items(
    sftp: Any,
    remote_dir: str,
    local_dir: Path,
    file_names: list[str] | tuple[str, ...],
) -> list[TransferItem]:
    """Collect a selected-file transfer plan, skipping missing optionals."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    transfer_plan: list[TransferItem] = []
    for name in file_names:
        remote_path = f"{str(remote_dir).rstrip('/')}/{name}"
        local_path = local_dir / name
        try:
            entry = sftp.stat(remote_path)
        except Exception:
            continue
        if _stat.S_ISDIR(entry.st_mode):
            continue
        transfer_plan.append(
            TransferItem(
                remote_path=remote_path,
                local_path=local_path,
                size_bytes=int(getattr(entry, "st_size", 0)),
            )
        )
    return transfer_plan


def execute_transfer_plan(
    sftp: Any,
    transfer_plan: list[TransferItem],
    *,
    hooks: SFTPSyncHooks,
    desc_prefix: str,
) -> None:
    """Execute one SFTP transfer plan with notebook-supplied progress hooks."""
    total_files = len(transfer_plan)
    total_bytes = sum(item.size_bytes for item in transfer_plan)
    transferred_bytes = 0
    progress = hooks.progress_factory(total_bytes)

    if total_files:
        hooks.progress_write(
            f"{desc_prefix} {total_files} files from Sol ({hooks.format_bytes(total_bytes)})...",
        )

    for index, item in enumerate(transfer_plan, start=1):
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        hooks.progress_write(
            f"[OBGPU load] Syncing {index}/{total_files}: {item.local_path.name} "
            f"({hooks.format_bytes(item.size_bytes)})",
        )
        base_bytes = transferred_bytes

        def callback(current_file_bytes: int, _current_file_total: int) -> None:
            overall_bytes = base_bytes + current_file_bytes
            progress.update_to(overall_bytes)

        hooks.replace_file_via_temp_copy(
            lambda temp_path: sftp.get(item.remote_path, str(temp_path), callback=callback),
            item.local_path,
        )
        transferred_bytes += item.size_bytes
        progress.update_to(transferred_bytes)

    if total_files:
        hooks.progress_write(
            f"[OBGPU load] Sync complete {hooks.render_progress_bar(total_bytes, total_bytes)} "
            f"{hooks.format_bytes(total_bytes)} / {hooks.format_bytes(total_bytes)}",
        )
    progress.close()


def sftp_copy_tree(
    sftp: Any,
    remote_dir: str,
    local_dir: Path,
    *,
    hooks: SFTPSyncHooks,
) -> None:
    """Recursively copy one remote directory tree through SFTP."""
    transfer_plan = collect_tree_transfer_items(sftp, remote_dir, Path(local_dir))
    execute_transfer_plan(
        sftp,
        transfer_plan,
        hooks=hooks,
        desc_prefix="[OBGPU load] Syncing",
    )


def sftp_copy_files(
    sftp: Any,
    remote_dir: str,
    local_dir: Path,
    file_names: list[str] | tuple[str, ...],
    *,
    hooks: SFTPSyncHooks,
) -> None:
    """Copy a selected set of remote files through SFTP."""
    transfer_plan = collect_selected_transfer_items(sftp, remote_dir, Path(local_dir), file_names)
    execute_transfer_plan(
        sftp,
        transfer_plan,
        hooks=hooks,
        desc_prefix="[OBGPU load] Syncing selected",
    )
