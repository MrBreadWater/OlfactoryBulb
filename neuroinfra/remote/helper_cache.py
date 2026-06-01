"""Reusable helpers for remote helper-bundle cache layout and probing."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shlex

from .helper_bundle import (
    HelperBundleEntry,
    bundle_entries_by_path,
    helper_bundle_manifest,
    helper_bundle_parent_dirs,
)


def helper_cache_runtime_key(
    *,
    connection_key: str,
    results_root: PurePosixPath,
    signature: str,
) -> str:
    """Return the runtime cache key for one uploaded helper bundle."""
    return "{}::{}::{}".format(str(connection_key), results_root.as_posix(), str(signature))


def helper_cache_dir(
    *,
    results_root: PurePosixPath,
    signature: str,
) -> PurePosixPath:
    """Return the remote directory that stores one helper cache."""
    return results_root / ".obgpu-helper-cache" / str(signature)


def helper_cache_manifest_path(remote_dir: PurePosixPath) -> PurePosixPath:
    """Return the manifest path for one helper cache directory."""
    return remote_dir / "manifest.json"


def helper_cache_probe_command(manifest_path: PurePosixPath) -> str:
    """Build the shell command that prints one helper-cache manifest when present."""
    quoted = shlex.quote(manifest_path.as_posix())
    return "if test -f {path}; then cat {path}; fi".format(path=quoted)


def helper_cache_probe_matches(stdout_text: str, *, expected_signature: str) -> bool:
    """Return whether one manifest probe payload matches the requested signature."""
    text = str(stdout_text or "").strip()
    if not text:
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("signature") == expected_signature


def helper_cache_mkdir_targets(
    *,
    remote_dir: PurePosixPath,
    entries: tuple[HelperBundleEntry, ...] | list[HelperBundleEntry],
) -> tuple[str, ...]:
    """Return the directories that must exist before uploading one helper cache."""
    return (
        remote_dir.as_posix(),
        *[(remote_dir / parent).as_posix() for parent in helper_bundle_parent_dirs(entries)],
    )


def helper_cache_upload_payload(
    *,
    remote_dir: PurePosixPath,
    entries: tuple[HelperBundleEntry, ...] | list[HelperBundleEntry],
    signature: str,
) -> tuple[dict[str, Path], dict[str, object], PurePosixPath]:
    """Return the local sources, manifest payload, and manifest path for one upload."""
    return (
        bundle_entries_by_path(entries),
        helper_bundle_manifest(entries, signature=signature),
        helper_cache_manifest_path(remote_dir),
    )
