"""Generic manifest/signature helpers for uploaded remote helper bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class HelperBundleEntry:
    """One local file that should appear at one relative path in a helper bundle."""

    relative_path: str
    source_path: Path

    def normalized_relative_path(self) -> str:
        """Return the validated POSIX relative path for this entry."""
        return normalize_helper_relative_path(self.relative_path)


def normalize_helper_relative_path(relative_path: str) -> str:
    """Normalize and validate one bundle-relative path."""
    text = str(relative_path).strip().replace("\\", "/")
    if not text:
        raise ValueError("Helper bundle relative_path must not be empty")
    path = PurePosixPath(text)
    if path.is_absolute():
        raise ValueError(f"Helper bundle relative_path must be relative: {relative_path!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Helper bundle relative_path contains invalid segments: {relative_path!r}")
    return path.as_posix()


def bundle_entries_by_path(entries: tuple[HelperBundleEntry, ...] | list[HelperBundleEntry]) -> dict[str, Path]:
    """Return one ordered mapping from relative bundle path to local source path."""
    mapping: dict[str, Path] = {}
    for entry in entries:
        relative_path = entry.normalized_relative_path()
        if relative_path in mapping:
            raise ValueError(f"Duplicate helper bundle relative path: {relative_path}")
        mapping[relative_path] = Path(entry.source_path)
    return mapping


def helper_bundle_signature(entries: tuple[HelperBundleEntry, ...] | list[HelperBundleEntry]) -> str:
    """Return a content-addressed signature for one helper bundle."""
    digest = sha256()
    for relative_path, source_path in sorted(bundle_entries_by_path(entries).items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def helper_bundle_parent_dirs(entries: tuple[HelperBundleEntry, ...] | list[HelperBundleEntry]) -> tuple[str, ...]:
    """Return parent directories that must exist before uploading one helper bundle."""
    parents = {
        str(PurePosixPath(relative_path).parent)
        for relative_path in bundle_entries_by_path(entries)
        if str(PurePosixPath(relative_path).parent) not in {"", "."}
    }
    return tuple(sorted(parents))


def helper_bundle_manifest(
    entries: tuple[HelperBundleEntry, ...] | list[HelperBundleEntry],
    *,
    signature: str | None = None,
) -> dict[str, object]:
    """Return the portable manifest payload for one helper bundle."""
    paths = sorted(bundle_entries_by_path(entries).keys())
    return {
        "signature": str(signature or helper_bundle_signature(entries)),
        "files": paths,
        "parent_dirs": list(helper_bundle_parent_dirs(entries)),
    }
