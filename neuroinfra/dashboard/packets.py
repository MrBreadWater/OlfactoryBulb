"""Generic manifest-backed packet discovery and cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence


PacketKeepPredicate = Callable[[Path, dict[str, Any]], bool]
PacketImageExcludePredicate = Callable[[Path], bool]
PacketHiddenDirPredicate = Callable[[Path], bool]
PacketCandidateIdResolver = Callable[[Path, dict[str, Any]], str | None]


@dataclass(frozen=True)
class PacketInfo:
    candidate_id: str
    packet_dir: Path
    contact_sheet: Path | None
    images: tuple[Path, ...]
    manifest: dict[str, Any]
    mtime: float


def read_json_dict(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def candidate_id_from_path(path: str | Path, *, pattern: str = r"(C\d+)") -> str | None:
    match = re.search(pattern, Path(path).name)
    return match.group(1) if match else None


def packet_mtime(paths: Sequence[Path]) -> float:
    mtimes: list[float] = []
    for path in paths:
        try:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        except (FileNotFoundError, OSError):
            continue
    return max(mtimes) if mtimes else 0.0


def discover_packets(
    root_dir: str | Path,
    *,
    manifest_name: str = "manifest.json",
    packet_keep_predicate: PacketKeepPredicate | None = None,
    hidden_dir_predicate: PacketHiddenDirPredicate | None = None,
    exclude_image_predicate: PacketImageExcludePredicate | None = None,
    candidate_id_resolver: PacketCandidateIdResolver | None = None,
    contact_sheet_names: Sequence[str] = ("contact_sheet.png", "00_contact_sheet.png"),
) -> dict[str, PacketInfo]:
    packet_root = Path(root_dir)
    if not packet_root.exists():
        return {}

    keep_packet = packet_keep_predicate or (lambda _packet_dir, _manifest: True)
    hidden_dir = hidden_dir_predicate or (lambda _path: False)
    exclude_image = exclude_image_predicate or (lambda _path: False)

    def resolve_candidate_id(packet_dir: Path, manifest: dict[str, Any]) -> str | None:
        if candidate_id_resolver is not None:
            return candidate_id_resolver(packet_dir, manifest)
        value = manifest.get("candidate_id")
        if isinstance(value, str) and value:
            return value
        return candidate_id_from_path(packet_dir)

    packets: dict[str, PacketInfo] = {}
    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir()):
        if hidden_dir(packet_dir):
            continue
        manifest_path = packet_dir / manifest_name
        manifest = read_json_dict(manifest_path) if manifest_path.exists() else {}
        if not keep_packet(packet_dir, manifest):
            continue
        candidate_id = str(resolve_candidate_id(packet_dir, manifest) or "")
        if not candidate_id:
            continue
        contact_sheet = None
        for name in contact_sheet_names:
            candidate = packet_dir / str(name)
            if candidate.exists():
                contact_sheet = candidate
                break
        images = tuple(
            sorted(
                path
                for path in packet_dir.glob("*.png")
                if path != contact_sheet and not exclude_image(path)
            )
        )
        packet = PacketInfo(
            candidate_id=candidate_id,
            packet_dir=packet_dir,
            contact_sheet=contact_sheet,
            images=images,
            manifest=manifest,
            mtime=packet_mtime([manifest_path, contact_sheet or packet_dir, *images]),
        )
        previous = packets.get(candidate_id)
        if previous is None or packet.mtime >= previous.mtime:
            packets[candidate_id] = packet
    return packets


def cleanup_packet_dirs(
    root_dir: str | Path,
    *,
    manifest_name: str = "manifest.json",
    keep_packet_predicate: PacketKeepPredicate,
    hidden_dir_predicate: PacketHiddenDirPredicate | None = None,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    packet_root = Path(root_dir)
    if not packet_root.exists():
        return 0, []

    hidden_dir = hidden_dir_predicate or (lambda _path: False)
    removed: list[str] = []
    for packet_dir in sorted(path for path in packet_root.iterdir() if path.is_dir()):
        if hidden_dir(packet_dir):
            continue
        manifest_path = packet_dir / manifest_name
        manifest = read_json_dict(manifest_path) if manifest_path.exists() else {}
        if keep_packet_predicate(packet_dir, manifest):
            continue
        removed.append(packet_dir.name)
        if dry_run:
            continue
        try:
            shutil.rmtree(packet_dir)
        except FileNotFoundError:
            continue
    return len(removed), removed


__all__ = [
    "PacketInfo",
    "candidate_id_from_path",
    "cleanup_packet_dirs",
    "discover_packets",
    "packet_mtime",
    "read_json_dict",
]
