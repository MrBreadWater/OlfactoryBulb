"""Regression tests for generic dashboard packet discovery helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from neuroinfra.dashboard.packets import (
    PacketInfo,
    candidate_id_from_path,
    cleanup_packet_dirs,
    discover_packets,
    packet_mtime,
    read_json_dict,
)


assert candidate_id_from_path("packet_C00042") == "C00042"
assert candidate_id_from_path("nope") is None
assert read_json_dict("/tmp/does-not-exist.json") == {}

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    figures = root / "figures"
    figures.mkdir()

    hidden = figures / ".tmp_packet"
    hidden.mkdir()
    (hidden / "manifest.json").write_text("{}")

    older = figures / "packet_C00042_old"
    older.mkdir()
    (older / "manifest.json").write_text(json.dumps({"candidate_id": "C00042", "keep": True}))
    (older / "image.png").write_bytes(b"old")

    newer = figures / "packet_C00042_new"
    newer.mkdir()
    (newer / "manifest.json").write_text(json.dumps({"candidate_id": "C00042", "keep": True}))
    (newer / "image.png").write_bytes(b"new")
    (newer / "contact_sheet.png").write_bytes(b"contact")
    (newer / "legacy.png").write_bytes(b"legacy")
    os.utime(older / "manifest.json", (10, 10))
    os.utime(older / "image.png", (10, 10))
    os.utime(older, (10, 10))
    os.utime(newer / "manifest.json", (20, 20))
    os.utime(newer / "image.png", (20, 20))
    os.utime(newer / "contact_sheet.png", (20, 20))
    os.utime(newer / "legacy.png", (20, 20))
    os.utime(newer, (20, 20))

    stale = figures / "packet_C00043"
    stale.mkdir()
    (stale / "manifest.json").write_text(json.dumps({"candidate_id": "C00043", "keep": False}))
    (stale / "image.png").write_bytes(b"stale")

    discovered = discover_packets(
        figures,
        packet_keep_predicate=lambda _packet_dir, manifest: bool(manifest.get("keep")),
        hidden_dir_predicate=lambda path: path.name.startswith("."),
        exclude_image_predicate=lambda path: path.name == "legacy.png",
    )
    assert set(discovered) == {"C00042"}
    packet = discovered["C00042"]
    assert isinstance(packet, PacketInfo)
    assert packet.packet_dir == newer
    assert packet.contact_sheet == (newer / "contact_sheet.png")
    assert packet.images == (newer / "image.png",)
    assert packet.mtime == packet_mtime([newer / "manifest.json", newer / "contact_sheet.png", newer / "image.png"])

    removed_count, removed_names = cleanup_packet_dirs(
        figures,
        keep_packet_predicate=lambda _packet_dir, manifest: bool(manifest.get("keep")),
        hidden_dir_predicate=lambda path: path.name.startswith("."),
        dry_run=True,
    )
    assert removed_count == 1
    assert removed_names == ["packet_C00043"]
    assert stale.exists()

    removed_count, removed_names = cleanup_packet_dirs(
        figures,
        keep_packet_predicate=lambda _packet_dir, manifest: bool(manifest.get("keep")),
        hidden_dir_predicate=lambda path: path.name.startswith("."),
        dry_run=False,
    )
    assert removed_count == 1
    assert removed_names == ["packet_C00043"]
    assert not stale.exists()
