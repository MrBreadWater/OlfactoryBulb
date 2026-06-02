"""Shared path helpers for the structured test tree."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_TOOLS_DIR = REPO_ROOT / "tools" / "remote"
