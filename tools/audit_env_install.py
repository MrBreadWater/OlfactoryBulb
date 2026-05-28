"""Compatibility wrapper for the environment/install audit."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.audit.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["env_install", *sys.argv[1:]]))
