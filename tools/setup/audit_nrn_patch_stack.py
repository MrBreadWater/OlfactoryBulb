"""Audit a vendored NEURON checkout against the maintained OBGPU patch stack.

This catches a specific failure mode we have already hit: a local source file
exists under ``external/nrn-9.0.1`` and the build works in one checkout, but the
file was never captured in ``third_party_patches/nrn`` so a clean rebuild on a
different machine fails.

The audit is intentionally simple:
- every tracked modified file in the source tree must be mentioned by at least
  one patch in the manifest
- every non-build untracked file in the source tree must also be mentioned by
  at least one patch in the manifest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str], *, cwd: Path) -> str:
    """Run a command and return stdout, raising on failure."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Command failed: {rendered}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the JSON patch manifest."""
    return json.loads(path.read_text())


def load_patch_texts(manifest: dict[str, Any], manifest_dir: Path) -> dict[str, str]:
    """Return patch contents keyed by patch filename."""
    patch_texts: dict[str, str] = {}
    for patch in manifest.get("patches", []):
        patch_path = manifest_dir / patch["file"]
        patch_texts[patch["file"]] = patch_path.read_text()
    return patch_texts


def normalize_paths(paths: list[str]) -> list[str]:
    """Drop generated/build paths and keep stable source paths only."""
    filtered: list[str] = []
    for path in paths:
        if not path:
            continue
        if path.startswith("build-ob-modern"):
            continue
        if path.startswith(".git/"):
            continue
        filtered.append(path)
    return filtered


def find_untracked_paths(source_tree: Path) -> list[str]:
    """Return non-build untracked files in the vendored source tree."""
    stdout = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=source_tree)
    return normalize_paths([line.strip() for line in stdout.splitlines()])


def find_modified_paths(source_tree: Path) -> list[str]:
    """Return tracked modified files in the vendored source tree."""
    stdout = run(["git", "diff", "--name-only"], cwd=source_tree)
    return normalize_paths([line.strip() for line in stdout.splitlines()])


def match_paths_to_patches(paths: list[str], patch_texts: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Match each path to a patch that mentions it, returning unmatched paths."""
    unmatched: list[str] = []
    matched_by: dict[str, str] = {}
    for path in paths:
        matched_patch = next((name for name, text in patch_texts.items() if path in text), None)
        if matched_patch is None:
            unmatched.append(path)
        else:
            matched_by[path] = matched_patch
    return unmatched, matched_by


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-tree",
        default=Path(__file__).resolve().parents[2] / "external" / "nrn-9.0.1",
        type=Path,
        help="Vendored NEURON checkout to audit.",
    )
    parser.add_argument(
        "--manifest",
        default=Path(__file__).resolve().parents[2] / "third_party_patches" / "nrn" / "manifest.json",
        type=Path,
        help="Patch manifest to audit against.",
    )
    return parser.parse_args()


def main() -> None:
    """Audit the current vendored NEURON checkout and emit a readable summary."""
    args = parse_args()
    source_tree = args.source_tree.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    patch_texts = load_patch_texts(manifest, manifest_path.parent)

    modified_paths = find_modified_paths(source_tree)
    untracked_paths = find_untracked_paths(source_tree)

    unmatched_modified, modified_matches = match_paths_to_patches(modified_paths, patch_texts)
    unmatched_untracked, untracked_matches = match_paths_to_patches(untracked_paths, patch_texts)

    summary = {
        "ok": not unmatched_modified and not unmatched_untracked,
        "source_tree": str(source_tree),
        "manifest": str(manifest_path),
        "modified_paths": modified_matches,
        "untracked_paths": untracked_matches,
        "unmatched_modified_paths": unmatched_modified,
        "unmatched_untracked_paths": unmatched_untracked,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if unmatched_modified or unmatched_untracked:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI failure formatting
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True))
        raise
