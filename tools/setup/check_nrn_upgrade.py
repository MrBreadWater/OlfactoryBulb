"""Validate a candidate upstream NEURON ref against the local OBGPU patch stack.

This helper is the upgrade gate for the maintained OBGPU NEURON/CoreNEURON
integration. It checks out a clean candidate upstream tree, reapplies the repo's
patch manifest, and can optionally rebuild the environment plus run smoke
commands. The supported version only changes when this script succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one command and raise a readable error on failure."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
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
    return completed


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the pinned NEURON patch manifest."""
    with open(path) as handle:
        return json.load(handle)


def write_candidate_manifest(
    *,
    source_manifest: dict[str, Any],
    candidate_ref: str,
    destination: Path,
) -> Path:
    """Write a temporary manifest identical to the source one except for the ref."""
    updated = dict(source_manifest)
    updated["upstream_ref"] = candidate_ref
    destination.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    return destination


def clone_candidate_tree(*, upstream_repo: str, candidate_ref: str, destination: Path) -> Path:
    """Clone a clean upstream tree at the requested ref."""
    run(["git", "clone", "--recursive", upstream_repo, str(destination)])
    run(["git", "fetch", "--tags", "--force", "origin"], cwd=destination)
    run(["git", "checkout", "--force", candidate_ref], cwd=destination)
    run(["git", "submodule", "sync", "--recursive"], cwd=destination)
    run(["git", "submodule", "update", "--init", "--recursive", "--force"], cwd=destination)
    return destination


def apply_patch_stack(*, repo_root: Path, source_tree: Path, manifest: dict[str, Any], manifest_dir: Path) -> None:
    """Apply the repo-managed patch stack to a clean candidate source tree."""
    for patch_entry in manifest.get("patches", []):
        patch_name = patch_entry["file"]
        patch_candidates = [
            manifest_dir / patch_name,
            repo_root / "third_party_patches" / "nrn" / patch_name,
            Path(patch_name),
        ]
        patch_path = next((candidate for candidate in patch_candidates if candidate.exists()), None)
        if patch_path is None:
            raise FileNotFoundError(
                f"Patch file not found. Tried: {', '.join(str(candidate) for candidate in patch_candidates)}"
            )
        run(["git", "apply", "--check", str(patch_path)], cwd=source_tree)
        run(["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=source_tree)


def default_mpi_exec() -> str:
    """Return the preferred smoke-test MPI launcher for the current shell."""
    configured = os.environ.get("OB_MPIEXEC")
    if configured:
        return configured

    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        slurm_mpi_type = os.environ.get("OB_SLURM_MPI_TYPE", "pmix").strip()
        if slurm_mpi_type:
            return f"srun --mpi={slurm_mpi_type}"
        return "srun"

    return "mpiexec"


def default_smoke_commands(*, enable_gpu: bool) -> list[str]:
    """Return the minimal default smoke matrix for an upgrade candidate."""
    command = (
        f"{default_mpi_exec()} -n 1 nrniv -mpi -python tools/benchmarks/benchmark_ob.py "
        "--label nrn_upgrade_smoke_onems --paramset OneMsTest --coreneuron"
    )
    if enable_gpu:
        command += " --coreneuron-gpu"
    return [
        'python -c "import neuron; from neuron import coreneuron; print(neuron.__version__)"',
        command,
    ]


def run_setup_and_smokes(
    *,
    repo_root: Path,
    source_tree: Path,
    manifest_path: Path,
    enable_gpu: bool,
    env_name: str,
    smoke_commands: list[str],
) -> list[dict[str, Any]]:
    """Build OBGPU against the candidate tree and then run the smoke commands."""
    env = os.environ.copy()
    env["ENV_NAME"] = env_name
    env["ENABLE_GPU"] = "1" if enable_gpu else "0"
    env["NRN_SRC_DIR"] = str(source_tree)
    env["NRN_PATCH_MANIFEST"] = str(manifest_path)
    run([str(repo_root / "tools" / "setup" / "setup_ob_modern.sh")], cwd=repo_root, env=env)

    results: list[dict[str, Any]] = []
    for smoke_command in smoke_commands:
        completed = run(["bash", "-lc", smoke_command], cwd=repo_root, env=env)
        results.append(
            {
                "command": smoke_command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the upgrade gate helper."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ref", required=True, help="Upstream NEURON tag or commit to test.")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Repo root containing the patch manifest and setup script.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        type=Path,
        help="Patch manifest to use. Defaults to third_party_patches/nrn/manifest.json.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        type=Path,
        help="Persistent working directory. Defaults to a temporary directory.",
    )
    parser.add_argument("--env-name", default="OBGPU-upgrade-check", help="Conda env name used for the test build.")
    parser.add_argument("--enable-gpu", action="store_true", help="Run the candidate build with ENABLE_GPU=1.")
    parser.add_argument("--skip-build", action="store_true", help="Stop after clone + patch application.")
    parser.add_argument(
        "--smoke-command",
        action="append",
        default=[],
        help="Extra shell command to run after a successful build. May be repeated.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the temporary workdir even when --workdir was not provided.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the upgrade check and emit a JSON summary."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else repo_root / "third_party_patches" / "nrn" / "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    temp_root_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir is None:
        temp_root_obj = tempfile.TemporaryDirectory(prefix="nrn-upgrade-check-")
        workdir = Path(temp_root_obj.name)
    else:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    source_tree = workdir / "nrn"
    candidate_manifest_path = workdir / "candidate_manifest.json"

    summary: dict[str, Any] = {
        "candidate_ref": args.candidate_ref,
        "repo_root": str(repo_root),
        "source_tree": str(source_tree),
        "manifest_path": str(manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "build_ran": False,
        "smokes_ran": [],
    }

    try:
        if source_tree.exists():
            shutil.rmtree(source_tree)
        clone_candidate_tree(
            upstream_repo=str(manifest["upstream_repo"]),
            candidate_ref=str(args.candidate_ref),
            destination=source_tree,
        )
        write_candidate_manifest(
            source_manifest=manifest,
            candidate_ref=str(args.candidate_ref),
            destination=candidate_manifest_path,
        )
        apply_patch_stack(
            repo_root=repo_root,
            source_tree=source_tree,
            manifest=manifest,
            manifest_dir=manifest_dir,
        )

        if not args.skip_build:
            smoke_commands = args.smoke_command or default_smoke_commands(enable_gpu=args.enable_gpu)
            summary["smokes_ran"] = run_setup_and_smokes(
                repo_root=repo_root,
                source_tree=source_tree,
                manifest_path=candidate_manifest_path,
                enable_gpu=bool(args.enable_gpu),
                env_name=str(args.env_name),
                smoke_commands=smoke_commands,
            )
            summary["build_ran"] = True

        summary["ok"] = True
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if temp_root_obj is not None and not args.keep_workdir:
            temp_root_obj.cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI failure formatting
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True))
        raise
