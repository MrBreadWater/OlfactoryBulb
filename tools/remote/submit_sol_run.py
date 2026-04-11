"""Submit one timestamped OBGPU benchmark run to Slurm on a remote host."""

import argparse
import json
import shlex
import subprocess
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict, List, Optional


def decode_command(payload_b64):
    """Decode a base64-encoded JSON command list."""
    command = json.loads(b64decode(payload_b64).decode("utf-8"))
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError("Decoded benchmark command must be a JSON list of strings")
    return command


def slurm_directives(args, label):
    """Return `#SBATCH` header lines for a generated batch script."""
    directives = ["#SBATCH --job-name={}".format(label[:120])]
    if args.partition:
        directives.append("#SBATCH --partition={}".format(args.partition))
    if args.account:
        directives.append("#SBATCH --account={}".format(args.account))
    if args.time:
        directives.append("#SBATCH --time={}".format(args.time))
    if args.gpus is not None:
        directives.append("#SBATCH --gpus={}".format(args.gpus))
    if args.cpus_per_task is not None:
        directives.append("#SBATCH --cpus-per-task={}".format(args.cpus_per_task))
    if args.mem:
        directives.append("#SBATCH --mem={}".format(args.mem))
    for extra in args.sbatch_arg:
        directives.append("#SBATCH {}".format(extra))
    return directives


def write_batch_script(
    repo_root,
    result_dir,
    label,
    conda_activate_cmd,
    benchmark_command,
    git_ref,
    git_fetch,
    git_remote,
    args,
):
    """Write the Slurm batch script that launches one benchmark run."""
    batch_path = result_dir / "slurm_job.sh"
    wrapper = repo_root / "tools" / "remote" / "run_obgpu_batch.sh"
    lines = [
        "#!/usr/bin/env bash",
        *slurm_directives(args, label),
        "set -euo pipefail",
        shlex.join(
            [
                str(wrapper),
                "--repo-root",
                str(repo_root),
                "--result-dir",
                str(result_dir),
                "--conda-activate-cmd",
                conda_activate_cmd,
                "--git-ref",
                git_ref or "",
                "--git-fetch",
                "1" if git_fetch else "0",
                "--git-remote",
                git_remote,
                "--",
                *benchmark_command,
            ]
        ),
        "",
    ]
    batch_path.write_text("\n".join(lines))
    batch_path.chmod(0o755)
    return batch_path


def submit_batch(batch_path):
    """Submit a generated Slurm script and return the parsed job id."""
    completed = subprocess.run(
        ["sbatch", "--parsable", str(batch_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "sbatch failed:\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                completed.stdout,
                completed.stderr,
            )
        )
    job_id = (completed.stdout or "").strip().split(";", 1)[0].strip()
    if not job_id:
        raise RuntimeError(
            "Could not parse Slurm job id from sbatch output: {!r}".format(completed.stdout)
        )
    return job_id


def main():
    """Parse CLI args, write the batch script, optionally submit it, and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--results-base", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--benchmark-command-b64", required=True)
    parser.add_argument("--conda-activate-cmd", required=True)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--git-fetch", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--sbatch-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result_dir = Path(args.results_base).expanduser().resolve() / args.label
    result_dir.mkdir(parents=True, exist_ok=True)

    benchmark_command = decode_command(args.benchmark_command_b64)
    batch_path = write_batch_script(
        repo_root=repo_root,
        result_dir=result_dir,
        label=args.label,
        conda_activate_cmd=args.conda_activate_cmd,
        benchmark_command=benchmark_command,
        git_ref=args.git_ref,
        git_fetch=bool(args.git_fetch),
        git_remote=str(args.git_remote),
        args=args,
    )

    payload = {
        "submitted": False,
        "job_id": None,
        "label": args.label,
        "result_dir": str(result_dir),
        "batch_script": str(batch_path),
        "stdout_path": str(result_dir / "stdout.txt"),
        "stderr_path": str(result_dir / "stderr.txt"),
        "slurm_stdout_path": str(result_dir / "slurm-%j.out"),
        "benchmark_command": benchmark_command,
        "git_ref": args.git_ref,
        "git_fetch": bool(args.git_fetch),
        "git_remote": str(args.git_remote),
    }

    if not args.dry_run:
        payload["job_id"] = submit_batch(batch_path)
        payload["submitted"] = True

    (result_dir / "remote_submit.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
