"""Submit one reusable Slurm allocation for notebook-managed follow-on job steps."""

import argparse
import json
import subprocess
from pathlib import Path


def normalize_sbatch_args(values):
    """Normalize raw sbatch args so split flag/value pairs become one directive."""
    normalized = []
    index = 0
    values = [str(value) for value in values]
    while index < len(values):
        current = values[index]
        if current.startswith("-") and "=" not in current and index + 1 < len(values):
            next_value = values[index + 1]
            if not next_value.startswith("-"):
                normalized.append("{} {}".format(current, next_value))
                index += 2
                continue
        normalized.append(current)
        index += 1
    return normalized


def slurm_directives(args, name):
    """Return ``#SBATCH`` header lines for one generated allocation holder script."""
    directives = ["#SBATCH --job-name={}".format(name[:120])]
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
    for extra in normalize_sbatch_args(args.sbatch_arg):
        directives.append("#SBATCH {}".format(extra))
    return directives


def write_holder_script(args, alloc_root):
    """Write the long-lived batch script that keeps one allocation open."""
    alloc_root = Path(alloc_root).expanduser().resolve()
    script_path = alloc_root / "allocation_job.sh"
    slurm_log_path = alloc_root / "allocation-%j.out"
    lines = [
        "#!/usr/bin/env bash",
        *slurm_directives(args, args.name),
        "#SBATCH --output={}".format(slurm_log_path),
        "#SBATCH --error={}".format(slurm_log_path),
        "set -Eeuo pipefail",
        "trap 'exit 0' TERM INT HUP",
        "printf '%s\\n' \"$SLURM_JOB_ID\" > {}".format(alloc_root / "job_id.txt"),
        "printf '%s\\n' \"${{SLURM_JOB_NODELIST:-}}\" > {}".format(alloc_root / "nodelist.txt"),
        "while true; do sleep 300; done",
    ]
    alloc_root.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(str(line) for line in lines))
    script_path.chmod(0o755)
    return script_path, slurm_log_path


def submit_batch(script_path):
    """Submit one generated holder script and return the parsed Slurm job id."""
    completed = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
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
    """Parse CLI args, write the holder script, submit it, and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--sbatch-arg", action="append", default=[])
    args = parser.parse_args()

    alloc_root = Path(args.alloc_root).expanduser().resolve()
    script_path, slurm_log_path = write_holder_script(args, alloc_root)
    job_id = submit_batch(script_path)
    payload = {
        "job_id": str(job_id),
        "name": str(args.name),
        "allocation_root": str(alloc_root),
        "batch_script": str(script_path),
        "slurm_log_pattern": str(slurm_log_path),
    }
    (alloc_root / "allocation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
