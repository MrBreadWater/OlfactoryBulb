"""Submit one reusable Slurm allocation for notebook-managed follow-on job steps."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from slurm_common import slurm_directives


def write_holder_script(args, alloc_root):
    """Write the long-lived batch script that keeps one allocation open."""
    alloc_root = Path(alloc_root).expanduser().resolve()
    script_path = alloc_root / "allocation_job.sh"
    slurm_log_path = alloc_root / "allocation-%j.out"
    heartbeat_path = alloc_root / "notebook-heartbeat.txt"
    lease_expired_path = alloc_root / "lease-expired.txt"
    heartbeat_timeout_s = max(int(args.heartbeat_timeout_s), 0)
    lines = [
        "#!/usr/bin/env bash",
        *slurm_directives(args, args.name),
        "#SBATCH --output={}".format(slurm_log_path),
        "#SBATCH --error={}".format(slurm_log_path),
        "set -Eeuo pipefail",
        "trap 'exit 0' TERM INT HUP",
        "printf '%s\\n' \"$SLURM_JOB_ID\" > {}".format(alloc_root / "job_id.txt"),
        "printf '%s\\n' \"${{SLURM_JOB_NODELIST:-}}\" > {}".format(alloc_root / "nodelist.txt"),
        "heartbeat_path={}".format(shlex.quote(str(heartbeat_path))),
        "heartbeat_timeout_s={}".format(heartbeat_timeout_s),
        "lease_expired_path={}".format(shlex.quote(str(lease_expired_path))),
        "touch \"$heartbeat_path\"",
        "while true; do",
        "  sleep 10",
        "  if [[ \"$heartbeat_timeout_s\" -le 0 ]]; then",
        "    continue",
        "  fi",
        "  now=$(date +%s)",
        "  if [[ -e \"$heartbeat_path\" ]]; then",
        "    last=$(stat -c %Y \"$heartbeat_path\" 2>/dev/null || echo 0)",
        "  else",
        "    last=0",
        "  fi",
        "  age=$((now - last))",
        "  if [[ \"$age\" -gt \"$heartbeat_timeout_s\" ]]; then",
        "    printf '[OBGPU allocation] notebook heartbeat expired after %ss at %s\\n' \"$age\" \"$(date -Is)\" > \"$lease_expired_path\"",
        "    exit 0",
        "  fi",
        "done",
    ]
    alloc_root.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text("")
    script_path.write_text("\n".join(str(line) for line in lines))
    script_path.chmod(0o755)
    return script_path, slurm_log_path, heartbeat_path


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
    parser.add_argument("--heartbeat-timeout-s", type=int, default=120)
    parser.add_argument("--sbatch-arg", action="append", default=[])
    args = parser.parse_args()

    alloc_root = Path(args.alloc_root).expanduser().resolve()
    script_path, slurm_log_path, heartbeat_path = write_holder_script(args, alloc_root)
    job_id = submit_batch(script_path)
    payload = {
        "job_id": str(job_id),
        "name": str(args.name),
        "allocation_root": str(alloc_root),
        "batch_script": str(script_path),
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_timeout_s": max(int(args.heartbeat_timeout_s), 0),
        "slurm_log_pattern": str(slurm_log_path),
    }
    (alloc_root / "allocation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
