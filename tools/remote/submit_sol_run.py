"""Submit one timestamped OBGPU benchmark run to Slurm on a remote host."""

import argparse
import json
import shlex
import subprocess
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict, List, Optional


def shell_join(parts):
    """Portable equivalent of shlex.join for older Python versions."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def path_is_within(path_value, root_value):
    """Return whether one string path is equal to or nested under another."""
    root_text = str(root_value).rstrip("/")
    path_text = str(path_value)
    if not root_text:
        return False
    return path_text == root_text or path_text.startswith(root_text + "/")


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


def relocate_benchmark_command(benchmark_command, repo_root, worktree_root, preserved_roots):
    """Rewrite repo-root paths in the benchmark command to the per-run worktree."""
    repo_root_text = str(repo_root).rstrip("/")
    worktree_root_text = str(worktree_root).rstrip("/")
    preserved = [str(root).rstrip("/") for root in preserved_roots if str(root).strip()]
    relocated = []
    for part in benchmark_command:
        if any(path_is_within(part, root) for root in preserved):
            relocated.append(part)
            continue
        if path_is_within(part, repo_root_text):
            relocated.append(worktree_root_text + part[len(repo_root_text):])
            continue
        relocated.append(part)
    return relocated


def write_batch_script(
    repo_root,
    result_dir,
    label,
    worktree_root,
    conda_activate_cmd,
    benchmark_command,
    git_ref,
    git_fetch,
    git_remote,
    args,
):
    """Write the Slurm batch script that launches one benchmark run."""
    batch_path = result_dir / "slurm_job.sh"
    worktree_command = relocate_benchmark_command(
        benchmark_command,
        repo_root=repo_root,
        worktree_root=worktree_root,
        preserved_roots=[result_dir.parent],
    )
    benchmark_shell = shell_join(worktree_command)
    slurm_log_path = result_dir / "slurm-%j.out"
    bootstrap_log_path = result_dir / "bootstrap.log"
    git_ref_value = git_ref or ""
    lines = [
        "#!/usr/bin/env bash",
        *slurm_directives(args, label),
        "#SBATCH --output={}".format(slurm_log_path),
        "#SBATCH --error={}".format(slurm_log_path),
        "set -euo pipefail",
        "mkdir -p {}".format(shlex.quote(str(result_dir))),
        "shared_repo_root={}".format(shlex.quote(str(repo_root))),
        "job_worktree={}".format(shlex.quote(str(worktree_root))),
        "bootstrap_log={}".format(shlex.quote(str(bootstrap_log_path))),
        "cleanup_worktree() {",
        "  set +e",
        "  cd \"$shared_repo_root\" || true",
        "  if [[ -e \"$job_worktree\" ]]; then",
        "    git -C \"$shared_repo_root\" worktree remove --force \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || rm -rf \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  fi",
        "  git -C \"$shared_repo_root\" worktree prune >> \"$bootstrap_log\" 2>&1 || true",
        "}",
        "on_exit() {",
        "  local exit_code=$?",
        "  trap - EXIT",
        "  cleanup_worktree",
        "  exit \"$exit_code\"",
        "}",
        "on_signal() {",
        "  local signal=${1:-TERM}",
        "  trap - EXIT HUP INT TERM",
        "  if [[ -n \"${benchmark_pid:-}\" ]]; then",
        "    kill -\"$signal\" \"$benchmark_pid\" 2>/dev/null || kill \"$benchmark_pid\" 2>/dev/null || true",
        "    wait \"$benchmark_pid\" 2>/dev/null || true",
        "  fi",
        "  cleanup_worktree",
        "  exit 128",
        "}",
        "trap on_exit EXIT",
        "trap 'on_signal HUP' HUP",
        "trap 'on_signal INT' INT",
        "trap 'on_signal TERM' TERM",
        "{",
        "mkdir -p {}".format(shlex.quote(str(worktree_root.parent))),
        "cd \"$shared_repo_root\"",
        'if [[ "{}" == "1" ]]; then git fetch --tags --prune {}; fi'.format(
            "1" if git_fetch else "0",
            shlex.quote(str(git_remote)),
        ),
        "git worktree remove --force \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "rm -rf \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        'if [[ -n "{}" ]]; then job_git_ref={}; else job_git_ref=HEAD; fi'.format(
            git_ref_value,
            shlex.quote(git_ref_value),
        ),
        "git worktree add --force --detach \"$job_worktree\" \"$job_git_ref\"",
        "cd \"$job_worktree\"",
        "git rev-parse HEAD > {}".format(shlex.quote(str(result_dir / "git_commit.txt"))),
        'if [[ -n "{}" ]]; then printf "%s\\n" {} > {}; fi'.format(
            git_ref_value,
            shlex.quote(git_ref_value),
            shlex.quote(str(result_dir / "git_ref.txt")),
        ),
        "eval {}".format(shlex.quote(conda_activate_cmd)),
        "} >> \"$bootstrap_log\" 2>&1",
        "printf '%s\\n' {} > {}".format(
            shlex.quote(benchmark_shell),
            shlex.quote(str(result_dir / "command.txt")),
        ),
        "{} > {} 2> {} &".format(
            benchmark_shell,
            shlex.quote(str(result_dir / "stdout.txt")),
            shlex.quote(str(result_dir / "stderr.txt")),
        ),
        "benchmark_pid=$!",
        "wait \"$benchmark_pid\"",
        "benchmark_rc=$?",
        "benchmark_pid=",
        "exit \"$benchmark_rc\"",
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
    worktree_root = repo_root.parent / ".obgpu-worktrees" / args.label
    result_dir.mkdir(parents=True, exist_ok=True)

    benchmark_command = decode_command(args.benchmark_command_b64)
    batch_path = write_batch_script(
        repo_root=repo_root,
        result_dir=result_dir,
        label=args.label,
        worktree_root=worktree_root,
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
        "worktree_path": str(worktree_root),
        "batch_script": str(batch_path),
        "stdout_path": str(result_dir / "stdout.txt"),
        "stderr_path": str(result_dir / "stderr.txt"),
        "slurm_stdout_path": str(result_dir / "slurm-%j.out"),
        "benchmark_command": relocate_benchmark_command(
            benchmark_command,
            repo_root=repo_root,
            worktree_root=worktree_root,
            preserved_roots=[result_dir.parent],
        ),
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
