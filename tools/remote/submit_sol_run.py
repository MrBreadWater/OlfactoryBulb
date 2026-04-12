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
    repo_mode,
    worktree_root,
    conda_activate_cmd,
    benchmark_command,
    mpi_exec,
    git_ref,
    git_fetch,
    git_remote,
    args,
):
    """Write the Slurm batch script that launches one benchmark run."""
    wrapper_dir = result_dir.parent / ".obgpu-wrapper" / label
    batch_path = wrapper_dir / "slurm_job.sh"
    if repo_mode == "snapshot":
        effective_command = relocate_benchmark_command(
            benchmark_command,
            repo_root=repo_root,
            worktree_root=worktree_root,
            preserved_roots=[result_dir.parent],
        )
    elif repo_mode == "shared":
        effective_command = list(benchmark_command)
    else:
        raise ValueError("Unsupported repo_mode={!r}".format(repo_mode))
    benchmark_shell = shell_join(effective_command)
    benchmark_suffix = list(effective_command)
    requested_mpi_parts = shlex.split(mpi_exec) if mpi_exec else []
    replace_mpi_exec = bool(requested_mpi_parts) and effective_command[: len(requested_mpi_parts)] == requested_mpi_parts
    if replace_mpi_exec:
        benchmark_suffix = effective_command[len(requested_mpi_parts):]
    slurm_log_path = wrapper_dir / "slurm-%j.out"
    bootstrap_log_path = wrapper_dir / "bootstrap.log"
    git_ref_value = git_ref or ""
    lines = [
        "#!/usr/bin/env bash",
        *slurm_directives(args, label),
        "#SBATCH --output={}".format(slurm_log_path),
        "#SBATCH --error={}".format(slurm_log_path),
        "set -Eeuo pipefail",
        "mkdir -p {}".format(shlex.quote(str(result_dir))),
        "mkdir -p {}".format(shlex.quote(str(wrapper_dir))),
        "result_dir={}".format(shlex.quote(str(result_dir))),
        "wrapper_dir={}".format(shlex.quote(str(wrapper_dir))),
        "shared_repo_root={}".format(shlex.quote(str(repo_root))),
        "job_repo_mode={}".format(shlex.quote(str(repo_mode))),
        "job_worktree={}".format(shlex.quote(str(worktree_root))),
        "bootstrap_log={}".format(shlex.quote(str(bootstrap_log_path))),
        "touch \"$bootstrap_log\"",
        "sync_wrapper_artifacts() {",
        "  set +e",
        "  mkdir -p \"$result_dir\" || true",
        "  shopt -s nullglob",
        "  local artifact",
        "  for artifact in \"$wrapper_dir\"/bootstrap.log \"$wrapper_dir\"/command.txt \"$wrapper_dir\"/stdout.txt \"$wrapper_dir\"/stderr.txt \"$wrapper_dir\"/slurm-*.out; do",
        "    [[ -e \"$artifact\" ]] || continue",
        "    cp -f \"$artifact\" \"$result_dir\"/ 2>/dev/null || true",
        "  done",
        "  shopt -u nullglob",
        "}",
        "on_err() {",
        "  local rc=$?",
        "  printf '[OBGPU batch] failed at line %s: %s (exit %s)\\n' \"${BASH_LINENO[0]:-?}\" \"${BASH_COMMAND:-<unknown>}\" \"$rc\" >> \"$bootstrap_log\"",
        "  sync_wrapper_artifacts",
        "  exit \"$rc\"",
        "}",
        "cleanup_worktree() {",
        "  set +e",
        "  if [[ \"$job_repo_mode\" != \"snapshot\" ]]; then",
        "    return 0",
        "  fi",
        "  cd \"$shared_repo_root\" || true",
        "  if [[ -e \"$job_worktree\" ]]; then",
        "    git -C \"$shared_repo_root\" worktree remove --force \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || rm -rf \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  fi",
        "  git -C \"$shared_repo_root\" worktree prune >> \"$bootstrap_log\" 2>&1 || true",
        "}",
        "restore_shared_repo() {",
        "  set +e",
        "  if [[ \"$job_repo_mode\" != \"shared\" ]]; then",
        "    return 0",
        "  fi",
        "  if [[ \"${shared_repo_needs_restore:-0}\" != \"1\" ]]; then",
        "    return 0",
        "  fi",
        "  cd \"$shared_repo_root\" || true",
        "  if [[ -n \"${shared_repo_original_branch:-}\" ]]; then",
        "    git checkout --force \"$shared_repo_original_branch\" >> \"$bootstrap_log\" 2>&1 || git checkout --detach \"$shared_repo_original_commit\" >> \"$bootstrap_log\" 2>&1 || true",
        "  else",
        "    git checkout --detach \"$shared_repo_original_commit\" >> \"$bootstrap_log\" 2>&1 || true",
        "  fi",
        "}",
        "on_exit() {",
        "  local exit_code=$?",
        "  trap - EXIT",
        "  sync_wrapper_artifacts",
        "  restore_shared_repo",
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
        "  sync_wrapper_artifacts",
        "  restore_shared_repo",
        "  cleanup_worktree",
        "  exit 128",
        "}",
        "trap on_err ERR",
        "trap on_exit EXIT",
        "trap 'on_signal HUP' HUP",
        "trap 'on_signal INT' INT",
        "trap 'on_signal TERM' TERM",
        "{",
        "printf '%s\\n' '[OBGPU batch] bootstrap start'",
        "cd \"$shared_repo_root\"",
        'if [[ "{}" == "1" ]]; then git fetch --tags --prune {}; fi'.format(
            "1" if git_fetch else "0",
            shlex.quote(str(git_remote)),
        ),
        'if [[ -n "{}" ]]; then job_git_ref={}; else job_git_ref=HEAD; fi'.format(
            git_ref_value,
            shlex.quote(git_ref_value),
        ),
        "if [[ \"$job_repo_mode\" == \"snapshot\" ]]; then",
        "  mkdir -p {}".format(shlex.quote(str(worktree_root.parent))),
        "  git worktree remove --force \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  rm -rf \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  git worktree add --force --detach \"$job_worktree\" \"$job_git_ref\"",
        "  job_repo_root=\"$job_worktree\"",
        "else",
        "  shared_repo_original_commit=$(git rev-parse HEAD)",
        "  shared_repo_original_branch=$(git symbolic-ref --quiet --short HEAD || true)",
        "  shared_repo_needs_restore=0",
        "  if [[ -n \"$job_git_ref\" && \"$job_git_ref\" != \"HEAD\" ]]; then",
        "    if [[ \"$shared_repo_original_commit\" != \"$job_git_ref\" ]]; then",
        "      if [[ -n \"$(git status --porcelain --untracked-files=no)\" ]]; then",
        "        printf '%s\\n' '[OBGPU batch] shared repo has tracked-file modifications; refusing to checkout requested git ref' >> \"$bootstrap_log\"",
        "        git status --short --untracked-files=no >> \"$bootstrap_log\" 2>&1 || true",
        "        exit 2",
        "      fi",
        "      git checkout --force --detach \"$job_git_ref\"",
        "      shared_repo_needs_restore=1",
        "    fi",
        "  fi",
        "  job_repo_root=\"$shared_repo_root\"",
        "fi",
        "cd \"$job_repo_root\"",
        "git rev-parse HEAD > {}".format(shlex.quote(str(result_dir / "git_commit.txt"))),
        'if [[ -n "{}" ]]; then printf "%s\\n" {} > {}; fi'.format(
            git_ref_value,
            shlex.quote(git_ref_value),
            shlex.quote(str(result_dir / "git_ref.txt")),
        ),
        "export OBGPU_RUNTIME_ONLY=1",
        "export OBGPU_SHARED_REPO_ROOT=\"$shared_repo_root\"",
        "eval {}".format(shlex.quote(conda_activate_cmd)),
        "printf '%s\\n' '[OBGPU batch] bootstrap complete'",
        "} >> \"$bootstrap_log\" 2>&1",
    ]
    if replace_mpi_exec:
        lines.extend(
            [
                "resolved_mpi_exec=${OB_MPIEXEC:-" + shlex.quote(mpi_exec) + "}",
                "read -r -a _obgpu_mpi_parts <<< \"$resolved_mpi_exec\"",
                "obgpu_command=(" + " ".join(shlex.quote(part) for part in benchmark_suffix) + ")",
                "obgpu_command=(\"${_obgpu_mpi_parts[@]}\" \"${obgpu_command[@]}\")",
                "touch " + shlex.quote(str(wrapper_dir / "command.txt")) + " "
                + shlex.quote(str(wrapper_dir / "stdout.txt")) + " "
                + shlex.quote(str(wrapper_dir / "stderr.txt")),
                "printf '%s\\n' \"$(printf '%q ' \"${obgpu_command[@]}\")\" > "
                + shlex.quote(str(wrapper_dir / "command.txt")),
                "printf '%s\\n' '[OBGPU batch] launching command' >> \"$bootstrap_log\"",
                "ls -lah \"$result_dir\" >> \"$bootstrap_log\" 2>&1 || true",
                "if \"${obgpu_command[@]}\" > "
                + shlex.quote(str(wrapper_dir / "stdout.txt"))
                + " 2> "
                + shlex.quote(str(wrapper_dir / "stderr.txt"))
                + "; then",
                "  benchmark_rc=0",
                "else",
                "  benchmark_rc=$?",
                "fi",
            ]
        )
    else:
        lines.extend(
            [
                "touch " + shlex.quote(str(wrapper_dir / "command.txt")) + " "
                + shlex.quote(str(wrapper_dir / "stdout.txt")) + " "
                + shlex.quote(str(wrapper_dir / "stderr.txt")),
                "printf '%s\\n' {} > {}".format(
                    shlex.quote(benchmark_shell),
                    shlex.quote(str(wrapper_dir / "command.txt")),
                ),
                "printf '%s\\n' '[OBGPU batch] launching command' >> \"$bootstrap_log\"",
                "ls -lah \"$result_dir\" >> \"$bootstrap_log\" 2>&1 || true",
                "if {} > {} 2> {}; then".format(
                    benchmark_shell,
                    shlex.quote(str(wrapper_dir / "stdout.txt")),
                    shlex.quote(str(wrapper_dir / "stderr.txt")),
                ),
                "  benchmark_rc=0",
                "else",
                "  benchmark_rc=$?",
                "fi",
            ]
        )
    lines.extend(
        [
        "printf '%s\\n' \"[OBGPU batch] benchmark rc=${benchmark_rc}\" >> \"$bootstrap_log\"",
        "ls -lah \"$result_dir\" >> \"$bootstrap_log\" 2>&1 || true",
        "sync_wrapper_artifacts",
        "exit \"$benchmark_rc\"",
        "",
        ]
    )
    batch_path.parent.mkdir(parents=True, exist_ok=True)
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


def submit_allocation_step(batch_path, allocation_job_id, wrapper_dir):
    """Launch the generated script as a reusable step inside an existing allocation."""
    wrapper_dir = Path(wrapper_dir)
    step_id_path = wrapper_dir / "srun-step-id.txt"
    launcher_stderr_path = wrapper_dir / "srun-launch.stderr"
    launcher_pid_path = wrapper_dir / "srun-launch.pid"
    slurm_log_path = wrapper_dir / "slurm-%j-%s.out"

    step_id_path.write_text("")
    launcher_stderr_path.write_text("")

    shell_script = """
set -euo pipefail
step_id_path={step_id_path}
launcher_stderr_path={launcher_stderr_path}
launcher_pid_path={launcher_pid_path}
batch_path={batch_path}
allocation_job_id={allocation_job_id}
slurm_log_path={slurm_log_path}

srun --jobid "$allocation_job_id" --overlap --parsable --output "$slurm_log_path" --error "$slurm_log_path" bash "$batch_path" > "$step_id_path" 2> "$launcher_stderr_path" &
launcher_pid=$!
printf '%s\\n' "$launcher_pid" > "$launcher_pid_path"

for _ in $(seq 1 100); do
  if [[ -s "$step_id_path" ]]; then
    break
  fi
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

if [[ ! -s "$step_id_path" ]]; then
  wait "$launcher_pid" || true
fi

if [[ ! -s "$step_id_path" ]]; then
  printf '%s' 'NO_STEP_ID'
  exit 1
fi

head -n 1 "$step_id_path"
""".format(
        step_id_path=shlex.quote(str(step_id_path)),
        launcher_stderr_path=shlex.quote(str(launcher_stderr_path)),
        launcher_pid_path=shlex.quote(str(launcher_pid_path)),
        batch_path=shlex.quote(str(batch_path)),
        allocation_job_id=shlex.quote(str(allocation_job_id)),
        slurm_log_path=shlex.quote(str(slurm_log_path)),
    )
    completed = subprocess.run(
        ["bash", "-lc", shell_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        launch_stderr = launcher_stderr_path.read_text() if launcher_stderr_path.exists() else ""
        raise RuntimeError(
            "srun within existing allocation failed:\nSTDOUT:\n{}\nSTDERR:\n{}\nLAUNCH STDERR:\n{}".format(
                completed.stdout,
                completed.stderr,
                launch_stderr,
            )
        )
    job_id = (completed.stdout or "").strip().split(";", 1)[0].strip()
    if not job_id:
        raise RuntimeError(
            "Could not parse Slurm step id from srun output: {!r}".format(completed.stdout)
        )
    return job_id


def main():
    """Parse CLI args, write the batch script, optionally submit it, and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--results-base", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--benchmark-command-b64", required=True)
    parser.add_argument("--repo-mode", default="shared")
    parser.add_argument("--mpi-exec", default="")
    parser.add_argument("--conda-activate-cmd", required=True)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--git-fetch", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--allocation-job-id", default=None)
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
    wrapper_dir = result_dir.parent / ".obgpu-wrapper" / args.label
    worktree_root = repo_root.parent / ".obgpu-worktrees" / args.label
    repo_mode = str(args.repo_mode).strip().lower()
    result_dir.mkdir(parents=True, exist_ok=True)
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    benchmark_command = decode_command(args.benchmark_command_b64)
    batch_path = write_batch_script(
        repo_root=repo_root,
        result_dir=result_dir,
        label=args.label,
        repo_mode=repo_mode,
        worktree_root=worktree_root,
        conda_activate_cmd=args.conda_activate_cmd,
        benchmark_command=benchmark_command,
        mpi_exec=str(args.mpi_exec),
        git_ref=args.git_ref,
        git_fetch=bool(args.git_fetch),
        git_remote=str(args.git_remote),
        args=args,
    )

    payload = {
        "submitted": False,
        "job_id": None,
        "allocation_job_id": args.allocation_job_id,
        "label": args.label,
        "repo_mode": repo_mode,
        "result_dir": str(result_dir),
        "wrapper_dir": str(wrapper_dir),
        "batch_script": str(batch_path),
        "stdout_path": str(result_dir / "stdout.txt"),
        "stderr_path": str(result_dir / "stderr.txt"),
        "slurm_stdout_path": str(wrapper_dir / "slurm-%j.out"),
        "benchmark_command": (
            relocate_benchmark_command(
                benchmark_command,
                repo_root=repo_root,
                worktree_root=worktree_root,
                preserved_roots=[result_dir.parent],
            )
            if repo_mode == "snapshot"
            else list(benchmark_command)
        ),
        "git_ref": args.git_ref,
        "git_fetch": bool(args.git_fetch),
        "git_remote": str(args.git_remote),
    }
    if repo_mode == "snapshot":
        payload["worktree_path"] = str(worktree_root)

    if not args.dry_run:
        if args.allocation_job_id not in (None, ""):
            payload["job_id"] = submit_allocation_step(batch_path, args.allocation_job_id, wrapper_dir)
        else:
            payload["job_id"] = submit_batch(batch_path)
        payload["submitted"] = True

    (result_dir / "remote_submit.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
