"""Submit one timestamped OBGPU benchmark run to Slurm on a remote host."""

import argparse
import json
import shlex
import subprocess
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict, List, Optional

from slurm_common import path_is_within, requested_mpi_rank_count, shell_join, slurm_directives


def decode_command(payload_b64):
    """Decode a base64-encoded JSON command list."""
    command = json.loads(b64decode(payload_b64).decode("utf-8"))
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError("Decoded benchmark command must be a JSON list of strings")
    return command


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


def neuron_mpi_preflight_suffix(benchmark_suffix):
    """Build a cheap nrniv command suffix that verifies NEURON's MPI world."""
    try:
        nrniv_index = benchmark_suffix.index("nrniv")
    except ValueError:
        return None
    code = (
        "import os\n"
        "from neuron import h\n"
        "pc = h.ParallelContext()\n"
        "rank = int(pc.id())\n"
        "nhost = int(pc.nhost())\n"
        "expected = int(os.environ.get('OBGPU_EXPECTED_NRANKS') or '0')\n"
        "if rank == 0:\n"
        "    print('OBGPU MPI preflight: ParallelContext.nhost()=%d expected=%d' % (nhost, expected), flush=True)\n"
        "if expected > 1 and nhost != expected:\n"
        "    raise RuntimeError('NEURON MPI preflight saw %d ranks, expected %d' % (nhost, expected))\n"
        "pc.barrier()\n"
    )
    return benchmark_suffix[: nrniv_index + 1] + ["-mpi", "-python", "-c", code]


def write_batch_script(
    repo_root,
    result_dir,
    label,
    repo_mode,
    worktree_root,
    conda_activate_cmd,
    runtime_profiles_b64,
    fallback_conda_activate_cmd,
    fast_node_feature,
    mechanism_profile,
    fallback_mechanism_profile,
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
    heartbeat_path = wrapper_dir / "notebook-heartbeat.txt"
    step_id_path = wrapper_dir / "srun-step-id.txt"
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
    expected_nhosts = requested_mpi_rank_count(effective_command) or 1
    preflight_suffix = neuron_mpi_preflight_suffix(benchmark_suffix) if replace_mpi_exec else None
    slurm_log_path = wrapper_dir / "slurm-%j.out"
    bootstrap_log_path = wrapper_dir / "bootstrap.log"
    git_ref_value = git_ref or ""
    heartbeat_timeout_s = max(int(getattr(args, "heartbeat_timeout_s", 120)), 0)
    needs_coreneuron = any(
        flag in effective_command
        for flag in ("--coreneuron", "--coreneuron-gpu", "--coreneuron-file-mode")
    )
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.write_text("")
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
        "step_id_path={}".format(shlex.quote(str(step_id_path))),
        "shared_repo_root={}".format(shlex.quote(str(repo_root))),
        "job_repo_mode={}".format(shlex.quote(str(repo_mode))),
        "job_worktree={}".format(shlex.quote(str(worktree_root))),
        "job_needs_coreneuron={}".format("1" if needs_coreneuron else "0"),
        "primary_conda_activate_cmd={}".format(shlex.quote(conda_activate_cmd)),
        "runtime_profiles_b64={}".format(shlex.quote(runtime_profiles_b64 or "")),
        "fallback_conda_activate_cmd={}".format(shlex.quote(fallback_conda_activate_cmd or "")),
        "fast_node_feature={}".format(shlex.quote(fast_node_feature or "")),
        "primary_mechanism_profile={}".format(shlex.quote(mechanism_profile or "default")),
        "fallback_mechanism_profile={}".format(shlex.quote(fallback_mechanism_profile or "portable")),
        "selected_conda_activate_cmd=\"$primary_conda_activate_cmd\"",
        "selected_mechanism_profile=\"$primary_mechanism_profile\"",
        "job_mechanism_root=\"\"",
        "git_lock_path=\"$shared_repo_root/.obgpu-git.lock\"",
        "bootstrap_log={}".format(shlex.quote(str(bootstrap_log_path))),
        "notebook_heartbeat_path={}".format(shlex.quote(str(heartbeat_path))),
        "notebook_heartbeat_timeout_s={}".format(heartbeat_timeout_s),
        "heartbeat_watchdog_pid=\"\"",
        "benchmark_pid=\"\"",
        "touch \"$bootstrap_log\"",
        "touch \"$notebook_heartbeat_path\"",
        "step_id_text=\"${SLURM_STEP_ID:-${SLURM_STEPID:-}}\"",
        "if [[ -n \"$step_id_text\" && \"$step_id_text\" != \"batch\" && \"$step_id_text\" != \"extern\" ]]; then",
        "  if [[ \"$step_id_text\" == *.* ]]; then",
        "    printf '%s\\n' \"$step_id_text\" > \"$step_id_path\"",
        "  else",
        "    printf '%s.%s\\n' \"$SLURM_JOB_ID\" \"$step_id_text\" > \"$step_id_path\"",
        "  fi",
        "fi",
        "run_git_locked() {",
        "  if command -v flock >/dev/null 2>&1; then",
        "    flock \"$git_lock_path\" \"$@\"",
        "  else",
        "    \"$@\"",
        "  fi",
        "}",
        "file_sha256() {",
        "  sha256sum \"$1\" | awk '{print $1}'",
        "}",
        "collect_allocated_node_info() {",
        "  if [[ -n \"${SLURM_JOB_NODELIST:-}\" ]] && command -v scontrol >/dev/null 2>&1; then",
        "    local saw_node=0",
        "    local node node_info",
        "    while IFS= read -r node; do",
        "      [[ -n \"$node\" ]] || continue",
        "      saw_node=1",
        "      node_info=\"$(scontrol show node -o \"$node\" 2>/dev/null || true)\"",
        "      if [[ -n \"$node_info\" ]]; then",
        "        printf '%s\\n' \"$node_info\"",
        "      else",
        "        printf 'NodeName=%s Arch=%s\\n' \"$node\" \"$(uname -m)\"",
        "      fi",
        "    done < <(scontrol show hostnames \"$SLURM_JOB_NODELIST\" 2>/dev/null || true)",
        "    if [[ \"$saw_node\" == \"1\" ]]; then",
        "      return 0",
        "    fi",
        "  fi",
        "  printf 'NodeName=%s Arch=%s\\n' \"$(hostname)\" \"$(uname -m)\"",
        "}",
        "sanitize_mechanism_profile() {",
        "  local profile=\"$1\"",
        "  if [[ -z \"$profile\" ]]; then",
        "    printf '%s\\n' default",
        "    return 0",
        "  fi",
        "  if [[ ! \"$profile\" =~ ^[A-Za-z0-9._-]+$ ]]; then",
        "    printf 'Invalid mechanism profile: %s\\n' \"$profile\" >&2",
        "    return 2",
        "  fi",
        "  printf '%s\\n' \"$profile\"",
        "}",
        "allocated_nodes_have_feature() {",
        "  local feature_lc",
        "  feature_lc=\"$(printf '%s' \"$1\" | tr '[:upper:]' '[:lower:]')\"",
        "  if [[ -z \"$feature_lc\" ]]; then",
        "    return 0",
        "  fi",
        "  if [[ -z \"${SLURM_JOB_NODELIST:-}\" ]] || ! command -v scontrol >/dev/null 2>&1; then",
        "    return 1",
        "  fi",
        "  local saw_node=0",
        "  local node node_info node_info_lc",
        "  while IFS= read -r node; do",
        "    [[ -n \"$node\" ]] || continue",
        "    saw_node=1",
        "    node_info=\"$(scontrol show node -o \"$node\" 2>/dev/null || true)\"",
        "    node_info_lc=\"$(printf '%s' \"$node_info\" | tr '[:upper:]' '[:lower:]')\"",
        "    if [[ \"$node_info_lc\" != *\"$feature_lc\"* ]]; then",
        "      printf '[OBGPU batch] node %s does not advertise feature %s\\n' \"$node\" \"$feature_lc\" >> \"$bootstrap_log\"",
        "      return 1",
        "    fi",
        "  done < <(scontrol show hostnames \"$SLURM_JOB_NODELIST\" 2>/dev/null || true)",
        "  [[ \"$saw_node\" == \"1\" ]]",
        "}",
        "select_runtime_profile_from_json() {",
        "  local node_info=\"$1\"",
        "  local python_exec",
        "  python_exec=\"$(command -v python3 || command -v python || true)\"",
        "  [[ -n \"$python_exec\" ]] || return 1",
        "  \"$python_exec\" - \"$runtime_profiles_b64\" \"$node_info\" <<'PY'",
        "import base64",
        "import json",
        "import shlex",
        "import sys",
        "",
        "profiles = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))",
        "node_lines = [line.strip() for line in sys.argv[2].splitlines() if line.strip()]",
        "if not node_lines:",
        "    node_lines = ['']",
        "",
        "def as_list(value):",
        "    if value in (None, ''):",
        "        return []",
        "    if isinstance(value, (list, tuple)):",
        "        return [str(item).lower() for item in value if str(item).strip()]",
        "    return [str(value).lower()]",
        "",
        "def arch_matches(text, arch_values):",
        "    if not arch_values:",
        "        return True",
        "    lowered = text.lower()",
        "    for value in arch_values:",
        "        if 'arch=' + value in lowered or 'architecture=' + value in lowered:",
        "            return True",
        "        if value in lowered:",
        "            return True",
        "    return False",
        "",
        "def profile_matches(profile):",
        "    match_arch = as_list(profile.get('match_arch'))",
        "    match_all = as_list(profile.get('match_all'))",
        "    match_any = as_list(profile.get('match_any'))",
        "    reject_any = as_list(profile.get('reject_any'))",
        "    for line in node_lines:",
        "        lowered = line.lower()",
        "        if reject_any and any(token in lowered for token in reject_any):",
        "            return False",
        "        if not arch_matches(lowered, match_arch):",
        "            return False",
        "        if match_all and not all(token in lowered for token in match_all):",
        "            return False",
        "        if match_any and not any(token in lowered for token in match_any):",
        "            return False",
        "    return True",
        "",
        "for profile in profiles:",
        "    if not isinstance(profile, dict) or not profile_matches(profile):",
        "        continue",
        "    name = str(profile.get('name') or 'runtime-profile')",
        "    cmd = str(profile.get('conda_activate_cmd') or '')",
        "    mechanism_profile = str(profile.get('mechanism_profile') or name)",
        "    print('selected_runtime_profile_name=' + shlex.quote(name))",
        "    if cmd:",
        "        print('selected_conda_activate_cmd=' + shlex.quote(cmd))",
        "    print('selected_mechanism_profile=' + shlex.quote(mechanism_profile))",
        "    raise SystemExit(0)",
        "raise SystemExit(1)",
        "PY",
        "}",
        "select_runtime_profile() {",
        "  selected_conda_activate_cmd=\"$primary_conda_activate_cmd\"",
        "  selected_mechanism_profile=\"$primary_mechanism_profile\"",
        "  selected_runtime_profile_name=\"primary\"",
        "  local node_info profile_assignments",
        "  node_info=\"$(collect_allocated_node_info)\"",
        "  printf '[OBGPU batch] allocated node info:\\n%s\\n' \"$node_info\" >> \"$bootstrap_log\"",
        "  if [[ -n \"$runtime_profiles_b64\" ]]; then",
        "    if profile_assignments=\"$(select_runtime_profile_from_json \"$node_info\")\"; then",
        "      eval \"$profile_assignments\"",
        "      printf '[OBGPU batch] selected runtime profile %s using mechanism profile %s\\n' \"$selected_runtime_profile_name\" \"$selected_mechanism_profile\" >> \"$bootstrap_log\"",
        "      return 0",
        "    fi",
        "    printf '[OBGPU batch] no ordered runtime profile matched allocated nodes; trying primary/fallback runtime settings\\n' >> \"$bootstrap_log\"",
        "  fi",
        "  if [[ -z \"$fallback_conda_activate_cmd\" || -z \"$fast_node_feature\" ]]; then",
        "    printf '[OBGPU batch] using primary runtime profile %s\\n' \"$selected_mechanism_profile\" >> \"$bootstrap_log\"",
        "    return 0",
        "  fi",
        "  if allocated_nodes_have_feature \"$fast_node_feature\"; then",
        "    printf '[OBGPU batch] all allocated nodes match feature %s; using primary runtime profile %s\\n' \"$fast_node_feature\" \"$selected_mechanism_profile\" >> \"$bootstrap_log\"",
        "  else",
        "    selected_conda_activate_cmd=\"$fallback_conda_activate_cmd\"",
        "    selected_mechanism_profile=\"$fallback_mechanism_profile\"",
        "    printf '[OBGPU batch] allocated nodes do not all match feature %s; using fallback runtime profile %s\\n' \"$fast_node_feature\" \"$selected_mechanism_profile\" >> \"$bootstrap_log\"",
        "  fi",
        "}",
        "resolve_mechanism_root() {",
        "  local profile",
        "  profile=\"$(sanitize_mechanism_profile \"$selected_mechanism_profile\")\"",
        "  selected_mechanism_profile=\"$profile\"",
        "  if [[ \"$profile\" == \"default\" ]]; then",
        "    job_mechanism_root=\"$job_repo_root\"",
        "  else",
        "    job_mechanism_root=\"$shared_repo_root/.obgpu-mechanisms/$profile\"",
        "  fi",
        "  printf '[OBGPU batch] mechanism root: %s\\n' \"$job_mechanism_root\" >> \"$bootstrap_log\"",
        "}",
        "configure_neuron_launch() {",
        "  obgpu_mechanism_dll=\"$job_mechanism_root/$(uname -m)/libnrnmech.so\"",
        "  obgpu_neuron_launch_dir=\"$job_repo_root\"",
        "  obgpu_neuron_dll_args=()",
        "  if [[ \"$selected_mechanism_profile\" != \"default\" ]]; then",
        "    if [[ ! -f \"$obgpu_mechanism_dll\" ]]; then",
        "      printf '[OBGPU batch] expected mechanism dll is missing: %s\\n' \"$obgpu_mechanism_dll\" >> \"$bootstrap_log\"",
        "      return 2",
        "    fi",
        "    obgpu_neuron_launch_dir=\"$wrapper_dir/neuron-launch\"",
        "    mkdir -p \"$obgpu_neuron_launch_dir\"",
        "    obgpu_neuron_dll_args=(-dll \"$obgpu_mechanism_dll\")",
        "    export OBGPU_SKIP_H_QUIT=1",
        "    printf '[OBGPU batch] launching NEURON from neutral cwd with dll %s\\n' \"$obgpu_mechanism_dll\" >> \"$bootstrap_log\"",
        "  else",
        "    unset OBGPU_SKIP_H_QUIT || true",
        "  fi",
        "}",
        "inject_neuron_dll_args() {",
        "  local inserted=0",
        "  local part base",
        "  obgpu_injected_command=()",
        "  for part in \"$@\"; do",
        "    obgpu_injected_command+=(\"$part\")",
        "    base=\"$(basename -- \"$part\")\"",
        "    if [[ \"$inserted\" == \"0\" && \"$base\" == \"nrniv\" ]]; then",
        "      if [[ \"${#obgpu_neuron_dll_args[@]}\" -gt 0 ]]; then",
        "        obgpu_injected_command+=(\"${obgpu_neuron_dll_args[@]}\")",
        "      fi",
        "      inserted=1",
        "    fi",
        "  done",
        "}",
        "mechanism_fingerprint() {",
        "  {",
        "    printf 'mechanism_profile=%s\\n' \"$selected_mechanism_profile\"",
        "    printf 'mechanism_root=%s\\n' \"$job_mechanism_root\"",
        "    printf 'machine_arch=%s\\n' \"$(uname -m)\"",
        "    printf 'conda_prefix=%s\\n' \"${CONDA_PREFIX:-}\"",
        "    printf 'nrniv=%s\\n' \"$(command -v nrniv || true)\"",
        "    printf 'nrnivmodl=%s\\n' \"$(command -v nrnivmodl || true)\"",
        "    printf 'obgpu_cpu_target=%s\\n' \"${OBGPU_CPU_TARGET:-}\"",
        "    printf 'obgpu_cpu_cflags=%s\\n' \"${OBGPU_CPU_CFLAGS:-}\"",
        "    printf 'obgpu_cpu_cxxflags=%s\\n' \"${OBGPU_CPU_CXXFLAGS:-}\"",
        "    printf 'cflags=%s\\n' \"${CFLAGS:-}\"",
        "    printf 'cxxflags=%s\\n' \"${CXXFLAGS:-}\"",
        "    printf 'coreneuron=%s\\n' \"$job_needs_coreneuron\"",
        "    find \"$job_repo_root/prev_ob_models/Birgiolas2020/Mechanisms\" -maxdepth 1 -name '*.mod' -type f | sort | while read -r mod_file; do",
        "      printf 'mod=%s sha=%s\\n' \"${mod_file#\"$job_repo_root/\"}\" \"$(file_sha256 \"$mod_file\")\"",
        "    done",
        "  } | sha256sum | awk '{print $1}'",
        "}",
        "ensure_mechanisms_current() {",
        "  local arch_dir=\"$job_mechanism_root/$(uname -m)\"",
        "  local lib_path=\"$arch_dir/libnrnmech.so\"",
        "  local stamp_path=\"$arch_dir/.obgpu_remote_mechanisms_stamp\"",
        "  local expected_fingerprint",
        "  expected_fingerprint=\"$(mechanism_fingerprint)\"",
        "  if [[ -f \"$lib_path\" && -f \"$stamp_path\" ]] && [[ \"$(tr -d '\\n' < \"$stamp_path\")\" == \"$expected_fingerprint\" ]]; then",
        "    printf '%s\\n' '[OBGPU batch] mechanisms are current' >> \"$bootstrap_log\"",
        "    return 0",
        "  fi",
        "  printf '%s\\n' \"[OBGPU batch] rebuilding mechanisms for current checkout (coreneuron=${job_needs_coreneuron})\" >> \"$bootstrap_log\"",
        "  mkdir -p \"$job_mechanism_root\"",
        "  cd \"$job_mechanism_root\"",
        "  if [[ \"$job_needs_coreneuron\" == \"1\" ]]; then",
        "    run_git_locked env OMPI_CC=\"${OMPI_CC:-gcc}\" OMPI_CXX=\"${OMPI_CXX:-g++}\" CFLAGS=\"${OBGPU_CPU_CFLAGS:-${CFLAGS:-}}\" CXXFLAGS=\"${OBGPU_CPU_CXXFLAGS:-${CXXFLAGS:-}}\" nrnivmodl -coreneuron \"$job_repo_root/prev_ob_models/Birgiolas2020/Mechanisms\" >> \"$bootstrap_log\" 2>&1",
        "  else",
        "    run_git_locked env OMPI_CC=\"${OMPI_CC:-gcc}\" OMPI_CXX=\"${OMPI_CXX:-g++}\" CFLAGS=\"${OBGPU_CPU_CFLAGS:-${CFLAGS:-}}\" CXXFLAGS=\"${OBGPU_CPU_CXXFLAGS:-${CXXFLAGS:-}}\" nrnivmodl \"$job_repo_root/prev_ob_models/Birgiolas2020/Mechanisms\" >> \"$bootstrap_log\" 2>&1",
        "  fi",
        "  if [[ -x \"$job_repo_root/tools/setup/fix_nvhpc_libnrnmech.sh\" && -f \"$lib_path\" ]]; then",
        "    \"$job_repo_root/tools/setup/fix_nvhpc_libnrnmech.sh\" \"$lib_path\" >> \"$bootstrap_log\" 2>&1",
        "  fi",
        "  mkdir -p \"$arch_dir\"",
        "  printf '%s\\n' \"$expected_fingerprint\" > \"$stamp_path\"",
        "}",
        "sync_wrapper_artifacts() {",
        "  set +e",
        "  mkdir -p \"$result_dir\" || true",
        "  shopt -s nullglob",
        "  local artifact",
        "  for artifact in \"$wrapper_dir\"/bootstrap.log \"$wrapper_dir\"/command.txt \"$wrapper_dir\"/stdout.txt \"$wrapper_dir\"/stderr.txt \"$wrapper_dir\"/overrides.json \"$wrapper_dir\"/slurm-*.out; do",
        "    [[ -e \"$artifact\" ]] || continue",
        "    cp -f \"$artifact\" \"$result_dir\"/ 2>/dev/null || true",
        "  done",
        "  shopt -u nullglob",
        "}",
        "start_notebook_heartbeat_watchdog() {",
        "  if [[ \"$notebook_heartbeat_timeout_s\" -le 0 ]]; then",
        "    return 0",
        "  fi",
        "  (",
        "    set +e",
        "    while true; do",
        "      sleep 10",
        "      now=$(date +%s)",
        "      if [[ -e \"$notebook_heartbeat_path\" ]]; then",
        "        last=$(stat -c %Y \"$notebook_heartbeat_path\" 2>/dev/null || echo 0)",
        "      else",
        "        last=0",
        "      fi",
        "      age=$((now - last))",
        "      if [[ \"$age\" -gt \"$notebook_heartbeat_timeout_s\" ]]; then",
        "        printf '[OBGPU batch] notebook heartbeat expired after %ss at %s; terminating benchmark\\n' \"$age\" \"$(date -Is)\" >> \"$bootstrap_log\"",
        "        if [[ -n \"${benchmark_pid:-}\" ]]; then",
        "          kill -TERM \"$benchmark_pid\" 2>/dev/null || true",
        "          sleep 10",
        "          kill -KILL \"$benchmark_pid\" 2>/dev/null || true",
        "        fi",
        "        exit 0",
        "      fi",
        "    done",
        "  ) &",
        "  heartbeat_watchdog_pid=$!",
        "}",
        "stop_notebook_heartbeat_watchdog() {",
        "  if [[ -n \"${heartbeat_watchdog_pid:-}\" ]]; then",
        "    kill \"$heartbeat_watchdog_pid\" 2>/dev/null || true",
        "    wait \"$heartbeat_watchdog_pid\" 2>/dev/null || true",
        "    heartbeat_watchdog_pid=\"\"",
        "  fi",
        "  return 0",
        "}",
        "on_err() {",
        "  local rc=$?",
        "  stop_notebook_heartbeat_watchdog",
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
        "    run_git_locked git -C \"$shared_repo_root\" worktree remove --force \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || rm -rf \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  fi",
        "  run_git_locked git -C \"$shared_repo_root\" worktree prune >> \"$bootstrap_log\" 2>&1 || true",
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
        "    run_git_locked git checkout --force \"$shared_repo_original_branch\" >> \"$bootstrap_log\" 2>&1 || run_git_locked git checkout --detach \"$shared_repo_original_commit\" >> \"$bootstrap_log\" 2>&1 || true",
        "  else",
        "    run_git_locked git checkout --detach \"$shared_repo_original_commit\" >> \"$bootstrap_log\" 2>&1 || true",
        "  fi",
        "}",
        "on_exit() {",
        "  local exit_code=$?",
        "  trap - EXIT",
        "  stop_notebook_heartbeat_watchdog",
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
        "  stop_notebook_heartbeat_watchdog",
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
        'if [[ "{}" == "1" ]]; then run_git_locked git fetch --tags --prune {}; fi'.format(
            "1" if git_fetch else "0",
            shlex.quote(str(git_remote)),
        ),
        'if [[ -n "{}" ]]; then job_git_ref={}; else job_git_ref=HEAD; fi'.format(
            git_ref_value,
            shlex.quote(git_ref_value),
        ),
        "if [[ \"$job_repo_mode\" == \"snapshot\" ]]; then",
        "  mkdir -p {}".format(shlex.quote(str(worktree_root.parent))),
        "  run_git_locked git worktree remove --force \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  rm -rf \"$job_worktree\" >> \"$bootstrap_log\" 2>&1 || true",
        "  run_git_locked git worktree add --force --detach \"$job_worktree\" \"$job_git_ref\"",
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
        "      run_git_locked git checkout --force --detach \"$job_git_ref\"",
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
        "export OBGPU_STATUS_MODE=file",
        "export OBGPU_STATUS_INTERVAL_MS=${OBGPU_STATUS_INTERVAL_MS:-5}",
        "export OBGPU_SHARED_REPO_ROOT=\"$shared_repo_root\"",
        "export OBGPU_EXPECTED_NRANKS={}".format(int(expected_nhosts)),
        "select_runtime_profile",
        "eval \"$selected_conda_activate_cmd\"",
        "resolve_mechanism_root",
        "export OBGPU_MECHANISM_ROOT=\"$job_mechanism_root\"",
        "export CORENEURONLIB=\"$job_mechanism_root/$(uname -m)/libcorenrnmech.so\"",
        "export LD_LIBRARY_PATH=\"$job_mechanism_root/$(uname -m)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"",
        "ensure_mechanisms_current",
        "configure_neuron_launch",
        "printf '%s\\n' '[OBGPU batch] bootstrap complete'",
        "} >> \"$bootstrap_log\" 2>&1",
    ]
    if replace_mpi_exec:
        lines.extend(
            [
                "configured_mpi_exec=" + shlex.quote(mpi_exec),
                "if [[ \"$configured_mpi_exec\" == \"srun\" && -n \"${OB_MPIEXEC:-}\" ]]; then",
                "  resolved_mpi_exec=\"$OB_MPIEXEC\"",
                "else",
                "  resolved_mpi_exec=\"$configured_mpi_exec\"",
                "fi",
                "if [[ \" $resolved_mpi_exec \" == *\" srun \"* ]]; then",
                "  export PMIX_MCA_psec=\"${PMIX_MCA_psec:-native}\"",
                "  export OMPI_MCA_psec=\"${OMPI_MCA_psec:-native}\"",
                "fi",
                "read -r -a _obgpu_mpi_parts <<< \"$resolved_mpi_exec\"",
            ]
        )
        if preflight_suffix is not None:
            lines.extend(
                [
                    "obgpu_preflight=(" + " ".join(shlex.quote(part) for part in preflight_suffix) + ")",
                    "inject_neuron_dll_args \"${obgpu_preflight[@]}\"",
                    "obgpu_preflight=(\"${obgpu_injected_command[@]}\")",
                    "obgpu_preflight_command=(\"${_obgpu_mpi_parts[@]}\" \"${obgpu_preflight[@]}\")",
                    "printf '%s\\n' '[OBGPU batch] running NEURON MPI preflight' >> \"$bootstrap_log\"",
                    "printf '%s\\n' \"$(printf '%q ' \"${obgpu_preflight_command[@]}\")\" >> \"$bootstrap_log\"",
                    "if ! (cd \"$obgpu_neuron_launch_dir\" && \"${obgpu_preflight_command[@]}\") >> \"$bootstrap_log\" 2>&1; then",
                    "  printf '%s\\n' '[OBGPU batch] NEURON MPI preflight failed' >> \"$bootstrap_log\"",
                    "  exit 126",
                    "fi",
                ]
            )
        lines.extend(
            [
                "obgpu_command=(" + " ".join(shlex.quote(part) for part in benchmark_suffix) + ")",
                "inject_neuron_dll_args \"${obgpu_command[@]}\"",
                "obgpu_command=(\"${obgpu_injected_command[@]}\")",
                "obgpu_command=(\"${_obgpu_mpi_parts[@]}\" \"${obgpu_command[@]}\")",
                "touch " + shlex.quote(str(wrapper_dir / "command.txt")) + " "
                + shlex.quote(str(wrapper_dir / "stdout.txt")) + " "
                + shlex.quote(str(wrapper_dir / "stderr.txt")),
                "printf '%s\\n' \"$(printf '%q ' \"${obgpu_command[@]}\")\" > "
                + shlex.quote(str(wrapper_dir / "command.txt")),
                "printf '%s\\n' '[OBGPU batch] launching command' >> \"$bootstrap_log\"",
                "ls -lah \"$result_dir\" >> \"$bootstrap_log\" 2>&1 || true",
                "(cd \"$obgpu_neuron_launch_dir\" && \"${obgpu_command[@]}\") > "
                + shlex.quote(str(wrapper_dir / "stdout.txt"))
                + " 2> "
                + shlex.quote(str(wrapper_dir / "stderr.txt"))
                + " &",
                "benchmark_pid=$!",
                "start_notebook_heartbeat_watchdog",
                "if wait \"$benchmark_pid\"; then",
                "  benchmark_rc=0",
                "else",
                "  benchmark_rc=$?",
                "fi",
                "benchmark_pid=\"\"",
                "stop_notebook_heartbeat_watchdog",
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
                "({}) > {} 2> {} &".format(
                    benchmark_shell,
                    shlex.quote(str(wrapper_dir / "stdout.txt")),
                    shlex.quote(str(wrapper_dir / "stderr.txt")),
                ),
                "benchmark_pid=$!",
                "start_notebook_heartbeat_watchdog",
                "if wait \"$benchmark_pid\"; then",
                "  benchmark_rc=0",
                "else",
                "  benchmark_rc=$?",
                "fi",
                "benchmark_pid=\"\"",
                "stop_notebook_heartbeat_watchdog",
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
    launcher_stdout_path = wrapper_dir / "srun-launch.stdout"
    slurm_log_path = wrapper_dir / "slurm-%j-%s.out"
    step_name = "obgpu-step-{}".format(wrapper_dir.name)[:120]

    step_id_path.write_text("")
    launcher_stderr_path.write_text("")
    launcher_stdout_path.write_text("")

    shell_script = """
set -euo pipefail
step_id_path={step_id_path}
launcher_stderr_path={launcher_stderr_path}
launcher_pid_path={launcher_pid_path}
launcher_stdout_path={launcher_stdout_path}
batch_path={batch_path}
allocation_job_id={allocation_job_id}
slurm_log_path={slurm_log_path}
step_name={step_name}

srun --jobid "$allocation_job_id" --overlap --cpu-bind=none --job-name "$step_name" --output "$slurm_log_path" --error "$slurm_log_path" bash "$batch_path" > "$launcher_stdout_path" 2> "$launcher_stderr_path" &
launcher_pid=$!
printf '%s\\n' "$launcher_pid" > "$launcher_pid_path"

for _ in $(seq 1 300); do
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
        launcher_stdout_path=shlex.quote(str(launcher_stdout_path)),
        batch_path=shlex.quote(str(batch_path)),
        allocation_job_id=shlex.quote(str(allocation_job_id)),
        slurm_log_path=shlex.quote(str(slurm_log_path)),
        step_name=shlex.quote(step_name),
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
    parser.add_argument("--runtime-profiles-b64", default="")
    parser.add_argument("--fallback-conda-activate-cmd", default=None)
    parser.add_argument("--fast-node-feature", default=None)
    parser.add_argument("--mechanism-profile", default="default")
    parser.add_argument("--fallback-mechanism-profile", default="portable")
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
    parser.add_argument("--heartbeat-timeout-s", type=int, default=120)
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
        runtime_profiles_b64=args.runtime_profiles_b64,
        fallback_conda_activate_cmd=args.fallback_conda_activate_cmd,
        fast_node_feature=args.fast_node_feature,
        mechanism_profile=args.mechanism_profile,
        fallback_mechanism_profile=args.fallback_mechanism_profile,
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
        "heartbeat_path": str(wrapper_dir / "notebook-heartbeat.txt"),
        "heartbeat_timeout_s": max(int(args.heartbeat_timeout_s), 0),
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
