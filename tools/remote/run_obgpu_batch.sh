#!/usr/bin/env bash
set -euo pipefail

repo_root=""
result_dir=""
conda_activate_cmd=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --result-dir)
      result_dir="$2"
      shift 2
      ;;
    --conda-activate-cmd)
      conda_activate_cmd="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${repo_root}" || -z "${result_dir}" || -z "${conda_activate_cmd}" ]]; then
  echo "run_obgpu_batch.sh requires --repo-root, --result-dir, and --conda-activate-cmd" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "run_obgpu_batch.sh requires a benchmark command after --" >&2
  exit 2
fi

mkdir -p "${result_dir}"

eval "${conda_activate_cmd}"
cd "${repo_root}"

{
  printf '%q ' "$@"
  printf '\n'
} > "${result_dir}/command.txt"
exec "$@" > "${result_dir}/stdout.txt" 2> "${result_dir}/stderr.txt"
