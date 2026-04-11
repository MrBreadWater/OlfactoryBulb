#!/usr/bin/env bash
set -euo pipefail

repo_root=""
result_dir=""
conda_activate_cmd=""
git_ref=""
git_fetch="0"
git_remote="origin"

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
    --git-ref)
      git_ref="$2"
      shift 2
      ;;
    --git-fetch)
      git_fetch="$2"
      shift 2
      ;;
    --git-remote)
      git_remote="$2"
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

cd "${repo_root}"

if [[ "${git_fetch}" == "1" ]]; then
  git fetch --tags --prune "${git_remote}"
fi

if [[ -n "${git_ref}" ]]; then
  git checkout --force "${git_ref}"
fi

git rev-parse HEAD > "${result_dir}/git_commit.txt"
if [[ -n "${git_ref}" ]]; then
  printf '%s\n' "${git_ref}" > "${result_dir}/git_ref.txt"
fi

eval "${conda_activate_cmd}"

{
  printf '%q ' "$@"
  printf '\n'
} > "${result_dir}/command.txt"
exec "$@" > "${result_dir}/stdout.txt" 2> "${result_dir}/stderr.txt"
