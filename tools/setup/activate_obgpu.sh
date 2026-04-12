#!/usr/bin/env bash
# Source this helper on generic Linux clusters to activate OBGPU cleanly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source tools/setup/activate_obgpu.sh" >&2
  exit 1
fi

ENV_NAME="${1:-${ENV_NAME:-OBGPU}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/obgpu_conda_utils.sh"

if ! obgpu_activate_conda_env "${ENV_NAME}"; then
  return 1
fi

export OBGPU_SHARED_REPO_ROOT="${REPO_ROOT}"
export OBGPU_MECHANISM_ROOT="${REPO_ROOT}"

arch_dir="${REPO_ROOT}/$(uname -m)"
if [[ -f "${arch_dir}/libcorenrnmech.so" ]]; then
  export CORENEURONLIB="${arch_dir}/libcorenrnmech.so"
fi

if [[ -d "${arch_dir}" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${arch_dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${arch_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  export OB_MPIEXEC="${OB_MPIEXEC:-srun}"
elif command -v mpiexec >/dev/null 2>&1; then
  export OB_MPIEXEC="${OB_MPIEXEC:-mpiexec}"
fi
