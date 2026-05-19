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

case ":${LD_LIBRARY_PATH:-}:" in
  *":${CONDA_PREFIX}/lib:"*) ;;
  *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

if [[ -d "${arch_dir}" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${arch_dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${arch_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

detect_srun_mpi_exec() {
  if ! command -v srun >/dev/null 2>&1; then
    return 1
  fi

  local preferred_type="${OB_SLURM_MPI_TYPE:-pmix}"
  local mpi_list=""
  mpi_list="$(srun --mpi=list 2>/dev/null || true)"

  if [[ -n "${preferred_type}" ]] && grep -Eq "(^|[[:space:],])${preferred_type}([[:space:],]|$)" <<<"${mpi_list}"; then
    printf 'srun --mpi=%s --cpu-bind=none\n' "${preferred_type}"
    return 0
  fi

  if grep -Eq '(^|[[:space:],])pmix([[:space:],]|$)' <<<"${mpi_list}"; then
    printf 'srun --mpi=pmix --cpu-bind=none\n'
    return 0
  fi

  if grep -Eq '(^|[[:space:],])pmi2([[:space:],]|$)' <<<"${mpi_list}"; then
    printf 'srun --mpi=pmi2 --cpu-bind=none\n'
    return 0
  fi

  printf 'srun --cpu-bind=none\n'
}

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v mpiexec >/dev/null 2>&1; then
  export OB_MPIEXEC="${OB_MPIEXEC:-mpiexec --bind-to none}"
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  if [[ -z "${OB_MPIEXEC:-}" ]]; then
    if detected_mpi_exec="$(detect_srun_mpi_exec)"; then
      export OB_MPIEXEC="${detected_mpi_exec}"
    fi
  fi
elif command -v mpiexec >/dev/null 2>&1; then
  export OB_MPIEXEC="${OB_MPIEXEC:-mpiexec}"
fi
