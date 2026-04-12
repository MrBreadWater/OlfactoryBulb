#!/usr/bin/env bash
# Source this helper on Sol after obtaining an interactive allocation.
# It loads the expected toolchain modules and activates the OBGPU conda env.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source tools/setup/activate_sol_obgpu.sh" >&2
  exit 1
fi

ENV_NAME="${1:-${ENV_NAME:-OBGPU}}"
SOL_MODULE_PURGE="${SOL_MODULE_PURGE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/obgpu_sol_module_utils.sh"

ensure_env_activation_cmd() {
  if find_env_prefix "${ENV_NAME}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Could not locate the conda environment prefix for '${ENV_NAME}'." >&2
  return 1
}

find_env_prefix() {
  local env_name="$1"
  local candidate=""

  for candidate in \
    "${OBGPU_ENV_PREFIX:-}" \
    "${HOME}/.conda/envs/${env_name}" \
    "/home/${USER}/.conda/envs/${env_name}"; do
    [[ -z "${candidate}" ]] && continue
    if [[ -x "${candidate}/bin/python" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    candidate="$(
      conda env list 2>/dev/null | awk -v env_name="${env_name}" '
        $1 == env_name { print $NF; exit }
      '
    )"
    if [[ -n "${candidate}" && -x "${candidate}/bin/python" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  return 1
}

activate_env() {
  local env_name="$1"
  local env_prefix=""
  local hook_dir=""
  local hook_script=""

  env_prefix="$(find_env_prefix "${env_name}")" || {
    echo "Could not activate environment '${env_name}': prefix not found." >&2
    return 1
  }

  if [[ "${CONDA_PREFIX:-}" != "${env_prefix}" ]]; then
    export CONDA_PREFIX="${env_prefix}"
    export CONDA_DEFAULT_ENV="${env_name}"
    case ":${PATH}:" in
      *":${env_prefix}/bin:"*) ;;
      *) export PATH="${env_prefix}/bin:${PATH}" ;;
    esac
  fi

  if [[ "${_OBGPU_SOL_ACTIVE_PREFIX:-}" == "${env_prefix}" ]]; then
    return 0
  fi

  hook_dir="${env_prefix}/etc/conda/activate.d"
  if [[ -d "${hook_dir}" ]]; then
    shopt -s nullglob
    for hook_script in "${hook_dir}"/*.sh; do
      # shellcheck disable=SC1090
      source "${hook_script}"
    done
    shopt -u nullglob
  fi

  export _OBGPU_SOL_ACTIVE_PREFIX="${env_prefix}"
  return 0
}

detect_srun_mpi_exec() {
  if ! command -v srun >/dev/null 2>&1; then
    return 1
  fi

  local preferred_type="${OB_SLURM_MPI_TYPE:-pmix}"
  local mpi_list=""
  mpi_list="$(srun --mpi=list 2>/dev/null || true)"

  if [[ -n "${preferred_type}" ]] && grep -Eq "(^|[[:space:],])${preferred_type}([[:space:],]|$)" <<<"${mpi_list}"; then
    printf 'srun --mpi=%s\n' "${preferred_type}"
    return 0
  fi

  if grep -Eq '(^|[[:space:],])pmix([[:space:],]|$)' <<<"${mpi_list}"; then
    printf 'srun --mpi=pmix\n'
    return 0
  fi

  if grep -Eq '(^|[[:space:],])pmi2([[:space:],]|$)' <<<"${mpi_list}"; then
    printf 'srun --mpi=pmi2\n'
    return 0
  fi

  printf 'srun\n'
}

resolve_module_if_needed() {
  local prefix="$1"
  local explicit="${2:-}"
  local tool_name="${3:-}"

  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return 0
  fi

  if [[ -n "${tool_name}" ]] && command -v "${tool_name}" >/dev/null 2>&1; then
    return 0
  fi

  obgpu_sol_resolve_module "${prefix}" "" || true
}

need_module_cmd=0
if [[ "${SOL_MODULE_PURGE}" == "1" ]]; then
  need_module_cmd=1
fi
for explicit_module in "${SOL_MAMBA_MODULE:-}" "${SOL_NVHPC_MODULE:-}" "${SOL_CUDA_MODULE:-}"; do
  if [[ -n "${explicit_module}" ]]; then
    need_module_cmd=1
    break
  fi
done
for required_tool in nvc nvcc; do
  if ! command -v "${required_tool}" >/dev/null 2>&1; then
    need_module_cmd=1
    break
  fi
done

if [[ "${need_module_cmd}" == "1" ]]; then
  if ! obgpu_sol_ensure_module_cmd; then
    echo "Could not initialize the module command in this shell." >&2
    return 1
  fi
fi

if [[ "${SOL_MODULE_PURGE}" == "1" ]]; then
  module purge
fi

SOL_MAMBA_MODULE="${SOL_MAMBA_MODULE:-}"
SOL_NVHPC_MODULE="$(resolve_module_if_needed nvhpc "${SOL_NVHPC_MODULE:-}" nvc)"
SOL_CUDA_MODULE="$(resolve_module_if_needed cuda "${SOL_CUDA_MODULE:-}" nvcc)"

if ! obgpu_sol_maybe_load_module "${SOL_MAMBA_MODULE}"; then
  echo "Failed to load module '${SOL_MAMBA_MODULE}'." >&2
  return 1
fi
if ! obgpu_sol_maybe_load_module "${SOL_NVHPC_MODULE}"; then
  echo "Failed to load module '${SOL_NVHPC_MODULE}'." >&2
  return 1
fi
if ! obgpu_sol_maybe_load_module "${SOL_CUDA_MODULE}"; then
  echo "Failed to load module '${SOL_CUDA_MODULE}'." >&2
  return 1
fi

ensure_env_activation_cmd
activate_env "${ENV_NAME}"

if [[ -n "${SLURM_JOB_ID:-}" ]] && [[ -z "${OB_MPIEXEC:-}" ]]; then
  if detected_mpi_exec="$(detect_srun_mpi_exec)"; then
    export OB_MPIEXEC="${detected_mpi_exec}"
  fi
fi

missing_tools=()
for tool_name in nvc "nvc++" nvcc; do
  if ! command -v "${tool_name}" >/dev/null 2>&1; then
    missing_tools+=("${tool_name}")
  fi
done

export OBGPU_SOL_MODULES_LOADED=1

echo "Loaded Sol modules and activated ${ENV_NAME}." >&2
echo "Resolved modules: mamba=${SOL_MAMBA_MODULE:-<none>} nvhpc=${SOL_NVHPC_MODULE:-<none>} cuda=${SOL_CUDA_MODULE:-<none>}" >&2
echo "Loaded modules: ${LOADEDMODULES:-<unknown>}" >&2
if [[ -n "${OB_MPIEXEC:-}" ]]; then
  echo "MPI launcher: ${OB_MPIEXEC}" >&2
fi
if [[ "${#missing_tools[@]}" -gt 0 ]]; then
  echo "Warning: toolchain commands not on PATH after activation: ${missing_tools[*]}" >&2
fi
