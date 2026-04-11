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

ensure_conda_cmd() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  echo "conda is not on PATH after loading modules. Adjust SOL_MAMBA_MODULE if needed." >&2
  return 1
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
for required_tool in conda nvc nvcc; do
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

SOL_MAMBA_MODULE="$(resolve_module_if_needed mamba "${SOL_MAMBA_MODULE:-}" conda)"
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

ensure_conda_cmd
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

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
if [[ "${#missing_tools[@]}" -gt 0 ]]; then
  echo "Warning: toolchain commands not on PATH after activation: ${missing_tools[*]}" >&2
fi
