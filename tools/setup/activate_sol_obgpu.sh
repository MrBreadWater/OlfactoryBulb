#!/usr/bin/env bash
# Source this helper on Sol after obtaining an interactive allocation.
# It loads the expected toolchain modules and activates the OBGPU conda env.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source tools/setup/activate_sol_obgpu.sh" >&2
  exit 1
fi

set -euo pipefail

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

if ! obgpu_sol_ensure_module_cmd; then
  echo "Could not initialize the module command in this shell." >&2
  return 1
fi

if [[ "${SOL_MODULE_PURGE}" == "1" ]]; then
  module purge
fi

SOL_MAMBA_MODULE="${SOL_MAMBA_MODULE:-$(obgpu_sol_resolve_module mamba "${SOL_MAMBA_MODULE:-}")}"
SOL_NVHPC_MODULE="${SOL_NVHPC_MODULE:-$(obgpu_sol_resolve_module nvhpc "${SOL_NVHPC_MODULE:-}")}"
SOL_CUDA_MODULE="${SOL_CUDA_MODULE:-$(obgpu_sol_resolve_module cuda "${SOL_CUDA_MODULE:-}")}"

obgpu_sol_maybe_load_module "${SOL_MAMBA_MODULE}"
obgpu_sol_maybe_load_module "${SOL_NVHPC_MODULE}"
obgpu_sol_maybe_load_module "${SOL_CUDA_MODULE}"

ensure_conda_cmd
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

export OBGPU_SOL_MODULES_LOADED=1

echo "Loaded Sol modules and activated ${ENV_NAME}." >&2
echo "Resolved modules: mamba=${SOL_MAMBA_MODULE:-<none>} nvhpc=${SOL_NVHPC_MODULE:-<none>} cuda=${SOL_CUDA_MODULE:-<none>}" >&2
echo "Loaded modules: ${LOADEDMODULES:-<unknown>}" >&2
