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
SOL_MAMBA_MODULE="${SOL_MAMBA_MODULE:-mamba/latest}"
SOL_NVHPC_MODULE="${SOL_NVHPC_MODULE:-nvhpc/24.9}"
SOL_CUDA_MODULE="${SOL_CUDA_MODULE:-cuda/12.6.1}"

ensure_module_cmd() {
  if declare -F module >/dev/null 2>&1; then
    return 0
  fi

  local init_script
  for init_script in \
    /etc/profile.d/z00_lmod.sh \
    /etc/profile.d/modules.sh \
    /usr/share/lmod/lmod/init/bash \
    /usr/share/Modules/init/bash; do
    if [[ -f "${init_script}" ]]; then
      # shellcheck disable=SC1090
      source "${init_script}"
      break
    fi
  done

  if ! declare -F module >/dev/null 2>&1; then
    echo "Could not initialize the module command in this shell." >&2
    return 1
  fi
}

maybe_load_module() {
  local module_name="$1"
  if [[ -z "${module_name}" ]]; then
    return 0
  fi
  if [[ ":${LOADEDMODULES:-}:" == *":${module_name}:"* ]]; then
    return 0
  fi
  module load "${module_name}"
}

ensure_conda_cmd() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  echo "conda is not on PATH after loading modules. Adjust SOL_MAMBA_MODULE if needed." >&2
  return 1
}

ensure_module_cmd

if [[ "${SOL_MODULE_PURGE}" == "1" ]]; then
  module purge
fi

maybe_load_module "${SOL_MAMBA_MODULE}"
maybe_load_module "${SOL_NVHPC_MODULE}"
maybe_load_module "${SOL_CUDA_MODULE}"

ensure_conda_cmd
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

export OBGPU_SOL_MODULES_LOADED=1

echo "Loaded Sol modules and activated ${ENV_NAME}." >&2
echo "Loaded modules: ${LOADEDMODULES:-<unknown>}" >&2
