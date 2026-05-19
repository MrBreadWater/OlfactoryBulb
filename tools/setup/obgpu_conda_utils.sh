#!/usr/bin/env bash
# Shared conda bootstrap helpers for generic OBGPU setup and activation.

obgpu_try_source_conda_sh() {
  local conda_sh="$1"
  [[ -n "${conda_sh}" && -f "${conda_sh}" ]] || return 1
  # shellcheck disable=SC1090
  source "${conda_sh}"
  command -v conda >/dev/null 2>&1
}

obgpu_try_source_conda_base() {
  local conda_base="$1"
  [[ -n "${conda_base}" ]] || return 1
  obgpu_try_source_conda_sh "${conda_base}/etc/profile.d/conda.sh"
}

obgpu_bootstrap_conda() {
  local conda_sh=""
  local conda_bin=""
  local module_name=""

  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  for conda_sh in \
    "${CONDA_SH:-}" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    obgpu_try_source_conda_sh "${conda_sh}" && return 0
  done

  for conda_bin in \
    "${CONDA_EXE:-}" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/mambaforge/bin/conda" \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    [[ -x "${conda_bin}" ]] || continue
    obgpu_try_source_conda_base "$(cd "$(dirname "${conda_bin}")/.." 2>/dev/null && pwd || true)" && return 0
  done

  if type module >/dev/null 2>&1; then
    for module_name in mamba anaconda miniconda miniconda3 miniforge mambaforge; do
      module load "${module_name}" >/dev/null 2>&1 || true
      if command -v conda >/dev/null 2>&1; then
        return 0
      fi
    done
  fi

  command -v conda >/dev/null 2>&1
}

obgpu_find_env_prefix() {
  local env_name="$1"
  local candidate=""

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" || "$(basename "${CONDA_PREFIX}")" == "${env_name}" ]]; then
      printf '%s\n' "${CONDA_PREFIX}"
      return 0
    fi
  fi

  for candidate in \
    "${OBGPU_ENV_PREFIX:-}" \
    "${HOME}/.conda/envs/${env_name}" \
    "${HOME}/miniconda3/envs/${env_name}" \
    "${HOME}/mambaforge/envs/${env_name}" \
    "${HOME}/miniforge3/envs/${env_name}" \
    "${HOME}/anaconda3/envs/${env_name}"; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/python" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if obgpu_bootstrap_conda; then
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

obgpu_activate_conda_env() {
  local env_name="$1"
  local conda_bin=""
  local conda_base=""
  local conda_sh=""
  local hook_output=""

  if ! obgpu_bootstrap_conda; then
    echo "Could not activate environment '${env_name}': conda is not available." >&2
    return 1
  fi

  conda_bin="$(command -v conda)"
  for conda_base in \
    "$(cd "$(dirname "${conda_bin}")/.." 2>/dev/null && pwd || true)" \
    "$(conda info --base 2>/dev/null || true)"; do
    [[ -n "${conda_base}" ]] || continue
    conda_sh="${conda_base}/etc/profile.d/conda.sh"
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      conda activate "${env_name}" || return 1
      return 0
    fi
  done

  hook_output="$(conda shell.bash hook 2>/dev/null || true)"
  if [[ -n "${hook_output}" ]] && grep -q '__conda_' <<<"${hook_output}"; then
    eval "${hook_output}"
    conda activate "${env_name}" || return 1
    return 0
  fi

  if type activate >/dev/null 2>&1; then
    activate "${env_name}" || return 1
    return 0
  fi

  echo "Could not activate environment '${env_name}'." >&2
  return 1
}
