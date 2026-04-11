#!/usr/bin/env bash
# Shared Sol module-resolution helpers for OBGPU activation and setup.

obgpu_sol_ensure_module_cmd() {
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

  declare -F module >/dev/null 2>&1
}

obgpu_sol_pick_loaded_module() {
  local prefix="$1"
  local loaded entry
  loaded="${LOADEDMODULES:-}"
  IFS=':' read -r -a _obgpu_loaded_modules <<< "${loaded}"
  for entry in "${_obgpu_loaded_modules[@]}"; do
    if [[ "${entry}" == "${prefix}" || "${entry}" == "${prefix}/"* ]]; then
      printf '%s\n' "${entry}"
      return 0
    fi
  done
  return 1
}

obgpu_sol_list_available_modules() {
  local prefix="$1"
  local raw line token cleaned found_any
  raw="$(module -t avail "${prefix}" 2>&1 || true)"
  found_any=0
  while IFS= read -r line; do
    for token in ${line}; do
      cleaned="${token%%(*}"
      cleaned="${cleaned%%:*}"
      cleaned="${cleaned#"${cleaned%%[![:space:]]*}"}"
      cleaned="${cleaned%"${cleaned##*[![:space:]]}"}"
      [[ -z "${cleaned}" ]] && continue
      if [[ "${cleaned}" == "${prefix}" || "${cleaned}" == "${prefix}/"* ]]; then
        printf '%s\n' "${cleaned}"
        found_any=1
      fi
    done
  done <<< "${raw}"

  if [[ "${found_any}" == "1" ]]; then
    return 0
  fi

  if [[ -z "${raw//[[:space:]]/}" || "${found_any}" == "0" ]]; then
    raw="$(module spider "${prefix}" 2>&1 || true)"
  fi
  while IFS= read -r line; do
    for token in ${line}; do
      cleaned="${token%%(*}"
      cleaned="${cleaned%%:*}"
      cleaned="${cleaned#"${cleaned%%[![:space:]]*}"}"
      cleaned="${cleaned%"${cleaned##*[![:space:]]}"}"
      [[ -z "${cleaned}" ]] && continue
      if [[ "${cleaned}" == "${prefix}" || "${cleaned}" == "${prefix}/"* ]]; then
        printf '%s\n' "${cleaned}"
      fi
    done
  done <<< "${raw}"
}

obgpu_sol_pick_available_module() {
  local prefix="$1"
  local candidates latest_module
  candidates="$(obgpu_sol_list_available_modules "${prefix}" | sort -uV)"
  [[ -z "${candidates}" ]] && return 1

  latest_module="$(printf '%s\n' "${candidates}" | awk -v prefix="${prefix}" '$0 == prefix"/latest" { print; exit }')"
  if [[ -n "${latest_module}" ]]; then
    printf '%s\n' "${latest_module}"
    return 0
  fi

  printf '%s\n' "${candidates}" | tail -n 1
}

obgpu_sol_resolve_module() {
  local prefix="$1"
  local explicit="${2:-}"

  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return 0
  fi

  if obgpu_sol_pick_loaded_module "${prefix}" >/dev/null; then
    obgpu_sol_pick_loaded_module "${prefix}"
    return 0
  fi

  obgpu_sol_pick_available_module "${prefix}"
}

obgpu_sol_maybe_load_module() {
  local module_name="$1"
  [[ -z "${module_name}" ]] && return 0
  if [[ ":${LOADEDMODULES:-}:" == *":${module_name}:"* ]]; then
    return 0
  fi
  module load "${module_name}"
}
