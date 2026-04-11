#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/libnrnmech.so" >&2
  exit 1
fi

lib_path="$1"

if [[ ! -f "${lib_path}" ]]; then
  echo "library not found: ${lib_path}" >&2
  exit 1
fi

if ! command -v patchelf >/dev/null 2>&1; then
  echo "patchelf is required to repair ${lib_path}" >&2
  exit 1
fi

merge_origin_rpath() {
  local target="$1"
  local existing_rpath merged_rpath

  existing_rpath="$(patchelf --print-rpath "${target}" 2>/dev/null || true)"
  if [[ -z "${existing_rpath}" ]]; then
    merged_rpath='$ORIGIN'
  elif [[ ":${existing_rpath}:" == *':$ORIGIN:'* ]]; then
    merged_rpath="${existing_rpath}"
  else
    merged_rpath="\$ORIGIN:${existing_rpath}"
  fi

  patchelf --set-rpath "${merged_rpath}" "${target}"
}

repair_one_lib() {
  local target="$1"
  mapfile -t bogus_needed < <(patchelf --print-needed "${target}" | grep '^/tmp/pgcudafat' || true)

  for dep in "${bogus_needed[@]}"; do
    patchelf --remove-needed "${dep}" "${target}"
  done

  merge_origin_rpath "${target}"
}

repair_one_lib "${lib_path}"

if [[ "$(basename "${lib_path}")" == "libnrnmech.so" ]]; then
  patchelf --set-soname libnrnmech.so "${lib_path}"
fi

lib_dir="$(dirname "${lib_path}")"
corenrn_lib="${lib_dir}/libcorenrnmech.so"
if [[ -f "${corenrn_lib}" ]]; then
  repair_one_lib "${corenrn_lib}"
fi
