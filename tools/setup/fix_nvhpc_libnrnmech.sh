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

mapfile -t bogus_needed < <(patchelf --print-needed "${lib_path}" | grep '^/tmp/pgcudafat' || true)

if [[ ${#bogus_needed[@]} -eq 0 ]]; then
  exit 0
fi

for dep in "${bogus_needed[@]}"; do
  patchelf --remove-needed "${dep}" "${lib_path}"
done

patchelf --set-soname libnrnmech.so "${lib_path}"

