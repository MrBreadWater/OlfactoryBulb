#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on PATH to launch the OBGPU kernel" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OBGPU

exec python -Xfrozen_modules=off -m ipykernel_launcher "$@"
