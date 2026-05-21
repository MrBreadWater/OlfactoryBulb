#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./install-obgpu.sh

This is a thin wrapper around:
  tools/setup/setup_ob_modern.sh

Common installs:
  ENABLE_GPU=1 ENV_NAME=OBGPU ./install-obgpu.sh
  ENV_NAME=OBGPU-portable ENABLE_GPU=0 OBGPU_CPU_TARGET=portable ./install-obgpu.sh

See INSTALL.md for the full flow and prerequisites.
EOF
  exit 0
fi

cd "${REPO_ROOT}"
exec "${REPO_ROOT}/tools/setup/setup_ob_modern.sh" "$@"
