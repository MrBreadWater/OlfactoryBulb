#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JUPYTER_STATE_DIR="${REPO_ROOT}/.jupyter-ai-state"

mkdir -p "${JUPYTER_STATE_DIR}/config" "${JUPYTER_STATE_DIR}/data" "${JUPYTER_STATE_DIR}/runtime"

export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-${JUPYTER_STATE_DIR}/config}"
export JUPYTER_DATA_DIR="${JUPYTER_DATA_DIR:-${JUPYTER_STATE_DIR}/data}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-${JUPYTER_STATE_DIR}/runtime}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate jupyter-ai

# Keep notebook trust warnings from hiding the real issue.
jupyter trust "${REPO_ROOT}/basic analysis .ipynb" >/dev/null 2>&1 || true

exec jupyter lab "$@"
