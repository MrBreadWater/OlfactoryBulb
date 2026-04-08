#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate OB

exec python -Xfrozen_modules=off -m ipykernel_launcher "$@"
