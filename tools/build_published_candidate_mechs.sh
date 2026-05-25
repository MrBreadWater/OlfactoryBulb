#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <family> <outdir>" >&2
    echo "Examples:" >&2
    echo "  $0 Short2016 /tmp/Short2016_minimal_mechs" >&2
    echo "  $0 LiCleland2013 /tmp/LiCleland2013_minimal_mechs" >&2
    exit 2
fi

family="$1"
outdir="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

declare -a mod_files
case "$family" in
    Short2016|Short2016.PGC|Short2016.ETC)
        source_dir="${repo_root}/prev_ob_models/Short2016"
        mod_files=(
            CaPN.mod
            CaT.mod
            Can.mod
            KCa.mod
            Nicotin.mod
            cadecay2.mod
            hpg.mod
            kM.mod
            kamt.mod
            kdrmt.mod
            naxn.mod
        )
        ;;
    LiCleland2013|LiCleland2013.PGC|LiCleland2013.GC)
        source_dir="${repo_root}/prev_ob_models/LiCleland2013"
        mod_files=(
            CaPN.mod
            CaT.mod
            Can.mod
            KCa.mod
            Nicotin.mod
            cadecay2.mod
            hpg.mod
            kAmt.mod
            kM.mod
            KDRmt.mod
            Naxn.mod
            nmdanet.mod
        )
        ;;
    *)
        echo "Unsupported family: ${family}" >&2
        exit 2
        ;;
esac

if ! command -v nrnivmodl >/dev/null 2>&1; then
    echo "nrnivmodl not found on PATH. Run this from a NEURON-enabled environment." >&2
    exit 127
fi

mkdir -p "${outdir}"

if find "${outdir}" -maxdepth 1 -name '*.mod' | grep -q .; then
    echo "Refusing to reuse ${outdir}: it already contains .mod files." >&2
    echo "Use an empty build directory to avoid mixing incompatible mechanism sets." >&2
    exit 2
fi

for mod_file in "${mod_files[@]}"; do
    cp "${source_dir}/${mod_file}" "${outdir}/"
done

(
    cd "${outdir}"
    nrnivmodl
)
