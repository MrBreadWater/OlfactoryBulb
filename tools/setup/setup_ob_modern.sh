#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-OBGPU}"
ENABLE_GPU="${ENABLE_GPU:-0}"
PATCH_MANIFEST="${NRN_PATCH_MANIFEST:-${REPO_ROOT}/third_party_patches/nrn/manifest.json}"
PATCH_DIR="$(cd "$(dirname "${PATCH_MANIFEST}")" && pwd)"
NRN_SRC_DIR="${NRN_SRC_DIR:-${REPO_ROOT}/external/nrn-9.0.1}"
NRN_BUILD_DIR="${NRN_BUILD_DIR:-}"

if [[ ! -f "${PATCH_MANIFEST}" ]]; then
  echo "Patch manifest not found: ${PATCH_MANIFEST}" >&2
  exit 1
fi

mapfile -t manifest_lines < <(
  python - "${PATCH_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    manifest = json.load(f)

print(manifest["upstream_repo"])
print(manifest["upstream_ref"])
for patch in manifest.get("patches", []):
    print(patch["file"])
PY
)

if [[ "${#manifest_lines[@]}" -lt 2 ]]; then
  echo "Patch manifest ${PATCH_MANIFEST} is missing upstream metadata" >&2
  exit 1
fi

UPSTREAM_REPO="${manifest_lines[0]}"
UPSTREAM_REF="${manifest_lines[1]}"
PATCH_FILES=("${manifest_lines[@]:2}")

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on PATH" >&2
  exit 1
fi

prepare_nrn_source() {
  if [[ ! -d "${NRN_SRC_DIR}/.git" ]]; then
    git clone --recursive "${UPSTREAM_REPO}" "${NRN_SRC_DIR}"
  fi

  git -C "${NRN_SRC_DIR}" remote set-url origin "${UPSTREAM_REPO}"
  if ! git -C "${NRN_SRC_DIR}" fetch --tags --force origin; then
    echo "Warning: could not fetch ${UPSTREAM_REPO}; reusing local checkout if ${UPSTREAM_REF} exists." >&2
  fi

  if ! git -C "${NRN_SRC_DIR}" rev-parse --verify --quiet "${UPSTREAM_REF}^{commit}" >/dev/null; then
    echo "Pinned upstream ref ${UPSTREAM_REF} is not available in ${NRN_SRC_DIR}" >&2
    exit 1
  fi

  git -C "${NRN_SRC_DIR}" checkout --force "${UPSTREAM_REF}"
  git -C "${NRN_SRC_DIR}" reset --hard "${UPSTREAM_REF}"
  git -C "${NRN_SRC_DIR}" clean -fdx -e build-ob-modern -e build-ob-modern-gpu-*
  git -C "${NRN_SRC_DIR}" submodule sync --recursive
  git -C "${NRN_SRC_DIR}" submodule update --init --recursive --force

  for patch_file in "${PATCH_FILES[@]}"; do
    patch_path="${PATCH_DIR}/${patch_file}"
    if [[ ! -f "${patch_path}" ]]; then
      echo "Patch file not found: ${patch_path}" >&2
      exit 1
    fi
    git -C "${NRN_SRC_DIR}" apply --whitespace=nowarn "${patch_path}"
  done
}

find_latest_libnrnmech() {
  python - "${REPO_ROOT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = [
    path
    for path in root.glob("**/libnrnmech.so")
    if path.is_file() and ".git" not in path.parts and "external" not in path.parts
]
if not candidates:
    raise SystemExit(1)
candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
print(candidates[0])
PY
}

detect_python_runtime_paths() {
  python - <<'PY'
import sys
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])

candidates = []
libdir = sysconfig.get_config_var("LIBDIR")
ldlibrary = sysconfig.get_config_var("LDLIBRARY")
if libdir and ldlibrary:
    candidates.append(Path(libdir) / ldlibrary)

version = f"{sys.version_info.major}.{sys.version_info.minor}"
prefix = Path(sys.prefix)
base_prefix = Path(sys.base_prefix)
for root in (prefix / "lib", base_prefix / "lib"):
    candidates.extend(
        [
            root / f"libpython{version}.so",
            root / f"libpython{version}.so.1.0",
            root / f"libpython{version}.dylib",
        ]
    )

libpython = None
seen = set()
for candidate in candidates:
    candidate = candidate.resolve()
    if candidate in seen:
        continue
    seen.add(candidate)
    if candidate.exists():
        libpython = candidate
        break

if libpython is None:
    raise SystemExit(
        "Could not locate the active environment's libpython shared library. "
        f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )

print(purelib)
print(libpython)
PY
}

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environments/environment-modern.yml" --prune
else
  conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environments/environment-modern.yml"
fi

conda activate "${ENV_NAME}"

python -m pip install blenderneuron==2.0.4 lfpsimpy==0.1.1 natsort==8.4.0

prepare_nrn_source

python "${REPO_ROOT}/tools/setup/audit_nrn_patch_stack.py" \
  --source-tree "${NRN_SRC_DIR}" \
  --manifest "${PATCH_MANIFEST}"

export CC=gcc
export CXX=g++
export OMPI_CC=gcc
export OMPI_CXX=g++
export C_INCLUDE_PATH="${CONDA_PREFIX}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
gpu_cmake_args=()

if [[ "${ENABLE_GPU}" == "1" ]]; then
  NVHPC_SDK_ROOT="${NVHPC_SDK_ROOT:-${HOME}/.local/nvidia/hpc_sdk}"
  NVHPC_VERSION="${NVHPC_VERSION:-}"
  NVHPC_ARCH_ROOT="${NVHPC_SDK_ROOT}/$(uname -s)_$(uname -m)"
  if [[ -z "${NVHPC_C_COMPILER:-}" ]]; then
    if [[ -n "${NVHPC_VERSION}" && -x "${NVHPC_ARCH_ROOT}/${NVHPC_VERSION}/compilers/bin/nvc" ]]; then
      NVHPC_C_COMPILER="${NVHPC_ARCH_ROOT}/${NVHPC_VERSION}/compilers/bin/nvc"
    else
      NVHPC_C_COMPILER="$(command -v nvc || true)"
      if [[ -z "${NVHPC_C_COMPILER}" && -d "${NVHPC_ARCH_ROOT}" ]]; then
        NVHPC_C_COMPILER="$(find "${NVHPC_ARCH_ROOT}" -path '*/compilers/bin/nvc' -print 2>/dev/null | sort -V | tail -n 1 || true)"
      fi
    fi
  fi
  if [[ -z "${NVHPC_CXX_COMPILER:-}" ]]; then
    if [[ -n "${NVHPC_VERSION}" && -x "${NVHPC_ARCH_ROOT}/${NVHPC_VERSION}/compilers/bin/nvc++" ]]; then
      NVHPC_CXX_COMPILER="${NVHPC_ARCH_ROOT}/${NVHPC_VERSION}/compilers/bin/nvc++"
    else
      NVHPC_CXX_COMPILER="$(command -v nvc++ || true)"
      if [[ -z "${NVHPC_CXX_COMPILER}" && -d "${NVHPC_ARCH_ROOT}" ]]; then
        NVHPC_CXX_COMPILER="$(find "${NVHPC_ARCH_ROOT}" -path '*/compilers/bin/nvc++' -print 2>/dev/null | sort -V | tail -n 1 || true)"
      fi
    fi
  fi
  if [[ -z "${NVHPC_C_COMPILER}" || -z "${NVHPC_CXX_COMPILER}" ]]; then
    cat >&2 <<EOF
ENABLE_GPU=1 requires NVIDIA HPC SDK compilers (nvc and nvc++).
CUDA runtime/nvcc alone is not enough for a CoreNEURON GPU build.

Either:
  - load your cluster's NVHPC module(s), or
  - install the NVIDIA HPC SDK and export NVHPC_SDK_ROOT/NVHPC_VERSION.

Official NVIDIA HPC SDK downloads:
  https://developer.nvidia.com/hpc-sdk/downloads

Expected architecture on this host: $(uname -m)
EOF
    exit 1
  fi

  if [[ -z "${NVHPC_VERSION}" ]]; then
    NVHPC_VERSION="$(basename "$(dirname "$(dirname "$(dirname "${NVHPC_C_COMPILER}")")")")"
  fi
  if [[ -z "${NVHPC_CUDA_HOME:-}" ]]; then
    NVHPC_CUDA_ROOT="${NVHPC_ARCH_ROOT}/${NVHPC_VERSION}/cuda"
    if [[ -d "${NVHPC_CUDA_ROOT}" ]]; then
      NVHPC_CUDA_HOME="$(find "${NVHPC_CUDA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort -V | head -n 1)"
    fi
    if [[ -z "${NVHPC_CUDA_HOME:-}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
      NVHPC_CUDA_HOME="${CUDA_HOME}"
    fi
    if [[ -z "${NVHPC_CUDA_HOME:-}" && -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
      NVHPC_CUDA_HOME="${CUDA_PATH}"
    fi
    if [[ -z "${NVHPC_CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
      NVHPC_CUDA_HOME="/usr/local/cuda"
    fi
  fi
  if [[ -z "${CUDA_COMPILER:-}" ]]; then
    if [[ -n "${NVHPC_CUDA_HOME:-}" && -x "${NVHPC_CUDA_HOME}/bin/nvcc" ]]; then
      CUDA_COMPILER="${NVHPC_CUDA_HOME}/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
      CUDA_COMPILER="$(command -v nvcc)"
    else
      CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
    fi
  fi
  if [[ ! -x "${CUDA_COMPILER}" ]]; then
    echo "ENABLE_GPU=1 requires a working nvcc. Tried: ${CUDA_COMPILER}" >&2
    exit 1
  fi
  if [[ -z "${NVHPC_CUDA_HOME:-}" ]]; then
    NVHPC_CUDA_HOME="$(cd "$(dirname "${CUDA_COMPILER}")/.." && pwd)"
  fi
  GPU_BUILD_TAG="${GPU_BUILD_TAG:-${NVHPC_VERSION//./_}}"
  NRN_BUILD_DIR="${NRN_BUILD_DIR:-${NRN_SRC_DIR}/build-ob-modern-gpu-${GPU_BUILD_TAG}}"

  CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-$(
    python - "${NVHPC_CUDA_HOME:-}" <<'PY'
import ctypes
import ctypes.util
from pathlib import Path
import sys
from ctypes import byref, c_int, c_char, c_size_t, c_uint

cuda_root = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None

candidates = []
if cuda_root:
    candidates.extend(
        [
            cuda_root / "lib64" / "libcudart.so",
            cuda_root / "lib64" / "libcudart.so.12",
            cuda_root / "targets" / "x86_64-linux" / "lib" / "libcudart.so",
            cuda_root / "targets" / "sbsa-linux" / "lib" / "libcudart.so",
        ]
    )

found = ctypes.util.find_library("cudart")
if found:
    candidates.append(Path(found) if "/" in found else Path(found))

lib = None
load_errors = []
for candidate in candidates:
    try:
        lib = ctypes.CDLL(str(candidate))
        break
    except OSError as exc:
        load_errors.append(f"{candidate}: {exc}")

if lib is None:
    try:
        lib = ctypes.CDLL("libcudart.so")
    except OSError as exc:
        load_errors.append(f"libcudart.so: {exc}")
        raise SystemExit(
            "Could not load libcudart for CUDA architecture detection. "
            + "; ".join(load_errors)
        )

class cudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", c_char * 256),
        ("uuid", c_char * 16),
        ("luid", c_char * 8),
        ("luidDeviceNodeMask", c_uint),
        ("totalGlobalMem", c_size_t),
        ("sharedMemPerBlock", c_size_t),
        ("regsPerBlock", c_int),
        ("warpSize", c_int),
        ("memPitch", c_size_t),
        ("maxThreadsPerBlock", c_int),
        ("maxThreadsDim", c_int * 3),
        ("maxGridSize", c_int * 3),
        ("clockRate", c_int),
        ("totalConstMem", c_size_t),
        ("major", c_int),
        ("minor", c_int),
        ("rest", c_char * 2048),
    ]

count = c_int()
rc = lib.cudaGetDeviceCount(byref(count))
if rc != 0 or count.value < 1:
    raise SystemExit(1)
prop = cudaDeviceProp()
rc = lib.cudaGetDeviceProperties(byref(prop), 0)
if rc != 0:
    raise SystemExit(1)
print(f"{prop.major}{prop.minor}")
PY
  )}"

  NVHPC_COMPUTE_CAPABILITIES="${NVHPC_COMPUTE_CAPABILITIES:-${CUDA_ARCHITECTURES}}"
  mapfile -t supported_arches < <("${NVHPC_CXX_COMPILER}" -help -gpu 2>&1 | grep -oE 'cc[0-9]{2}' | sed 's/^cc//' | sort -uV)
  if [[ "${#supported_arches[@]}" -gt 0 ]]; then
    if ! printf '%s\n' "${supported_arches[@]}" | grep -qx "${NVHPC_COMPUTE_CAPABILITIES}"; then
      fallback_arch="$(printf '%s\n' "${supported_arches[@]}" | awk -v detected="${NVHPC_COMPUTE_CAPABILITIES}" '$1 <= detected { best=$1 } END { if (best != "") print best }')"
      if [[ -z "${fallback_arch}" ]]; then
        fallback_arch="${supported_arches[${#supported_arches[@]}-1]}"
      fi
      echo "Detected GPU architecture ${NVHPC_COMPUTE_CAPABILITIES} is not supported by ${NVHPC_CXX_COMPILER}; using ${fallback_arch} for NVHPC offload while keeping CUDA kernels at ${CUDA_ARCHITECTURES}." >&2
      NVHPC_COMPUTE_CAPABILITIES="${fallback_arch}"
    fi
  fi

  export CC="${NVHPC_C_COMPILER}"
  export CXX="${NVHPC_CXX_COMPILER}"
  if [[ -n "${NVHPC_CUDA_HOME:-}" ]]; then
    export NVHPC_CUDA_HOME
    export CUDA_HOME="${NVHPC_CUDA_HOME}"
  fi

  gpu_cmake_args+=(
    -DCORENRN_ENABLE_GPU=ON
    -DCMAKE_C_COMPILER="${NVHPC_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${NVHPC_CXX_COMPILER}"
    -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}"
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
    -DCORENRN_NVHPC_COMPUTE_CAPABILITIES="${NVHPC_COMPUTE_CAPABILITIES}"
  )
  if [[ -n "${NVHPC_CUDA_HOME:-}" ]]; then
    gpu_cmake_args+=(
      -DCUDAToolkit_ROOT="${NVHPC_CUDA_HOME}"
    )
  fi
else
  NRN_BUILD_DIR="${NRN_BUILD_DIR:-${NRN_SRC_DIR}/build-ob-modern}"
fi

cmake_args=(
  -G Ninja
  -S "${NRN_SRC_DIR}"
  -B "${NRN_BUILD_DIR}"
  -DCMAKE_MAKE_PROGRAM="$(command -v ninja)"
  -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}"
  -DPYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
  -DNRN_ENABLE_MPI=ON
  -DNRN_ENABLE_CORENEURON=ON
  -DNRN_ENABLE_INTERVIEWS=OFF
  -DNRN_ENABLE_RX3D=OFF
  -DNRN_ENABLE_DOCS=OFF
  -DNRN_ENABLE_TESTS=OFF
  -DCORENRN_ENABLE_REPORTING=ON
  -DCORENRN_ENABLE_LOCAL_REPORT_SHIM=ON
  -DMPI_C_COMPILER="${CONDA_PREFIX}/bin/mpicc"
  -DMPI_CXX_COMPILER="${CONDA_PREFIX}/bin/mpicxx"
)

if [[ "${ENABLE_GPU}" == "1" ]]; then
  cmake_args+=("${gpu_cmake_args[@]}")
fi

cmake "${cmake_args[@]}"

cmake --build "${NRN_BUILD_DIR}" --parallel 8
cmake --install "${NRN_BUILD_DIR}"

mapfile -t python_runtime_paths < <(detect_python_runtime_paths)
if [[ "${#python_runtime_paths[@]}" -ne 2 ]]; then
  echo "Could not determine Python runtime paths for ${CONDA_PREFIX}" >&2
  exit 1
fi
PYTHON_SITE_PACKAGES="${python_runtime_paths[0]}"
PYTHON_SHARED_LIB="${python_runtime_paths[1]}"

mkdir -p "${PYTHON_SITE_PACKAGES}"
printf '%s\n' "${CONDA_PREFIX}/lib/python" > "${PYTHON_SITE_PACKAGES}/ob_modern_neuron.pth"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/ob_modern_neuron.sh" <<EOF
export OMPI_MCA_opal_cuda_support=true
export NMODLHOME=${CONDA_PREFIX}
export NMODL_PYLIB=${PYTHON_SHARED_LIB}
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/ob_modern_neuron.sh" <<'EOF'
unset OMPI_MCA_opal_cuda_support
unset NMODLHOME
unset NMODL_PYLIB
EOF

(
  cd "${REPO_ROOT}"
  OMPI_CC=gcc OMPI_CXX=g++ nrnivmodl -coreneuron prev_ob_models/Birgiolas2020/Mechanisms
)

if [[ "${ENABLE_GPU}" == "1" ]]; then
  if libnrnmech_path="$(find_latest_libnrnmech)"; then
    "${REPO_ROOT}/tools/setup/fix_nvhpc_libnrnmech.sh" "${libnrnmech_path}"
  else
    echo "Warning: could not locate a generated libnrnmech.so to repair." >&2
  fi
fi

echo "OBGPU setup complete from ${UPSTREAM_REF}."
