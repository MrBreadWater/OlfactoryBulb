#!/usr/bin/env bash
set -Eeuo pipefail

log_step() {
  echo "[OBGPU setup] $*" >&2
}

trap 'status=$?; echo "[OBGPU setup] failed (exit ${status}) at line ${LINENO}: ${BASH_COMMAND}" >&2; exit ${status}' ERR

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

find_current_arch_libnrnmech() {
  python - "${REPO_ROOT}" "$(uname -m)" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
arch = sys.argv[2]
arch_dir = root / arch
candidates = [
    arch_dir / "libnrnmech.so",
    arch_dir / ".libs" / "libnrnmech.so",
]

for candidate in candidates:
    if candidate.is_file():
        print(candidate)
        raise SystemExit(0)

fallbacks = sorted(
    [
        path
        for path in arch_dir.glob("**/libnrnmech.so")
        if path.is_file()
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if fallbacks:
    print(fallbacks[0])
    raise SystemExit(0)

raise SystemExit(1)
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

file_sha256() {
  python - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

path = Path(sys.argv[1])
digest = hashlib.sha256()
digest.update(path.read_bytes())
print(digest.hexdigest())
PY
}

hash_stdin() {
  python - <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())
PY
}

hash_text() {
  python - "$1" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode()).hexdigest())
PY
}

detect_cuda_architectures() {
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
    raise SystemExit(
        "Could not auto-detect CUDA_ARCHITECTURES because no visible CUDA device was found. "
        "Set CUDA_ARCHITECTURES and NVHPC_COMPUTE_CAPABILITIES explicitly, or rerun on a GPU node."
    )
prop = cudaDeviceProp()
rc = lib.cudaGetDeviceProperties(byref(prop), 0)
if rc != 0:
    raise SystemExit("cudaGetDeviceProperties failed while detecting CUDA_ARCHITECTURES.")
print(f"{prop.major}{prop.minor}")
PY
}

detect_supported_nvhpc_arches() {
  local compiler="$1"
  local help_output

  help_output="$("${compiler}" -help -gpu 2>&1 || true)"
  printf '%s\n' "${help_output}" | sed -n 's/.*\(cc[0-9][0-9]\).*/\1/p' | sed 's/^cc//' | sort -uV
}

load_previous_build_meta() {
  local meta_path="$1"
  if [[ -f "${meta_path}" ]]; then
    # shellcheck disable=SC1090
    source "${meta_path}"
    if [[ -z "${CUDA_ARCHITECTURES:-}" && -n "${OBGPU_STAMP_CUDA_ARCHITECTURES:-}" ]]; then
      CUDA_ARCHITECTURES="${OBGPU_STAMP_CUDA_ARCHITECTURES}"
    fi
    if [[ -z "${NVHPC_COMPUTE_CAPABILITIES:-}" && -n "${OBGPU_STAMP_NVHPC_COMPUTE_CAPABILITIES:-}" ]]; then
      NVHPC_COMPUTE_CAPABILITIES="${OBGPU_STAMP_NVHPC_COMPUTE_CAPABILITIES}"
    fi
  fi
}

write_build_meta() {
  local meta_path="$1"
  mkdir -p "$(dirname "${meta_path}")"
  cat > "${meta_path}" <<EOF
OBGPU_STAMP_CUDA_ARCHITECTURES='${CUDA_ARCHITECTURES:-}'
OBGPU_STAMP_NVHPC_COMPUTE_CAPABILITIES='${NVHPC_COMPUTE_CAPABILITIES:-}'
OBGPU_STAMP_NVHPC_C_COMPILER='${NVHPC_C_COMPILER:-}'
OBGPU_STAMP_NVHPC_CXX_COMPILER='${NVHPC_CXX_COMPILER:-}'
OBGPU_STAMP_CUDA_COMPILER='${CUDA_COMPILER:-}'
OBGPU_STAMP_NVHPC_CUDA_HOME='${NVHPC_CUDA_HOME:-}'
EOF
}

build_stamp_fingerprint() {
  local payload
  payload="$(
    {
    printf 'upstream_repo=%s\n' "${UPSTREAM_REPO}"
    printf 'upstream_ref=%s\n' "${UPSTREAM_REF}"
    printf 'patch_manifest=%s\n' "${PATCH_MANIFEST}"
    printf 'patch_manifest_sha=%s\n' "$(file_sha256 "${PATCH_MANIFEST}")"
    for patch_file in "${PATCH_FILES[@]}"; do
      printf 'patch=%s sha=%s\n' "${patch_file}" "$(file_sha256 "${PATCH_DIR}/${patch_file}")"
    done
    printf 'enable_gpu=%s\n' "${ENABLE_GPU}"
    printf 'conda_prefix=%s\n' "${CONDA_PREFIX}"
    printf 'python_executable=%s\n' "${CONDA_PREFIX}/bin/python"
    printf 'cc=%s\n' "${CC}"
    printf 'cxx=%s\n' "${CXX}"
    printf 'ompi_cc=%s\n' "${OMPI_CC}"
    printf 'ompi_cxx=%s\n' "${OMPI_CXX}"
    printf 'nrn_src_dir=%s\n' "${NRN_SRC_DIR}"
    printf 'nrn_build_dir=%s\n' "${NRN_BUILD_DIR}"
    printf 'nvhpc_c_compiler=%s\n' "${NVHPC_C_COMPILER:-}"
    printf 'nvhpc_cxx_compiler=%s\n' "${NVHPC_CXX_COMPILER:-}"
    printf 'cuda_compiler=%s\n' "${CUDA_COMPILER:-}"
    printf 'nvhpc_cuda_home=%s\n' "${NVHPC_CUDA_HOME:-}"
    printf 'cuda_architectures=%s\n' "${CUDA_ARCHITECTURES:-}"
    printf 'nvhpc_compute_capabilities=%s\n' "${NVHPC_COMPUTE_CAPABILITIES:-}"
    printf 'gpu_build_tag=%s\n' "${GPU_BUILD_TAG:-}"
    printf 'cmake_arg=%s\n' "${cmake_args[@]}"
    }
  )"
  hash_text "${payload}"
}

mechanism_stamp_fingerprint() {
  local base_fingerprint="$1"
  local payload
  payload="$(
    {
    printf 'build_fingerprint=%s\n' "${base_fingerprint}"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'conda_prefix=%s\n' "${CONDA_PREFIX}"
    printf 'ompi_cc=%s\n' "${OMPI_CC}"
    printf 'ompi_cxx=%s\n' "${OMPI_CXX}"
    printf 'machine_arch=%s\n' "$(uname -m)"
    while IFS= read -r mod_file; do
      [[ -z "${mod_file}" ]] && continue
      printf 'mod=%s sha=%s\n' "${mod_file}" "$(file_sha256 "${mod_file}")"
    done < <(find "${REPO_ROOT}/prev_ob_models/Birgiolas2020/Mechanisms" -maxdepth 1 -name '*.mod' -type f | sort)
    }
  )"
  hash_text "${payload}"
}

stamp_matches() {
  local stamp_path="$1"
  local expected="$2"
  [[ -f "${stamp_path}" ]] && [[ "$(tr -d '\n' < "${stamp_path}")" == "${expected}" ]]
}

neuron_install_ok() {
  python - <<'PY' >/dev/null 2>&1
import neuron
from neuron import coreneuron
print(neuron.__version__, coreneuron)
PY
}

eval "$(conda shell.bash hook)"

log_step "Preparing conda environment ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environments/environment-modern.yml" --prune
else
  conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environments/environment-modern.yml"
fi

conda activate "${ENV_NAME}"

log_step "Installing required pip packages into ${ENV_NAME}"
python -m pip install blenderneuron==2.0.4 lfpsimpy==0.1.1 natsort==8.4.0

log_step "Preparing pinned NEURON source tree at ${NRN_SRC_DIR}"
prepare_nrn_source

log_step "Auditing local NEURON patch stack coverage"
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
  log_step "Resolving GPU toolchain configuration"
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
  NRN_BUILD_META_PATH="${NRN_BUILD_DIR}/.obgpu_build_meta.sh"
  load_previous_build_meta "${NRN_BUILD_META_PATH}"

  if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
    CUDA_ARCHITECTURES="$(detect_cuda_architectures)"
  fi

  NVHPC_COMPUTE_CAPABILITIES="${NVHPC_COMPUTE_CAPABILITIES:-${CUDA_ARCHITECTURES}}"
  mapfile -t supported_arches < <(detect_supported_nvhpc_arches "${NVHPC_CXX_COMPILER}")
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
  NRN_BUILD_META_PATH="${NRN_BUILD_DIR}/.obgpu_build_meta.sh"
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

NRN_BUILD_STAMP_PATH="${NRN_BUILD_DIR}/.obgpu_build_stamp"
NRN_BUILD_FINGERPRINT="$(build_stamp_fingerprint)"

if stamp_matches "${NRN_BUILD_STAMP_PATH}" "${NRN_BUILD_FINGERPRINT}" && neuron_install_ok; then
  log_step "Skipping NEURON/CoreNEURON build; matching successful stamp found at ${NRN_BUILD_STAMP_PATH}"
else
  log_step "Configuring and building NEURON/CoreNEURON in ${NRN_BUILD_DIR}"
  cmake "${cmake_args[@]}"
  cmake --build "${NRN_BUILD_DIR}" --parallel 8
  cmake --install "${NRN_BUILD_DIR}"
  mkdir -p "${NRN_BUILD_DIR}"
  printf '%s\n' "${NRN_BUILD_FINGERPRINT}" > "${NRN_BUILD_STAMP_PATH}"
  write_build_meta "${NRN_BUILD_META_PATH}"
fi

log_step "Resolving Python runtime paths for NMODL"
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
if [[ "\${OBGPU_AUTOLOAD_SOL_MODULES:-0}" == "1" ]]; then
  _obgpu_sol_helper="${REPO_ROOT}/tools/setup/obgpu_sol_module_utils.sh"
  if [[ -f "\${_obgpu_sol_helper}" ]]; then
    # shellcheck disable=SC1090
    source "\${_obgpu_sol_helper}"
  fi

  if declare -F obgpu_sol_ensure_module_cmd >/dev/null 2>&1 && obgpu_sol_ensure_module_cmd; then
    _obgpu_sol_mamba_module="\${OBGPU_SOL_MAMBA_MODULE:-\${SOL_MAMBA_MODULE:-}}"
    _obgpu_sol_nvhpc_module="\${OBGPU_SOL_NVHPC_MODULE:-\${SOL_NVHPC_MODULE:-}}"
    _obgpu_sol_cuda_module="\${OBGPU_SOL_CUDA_MODULE:-\${SOL_CUDA_MODULE:-}}"

    if declare -F obgpu_sol_resolve_module >/dev/null 2>&1; then
      _obgpu_sol_mamba_module="\$(obgpu_sol_resolve_module mamba "\${_obgpu_sol_mamba_module}")"
      _obgpu_sol_nvhpc_module="\$(obgpu_sol_resolve_module nvhpc "\${_obgpu_sol_nvhpc_module}")"
      _obgpu_sol_cuda_module="\$(obgpu_sol_resolve_module cuda "\${_obgpu_sol_cuda_module}")"
    fi

    if declare -F obgpu_sol_maybe_load_module >/dev/null 2>&1; then
      obgpu_sol_maybe_load_module "\${_obgpu_sol_mamba_module}"
      obgpu_sol_maybe_load_module "\${_obgpu_sol_nvhpc_module}"
      obgpu_sol_maybe_load_module "\${_obgpu_sol_cuda_module}"
    else
      module load "\${_obgpu_sol_mamba_module}"
      module load "\${_obgpu_sol_nvhpc_module}"
      module load "\${_obgpu_sol_cuda_module}"
    fi
  fi

  unset _obgpu_sol_helper
  unset _obgpu_sol_mamba_module
  unset _obgpu_sol_nvhpc_module
  unset _obgpu_sol_cuda_module
fi

export OMPI_MCA_opal_cuda_support=true
export NMODLHOME=${CONDA_PREFIX}
export NMODL_PYLIB=${PYTHON_SHARED_LIB}
if [[ -n "\${NRN_NMODL_PATH+x}" ]]; then
  export _OBGPU_OLD_NRN_NMODL_PATH="\${NRN_NMODL_PATH}"
fi
export NRN_NMODL_PATH=${REPO_ROOT}
if [[ -n "\${CORENEURONLIB+x}" ]]; then
  export _OBGPU_OLD_CORENEURONLIB="\${CORENEURONLIB}"
fi
export CORENEURONLIB=${REPO_ROOT}/$(uname -m)/libcorenrnmech.so
if [[ -d "${REPO_ROOT}/$(uname -m)" ]]; then
  if [[ -n "\${LD_LIBRARY_PATH:-}" ]]; then
    export _OBGPU_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${REPO_ROOT}/$(uname -m):\${LD_LIBRARY_PATH}"
  else
    unset _OBGPU_OLD_LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${REPO_ROOT}/$(uname -m)"
  fi
fi
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/ob_modern_neuron.sh" <<'EOF'
unset OMPI_MCA_opal_cuda_support
unset NMODLHOME
unset NMODL_PYLIB
if [[ -n "${_OBGPU_OLD_NRN_NMODL_PATH+x}" ]]; then
  export NRN_NMODL_PATH="${_OBGPU_OLD_NRN_NMODL_PATH}"
  unset _OBGPU_OLD_NRN_NMODL_PATH
else
  unset NRN_NMODL_PATH
fi
if [[ -n "${_OBGPU_OLD_CORENEURONLIB+x}" ]]; then
  export CORENEURONLIB="${_OBGPU_OLD_CORENEURONLIB}"
  unset _OBGPU_OLD_CORENEURONLIB
else
  unset CORENEURONLIB
fi
if [[ -n "${_OBGPU_OLD_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${_OBGPU_OLD_LD_LIBRARY_PATH}"
  unset _OBGPU_OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOF

# Make the current setup shell consistent with the activate hook we just wrote so follow-on
# nrnivmodl invocations in this same process do not depend on a manual conda reactivate.
export OMPI_MCA_opal_cuda_support=true
export NMODLHOME="${CONDA_PREFIX}"
export NMODL_PYLIB="${PYTHON_SHARED_LIB}"
export NRN_NMODL_PATH="${REPO_ROOT}"
export CORENEURONLIB="${REPO_ROOT}/$(uname -m)/libcorenrnmech.so"
if [[ -d "${REPO_ROOT}/$(uname -m)" ]]; then
  export LD_LIBRARY_PATH="${REPO_ROOT}/$(uname -m)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

(
  log_step "Checking Birgiolas mechanism build stamp"
  MECH_BUILD_STAMP_PATH="${REPO_ROOT}/$(uname -m)/.obgpu_mechanisms_stamp"
  MECH_BUILD_FINGERPRINT="$(mechanism_stamp_fingerprint "${NRN_BUILD_FINGERPRINT}")"

  if mechanism_lib_path="$(find_current_arch_libnrnmech 2>/dev/null)"; then
    mechanism_lib_present=1
  else
    mechanism_lib_path=""
    mechanism_lib_present=0
  fi

  if [[ "${mechanism_lib_present}" == "1" ]] && stamp_matches "${MECH_BUILD_STAMP_PATH}" "${MECH_BUILD_FINGERPRINT}"; then
    log_step "Skipping mechanism rebuild; matching successful stamp found at ${MECH_BUILD_STAMP_PATH}"
  else
    log_step "Building Birgiolas mechanisms with nrnivmodl -coreneuron"
    cd "${REPO_ROOT}"
    OMPI_CC=gcc OMPI_CXX=g++ nrnivmodl -coreneuron prev_ob_models/Birgiolas2020/Mechanisms
    mechanism_lib_path="$(find_current_arch_libnrnmech)"
    mkdir -p "$(dirname "${MECH_BUILD_STAMP_PATH}")"
    printf '%s\n' "${MECH_BUILD_FINGERPRINT}" > "${MECH_BUILD_STAMP_PATH}"
  fi

  if [[ "${ENABLE_GPU}" == "1" && -n "${mechanism_lib_path}" ]]; then
    log_step "Repairing NVHPC libnrnmech dependencies for ${mechanism_lib_path}"
    "${REPO_ROOT}/tools/setup/fix_nvhpc_libnrnmech.sh" "${mechanism_lib_path}"
  fi
)

if [[ "${ENABLE_GPU}" == "1" ]] && ! find_current_arch_libnrnmech >/dev/null 2>&1; then
  echo "Warning: could not locate a generated libnrnmech.so to repair." >&2
fi

log_step "OBGPU setup complete from ${UPSTREAM_REF}"
