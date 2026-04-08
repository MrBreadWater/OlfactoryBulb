#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-21.7}"
VERSION_MAJOR="${VERSION%%.*}"
VERSION_COMPACT="${VERSION//./}"
VERSION_TOKEN="20${VERSION_MAJOR}_${VERSION_COMPACT}"

detect_system_cuda_version() {
  local version=""
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    version="$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  fi
  if [[ -z "${version}" && -f /etc/nv_tegra_release ]]; then
    case "$(cat /etc/nv_tegra_release 2>/dev/null)" in
      *"R35 ("*) version="11.4" ;;
    esac
  fi
  printf '%s\n' "${version}"
}

SYSTEM_CUDA_VERSION="$(detect_system_cuda_version)"

CUDA_BUNDLE="${CUDA_BUNDLE:-}"
if [[ -z "${CUDA_BUNDLE}" ]]; then
  if [[ "${VERSION}" == "21.7" ]]; then
    CUDA_BUNDLE="11.4"
  else
    CUDA_BUNDLE="multi"
  fi
fi

CUDA_VERSION="${CUDA_VERSION:-}"
if [[ -z "${CUDA_VERSION}" ]]; then
  if [[ "${CUDA_BUNDLE}" == "multi" ]]; then
    case "${SYSTEM_CUDA_VERSION}" in
      11.8|12.*)
        CUDA_VERSION="${SYSTEM_CUDA_VERSION}"
        ;;
      *)
        # Newer Arm "multi" SDK bundles do not ship CUDA 11.4 even on JetPack 5 hosts.
        CUDA_VERSION="11.8"
        ;;
    esac
  else
    CUDA_VERSION="${CUDA_BUNDLE}"
  fi
fi

ARCHIVE_BASENAME="nvhpc_${VERSION_TOKEN}_Linux_aarch64_cuda_${CUDA_BUNDLE}"
DOWNLOAD_URL="https://developer.download.nvidia.com/hpc-sdk/${VERSION}/${ARCHIVE_BASENAME}.tar.gz"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/nvhpc-downloads}"
EXTRACT_ROOT="${EXTRACT_ROOT:-${CACHE_DIR}}"
INSTALL_ROOT="${INSTALL_ROOT:-${HOME}/.local/nvidia/hpc_sdk}"
EXTRACT_DIR="${EXTRACT_ROOT}/${ARCHIVE_BASENAME}"
ARCHIVE_PATH="${CACHE_DIR}/${ARCHIVE_BASENAME}.tar.gz"

mkdir -p "${CACHE_DIR}" "${EXTRACT_ROOT}" "${INSTALL_ROOT}"

if [[ -f "${ARCHIVE_PATH}" ]] && ! gzip -t "${ARCHIVE_PATH}" 2>/dev/null; then
  echo "Archive is corrupt or incomplete, removing: ${ARCHIVE_PATH}" >&2
  rm -f "${ARCHIVE_PATH}"
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  wget -c "${DOWNLOAD_URL}" -O "${ARCHIVE_PATH}"
fi

if [[ -d "${EXTRACT_DIR}" && ! -x "${EXTRACT_DIR}/install" ]]; then
  echo "Removing partial extract directory: ${EXTRACT_DIR}" >&2
  rm -rf "${EXTRACT_DIR}"
fi

if [[ ! -d "${EXTRACT_DIR}" ]]; then
  tar xpfz "${ARCHIVE_PATH}" -C "${EXTRACT_ROOT}"
fi

NVHPC_SILENT=true \
NVHPC_INSTALL_DIR="${INSTALL_ROOT}" \
NVHPC_INSTALL_TYPE=single \
NVHPC_DEFAULT_CUDA="${CUDA_VERSION}" \
  "${EXTRACT_DIR}/install"

NVARCH="$(uname -s)_$(uname -m)"
COMPILER_BIN="${INSTALL_ROOT}/${NVARCH}/${VERSION}/compilers/bin"

if [[ ! -x "${COMPILER_BIN}/nvc" || ! -x "${COMPILER_BIN}/nvc++" ]]; then
  echo "NVHPC install did not produce expected compiler binaries under ${COMPILER_BIN}" >&2
  exit 1
fi

echo "NVHPC installed."
echo "Root: ${INSTALL_ROOT}"
echo "Compilers: ${COMPILER_BIN}"
echo "Default CUDA: ${CUDA_VERSION}"
