#!/usr/bin/env bash
# Bootstraps build prerequisites for italy-rs on Debian/Ubuntu and runs the
# first build. Mirrors humans.md's "Prerequisites" and "Build" sections —
# keep the two in sync if either changes.
#
# What this does NOT do: download the OptiX SDK. NVIDIA gates that installer
# behind a free Developer Program login, so it can't be scripted — this
# script checks for it and prints the manual steps if it's missing, same as
# humans.md.
set -euo pipefail

CUDA_PACKAGE="cuda-toolkit-13-2"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
APT_BUILD_DEPS=(build-essential cmake ninja-build git pkg-config xorg-dev libgl1-mesa-dev g++-13)

log() { echo "==> $*"; }
die() { echo "error: $*" >&2; exit 1; }
# CMake's find_package(CUDAToolkit) locates nvcc via the standard
# /usr/local/cuda* prefixes on its own, independent of PATH — so check
# those too rather than requiring nvcc on this shell's PATH specifically.
find_nvcc() { command -v nvcc 2>/dev/null || compgen -G '/usr/local/cuda*/bin/nvcc' | head -1; }

command -v apt-get >/dev/null 2>&1 ||
  die "this script only supports Debian/Ubuntu (apt-get not found) — see humans.md for manual steps on other distros."

# --- NVIDIA driver -----------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 ||
  die "nvidia-smi not found — install the NVIDIA driver first (this project is NVIDIA-only: CUDA + OptiX, no AMD/Vulkan)."
nvidia-smi -L | grep -q . || die "nvidia-smi found no GPU."
log "NVIDIA driver OK: $(nvidia-smi -L | head -1)"

# --- Build tools + GLFW's Linux (X11) build deps ------------------------
missing=()
for pkg in "${APT_BUILD_DEPS[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ "${#missing[@]}" -gt 0 ]; then
  log "installing: ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi

# --- CUDA Toolkit --------------------------------------------------------
nvcc_bin="$(find_nvcc)"
if [ -z "$nvcc_bin" ]; then
  log "nvcc not found — installing ${CUDA_PACKAGE}"
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    tmp="$(mktemp -d)"
    curl -sSLo "$tmp/cuda-keyring.deb" "$CUDA_KEYRING_URL"
    sudo dpkg -i "$tmp/cuda-keyring.deb"
    rm -rf "$tmp"
  fi
  sudo apt-get update
  sudo apt-get install -y "$CUDA_PACKAGE"
  nvcc_bin="$(find_nvcc)"
fi
[ -n "$nvcc_bin" ] || die "nvcc still not found after installing ${CUDA_PACKAGE} — check /usr/local/cuda*/bin exists."
log "CUDA Toolkit OK: $("$nvcc_bin" --version | tail -1)"

# --- OptiX SDK: cannot be scripted, only checked -------------------------
if [ -z "${OPTIX_ROOT:-}" ] || [ ! -f "${OPTIX_ROOT}/include/optix.h" ] || [ ! -f "${OPTIX_ROOT}/SDK/sutil/vec_math.h" ]; then
  cat >&2 <<'EOF'
error: OPTIX_ROOT is not set (or doesn't point at a full SDK install).

NVIDIA gates the OptiX installer behind a free Developer Program login, so
this step is manual:
  1. Log in at https://developer.nvidia.com/designworks/optix/download
  2. Download the Linux .sh installer.
  3. chmod +x NVIDIA-OptiX-SDK-*-linux64-x86_64.sh
     sudo mkdir -p /opt/optix
     sudo chown "$(id -u):$(id -g)" /opt/optix
     ./NVIDIA-OptiX-SDK-*.sh --skip-license --prefix=/opt/optix
  4. export OPTIX_ROOT=/opt/optix   (add this to your shell rc file)
Then re-run this script.
EOF
  exit 1
fi
log "OptiX SDK OK: ${OPTIX_ROOT}"

# --- Build -----------------------------------------------------------
# claudia: no automatic CUDA-host-compiler fallback — humans.md notes
# g++-13 as a pinned fallback in case a newer default gcc breaks nvcc; if
# the configure/build below fails on a host-compiler compat error, retry
# with: cmake -B build -G Ninja -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
log "configuring"
cmake -B build -G Ninja
log "building"
cmake --build build

log "done — run ./build/italy-rs (or ./build/italy-rs assets/test.glb)"
