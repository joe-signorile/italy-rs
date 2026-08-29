#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Shares this machine's one GPU with s-rank's SUPIR/TripoSplat services;
# running both at once can OOM either side (see CLAUDE.md).

if [[ ! -f build/CMakeCache.txt ]]; then
  if [[ -z "${OPTIX_ROOT:-}" ]]; then
    echo "run.sh: OPTIX_ROOT is not set. See humans.md for setup." >&2
    exit 1
  fi
  cmake -B build -G Ninja
fi

cmake --build build

# On a hybrid-GPU laptop (Intel/AMD iGPU + NVIDIA dGPU), GLX defaults to the
# iGPU, which breaks CUDA-GL interop (see humans.md). Detect that case by
# checking for a non-NVIDIA VGA controller alongside the NVIDIA one, rather
# than hardcoding the offload vars for every machine this script runs on.
if lspci | grep -qi 'vga.*nvidia' && lspci | grep -Eqi 'vga.*(intel|amd|ati)'; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

exec ./build/italy-rs "$@"
