#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f build/CMakeCache.txt ]]; then
  if [[ -z "${OPTIX_ROOT:-}" ]]; then
    echo "run.sh: OPTIX_ROOT is not set. See humans.md for setup." >&2
    exit 1
  fi
  cmake -B build -G Ninja
fi

cmake --build build

if lspci | grep -qi 'vga.*nvidia' && lspci | grep -Eqi 'vga.*(intel|amd|ati)'; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

exec ./build/italy-rs "$@"
