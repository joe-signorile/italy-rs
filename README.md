# italy

A native volumetric pathtracing editor/renderer: emissive/reflective/refractive
materials, HDRI lighting, SPPM caustics, AgX tonemapping. MVP ingests a
textured GLB and resamples it into a voxel field and/or signed distance
field for orbit-camera pathtraced preview.

Full design/phasing: see the plan this repo was scaffolded from —
`/home/joe/.claude/plans/lets-make-a-new-immutable-marble.md`.

## Prerequisites

- NVIDIA GPU, Ada generation or newer recommended (RT cores required for the
  OptiX path).
- CUDA Toolkit 13.2 (`sudo apt-get install cuda-toolkit-13-2`, via NVIDIA's
  apt repo — see below).
- **OptiX SDK — manual step, cannot be scripted.** NVIDIA gates the
  installer behind a free Developer Program login:
  1. Log in at https://developer.nvidia.com/designworks/optix/download
  2. Download the Linux `.sh` installer.
  3. `chmod +x NVIDIA-OptiX-SDK-*-linux64-x86_64.sh && sudo ./NVIDIA-OptiX-SDK-*.sh --skip-license --prefix=/opt/optix`
  4. `export OPTIX_ROOT=/opt/optix` (CMake looks for this env var).
- CMake >= 3.24, Ninja, a C++20 compiler.
- Confirmed on this host: `nvcc` (CUDA 13.2) accepts the system default
  gcc/g++ 15.2 as host compiler with no flags needed — the "nvcc lags gcc"
  compatibility issue that's common with older CUDA releases didn't
  materialize here. `gcc-13`/`g++-13` are still installed as a fallback
  (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`) in case a newer gcc breaks
  this later.

To register NVIDIA's apt repo (this host is Ubuntu 26.04; NVIDIA hasn't
published a matching keyring yet, so the nearest published release works):

```
curl -sO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

## Build

```
cmake -B build -G Ninja
cmake --build build
./build/italy
```

GLFW, Dear ImGui, and glm are pulled via CMake `FetchContent` at configure
time — no system packages needed for those. `tinygltf`/`stb` are added the
same way once GLB loading lands (phase 3).

## Status

Phase 1 (window + ImGui shell + orbit camera, no rendering yet). See the
plan doc for the full phase list.
