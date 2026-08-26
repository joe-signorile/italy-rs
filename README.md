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
  3. The installer expects its target directory to already exist:
     `chmod +x NVIDIA-OptiX-SDK-*-linux64-x86_64.sh && sudo mkdir -p /opt/optix && sudo chown "$(id -u):$(id -g)" /opt/optix && ./NVIDIA-OptiX-SDK-*.sh --skip-license --prefix=/opt/optix`
  4. `export OPTIX_ROOT=/opt/optix` (CMake looks for this env var). The build
     also needs `${OPTIX_ROOT}/SDK/sutil` (vec_math.h, random.h, helpers.h) —
     that's part of the same installer, not a separate download.
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

GLFW, Dear ImGui, glm, stb, and tinygltf are pulled via CMake `FetchContent`
at configure time — no system packages needed for those.

To render a GLB instead of the built-in test scene:

```
./build/italy path/to/asset.glb
```

A small bundled test asset lives at `assets/test.glb`. Single mesh/primitive
only for now (no scene graph, no node transforms) — good enough for "one
textured object," which is the phase-3 ask.

To voxelize that mesh instead of rendering it as textured triangles:

```
./build/italy path/to/asset.glb --voxel=64
```

`64` is cells along the mesh's longest bounding-box axis (default if
`--voxel` is given with no `=N`). Voxel color is baked from the source
texture/material at conversion time, not sampled live.

### Tests

```
ctest --test-dir build
```

Currently just the voxelizer's triangle-box SAT test (`tests/voxelize_test.cpp`)
— the rendering core itself is verified visually (see "Debugging without
eyeballing the live window" below), per the plan doc's reasoning.

### Hybrid-GPU laptops (Intel iGPU + NVIDIA dGPU)

On this dev machine (Intel Iris Xe + RTX 4080 Laptop), GLX defaults to the
Intel iGPU, and CUDA-GL interop (`cudaGraphicsGLRegisterBuffer`) fails with
"invalid OpenGL or DirectX context" unless the GL context is actually backed
by the NVIDIA driver. Force it with PRIME render offload:

```
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/italy
```

Not needed on a desktop/single-GPU NVIDIA machine.

### Debugging without eyeballing the live window

Set `ITALY_DUMP_FRAME=/path/to/out.png` to have italy write the accumulated
render to a PNG after 128 subframes and exit — useful for checking render
correctness headlessly/from a script rather than watching the window.

## Status

Phase 2 done: OptiX 9.1 path tracer (NEE + power-heuristic MIS + Russian
roulette, iterative not recursive) rendering a hardcoded scene — diffuse,
mirror, and dielectric-glass spheres plus a quad area light, triangle ground
plane, mixed triangle/built-in-sphere geometry in one IAS — displayed live
in the ImGui viewport via CUDA-GL PBO interop, progressive accumulation that
resets on camera move. Verified by rendering to a PNG and inspecting it:
correct shadows, mirror reflections, glass refraction, and an emergent
caustic-bright patch on the ground under the glass sphere.

Phase 3 done: GLB loading (tinygltf) — positions/normals/UVs/base-color
texture, camera auto-framed to the loaded mesh's bounding sphere, rendered as
`MATERIAL_TEXTURED_DIFFUSE` triangles lit by a synthetic quad light sized to
the mesh (no HDRI yet). Verified against two real assets: a small untextured
primitive (correct smooth-normal shading, falls back to the material's flat
`baseColorFactor` when there's no texture) and a 272k-triangle textured
device model (correct UV-mapped texture — grille holes, buttons, panel seams
all land in the right places).

Phase 4 done: mesh -> sparse voxel grid (exact triangle/AABB SAT test, not
just bbox overlap — see `src/convert/voxelize.cpp`), rendered as
`MATERIAL_VOXEL` custom-AABB primitives with OptiX's hardware BVH doing the
sparse traversal (no hand-rolled DDA raymarch). Per-voxel color is baked from
the source texture/`baseColorFactor` at conversion time; shading normal comes
from which of the 6 box faces the intersection program's ray/slab test
entered. Verified three ways: a unit-test on the SAT logic itself (shell
occupied, interior empty), a visual check on the small primitive (correct
blocky silhouette matching the smooth original), and a visual check on the
272k-triangle textured device model (buttons/grille/seams still legible after
voxelization, with correctly baked colors).

Not yet done: SDF resampling, HDRI lighting, SPPM caustics, AgX tonemapping,
UI controls, denoiser. See the plan doc for the full phase list.
