# Italy R/S

A native volumetric pathtracing editor/renderer: emissive/reflective/refractive
materials, HDRI lighting, SPPM caustics, AgX tonemapping. MVP ingests a
textured GLB and resamples it into a voxel field and/or signed distance
field for orbit-camera pathtraced preview.

Full design/phasing: see the plan this repo was scaffolded from —
`/home/joe/.claude/plans/lets-make-a-new-immutable-marble.md`.

## Quick start

`scripts/setup-linux.sh` (Debian/Ubuntu) and `scripts/setup-windows.ps1`
automate everything below except the OptiX SDK download, which NVIDIA gates
behind a login and can't be scripted — both scripts check for it and print
the manual steps if it's missing. The Windows script is untested on real
hardware so far (this project has only been built on Linux); the manual
steps below are the fallback if it hits something machine-specific.

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
./build/italy-rs
```

GLFW, Dear ImGui, glm, stb, and tinygltf are pulled via CMake `FetchContent`
at configure time — no system packages needed for those.

To render a GLB instead of the built-in test scene:

```
./build/italy-rs path/to/asset.glb
```

A small bundled test asset lives at `assets/test.glb`. Single mesh/primitive
only for now (no scene graph, no node transforms) — good enough for "one
textured object," which is the phase-3 ask.

To voxelize that mesh instead of rendering it as textured triangles:

```
./build/italy-rs path/to/asset.glb --voxel=64
```

`64` is cells along the mesh's longest bounding-box axis (default if
`--voxel` is given with no `=N`). Voxel color is baked from the source
texture/material at conversion time, not sampled live.

To bake a signed distance field instead:

```
./build/italy-rs path/to/asset.glb --sdf=56
```

`56` is cells along the longest axis, same convention as `--voxel`. If both
flags are given, `--sdf` wins. The source mesh should be closed/watertight —
sign is determined by ray-parity counting, which isn't reliable on an open
mesh. See `src/convert/sdf_baker.h` for how baking works and its known
approximation (no true closest-point query exists in OptiX, so distance is
estimated via minimum hit distance over many random directions).

To light the scene with an HDRI instead of the built-in quad light (applies
to any of the above, including the built-in test scene — a good way to see
glass/mirror materials against real environment lighting):

```
./build/italy-rs path/to/asset.glb --env=overcast   # or midnight, noon
./build/italy-rs path/to/asset.glb --hdri=path/to/custom.hdr
```

All of the above (GLB path, representation, HDRI preset) are also live
ImGui controls once the window is open — the CLI flags just seed the same
state and are mainly useful for scripted verification now. The window also
has live sliders/toggles that don't need a rebuild: exposure,
samples-per-launch, the OptiX AI denoiser, and the tonemap operator (AgX
default; Reinhard/ACES/Hable/Clamp are debugging aids for comparison).

Scripted testing hooks for the two toggles that are otherwise UI-only:
`ITALY_FORCE_DENOISE=1` and `ITALY_TONEMAP=<agx|reinhard|aces|hable|clamp>`.

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
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/italy-rs
```

Not needed on a desktop/single-GPU NVIDIA machine.

### Debugging without eyeballing the live window

Set `ITALY_DUMP_FRAME=/path/to/out.png` to have italy write the accumulated
render to a PNG after 128 subframes and exit — useful for checking render
correctness headlessly/from a script rather than watching the window.

## Status

**Bug-fix pass after phase 4** caught two rendering-correctness bugs that had
been silently present since phase 2 (affecting every material/phase built on
top): NEE contributions were double-multiplying the surface's own albedo
(the raygen loop already applies it via the updated `attenuation`, so baking
it into `radiance` too darkened every diffuse/textured/voxel direct-lighting
sample), and light seen indirectly (via a mirror/glass bounce, or a
diffuse BSDF-sampled ray landing on the light) was added at full brightness
with no attenuation from the bounces that led to it. Also fixed: glTF
`baseColorTexture` is sRGB-encoded and was being read as if already linear,
systematically washing out colors — confirmed by re-rendering the same
device-bottom.glb before/after (vivid, correctly saturated red body and
distinct button colors after the fix, versus a washed-out pale render
before). All three fixes are in `pathtracer.cu`/`optix_renderer.cpp`/
`voxelize.cpp`; re-verified visually across the fixed test scene, the
textured mesh, and the voxelized mesh.

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

Phase 5 done: mesh -> dense signed distance field, baked by a second,
self-contained OptiX pipeline (`src/convert/sdf_baker.cpp` + `sdf_bake.cu`)
that reuses the mesh's own BVH: unsigned distance is the minimum hit
distance over many independent random directions per cell (OptiX has no
native closest-point query), sign is a ray-parity majority vote across
several directions. Rendered as a single `MATERIAL_SDF` custom-AABB
primitive (the whole grid's bbox) that `__intersection__sdf` sphere-traces
through, sampling the field with manual trilinear interpolation; shading
normal is the field's central-difference gradient.

Getting a clean render took a real debugging pass — worth recording since
the symptom was misleading. The first attempt showed structured diagonal
banding on the small test asset and a fully black front/side on the
device-bottom.glb asset. Three plausible-looking causes were tried and each
had *no effect whatsoever* on the artifact: switching the distance estimator
from a fixed Fibonacci-sphere direction set to per-cell-jittered to fully
independent random directions; adding linear-interpolation refinement for
sphere-tracing overshoot (this did fix a real bug in itself, tracked
separately below); and switching sign determination from a single fixed ray
to a 5-direction majority vote. The actual cause was shadow acne: sphere
tracing only locates the surface to within an epsilon tolerance (unlike
triangle/voxel geometry, which is exact), and the shared NEE/bounce-ray
`1e-3f` origin offset wasn't reliably larger than that tolerance, so
shadow/bounce rays were self-intersecting the surface they'd just come from.
Nudging the shading point outward along its normal by a fraction of a voxel
before tracing any secondary ray (`pathtracer.cu`'s `MATERIAL_SDF` branch)
fixed both symptoms completely on both assets. The refinement fix found
along the way was real too: the overshoot-correction formula was being
applied even on ordinary (non-overshoot) termination, where it *extrapolated
past* the current sample instead of using it — fixed by only interpolating
when the sampled distance actually goes negative.

Phase 6 done: HDRI/environment lighting (`src/render/environment.{h,cpp}`)
— `.hdr` loading via `stb_image`, importance-sampled via a PBRT-style
piecewise-constant 2D distribution (marginal CDF over rows, conditional CDF
per row), wired into both the miss shader and NEE. An environment
supersedes the quad light entirely when loaded, never blended with it.
Three real CC0 Poly Haven HDRIs (1k res) bundled under `assets/hdri/` as
the overcast/midnight/noon presets. Verified by rendering the fixed test
scene's mirror/glass materials under all three and confirming physically
distinct, plausible lighting — including a sharp sun disk correctly
captured in the noon preset's mirror reflection (proof the importance
sampling actually finds small bright features rather than losing them to
noise) — plus a pixel-identical regression check with no environment.

Phase 7 done: caustics via a global-radius progressive photon map (the
original Hachisuka/Ogaki/Jensen 2008 PPM formulation — one shared radius
shrunk each pass via `R_{i+1} = R_i * sqrt((i+alpha)/(i+1))`, not the later
per-visible-point Stochastic PPM refinement; a deliberate scope reduction,
see `pathtracer_params.h`). Photons are emitted from the quad light, bounce
through specular surfaces, and deposit only once they've had at least one
specular bounce (direct light is already handled by NEE — storing/gathering
non-caustic photons would just double-count it). Gather is a "long ray
through a sphere, any-hit accumulates" range query against an OptiX BVH
built over the deposited photons each pass — reusing ray-tracing hardware
for a non-primary-ray query, the same trick SDF baking uses. Verified: a
clear, progressively-sharpening caustic focus spot under the glass sphere.

Phase 9 done: live ImGui controls (GLB load, mesh/voxel/SDF representation
with a resolution slider, HDRI preset, exposure, samples-per-launch)
replacing the CLI-only workflow. CLI args now just seed the same
`AppState` the UI edits; both funnel through one `rebuildScene()` so they
can't drift apart. Representation/HDRI changes are one explicit "Apply"
rebuild rather than incremental scene patching (`OptixRenderer` has no
partial-rebuild API, and voxelizing/SDF-baking isn't cheap enough to redo
per slider-tick anyway).

Phase 10 done: the OptiX AI denoiser (color-only for now, no albedo/normal
guide layers — a real quality upgrade left as a natural follow-up). The
main launch writes the HDR accumulator only; a separate `optixDenoiserInvoke`
call denoises it into its own buffer (denoising in place would corrupt the
running progressive average used by future frames); a third raygen-only
launch (`__raygen__tonemap`, its own minimal SBT — reuses the existing
pipeline instead of a new CUDA compilation unit) reads the denoised result
into the display buffer. Verified with a forced on/off comparison at 4
samples/pixel: heavy noise without it, clean-but-still-detailed with it.

Phase 8 done: selectable tonemap operator (AgX default; Reinhard/ACES/
Hable/Clamp as alternates), replacing the bare `clamp(0,1)` + sRGB encode
every render used through phase 7. AgX is the widely-circulated community
GLSL approximation of Blender's AgX view transform (inset matrix -> log2
remap -> 6th-order polynomial contrast fit -> outset matrix), the same
approach Godot 4.3+/Bevy ship instead of pulling in OpenColorIO for one
transform — reproduced from memory of that fit, not diffed against
Blender's reference OCIO config byte-for-byte (no Blender install available
here), so verification is behavioral rather than pixel-exact: rendered the
fixed test scene under all 5 operators and confirmed AgX shows its
well-known signature look (lifted shadows instead of crushed blacks,
desaturated reds, soft highlight rolloff instead of hard clipping),
distinct from the other four, which are also each visibly distinct from
one another.

Along the way, a full clean rebuild (after this repo's directory was
renamed `italy` -> `italy-rs`) surfaced a real off-by-one in the voxelizer
that incremental builds had apparently been masking with a stale test
binary: a triangle vertex sitting exactly on the mesh's far bounding-box
edge computed a cell index one past the last valid one, silently adding a
phantom extra layer. Fixed by clamping cell indices to `[0, resolution-1]`;
the test's own assertion threshold turned out to be separately wrong too
(mathematically unsatisfiable for a correct shell at low resolution) and
was replaced with a resolution-derived bound.

All 10 planned MVP phases are now done. See the plan doc for the full
phase list and out-of-scope items (Gaussian-splat resampling, full VCM/
ReSTIR caustics, Metal backend — all explicitly deferred, not forgotten).
