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

Scripted testing hooks for the toggles that are otherwise UI-only:
`ITALY_FORCE_DENOISE=1`, `ITALY_TONEMAP=<agx|reinhard|aces|hable|clamp>`, and
`ITALY_FIREFLY_CLAMP=<value>` (0 disables). `ITALY_DUMP_AFTER_SUBFRAME=<n>`
sets how many subframes accumulate before `ITALY_DUMP_FRAME` writes.

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

**Correctness + doctrine pass after phase 10.** Wrote the project's ranked
priority order (beauty > performance > realism) into `CLAUDE.md` as a binding
ladder rather than leaving it implicit, and added the `// realism:` marker
convention for deliberate departures from physics. Then read all of `src/`
and fixed ten defects, listed worst-first:

- **Glass was never solid.** OptiX's built-in sphere primitive is documented
  hollow and back-face culled (Programming Guide 9.1, 9.6): a ray refracted
  into one never receives an exit intersection. So `MATERIAL_GLASS` was a
  single refracting interface with no interior, no total internal reflection,
  and nothing for absorption to act over. Dielectrics now use a custom
  `__intersection__sphere_solid` that reports the far root when the near one
  is behind the ray — the same route NVIDIA takes in `optixWhitted`. Beer-
  Lambert absorption came along with it, since the interior segment length is
  finally knowable. The glass sphere now shows its TIR horizon band and casts
  a coloured caustic.
- **AgX was sRGB-encoded twice** — see the phase-8 note below.
- **Every photon pass emitted identical photons.** `params.totalPhotonsEmitted`
  (the emission RNG seed) was uploaded to the device *after* the launch that
  reads it, so it was always 0. Deposited counts per pass were literally
  identical (2931, 2931, 2931); they now vary (2931, 3085, 2962).
- **Caustics were ~65,536x too dim**, from dividing by the photon batch size
  in both the emission and the gather. That is what the 200x
  `ITALY_DEBUG_CAUSTICS_ONLY` boost had been compensating for; it is gone.
- **Loading a GLB appended to the previously loaded one.** `loadGlb()` never
  reset its output, and every UI radio button triggers a reload, so the
  triangle count grew with each Apply while the bounds tracked only the newest
  load. Three consecutive loads gave 6, 12, 18 triangles; now 6, 6, 6.
- **Meshes without a NORMAL attribute shaded as if they all faced the sky.**
  The fallback was a constant up-vector rather than the triangle's own
  geometric normal — which `assets/test.glb` triggers, so the bundled demo
  asset was rendering with no form definition at all.
- **Voxel colours baked from a vertically flipped texture**, disagreeing with
  the GPU mesh path. Verified with a purpose-built asset whose texture is
  red-over-blue: mesh and voxel renders now agree.
- **`optixGetTriangleVertexData`/`optixGetSphereData` were called on
  acceleration structures built without `ALLOW_RANDOM_VERTEX_ACCESS`**, which
  the API requires — undefined by contract, working by luck.
- **No NaN guard and no firefly clamp**, so a single bad sample poisoned its
  pixel permanently. Both added; the clamp removes ~78% of isolated bright
  outliers at the default test scene.
- **`CLAUDE.md` mandated a `src/rhi/` seam that has never existed** (and
  `sdf_baker.cpp` bypassed it anyway). Docs now describe the actual seam,
  `src/render/optix_renderer.h`, and name the two sanctioned OptiX call sites.

Two pieces of housekeeping fell out: the `@monkey-boy/CLAUDE.md.snippet`
import at the top of `CLAUDE.md` pointed at a project that no longer exists,
and all seven deliberate-simplification markers still used the old
`monkey-boy:` prefix, so the tooling that greps for `claudia:` found none of
them.

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
here).

**This phase shipped with a bug, and the phase's own verification is what
missed it — worth recording as a cautionary tale.** The chain was missing its
closing `pow(2.2)`. That step exists because the contrast polynomial's output
is display-encoded, and every operator here shares one final sRGB encode in
`sutil::make_color()`; without it, the sRGB OETF ran on already-encoded
values. The verification at the time was "does AgX show its well-known
signature look — lifted shadows instead of crushed blacks, soft highlight
rolloff?" It did, and that was accepted. But *a double sRGB encode produces
exactly those symptoms too*, which is why a behavioural check against a
remembered description could not separate the two. The tell was available and
unused: the fixed test scene's background is a constant, `bgColor` = linear
(0.05, 0.06, 0.08), whose correct 8-bit sRGB value is computable in advance
as 62 — it was rendering as 139. Fixed and re-verified numerically at 63.
The lesson generalised into `CLAUDE.md`'s doctrine: prefer a check with a
predictable numeric answer over a check against a remembered look.

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

**Beauty features added alongside the correctness pass**, once the doctrine
ladder made "which of these is worth doing" an answerable question:

- **GGX metallic-roughness materials.** Every glTF-loaded surface was pure
  Lambert regardless of what the source material said — the single biggest
  gap between loaded output and something that reads as a real render. The
  loader now walks the full scene graph (previously only
  `meshes[0].primitives[0]`, so a 169-mesh asset like the bundled
  `assets/test.glb` rendered as one), collects every material/texture the
  scene actually references (de-duplicated by image, not by glTF texture
  index — several materials commonly share one multi-megabyte JPEG), and
  reads `metallicFactor`/`roughnessFactor`/`metallicRoughnessTexture`/
  `normalTexture`. The path tracer gained a proper two-lobe BSDF (Lambert +
  GGX with Heitz 2018 visible-normal sampling) with the combined pdf feeding
  every MIS weight, not just the new one — every existing `powerHeuristic`
  call site had to be re-derived, not just the new lobe's. Verified with a
  purpose-built furnace-test asset (5 metallic spheres, roughness 0.05 to
  1.0, under a uniform-ish HDRI): the specular highlight tightens
  monotonically from mirror-sharp to a soft blur, with no energy blowup or
  collapse across the sweep.
- **Solid dielectrics.** Written up above under "Glass was never solid" —
  the custom sphere intersector this needed is what made Beer-Lambert
  absorption possible, since the interior segment length only exists once
  there's a real interior.
- **Camera and sampling controls**, all artist-facing per the doctrine's own
  "artist controls are features" corollary: thin-lens depth of field
  (aperture + focus distance, defaulting to the orbit target so it's correct
  untouched), a tent reconstruction filter replacing the old box-filter
  jitter (less aliasing at the same sample count), and environment rotation
  — one shared `dirToEquirectUv`/`equirectUvToDir` pair feeds the lookup,
  the pdf, and the importance sampler, so the rotation can't reach only some
  of the three and silently break MIS.
- **Selectable render resolution and PNG export.** Resolution changes route
  through the same all-or-nothing scene rebuild everything else already
  uses — `OptixRenderer` owns the accumulator/PBO/texture/denoiser as one
  unit, so there was no reason to grow a second, partial resize path.
  Export reuses the same PNG-writing code the scripted `ITALY_DUMP_FRAME`
  hook already had. CLI resolution/dimension flags are now bounds-checked
  instead of trusting `atoi` — `--voxel=0` used to reach
  `glm::clamp(v, 0, -1)`.
- **Environment-lit caustics.** Photon mapping was hardcoded to the
  synthetic quad light and disabled outright the instant any HDRI loaded,
  even for the one scene that has always had specular geometry (mirror +
  glass) regardless of light source. `__raygen__photon` now emits from the
  environment when one is loaded — importance-sampled via the same
  distribution NEE uses, with the emission origin placed on a disk outside
  the scene's bounding sphere (PBRT's `InfiniteAreaLight::Sample_Le`
  construction). Deliberately *not* extended to loaded mesh/voxel/SDF
  scenes: none of their material types (`MATERIAL_TEXTURED_DIFFUSE` is GGX
  *reflectance*, no transmission; `MATERIAL_VOXEL`/`MATERIAL_SDF` are flat
  diffuse) register as specular to the photon closest-hit, so a loaded asset
  cannot seed a caustic no matter how emission is set up — adding the SBT
  plumbing for it now would be complexity in exchange for nothing visible.
  Marked as a real gap on `MATERIAL_TEXTURED_DIFFUSE`'s own doc comment
  (`// claudia:`), not silently dropped.
- **A shadow-catcher ground plane for loaded scenes, then a correction to
  it.** Mesh/voxel/SDF scenes previously had nothing to ground them — no
  contact shadow, no bounce surface, an object visibly floating. A plane
  shipped first at 6x the object's bounding radius, which — combined with
  the tighter 1.15x camera framing above — put its hard, perfectly flat
  edge inside the frame at ordinary orbit distances, and under the `noon`
  HDRI preset it blew out to pure white (confirmed under
  `ITALY_TONEMAP=clamp`, which rules out this being a tonemap artifact).
  The deeper problem wasn't just size: a flat synthetic grey plane
  structurally can't match an HDRI's own baked-in ground — wrong hue, hard
  silhouette, no falloff — so resizing and darkening it only reduced how
  much of the frame it ate, not the mismatch itself. Fixed by skipping it
  entirely once an environment map is loaded (the same reasoning the
  synthetic quad light already uses — an HDRI supplies its own ground and
  horizon) and shrinking it to a 1.4x-radius contact-shadow catcher for the
  remaining no-environment case.

Two of the above were caught mid-implementation by rendering, not by
reading the diff: the ground plane's failure mode only showed up as a wall
of white filling most of a `bike.glb` render, and a `voxelize_test` bounds
assertion (a test fixture invariant broken by the loader's new
multi-material `MeshAsset` shape) only showed up by running `ctest`. Both
are why this project's own stated verification philosophy — render it,
don't just reason about it — keeps paying for itself.
