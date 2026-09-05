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

To render an imported Gaussian-splat scene instead of a GLB (a distinct load
source, not a representation of a loaded mesh — the two flags aren't
combined):

```
./build/italy-rs --gsplat=path/to/scene.ply
```

Expects the standard 3D Gaussian Splatting `.ply` vertex layout (position,
scale, rotation, opacity, SH DC color — see `src/io/gsplat_ply_loader.h`).
Higher-order SH bands (view-dependent color) are read past but dropped;
splats render as flat-diffuse, alpha-blended via stochastic per-splat
transparency rather than screen-space rasterization — see `MATERIAL_GSPLAT`
in `src/render/kernels/pathtracer_params.h`.

To light the scene with an HDRI instead of the built-in quad light (applies
to any of the above, including the built-in test scene — a good way to see
glass/mirror materials against real environment lighting):

```
./build/italy-rs path/to/asset.glb --env=overcast   # or midnight, noon
./build/italy-rs path/to/asset.glb --hdri=path/to/custom.hdr
./build/italy-rs path/to/asset.glb --env=sky        # procedural Preetham sky, defaults to clear midday
```

The procedural sky's turbidity/sun elevation/sun azimuth are UI sliders once
loaded (shown under the "Procedural Sky" HDRI radio button); `--env=sky`
alone seeds a default clear-midday look for scripted use.

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
`ITALY_SKY_TURBIDITY`/`ITALY_SKY_ELEVATION`/`ITALY_SKY_AZIMUTH` seed the
procedural sky's params (see `--env=sky` above) the same way.

To composite a procedural NanoVDB fog volume onto whatever scene the other
flags select (a real absorbing/scattering participating medium, not a
billboard — Woodcock delta-tracking with multi-scatter and Henyey-Greenstein
phase, see `src/io/nvdb_loader.h` and the `MATERIAL_NVDB` branch in
`pathtracer.cu`):

```
./build/italy-rs --fog --env=sky
```

Radius, voxel size, extinction (`sigma_t`), single-scatter albedo, phase
asymmetry (`g`), and a density multiplier are live ImGui sliders under
Settings once the window is open (each edit re-bakes the grid and
reconstructs the renderer — this is scene geometry, not a `RenderSettings`
field). Scripted A/B verification: `ITALY_NVDB_SIGMA_T=<value>` (0 ~=
no volume), `ITALY_NVDB_DENSITY_SCALE=<value>`, `ITALY_NVDB_G=<-0.95..0.95>`.

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

`__intersection__sdf` accepts a hit on a **sign change** across the step, not
only on `|dist| < epsilon`. The old test could step clean over the surface:
`minStep` is `0.05 * voxelSize` against an `epsilon` of `0.02 * voxelSize`, so
a ray could go from `+0.06` to `-0.06` — never inside the band, never
reported — and carry on to the far wall. Which rays skipped depended on step
phase, which varies smoothly with ray direction, so the misses landed in
coherent curved bands: the arc lattice visible through the glass cup's shell.
Refinement is now 8 bisections plus a linear solve on the bracket rather than
one secant step, and the marcher advances by `0.85 * |dist|`, since trilinear
interpolation of a distance field is not 1-Lipschitz near cell diagonals and
the full distance can overshoot. Perf-neutral at 320^3 (8.02/8.92s before,
8.14/8.34s after) — the earlier acceptance pays for the shorter steps.

Ruled out before landing this, so it does not get re-litigated: **the artifact
is not a resolution or representation problem.** Baking the same cup at 640^3
reproduced the arcs at identical spacing and identical placement; a
discretization artifact would have halved its spacing. Moving to an SDF
hierarchy or a pure analytic-function evaluator would have cost the bake and
changed nothing here.

**Infinite ground plane, and the two bugs it exposed.** `addGroundPlane`
sized its quad at `radius * 1.4`, so every scene sat on a visible slab. The
extent is now `max(radius, 1) * 1e4`, which has no reachable edge. It stays a
triangle quad rather than becoming an analytic plane primitive: a new custom
primitive would need its own light-subpath hit group, and without one it
would trip the guard in `buildScene()` that disables light subpaths for
custom geometry — which would have silently turned the VCM work back off.
Precision is fine at that extent because the hit point comes from
`origin + t * dir` rather than barycentric interpolation, and the normal is
an exact axis-aligned cross product. Bounds are unaffected: `addGroundPlane`
runs after `boundsCenter`/`boundsRadius` are fixed, and the `extraSdf`
expansion reads grids, not objects.

The first thing it exposed was cost. An infinite floor fills the lower
hemisphere, so nearly every first hit is now diffuse, and each diffuse hit
was firing `maxConnectionsPerVertex = 128` occlusion rays. The material probe
went 9.0s -> 52.0s. Connection count turns out not to matter any more now
that merging works and the photon map is dense — caustic-region p99/p90/p50
are identical at 8, 32 and 128 connections, differing only in the fourth
decimal of the noise metric — so the default drops to 8. Scenes now run
7.6/8.6/12.9/7.9s, which is at or below where they were before the floor
changed.

The second was a real estimator bug that the old small floor had been
hiding. Photons were deposited at the *first* diffuse hit, including photons
that arrived straight from the light — exactly the transport the eye path's
NEE already computes, with no MIS weight between the two, so direct light on
diffuse surfaces was counted twice. With a small floor this was a uniform
brightness offset and invisible. With an infinite floor it became a
hard-edged bright disc: the photon emission disc is sized to
`sceneBoundsRadius`, so the double count stopped abruptly at its rim.
`realism: the photon map is now caustic-only — light vertices are deposited
only after the path has hit a specular or dielectric surface, tracked
through a new payload_11.` Diffuse-to-diffuse indirect goes back to the path
tracer, which already covers it. Caustic-region p90 drops 6.22 -> 3.95, and
that drop is the double count leaving: the bright ring reported in the
previous entry as a caustic was substantially emission-disc-bounded direct
light, not focused light. What is left is a smaller, correctly placed caustic
under the cup on a uniform floor.

`traceOcclusion`'s hardcoded `1e-3f` origin offset became `surfaceEpsilon(P)`
(`1e-3 + 1e-6 * max|P|`) while chasing far-field streaking on the big plane.
It did not fix the streaking — that turned out to be ordinary Monte Carlo
noise on newly visible geometry, 44% pixel-to-pixel spread in the linear
data, only visible under a 6x contrast stretch — but scaling a surface
epsilon with position is right at any scene scale and was kept.

Note that `maxConnectionsPerVertex = 0` still means "connect to every light
vertex", which with the current 262144-path batch is ~164K occlusion rays per
diffuse hit and takes minutes per frame. It is reachable from the UI now that
the caps are gone; the default of 8 keeps it out of the way.

**Tricubic C2 normals, and they turned out to be free.** SDF shading normals
came from a half-voxel central difference of the trilinear field. Trilinear
reconstruction is piecewise-linear per axis, so its second derivative is zero
inside a cell and a delta at cell faces — the surface had no curvature
anywhere, and curvature is what shapes a caustic. `sdfGradientNormal` now
evaluates a cubic B-spline (C2) and takes its gradient *analytically* from
the same 64 taps, using the derivative basis in one axis and the value basis
in the other two. That is the trick that makes it affordable: one 64-tap
gather yields all three derivatives, where the old finite difference already
paid six trilinear samples (48 taps) for a worse answer.

Sphere tracing deliberately stays trilinear. `sampleSdf` runs every march
step — up to 512 of them — so a 64-tap sample there would be an eight-fold
hit on the dominant cost, while the trace only needs a conservative distance
to find the surface. The normal is evaluated once per hit and is what
actually steers refraction.

Measured properly, three alternating runs per variant at 600 subframes by
swapping `pathtracer.optixir` under a fixed binary: trilinear 15.74s median,
tricubic 15.83s. Under 1%. An earlier single-sample reading of 24% was a cold
first run and nothing else — worth remembering that the first render after a
build is not a measurement.

The image gain is real but modest: refracted streaks across the cup wall are
continuous where trilinear broke them into high-frequency fragments. The
caustic itself did not change shape, and neither did dropping the merge
radius 4x — a smooth spherical bowl with a flat liquid surface produces a
smooth annular caustic, which is the correct answer for that geometry rather
than a renderer limit. Filaments need a shaped rim or a disturbed liquid
surface; that is scene authoring, not reconstruction.

**The cup is a scene now, not a checkbox.** "Caustics stress-test cup (SDF)"
was a Settings tickbox that layered `makeCupSdf` onto whatever was already
loaded, which meant its data arrived outside the project-load path that every
other scene goes through. The checkbox and the `--cup` flag are gone; the cup
is registry entry 3, "Caustics cup", and pipes in on load like the rest. Its
default framing is wide, because the cup sits at `centerXZ = (-1.6, 1.6)` and
expands the scene bounds the camera frames from — pre-existing, and a scene
authoring fix rather than a renderer one.

**The merge half of VCM never gathered a single photon.** Vertex merging
traced a ray from the shading point `P` along `N` with `tmin = 0` against a
GAS of spheres centred on each light vertex. But the photons worth gathering
are the ones whose sphere *contains* `P` — a ray starting inside a sphere
enters it at negative `t`, so every one of them was culled by `tmin`, and
`merged` was identically zero in every pixel of every frame. The `tmax =
2.02f * mergeRadius` constant is the fingerprint of the intent: that is
exactly the chord you need for a ray starting one radius *behind* `P`. The
origin was simply never moved back. Fixed by tracing from
`P - N * 1.01 * mergeRadius`, and the anyhit now also rejects on true
distance (it reconstructs `P` from the ray origin), since a ray along `N`
otherwise gathers photons up to a radius away *along the normal* and biases
the density estimate.

Finding it took three rounds of the same measurement, because the null
result was so complete it read as a measurement error: the frame was
byte-identical under a 128x photon-count change, under a 10x merge-radius
change, and with a debug build whose only output was `merged`. A merge
radius change alone scales `merged` by the square, so byte-identical output
across it is only possible if the term is exactly zero — that was the step
that pinned it.

Measured on the glass cup, floor region, 400 subframes, `ITALY_FIREFLY_CLAMP=0`:
caustic p90 4.22 -> 7.12, p50 0.785 -> 0.992, and relative noise *fell*
0.450 -> 0.309. More energy and less noise at once is the signature of a
density estimator that started working.

Two knock-on results:

- The firefly clamp is no longer in conflict with caustics, so the
  `fireflyClamp = 10` default is vindicated rather than merely tolerated.
  Before the fix the caustic lived entirely in the tail, so clamping the
  tail erased it (contrast 19.1 -> 6.1). Now the caustic arrives through the
  merge estimator as smooth energy: clamp 10 costs only p90 7.12 -> 6.53
  while dropping relative noise 0.309 -> 0.057. Clamped-and-fixed is the
  best image of the four combinations by a wide margin.
- The photon count finally does something, and it is nearly free, so
  `kLightSubpathBatchSize` goes 2048 -> 262144 with capacity sized to the
  ~164K vertices actually deposited per pass (524288, about 36MB) rather
  than the 4M used while testing. Relative noise 0.337 -> 0.309 for 7% more
  wall time. The gain is modest because the residual specks are eye-path
  caustic samples on *specular* hits, which merging cannot catch — merging
  only runs on diffuse.

`kInitialMergeRadiusFraction` was tested at 0.01 and returned to 0.02:
halving it raised noise 0.309 -> 0.327 with no visible sharpening, which is
consistent with the earlier finding that caustic sharpness is limited by the
field's missing curvature, not by the gather. The light-subpath log line
prints merge radius and GAS state now, which is what made the diagnosis
legible.

**HDR output, and what it immediately exposed.** `ITALY_DUMP_FRAME` and the
export button now dispatch on extension: `.hdr` writes the linear
accumulation buffer via `stbi_write_hdr`, anything else keeps the old
tonemapped 8-bit PNG path. The accumulation buffer already holds the
progressive running mean, so no division is needed; `readAccumulationRgb`
is the only new seam method and `app/` still sees no CUDA. Every A/B in
this repo before now was measured through AgX plus 8-bit crush, which is
worth remembering when re-reading older entries.

Three measurements followed immediately, all on the glass cup at 400
subframes, floor region only:

- `realism: fireflyClamp defaulting to 10 costs the caustic, not just the
  specks.` Caustic-region peak radiance falls 336 -> 4.8, p99 15.0 -> 4.4,
  and contrast (p99/p50) 19.1 -> 6.1 with the clamp on. A caustic *is* the
  tail of the distribution, so clamping the tail is clamping the caustic.
  The default stays at 10 because it is right for general preview work, but
  caustic tuning has to run with `ITALY_FIREFLY_CLAMP=0` and reduce variance
  rather than truncate it.
- The photon count is currently inert. Raising `kLightSubpathBatchSize`
  2048 -> 262144 lifts deposited vertices per pass from ~1.6K to ~164K for
  6% more wall time, and changes the frame by less than RGBE precision
  (zero differing channels) — including the caustic region's noise metric,
  which stays at 0.4492 relative. The estimator normalisation is correct
  (per-photon throughput goes as 1/batch, `reweight` as batch, so the mean
  is invariant by construction) but the *variance* should have dropped
  visibly and did not. The caustic is being carried almost entirely by
  brute-force eye-path sampling; the VCM connect/merge layer contributes
  little enough to hide under 0.4%. Constants reverted to 2048/32768 pending
  that being understood — raising them is the first thing to redo once
  merging actually carries weight, since it is nearly free.
- Merging is running (`mergeGas yes`, radius 0.0182 falling per the PPM
  schedule) but the radius is 2% of the 0.909 scene radius, about five cup
  voxels, which is a hard blur floor on fine caustic filaments. The
  light-subpath log line now prints radius and GAS state alongside the
  deposit count.

**Why the caustic can't have shape yet, from the field side.** A caustic is
the focusing of refracted rays, so its structure is governed by surface
curvature — the derivative of the normal, the *second* derivative of the
SDF. Trilinear reconstruction is piecewise-linear in each axis: its second
derivative is zero inside a cell and a delta at cell faces. The surface has,
numerically, no curvature anywhere. That is a ceiling no sample count
reaches past, and it is separate from the estimator problem above. The fix
is a C2 reconstruction — cubic B-spline tricubic, eight hardware-trilinear
fetches out of a 3D texture — and it should be perf neutral or better,
because `sampleSdf` is currently eight uncached global loads and
`sdfGradientNormal` calls it six times, so a hit pays ~56 scalar global
loads where a texture path pays a handful of cached filtered fetches.
`uploadTexture` in `optix_renderer.cpp` is already the pattern to follow.
A cheaper procedural-only complement: bake the analytic gradient into a
second grid, since `bakeAnalyticSdf*` has the exact CSG function in hand —
exact normals from one fetch, replacing six `sampleSdf` calls.

Blocker on the stated next goal: `cup.glb` does not exist. `assets/`
contains only `test.glb` and the three HDRIs; the current cup is procedural
(`makeGlassCupSdf`), not a glTF import.

**Cup rings are the bake, not the tracer or the camera.** The glass cup is
an analytic CSG hierarchy (two spheres, `opSubtract`, `opIntersect` against
the rim plane, `opSmoothIntersect` for the liquid) that exists only on the
CPU: `bakeAnalyticSdfPalette` evaluates it once per cell into a flat float
array and the kernel sphere-traces the trilinear interpolation of that
array. The exact function never reaches the device, so the residual rings
are reconstruction error and no tracer tuning removes them. Confirmed by
doubling `kGlassCupResolution` 160 -> 320 (grid 164x134x164 ->
324x264x324): rings drop and the meniscus sharpens, which is grid-locked
behaviour, not view-locked. Shipped at 320 — 9.3s -> 11.3s wall including
bake, about 139MB of device memory. Tracing the hierarchy analytically on
device would remove them outright and is the real fix for procedural
content; it is deferred, since real mesh imports still have to bake and
that means carrying a second SDF material path.

**Artist-facing upper bounds removed.** Every UI/CLI ceiling is gone —
resolution (was 512), render size (8192), export samples (4096), samples
per launch (16), exposure (8), firefly clamp (50), sun intensity (200000),
BDPT connections (4096), aperture (a quarter of scene radius), focus
distance (eight times scene radius), sky turbidity (10), and the wrapping
angle clamps on sun elevation/azimuth and environment rotation. Lower
bounds survive only where a smaller value is undefined rather than merely
ugly: resolution and render size at 1, non-negative for exposure/intensity/
aperture/clamp, 1.9 turbidity (below that the Preetham fit breaks, in
`procedural_sky.cpp` too), and `sunAngularRadiusDeg` in (0, 90) since
`cosAngularRadius` and the `2*pi*(1-cos)` solid-angle pdf go to zero or
negative outside it. The `ITALY_SCENE` clamp stays as well; it bounds a
vector index, not a preference.

**Surfacing pass 2, plus two dead controls.** After the classification fix
the cup still carried faint ring remnants; the cause was the sphere
tracer's hit tolerance. `__intersection__sdf` accepted a hit at
`|dist| < 0.1 * voxel` and only refined it when the sign flipped between
steps, so near-tangent rays — which never bracket a sign change — reported
positions up to a tenth of a voxel off the isosurface, and the normal was
then evaluated at that offset. Tightening to `0.02 * voxel` (iteration cap
256 -> 512 to pay for it) removed them; the frame cost about 11% more wall
time, which rung 2 affords for a rung 1 gain this size. A wider gradient
stencil was tested at the same time and made no difference, so it stayed at
half a voxel — the residual was never normal reconstruction.
`realism: fireflyClamp now defaults to 10 instead of 0 (off) — the clamp
existed but nothing enabled it, so every render shipped with permanent
white specks; 10 removes them while keeping visibly more caustic energy
than 3 did.`

Two controls did nothing. The viewport's `InvisibleButton` was created with
default flags, which accept the left button only, so `IsItemActive()` was
never true for a middle-drag and pan was unreachable — now created with the
left/middle/right flags, and pan accepts either middle or right. The HDRI
radio buttons only assigned `state.envChoice` and waited for the next full
bake, even though the procedural-sky sliders directly beneath them rebuild
live; they now call `rebuildEnvironmentLive` on change, guarded so that
selecting Custom with an empty path doesn't tear down the environment.
`--env`/`--hdri` were also being silently overwritten by the scene
registry's own `configure()` in the headless dump path, which is what made
this untestable from a script; an explicit flag now survives the registry,
and rendering the glass cup under overcast/noon/midnight confirms all three
load and light differently (23-39% of pixels differ between presets).

**Glass ringing was a reconstruction/classification mismatch, not
voxelization.** The glass cup and the probe's dielectric spheres rendered
covered in dense concentric moire rings that refraction amplified into a
stippled mess. It was not under-resolution: at `kGlassCupResolution = 160`
the 0.09 wall is about 13 voxels thick. It was not gradient faceting from
C0 trilinear reconstruction either — swapping `sampleSdf`'s weights for
quintic (C1) ones changed the rings' frequency and nothing else, which
ruled that out in one render. The actual cause: `__intersection__sdf`
traces the *trilinearly interpolated* field, but `sdfBranchAt` decided
inside-vs-outside from the *nearest cell's stored* distance sign. Those two
surfaces disagree within a one-voxel band around the isosurface, which is
exactly the band every hit point sits in, so on a curved shell the
misclassified cells tile out as rings on the lattice — each one refracting
with the wrong IOR or not refracting at all. `sdfBranchAt` now takes its
sign from `sampleSdf`, the same field the intersector traced; the branch
label still comes from the nearest cell, which is a lookup, not a decision.
The rings are gone.

The material probes were also offset along `rayDir`, which degenerates at
grazing incidence — both probes land on the same side and the interface
disappears. They now offset along +/-N (the ray arrives from the +N side,
since N is faceforwarded against it), which is well-conditioned at any
incidence and removed the residual silhouette arcs. Both call sites went
through one new `sdfProbeMaterials` helper rather than being fixed twice.

**Correctness review pass: MIS mixture, sun/env sync, SDF specular.**
Five defects found by tracing the new sun/palette-SDF code and confirmed
with `ITALY_DUMP_FRAME` A/Bs against a binary built from the pre-fix tree.
(1) The quad-light BSDF-hit MIS branch computed a bare solid-angle
`pdfLight` while NEE scaled its own by `(1 - pSun)`, so with the sun on the
two heuristic weights no longer summed to 1 and direct quad-light energy
was lost; it now applies the same mixture factor the env-miss branch
already did. (2) `params.sun.direction` was uploaded unrotated while the
environment lookup rotates by `envRotation` — the analytic disk, its
shadows and its caustics slid off the baked sky glow as the slider moved.
Rendering the material probe at rotation 0 vs 90 degrees proved it: before,
90 degrees of rotation moved 0.67% of pixels by at most 17 levels (the
glow only); after, 6.4% by up to 110 (the sun and its shadows track the
sky, as intended). (3) Opaque palette SDF hits set only `diffuse` and never
`hasSpecular`, so the probe's 0.05-to-1.0 roughness row rendered as five
identical Lambertian spheres; both SDF material paths now go through the
shared `materialFromMetallicRoughness` (renamed from
`materialFromLightVertex`, which no longer described its callers).
(4) `schlickFresnel` returned ~1 at grazing angles for `iorFrom ==
iorTo` — a full reflection off a boundary that isn't there, reachable
whenever both SDF probes land in the same dielectric; it now returns 0 when
the interface has no index step, which fixes every caller at once.
(5) `classifyPrimitive` ran the sphere test first and accepted on
equidistance from the centroid alone, which every 8-corner box satisfies
exactly, so a default cube imported as a Sphere; the box test now runs
first (it is the stricter fit) and `mesh_primitive_classify_test` gained
unsubdivided-cube cases that reproduce the old failure. Also: the bake
log's "Showing the built-in test scene." line overwrote the scene
registry's own status text for every path-free scene, mislabelling the
material probe and glass cup bakes.

**Review pass: sun-power bug, live sky re-bake, dead state cleanup.**
Rendering all three registry scenes and looking (`ITALY_DUMP_FRAME` +
a new `ITALY_SCENE=<index>` to pick which registry entry the headless
dump hook bakes) surfaced a real bug the unit tests couldn't catch:
`emitFromSun` was missing the `/pdfDir` term `emitFromEnvironment` has —
inconsistent with the eye-side NEE estimator (which does divide by the
sun's solid-angle pdf), so light-subpath energy from the sun didn't match
what NEE assumed. Fixed to mirror `emitFromEnvironment`'s shape exactly.
Combined with the default `sunAngularRadiusDeg`/`sunIntensity` (0.4°/15)
being far too small/dim for a physically tiny-solid-angle emitter to
contribute anything visible once correctly Ω-scaled, the glass cup's
caustic was reading as completely flat — confirmed by rendering at 1500
subframes and contrast-boosting the floor region, next to a same-treatment
render of the existing bring-up scene's glass sphere (whose caustic *was*
clearly visible under the same boost, ruling out a display/exposure
explanation and pointing at the sun path specifically). `realism: sun
defaults widened to 2.0deg angular radius / 10000 intensity (from a more
physically-real 0.4deg/15) — a tiny accurate solid angle needs
five-figure radiance to contribute anything once NEE divides by its own
Ω, which is impractical to dial in by hand; the wider, brighter default
reads as a comparably tight sun while staying in a sane authoring range.`
Re-rendering after the fix shows real caustic structure (concentric rings,
a focused bright/dark patch distinct from ambient floor lighting) instead
of flatness — the `todo` bar. `groundOffset` (the stand-off gap under the
cup) was ruled out as a contributing cause by testing flush-vs-offset
before finding the actual bug — kept at `0.2 * outerRadius` per the
original design.

Also fixed while reviewing the wiring pass: the procedural-sky sliders
(turbidity/elevation/azimuth) and the "Sun enabled" checkbox previously
only called `resetAccumulation()`, which cannot make a live environment
change visible — `OptixRenderer` has no partial-update path for its baked
env texture/CDF, only full reconstruction at construction time. They now
call a new `rebuildEnvironmentLive()` (re-bake environment, reconstruct
`OptixRenderer` without re-framing the camera) instead, so moving the sun
actually re-bakes the sky glow to match, per the original decision. This
reconstructs from the Settings window's own GL context rather than
Viewport's — safe post the shareAnchor refactor (proven already: the
Project window's bake step already reconstructs from its own, different,
context). `bakeStepConstructRenderer` also stopped depending on
`ProjectBakeState::haveSdf`/`haveVoxels` (transient, only valid right
after a full `bakeStepRepresentation` pass) in favor of deriving the same
condition from `state.representation`/`state.haveMesh` directly — the two
were always equivalent, and the transient version silently broke a
partial live-rebuild path (this one) called without re-running the full
bake sequence. The now-dead `ProjectBakeState::haveSdf`/`haveVoxels`
fields were removed.

**Material probe scene was rendering on top of the bring-up scene.**
`extraSdf`-only content (no `mesh`/`sdf`/`voxel`/`splats` set) fell
through `buildScene()`'s dispatch to `buildFixedTestScene()` — correct,
deliberate behavior for `--cup`'s ceramic bowl (which wants to layer onto
the known-good glass/mirror spheres for caustic stress-testing) but wrong
for the material probe, which wants a clean, isolated scene and was
instead rendering its 9 probe spheres on top of the bring-up scene's red/
mirror/glass spheres — confirmed by rendering and looking; the "red
sphere" visible in an early probe render wasn't part of the probe at all.
Fixed with a new opt-in `SceneSource::emptyBase` (+`emptyBaseFloorY`):
`buildScene()` seeds a near-zero bounds box instead of the fixed scene,
lets the existing `extraSdf` loop expand bounds from the real content as
usual, then places the ground plane and key light *after* that expansion
using the now-correctly-sized bounds — placing them before, sized from a
guessed default radius, produced a technically-clean but comically
oversized floor with the probe spheres looking tiny and distant.

**Device kernel: dielectric SDF, sun disk, solid background.**
`evalDielectricBounce`/`schlickFresnel` dropped the `entering` bool and
now take explicit `iorFrom`/`iorTo`/`extinctionFrom` — strictly more
correct than the old air-relative-only version even for the pre-existing
glass sphere and glTF transmissive materials (both call sites updated
mechanically, same rendered output). `MATERIAL_SDF` hits now sample the
grid's branch/palette on both sides of the ray (`sdfBranchAt`,
`sdfPaletteOrAir`, `sdfNearestCellIndex` factored out of the pre-existing
nearest-cell material lookup) — a branch that's `Dielectric` on either
side routes the hit into the dielectric dispatch instead of diffuse
shading, with no medium stack: the glass shell and liquid body are one
grid, baked with the liquid's cavity-facing sphere geometrically
coincident with the shell's own cavity, so "which side is which" is
answerable locally by sampling just behind/ahead of the hit point along
the ray. `realism: nested-dielectric IOR resolution uses same-grid
branch lookup instead of a Schmidt-Budge priority stack — correct for one
body containing another in a single bake, wrong the moment two
separately-baked dielectric objects overlap; upgrade trigger is exactly
that.` `__intersection__sdf` no longer treats "any negative distance" as
an instant hit (the bug that made a transmitted ray sphere-trace into a
dielectric SDF report a false hit at its own origin): it now marches on
`|distance|`, detects on proximity rather than sign (needed for the
glass/liquid contact surface, where the union distance never actually
changes sign), and gates the first reportable hit behind a
march-distance guard instead of the old opaque-only outward shading-point
nudge, which is now removed. `geometricNormalFor` gained an SDF branch
(`sdfGradientNormal`) — it previously fell through to a sphere/triangle
path that's undefined for an SDF custom-primitive hit, silently wrong for
the `entering` test on any non-vertical glass SDF hit and for every
`__closesthit__lightSubpath` call (which uses it unconditionally).
`__closesthit__lightSubpath` gained a `MATERIAL_SDF` branch mirroring
`MATERIAL_GLASS` (refract-and-continue, no vertex deposit — matches how
SDF has no BSDF-pdf machinery for a diffuse deposit either). A new
`SunLight` (`Params::sun`) is a uniform-cone-sampled analytic light,
NEE'd and MIS'd via a fixed 50/50 mixture with whichever of env/quad was
already active, in both the eye-path NEE block and the light-subpath
emitter (`emitFromSun`, mirrors `emitFromEnvironment`); `__miss__radiance`
adds its contribution on top of the env lookup (both can be lit at once)
with its own MIS term against `sunConePdf`. `realism: analytic sun disk
and the procedural sky's own baked disk (buildProceduralSky's
bakeSunDisk parameter) are mutually exclusive by construction to avoid
double-counting — an artist-loaded HDRI with a real visible sun plus an
analytic sun aimed at it will still double-count; no auto-detection.`
`__miss__radiance` also now returns `Params::backgroundColor` for
`depth == 0` rays only (not `prevBsdfPdf < 0`, which is also true right
after a delta bounce and would have flattened the HDRI seen through
glass/mirrors) — `MissData::bgColor` and its one write site are removed,
fully superseded by the live per-launch `Params` field.

**Renderer host-side plumbing for dielectric SDF materials.** `SdfGrid`'s
new `branch`/`palette` arrays (see the convert-layer entry below) upload
into two new `HitGroupData` fields (`sdfBranch`, `sdfPalette`/
`sdfPaletteCount`) only when non-empty, alongside a guard added to the
pre-existing `sdfBaseColor`/`sdfMetallic`/`sdfRoughness` upload (previously
unconditional — a palette-baked grid leaves those empty by design, and an
unconditional upload of a zero-length host vector was a latent bug this
change happened to expose, not something introduced by it). A dielectric
SDF grid (any palette entry with `kind == Dielectric`) now opts the scene
into `enableLightSubpaths`, and the light-subpath refusal guard in
`buildScene()` no longer trips on `GeometryKind::Sdf` — only Voxel/Gsplat
still refuse, since only they lack a light-subpath hit-group
(`lightSubpathHitSdfPG`, new, mirrors the existing sphere/solid-sphere
ones: CH=`__closesthit__lightSubpath`, IS=`__intersection__sdf`). Fixed a
real bug while here: `addGroundPlane` silently suppressed the ground plane
whenever an environment map was loaded (`if (!wantGroundPlane ||
hasEnvironment) return;`) — the rationale ("an HDRI already bakes in its
own ground/horizon") stops holding once the miss shader can return a flat
background color for camera rays instead of the HDRI directly, and even
before that landed, an HDRI-lit scene that explicitly wants a ground plane
(the glass cup needs one to catch its caustic) had no way to ask for one.
`groundPlane` is now purely `source.groundPlane`'s explicit choice; a new
`SceneSource::groundOffset` (default 0, today's flush-to-object behavior)
lets a scene stand its floor off from an object's bounds, for a lensing
object whose focus lands below its base. `RenderSettings` gained
`backgroundColor`/`sunEnabled`/`sunDirection`/`sunAngularRadiusDeg`/
`sunRadiance`, uploaded into new `Params::backgroundColor`/`Params::sun`
every launch (not SBT-baked) so they're live-editable post-bake like
exposure; `MissData::bgColor` is now dead weight, left in place until the
device-side pass (next) removes its one remaining reader.
`buildProceduralSky` gained a `bool bakeSunDisk = true` parameter (default
preserves today's baked-in sun disk) — the device-side pass flips it to
`false` once the analytic sun light exists, so the two don't double-count
the same sun.

**Project window: gated startup, reopenable panels.** `main()` no longer
pre-arms straight into the viewport. A hidden 1x1 GLFW window (never shown,
never in any panel slot, destroyed after the main loop) anchors the shared
GL namespace, so Project/Viewport/Settings share against it instead of
against each other — the previous scheme shared against whichever
`AppWindow` happened to be built first, which made that window undestroyable
without dangling every other panel's GL objects. Three named `AppWindow`
slots (`projectWindow`/`viewportWindow`/`settingsWindow`) plus three
`want*` bools replace the old flat window vector: a slot is constructed
when its `want` flag is true and the pointer is null, and destroyed when
its own `want` flag goes false; `shouldClose()` (OS chrome X included)
clears that panel's own `want` flag before the destroy check runs, so the
native close button and the checkbox row both durably close a panel the
same way — reopening either one only happens through an explicit
`want = true` (the checkbox row, or Project's own post-bake handoff). App
exits once all three `want`
flags are false and all three pointers are null. `rebuildScene()` split
into five named phases (load → classify/validate → bake representation →
build environment → construct `OptixRenderer`) driven one-per-frame by the
new Project window, mirroring the existing `exportPending`/`subframeIndex`
polling shape — each phase appends a timestamped line to an in-memory,
auto-scrolling `bakeLog` and to `stderr` via the usual `italy: ` prefix, so
a long bake reads as progress instead of a frozen window. A 3-entry
`SceneDescriptor` registry (name/description/`configure(AppState&)`)
backs the Project window's scene picker; only "Bring-up scene" is real
today, "Material probe" and "Glass cup" are stubs that just mark
`statusLine` — the SDF/material content they need lands in `src/convert`/
`src/render` in a later pass. Camera auto-frame now also fires for an
SDF-only scene (`source.sdf != nullptr || !source.extraSdf.empty()`), not
just mesh/gsplat — previously an SDF-only bake left the camera unframed.
`claudia: bake-log progress is coarse (phase-level, five lines per bake),
not GPU-launch-level — sdf_baker.cpp's optixLaunch stays one call; would
need Z-slab slicing to show live percentage within a phase, not worth it
until a bake is slow enough to want that.`
`claudia: window position/size are fixed defaults, not persisted across
restarts — revisit if users find re-arranging windows every launch
annoying.`

**Content pivot: hybrid SDF + gsplat, procedural content first.** The
roadmap direction changed here. italy-rs stops treating mesh/GLB import as
its primary content path: SDF carries silhouette (and later
animation/deformation), Gaussian splats carry fine detail and accents,
both procedurally generated (scanned/photogrammetry ingestion is deferred),
traced together in one spatial structure. GLB import stays available but is
now gated to genuine scene primitives.

- **Primitive-fit import gate.** `src/io/mesh_primitive_classify.{h,cpp}`
  classifies a loaded mesh as box/sphere/cylinder or rejects it — an
  analytic least-squares-style fit (welded vertices, PCA oriented frame via
  glm's `gtx/pca.hpp` symmetric eigensolver, RMS-not-max relative residual
  per candidate shape), not a triangle-count heuristic, so a low-poly and a
  high-poly sphere classify identically and an organic blob is rejected
  regardless of tessellation. Cylinder tests all three PCA axes (a squat
  cylinder's true axis is the smallest-eigenvalue one). Wired into
  `rebuildScene()` alongside the existing watertightness check, same
  load→validate→reject-with-status-line pattern. Capsule/cone and
  arbitrarily-rotated boxes (PCA frame is undefined for an isotropic cube
  covariance) are documented v1 cuts. The vertex weld shared with
  `mesh_validate.cpp` was factored into `src/io/mesh_weld.{h,cpp}`. New
  `mesh_primitive_classify_test` (cube, low/high-poly sphere, tall/squat
  cylinder, beveled cube, ambiguous near-sphere, organic-reject, degenerate)
  — `ctest` now 7 tests. `assets/test.glb` (a device model) is now
  rejected by this gate; the bundled primitive path is `/tmp`-built cubes or
  a real prim export.
- **Procedural SDF authoring library.** `src/convert/sdf_primitives.h`
  (header-only, IQ-transcribed `sdSphere`/`sdBox`/`sdCylinder`/`sdTorus` +
  `opUnion`/`opSubtract`/`opIntersect`/`opSmoothUnion`) and
  `bakeAnalyticSdf()` in `sdf_procedural.{h,cpp}` — a reusable baker that
  samples a caller-supplied distance function (returning a CSG branch id
  alongside the distance, so a union of differently-colored primitives has a
  defined per-point material) at every cell center, reusing the exact
  grid-sizing/padding/cell-centered-sampling `makeCupSdf` had hand-inlined.
  `makeCupSdf` retrofitted onto it, renders pixel-identical to before.
- **Deliberate hybrid SDF + gsplat composition.** `SceneSource::extraSdf`
  is now `std::vector<const SdfGrid*>` — `buildScene()` loops each into the
  IAS alongside whatever base scene (mesh / gsplat / fixed) is active, and
  **expands `sceneBoundsCenter`/`sceneBoundsRadius` to enclose every entry**
  (not just camera framing — `pathtracer.cu`'s environment-light emission
  rays originate from a disk sized by these, so a piece outside the primary
  bounds would be under-lit). The per-object SDF device buffers became
  vectors so a multi-SDF scene rebuild doesn't leak all but the last grid.
- **Light-subpath SBT guard.** `traceLightSubpathPass()`'s SBT builder only
  knows Triangle/Sphere/SolidSphere hit-groups; a Voxel/Sdf/Gsplat object
  would silently bind to the triangle program group (no intersection
  program for a custom primitive — OptiX has no empty hit-group). Rather
  than add the missing per-kind program groups (real but out-of-scope —
  VCM connectability for these kinds is itself deferred), `buildScene()`
  now refuses `enableLightSubpaths` outright if the scene contains any such
  object, with a stderr line. This fires for the fixed test scene + an
  `extraSdf` cup.

Verified: `ctest` green (7 tests); `ITALY_DUMP_FRAME` of the fixed scene +
procedural cup renders correct compositing/occlusion/contact-shadow in one
IAS with the light-subpath guard firing as designed; a hand-built cube GLB
is accepted and rendered while `assets/test.glb` is rejected with a clear
status line; the all-mesh caustics case is unaffected (guard only trips on
custom-primitive kinds).

Still deferred (named, not forgotten): scanned/photogrammetry ingestion;
SDF animation/deformation (needs a primitive-SDF scene graph with live
per-primitive transforms, not a re-baked dense grid per frame — the CSG
library here is its seed); gsplat↔SDF parent/attachment transforms;
SDF BSDF-pdf / VCM-connectability; per-kind light-subpath hit-groups;
capsule/cone primitive kinds.

**VCM merge-term perf + correctness pass.** Decoupled the
O(lightVertexCount) brute-force merge scan onto a real GAS spatial query;
found and fixed a missing
`OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS` pipeline flag (silently
no-op'd the merge query with no host-visible error unless OptiX validation
mode is on) and a cross-surface light-leak bug (no surface-normal
compatibility check on the merge). `maxConnectionsPerVertex` capped and
defaulted to 128 from a live benchmark sweep (~12.5x speedup, confirmed
unbiased). Default scene light brightness 3x. `src/convert/sdf_procedural`
landed here too as the first procedural (non-mesh-baked) `SdfGrid` — a
hand-authored CSG bowl, `--cup` flag / UI checkbox, layered via the new
`SceneSource::extraSdf` hook for caustics stress-testing.

**Roadmap phase 3: Gaussian-splat import.** Imports pre-generated 3D
Gaussian Splatting `.ply` scenes (the format `s-rank`'s triposplat and most
3DGS trainers/exporters write) and renders them path-traced, not fit from a
mesh — the design doc explicitly deferred mesh→3DGS fitting as its own
research-grade problem. `src/io/gsplat_asset.h`/`gsplat_ply_loader.{h,cpp}`
parse the standard vertex layout (position, scale_0-2, rot_0-3, opacity,
f_dc_0-2 — higher SH bands are read past but dropped, see the `realism:`
note on `MATERIAL_GSPLAT`) and apply its activation functions once at
import. Each splat becomes a custom-primitive ellipsoid (3-sigma bound,
`convert/gsplat_bounds.h`); `__intersection__gsplat` finds the ray/ellipsoid
crossing and `__anyhit__gsplat` stochastically accepts or rejects it against
the splat's actual Gaussian density — alpha-correct compositing of
overlapping/translucent splats with no depth sort, unbiased in expectation
(doctrine: unbiased transport kept for beauty, not realism). Diffuse-only
shading via an ellipsoid gradient normal, same tier as MATERIAL_VOXEL/
MATERIAL_SDF (no caustic eligibility). New `Representation::Gsplat` UI radio
button + its own path field (a splat file is a distinct load source, not a
resampling of a loaded GLB), `--gsplat=<path>` CLI flag, and
`ITALY_GSPLAT_PATH` scripted-testing hook. Verified with a synthetic
sphere-of-splats `.ply` (`ITALY_DUMP_FRAME`): renders as a soft, textured
cloud with alpha-blended edges, not the hard-edged blocky look a
voxel-resampling fallback would have produced. Not yet verified against a
real captured/trained splat scene — no such asset lives in this repo yet.

**Roadmap phase 2: procedural sky (Preetham/Perez analytic daylight
model).** `src/render/procedural_sky.{h,cpp}` synthesizes an equirect sky
image (sun elevation/azimuth + turbidity as artist controls) and feeds it
through the exact same `buildEnvironmentCdf` a loaded `.hdr` file already
uses, so it needed zero changes downstream — GPU upload, NEE, importance
sampling, and photon emission all already treat "an environment" generically.
New `EnvChoice::ProceduralSky` UI radio button (with turbidity/elevation/
azimuth sliders alongside the existing Overcast/Midnight/Noon/Custom
presets), `--env=sky` CLI flag, and `ITALY_SKY_TURBIDITY`/`ITALY_SKY_
ELEVATION`/`ITALY_SKY_AZIMUTH` scripted-testing hooks matching the existing
`ITALY_ENV_ROTATION`-style pattern.

Preetham was picked over the higher-fidelity Hosek-Wilkie model specifically
because it's small enough to be fully closed-form (a page of published
polynomial coefficients) rather than Hosek-Wilkie's large fitted dataset,
which would have to be transcribed from a reference implementation with no
way to diff against it here — this project already shipped one bug from
exactly that failure mode (AgX's missing `pow(2.2)`, see below). Preetham
wasn't actually immune to it: a first draft's zenith-chromaticity polynomial
had the coefficient matrix transposed (grouped by power of view angle
instead of power of turbidity — superficially similar, silently permutes
which number multiplies which term) and rendered an entire sky in
olive-green instead of blue. The unit test (`procedural_sky_test.cpp`)
didn't catch it — it only checked that the sky is brighter toward the sun
than away from it, not hue — so this was caught by eye on the first
`ITALY_DUMP_FRAME` render, then fixed by checking the coefficients
digit-for-digit against two independent open-source implementations
(`ebruneton/clear-sky-models` and `diharaw/sky-models`, both transcribing
the same paper) rather than trusting a second guess. Re-verified after the
fix: a default clear-midday render shows a plausible blue sky, a visible
sun highlight on the mirror sphere, and sky-tinted glass; a low-elevation/
higher-turbidity render shows a distinctly different, warmer gradient,
confirming the parameters actually drive the output rather than the fix
having only patched the one case that was eyeballed.

**Roadmap phase 1: watertight-mesh import gate + caustics on loaded
meshes.** First of a larger agreed roadmap (procedural sky, ray-traced
Gaussian-splat import, then VCM/ReSTIR acceleration). Unified-SDF / hybrid
SDF+gsplat rendering was listed "last/deferred" here but has since moved to
the active direction — see the content-pivot entry at the top of this
section. This phase was picked first because it was small and already
unblocked.
`src/io/mesh_validate.{h,cpp}` rejects any imported mesh that isn't a
closed 2-manifold (every edge shared by exactly two triangles, checked
after welding the loader's unindexed triangle soup back into shared
vertices by position) — `rebuildScene()` in `main.cpp` calls it right
after `loadGlb()` and refuses the load with a diagnostic boundary/non-
manifold edge count on failure, the same status-line pattern a failed GLB
parse already used. That closedness guarantee is what a real dielectric
material on triangle geometry needs (a well-defined interior), unlike
`MATERIAL_GLASS`'s dedicated solid-sphere intersector — so loaded glTF
materials now parse `KHR_materials_transmission`/`_ior`/`_volume` and, when
`transmission > 0`, get a stochastic per-sample dielectric bounce (same
Fresnel/refraction/Beer-Lambert device functions `MATERIAL_GLASS` already
had, now parameterized per-material instead of per-object) in both the
main path tracer and the photon pass. A cheap second win shipped
alongside it: a near-zero-roughness, high-metallic textured surface (e.g.
chrome trim) is now treated as specular-enough to carry a caustic bounce
too, no transmission needed. `enablePhotonMapping` now turns on for a
loaded mesh scene whenever it actually has one of these two material
kinds (checked once at load time so an ordinary all-Lambert asset doesn't
pay for a photon pipeline that would only ever produce an empty map).
Voxel/SDF representations are unaffected — flat-diffuse-only, no
per-primitive material to carry a transmission factor.

Verified: a hand-rolled closed-cube-shell fixture accepted, the same cube
with two triangles deleted rejected with a boundary-edge count (both
exercised by the new `mesh_validate_test`); a downloaded Khronos
conformance asset that deliberately mixes closed spheres with flat
decorative geometry (text labels, a backdrop plane) correctly rejected —
confirms the gate isn't just accepting everything; a hand-built watertight
glass cube (`KHR_materials_transmission=1`, tinted `KHR_materials_volume`
attenuation) renders visible refraction bending, depth-dependent tinted
absorption, and no light leaking through the shell, with the photon pass
depositing a different, non-zero count than the fixed demo scene
(confirming it's actually tracing the loaded asset); a hand-built chrome
cube (metallic 1.0, roughness 0.02, no transmission) also deposits
photons via the new mirror-threshold branch. `ctest` grew from 3 to 4
tests (`mesh_validate_test` added) and stays green; `assets/test.glb`
(an ordinary opaque textured asset, transmission defaults to 0) renders
pixel-unchanged and correctly does *not* build a photon pipeline.

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

**Material probe and Glass cup scenes wired into the Project registry**,
replacing their stub `configure` callbacks. Both are pure-SDF scenes with
no loaded mesh: Glass cup bakes `makeGlassCupSdf` into `source.sdf` in
`bakeStepRepresentation` (ahead of `bakeStepConstructRenderer`, unlike the
mesh-derived Voxel/SDF paths which still bake at Representation time off an
already-loaded mesh) and sets `source.groundOffset` to 20% of the cup's own
outer radius so the floor caustic has room to converge below the bowl.
Material probe bakes 9 `makeMaterialProbeSdfs` grids into `source.extraSdf`
against the same unmodified built-in base scene `wantCup`'s ceramic bowl
already layers onto — nothing new to build there. The analytic sun
(`RenderSettings::sunEnabled/sunDirection/sunAngularRadiusDeg/sunRadiance`)
reuses the existing procedural-sky `skySunElevationDeg`/`skySunAzimuthDeg`
fields instead of growing a second pair of angle sliders — moving either
slider now moves the sky glow (next bake) and the analytic disk (live,
`syncSunRenderSettings`) off the same numbers, so they can't drift apart by
construction, which is the exact failure the design doc's "moving the sun
moves both together" decision was warning against. `bakeStepEnvironment`
now bakes the procedural sky with `bakeSunDisk = !state.sunEnabled` so the
two suns don't double-count.

`realism: the sun UI is one pair of angle fields shared by
the procedural-sky bake and the live analytic disk, rather than two
independently adjustable lights — an artist wanting the sun aimed one way
and the sky glow's own baked-in sun disk aimed another can't currently do
that; also means the analytic sun's angle is only reachable from the
Settings panel while Procedural Sky is the selected HDRI, since that's the
only place skySunElevationDeg/skySunAzimuthDeg are drawn. Both scenes ship
with Procedural Sky selected by default so this doesn't block either one.`

Also found and fixed a real crash while verifying Glass cup end-to-end
(render it, don't just reason about it, again): `__closesthit__radiance`'s
`MATERIAL_SDF` branch fell back to `sdfNearestCellMaterial` — which reads
`rt->sdfBaseColor/sdfMetallic/sdfRoughness` — whenever neither near-side
palette sample came back dielectric, with no check that those buffers
exist. They don't, for any *palette*-baked grid (`bakeAnalyticSdfPalette`
only fills `branch`/`palette`, not the legacy per-cell `baseColor`
array) — `buildSdfObject` only uploads `sdfBaseColor` et al. when
`grid.baseColor` is non-empty. Every previously-shipped primary-SDF or
extra-SDF scene used the non-palette bake (`makeCupSdf`/mesh-baked SDF),
so this null-pointer path was never reachable before Glass cup's
dielectric palette grid became the first *primary* `source.sdf` with
palette data and a two-branch dielectric material, and it's hit constantly
at the shell's silhouette (both bias samples land in "air" at grazing
angles). Fixed at the call site: when `rt->sdfBranch` is set (a palette
grid), use the palette material's own `baseColor` instead of the
legacy per-cell lookup; the legacy path is unchanged for non-palette
grids. No test covers OptiX device code (no GPU in CI), so this was only
catchable by actually rendering the new scene, which is what surfaced it.

### Deliberate departures (markers)

realism: src/render/procedural_sky.cpp — the real sun subtends ~0.25°,
  too small at these equirect resolutions to sample cleanly; the disk is
  widened and its brightness is a flat artist-tuned boost, not physical
  solar irradiance.
realism: src/render/procedural_sky.cpp — Preetham's zenith luminance
  (kcd/m^2, for a display tone-mapping pipeline) is replaced by a by-eye
  match to this renderer's scene-linear HDRI scale, not a radiometric
  conversion.
claudia: src/render/optix_renderer.cpp — still an unbenchmarked guess,
  not a retuned value; benchmark frame time across sample counts and
  retune before trusting this for anything perf-sensitive.
claudia: src/render/optix_renderer.h — single-backend seam, one concrete
  OptixRenderer class not a generic multi-backend interface split;
  upgrade to a real src/rhi/ if a second backend starts (see CLAUDE.md
  seam section).
realism: src/render/optix_renderer.h — clamps a single sample's radiance
  to suppress fireflies at the cost of a little highlight energy; <= 0
  disables.
realism: src/render/kernels/pathtracer_params.h — splat colorDC is
  treated as Lambertian albedo relit by the scene, not rendered unlit as
  trained; double-counts capture lighting, but there's no unlit material
  path and unlit splats look flat beside path-traced geometry.
realism: src/render/kernels/pathtracer_params.h — ceiling on a single
  sample's radiance, applied before accumulation (see clampFirefly in
  pathtracer.cu); trades highlight energy for no permanent white specks;
  <= 0 disables.
claudia: src/render/kernels/pathtracer_params.h — scoped down from the
  plan's full Georgiev et al. 2012 combined-MIS weight to a
  radius-partition scheme; revisit if VCM needs the full recursive
  pdf-history accumulators.
realism: src/render/kernels/pathtracer.cu — caps a single sample's
  radiance against fireflies; scales by max-component to keep hue;
  doctrine rung 1 over rung 3, params.fireflyClamp <= 0 disables.
claudia: src/render/gl_ext.h — hand-resolves 4 GL function pointers
  instead of pulling in a full loader (GLAD/GLEW); upgrade to one if
  later phases need much more of modern GL than this.
claudia: src/io/gltf_loader.cpp — only FLOAT-componentType accessors are
  read; normalized-integer and sparse accessors are detected and reported
  as absent (caller falls back), not misread; upgrade if a real asset
  needs them.
realism: src/io/gsplat_asset.h — only the SH DC term (view-independent
  color) is kept, f_rest_* view-dependent bands dropped; flat color reads
  close enough at single-object splat density and avoids a per-hit SH
  kernel.
claudia: src/io/mesh_primitive_classify.cpp — PCA oriented frame; a
  cube's isotropic covariance can give an arbitrary frame the Box test
  misses; upgrade to a brute-force box-orientation fit if real assets
  ship rotated prims this rejects.
claudia: src/convert/sdf_baker.cpp — builds/tears down its own
  OptixDeviceContext instead of sharing OptixRenderer's; baking is
  one-shot at load time; revisit if baking goes interactive.
realism: src/convert/sdf_procedural.cpp — no per-material authoring tool
  for procedural SDF content yet, so a plausible warm off-white ceramic is
  hand-picked; an artist control (doctrine rung 1).
claudia: src/render/kernels/pathtracer.cu — __intersection__causticSplat
  duplicates __intersection__gsplat's local-frame ellipsoid quadratic
  (factored into a shared reportEllipsoidIntersection helper) rather than
  parameterizing one kernel over both; the two anyhit accept rules differ
  enough (stochastic alpha-composite for opacity vs. deterministic
  always-ignore weighted accumulation for caustic density estimation)
  that a shared entry point would need a runtime mode branch on the hot
  path; revisit only if a third ellipsoid-intersection use case appears.
claudia: src/render/kernels/pathtracer.cu — accumAlbedoBuffer/
  accumNormalBuffer (OptiX AI Denoiser guide layers) hold the first
  sample's first-hit values per pixel per subframe, written verbatim every
  subframe with no lerp/average against the previous subframe, unlike
  accumBuffer; don't read them as a progressively-converging buffer. The
  miss-case normal (-rayDirection) is a synthesized sentinel for
  background pixels, not a physical surface normal.
claudia: src/render/kernels/pathtracer.cu — NanoVDB volumetric medium
  (Phase 2) uses a single global-max-density majorant (grid stats'
  `tree().root().maximum()` times sigma_t times the artist density-scale
  slider) for both the primary Woodcock/delta-tracking random walk and the
  shadow-ray ratio-tracking transmittance estimator. This is unbiased —
  every rejected null-collision is exactly compensated by the free-flight
  pdf — but slower than it needs to be: a uniform-density fog sphere has
  no empty space to skip, so the majorant is tight everywhere, but a
  sparser/wispier grid would waste most candidate collisions on nearly-air
  regions this global bound can't distinguish from dense ones. NanoVDB
  ships per-leaf/per-tile min/max stats specifically to tighten the
  majorant locally (and its HDDA traversal to skip empty leaves entirely);
  wiring that in is a natural, deliberately-deferred fast-follow once a
  non-trivial (not analytically-uniform) density field is authored — no
  correctness change, purely a null-collision-rate/performance win.
realism: src/render/kernels/pathtracer.cu — the medium's extinction
  `sigma_t` is tracked as a single scalar (the mean of the three
  channels), not per-channel/spectral; using different sigma_t per RGB
  channel would need a hero-wavelength or spectral-MIS scheme to stay
  unbiased, which is out of scope for the first volumetric primitive.
  Color in the fog comes entirely from `scatterAlbedo` (applied via an
  unbiased albedo/albedoAvg ratio at each real scattering event, the same
  trick used for the continuation throughput) — a grey-extinction,
  colored-single-scatter-albedo medium is a common, physically legitimate
  simplification (real smoke/steam skews this way already), not a fudge
  of the free-flight estimator itself.
realism: src/render/kernels/pathtracer.cu — light reaching a diffuse/
  glass/etc. surface through the volume is ratio-traced
  (`mediumTransmittance`) in the three existing NEE branches (sun/env/quad
  light), but the bidirectional light-subpath/vertex-merging pass has no
  hit-group for NanoVDB's custom-primitive AABB and is disabled outright
  whenever an Nvdb object is in the scene (same pre-existing guard as
  Voxel/Gsplat) — caustics-through-fog isn't part of this phase.
claudia: src/render/optix_renderer.h — `SceneSource::volume` is a single
  `const NvdbVolume *`, not a vector like `extraSdf`, even though the
  plan's stated pattern was the plural-list one; end-to-end delta
  tracking/NEE transmittance only reads one active medium
  (`Params::volume`), so a vector would have implied multi-volume support
  that isn't actually wired up. `GeometryKind::Nvdb`/`buildNvdbObject` can
  add more than one medium *object* to the GAS list today, but only the
  most recently built one's parameters end up in `Params`; upgrade to a
  real multi-volume `Params` array (and a compositing rule for overlapping
  media) if a scene ever needs more than one fog volume at once.
claudia: src/io/nvdb_loader.cpp — the plan's Phase 2 spec asked for the
  `cudaMalloc`/`cudaMemcpy` device upload to live in this file, but
  CLAUDE.md's OptiX seam rule is explicit that `src/io/` and `src/app/`
  never include `optix.h`/`cuda_runtime.h` — only `src/render/` and
  `src/convert/sdf_baker.cpp` are sanctioned CUDA/OptiX call sites. Kept
  the seam: this loader builds the NanoVDB grid entirely on the host
  (`nanovdb::tools::createFogVolumeSphere`) and returns the raw serialized
  bytes in a `std::vector<uint8_t>`; `OptixRenderer::Impl::buildNvdbObject`
  (src/render/optix_renderer.cpp) does the actual device upload, the same
  division of labor `VoxelGrid`/`SdfGrid`/`GsplatAsset` already use.
realism: src/render/kernels/pathtracer.cu, src/render/optix_renderer.{h,cpp}
  — Phase 3 adds RTXDI's *spatial-only* resampled importance sampling
  (RIS) idea to surface NEE, reimplemented directly from the published
  weighted-reservoir-sampling math (Talbot/Bitterli), not the RTXDI SDK.
  This is explicitly not RTXDI-the-SDK: RTXDI assumes a fixed-frame-budget
  G-buffer renderer that reuses reservoirs *across rendered frames* via
  motion-vector reprojection, and italy has none of that (no G-buffer, no
  motion vectors, a static-camera progressive accumulator, not a
  real-time many-bounce-many-light renderer). Carrying a reservoir across
  subframes into the progressively-averaged `accumBuffer` would
  double-count samples the same way naive photon-radius reuse would, so
  there is zero temporal reservoir carry-over — `__raygen__reservoirBuild`
  reruns from scratch every subframe. What's kept is the *spatial* half:
  a per-subframe build pass draws `kReservoirCandidates` (6) candidate
  light samples per pixel (same sun/env/quad strategies as plain NEE),
  weights them via RIS, then the main pass mixes in `kReservoirSpatialNeighbors`
  (4) neighbor pixels' reservoirs from the *same* subframe launch before
  consuming the winning sample through the existing NEE/`traceOcclusion`/
  `mediumTransmittance`/BSDF machinery unchanged — only which light
  sample gets shadow-tested changes. Gated behind `RenderSettings::reservoirNEE`
  (default false, `ITALY_RESERVOIR_NEE` env var), matching the
  `lightSubpaths` on/off precedent.
claudia: src/render/kernels/pathtracer.cu — RIS/reservoir reuse is scoped
  to the primary-hit surface NEE only (`depth == 0` in the diffuse-family
  branch of `__closesthit__radiance`); it deliberately does not extend to
  the NanoVDB volume's own NEE branch, even though Phase 2's multi-light-
  in-fog scenario is the stated motivation for Phase 3. Reason: the
  volume branch's scatter position is itself resampled by Woodcock/delta
  tracking on every trace, so the reservoir-build pass's scatter point and
  the main pass's independently-sampled scatter point would generally be
  different points in the medium — reusing a reservoir built at one
  stochastic scatter location to shade a different one breaks the
  domain-consistency the RIS math depends on (the same failure mode this
  phase's `evalLightSampleAtPoint`/`combineReservoirs` machinery exists to
  avoid for *surface* points). Making that sound would mean caching or
  otherwise sharing a single deterministic scatter position per pixel per
  subframe between the build and shading passes — a bigger structural
  change than this phase's scope. The volume's own quad-light NEE was
  still generalized to `pickQuadLightUniform()` so the many-light test
  scene lights it correctly; it just isn't RIS-resampled. Upgrade trigger:
  a scene where volume-NEE light-selection noise is visibly the dominant
  artifact (not yet observed — Phase 2's single-quad-light fog scene
  doesn't stress this).
claudia: src/render/optix_renderer.{h,cpp} — the "several simultaneous
  emitters" scene needed to demonstrate RIS's benefit (`RenderSettings::
  extraTestLightCount`, `ITALY_TEST_EXTRA_LIGHTS`) is implemented as
  purely virtual, non-intersectable quad lights (`Params::extraLights`,
  up to `kMaxExtraLights` = 3): NEE-only positions/normals/emission with
  no backing geometry or SBT hit-group entries. This sidesteps needing a
  general multi-light-geometry authoring feature (accel structure
  rebuilds, per-light HitGroupData, BSDF-ray-hits-a-light MIS bookkeeping
  for each) that the plan explicitly said not to build. It's not a
  compromise on physical correctness: a light with no geometry that can
  never be hit by a BSDF-sampled ray needs no MIS weight against a
  BSDF-sampling technique (there is no such technique for it), so its
  contribution is exactly `Le * BSDF * cos * V / pdf`, the same math sun/
  environment lighting already use in this renderer for the same reason.
  Upgrade to real emissive geometry only if a scene needs these lights to
  be directly visible/hittable, not just NEE sources.
realism: Phase 4 (RTXGI) — scoped out fully, no code. DDGI light-probe
  volumes amortize indirect lighting across many rendered frames of a
  moving real-time scene; italy already computes converged global
  illumination directly and unbiased over ~128 progressive subframes.
  Adopting probes would mean trading correctness italy already has for a
  performance win a progressive/offline renderer doesn't need — probe
  interpolation and light leaking are exactly the kind of transport bias
  the doctrine says not to introduce to save time nobody's asking to save.
  Phase 2's many-light-in-fog case, solved via Phase 3's unbiased spatial
  RIS instead of a biased probe cache, is itself supporting evidence: the
  actual variance problem didn't need this.
realism: Phase 4 (NRD) — scoped out fully, no code. NRD (ReBLUR/ReLAX/
  SIGMA) is built for 1-spp-or-fewer real-time ray tracing, denoising via
  heavy temporal accumulation over motion vectors and disocclusion
  handling. italy's actual scenario is static-camera progressive
  denoising with no per-frame motion history to hand it in the first
  place; the OptiX AI Denoiser already integrated pre-Phase-1, now with
  Phase 1's albedo/normal guide layers, is architecturally the correct
  tool for that scenario already.
realism: Phase 4 (DLSS Frame Generation) — scoped out fully, no exception.
  Interpolates frames for perceived-motion smoothing in a bounded-frame-
  time real-time context; italy has no frame-rate target at all.
claudia: Phase 4 (DLSS Super-Resolution) — the one plausible fit
  (upscaling the live camera-navigation preview before convergence,
  decoupled from the full-res converged accumulation buffer), evaluated
  and deliberately deferred rather than implemented. No evidence in the
  codebase or from Phases 1-3's work that native-resolution navigation
  preview is actually slow on target hardware (RTX 4080 Laptop per
  README); pulling in a third vendor SDK (its own model weights, runtime
  dependency footprint) to solve an unverified performance problem is the
  speculative-building this project's own plan doctrine warns against —
  same standard already applied to RTXDI before Phase 3, now applied here.
  Upgrade trigger: navigation empirically shown to be sluggish, and even
  then try a zero-dependency bilinear low-res preview first before
  reaching for DLSS-SR specifically.
