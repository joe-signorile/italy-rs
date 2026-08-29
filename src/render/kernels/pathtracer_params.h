// Shared between host (optix_renderer.cpp) and device (pathtracer.cu) code.
// Keep this CUDA/OptiX-only (no glm, no C++ STL) since it's included by nvcc
// when compiling the device program too.
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

// Threshold for treating a GGX metallic-roughness triangle as "mirror-like
// enough" to carry a delta (specular) light-subpath bounce rather than a
// connectable one — see __closesthit__lightSubpath's MATERIAL_TEXTURED_
// DIFFUSE branch in pathtracer.cu. Also used host-side by optix_renderer.cpp
// to decide up front whether a loaded mesh has anything that could seed a
// light subpath at all — shared here so the two checks can't drift apart.
//
// VCM Step 1 (see /home/joe/.claude/plans/what-is-the-next-wise-flurry.md)
// reads this the same threshold as before but with the opposite meaning at
// the point of use: what used to gate "specular enough to deposit a caustic
// photon after" now gates "specular enough that a light-subpath vertex must
// NOT be stored/connected here at all" (delta BSDFs have no finite pdf, so
// Georgiev's own VCM formulation excludes them from connection eligibility
// the same way). The eye side (__closesthit__radiance's BDPT connection
// loop) never needs to consult this threshold directly — it just never
// finds a vertex to connect to at a delta surface, since none was ever
// stored there.
inline constexpr float kCausticMirrorMetallic = 0.9f;
inline constexpr float kCausticMirrorRoughness = 0.1f;

enum MaterialType : unsigned int {
  MATERIAL_DIFFUSE = 0,
  MATERIAL_MIRROR = 1,
  MATERIAL_GLASS = 2,
  MATERIAL_LIGHT = 3,
  // Lambert diffuse + GGX metallic-roughness specular (see ShadingMaterial/
  // evalBsdf/sampleBsdf in pathtracer.cu), textured per glTF material —
  // originally phase-3's flat-textured-diffuse-only material, extended in
  // the GGX pass. albedo/uvs/normals/tangents/materials/triangleMaterial on
  // HitGroupData carry everything a triangle needs to look itself up.
  //
  // Gained a real transmission/dielectric variant once the watertight-mesh
  // import gate (mesh_validate.h) made "this triangle mesh has a
  // well-defined interior" a guarantee rather than a hope: a per-triangle
  // KHR_materials_transmission/ior/volume material (GpuMaterial's
  // transmission/ior/attenuation* fields) refracts through the mesh the
  // same way MATERIAL_GLASS refracts through a solid sphere — see the
  // merged dielectric branch at the end of __closesthit__radiance and the
  // matching branch in __closesthit__lightSubpath. A mirror-like
  // metallic/rough combo (kCausticMirrorMetallic/kCausticMirrorRoughness
  // above) gets the same delta-bounce treatment without needing transmission
  // at all. Voxel/SDF scenes still can't (no per-primitive material to carry
  // either property) — see enableLightSubpaths' comment in optix_renderer.cpp.
  MATERIAL_TEXTURED_DIFFUSE = 4,
  // Diffuse BRDF, custom-AABB voxel primitive; albedo is a per-voxel baked
  // color, shading normal comes from which of the 6 box faces was entered
  // (see __intersection__voxel) — phase-4 voxel-resampled meshes.
  MATERIAL_VOXEL = 5,
  // Diffuse BRDF, single custom-AABB primitive covering the whole SDF grid's
  // bounding box; __intersection__sdf sphere-traces through a trilinearly
  // sampled distance field, shading normal is the field's gradient — phase-5
  // SDF-resampled meshes. Albedo is a single flat tint (the source mesh's
  // baseColorFactor) — no per-surface-point color field is baked, unlike the
  // voxel path; see sdf_baker.h for why.
  MATERIAL_SDF = 6,
  // Custom-AABB-per-splat primitive (one 3-sigma ellipsoid bound per splat,
  // see gsplat_bounds.h) — roadmap phase 3, imported 3D Gaussian Splatting
  // `.ply` scenes (gsplat_ply_loader.h). __intersection__gsplat solves a
  // ray/ellipsoid test per splat; __anyhit__gsplat then stochastically
  // accepts or rejects that candidate against the splat's actual Gaussian
  // density (russian-roulette alpha test, unbiased in expectation) so
  // overlapping splats composite correctly with no depth sort. Diffuse BRDF
  // only, like MATERIAL_VOXEL/MATERIAL_SDF — see HitGroupData::splatColors.
  //
  // realism: splat color (colorDC) is treated as a Lambertian diffuse
  // albedo and relit by the scene's real light (HDRI/quad light), not
  // rendered unlit as trained. A trained splat's color already bakes in its
  // capture's original lighting, so this double-counts illumination — but
  // this renderer has no unlit/emissive-facsimile material path, and an
  // unlit splat would look flat and unshaded next to path-traced geometry
  // under a different HDRI. Good enough for "import and light like
  // everything else"; a dedicated unlit splat mode is future work if this
  // reads badly in practice.
  MATERIAL_GSPLAT = 7,
};

// A single rectangular area light — the phase 2..5 default. Superseded by
// environment/HDRI lighting when one is loaded (see Params::envTex below):
// the two aren't blended together, envTex!=0 means "ignore `light`, this
// scene has no quad light object at all." Multiple/arbitrary point/area
// lights alongside an environment are future work.
struct QuadLight {
  float3 corner;
  float3 v1, v2; // edge vectors from corner
  float3 normal; // must be normalize(cross(v1, v2)), pointing into the scene
  float3 emission;
};

// One stored light-subpath vertex — VCM Step 1's replacement for the old
// phase-7 `Photon` (see Params::lightVertices' doc comment below for the
// pipeline this feeds). Deposited by __closesthit__lightSubpath at every
// non-delta (diffuse) bounce of the light-subpath walk; mirror/glass/
// mirror-like-metallic bounces never reach here at all (kCausticMirrorMetallic/
// kCausticMirrorRoughness above gate that at the point of deposit), so every
// vertex in this buffer is guaranteed connectable.
struct LightVertex {
  float3 position;
  float3 normal; // shading normal at the vertex — needed to evaluate the light-side BSDF at connection time
  // Direction the subpath was traveling when it reached this vertex — same
  // convention Photon::direction used (not a reflection/half-vector).
  float3 direction;
  // Subpath throughput up to and including this vertex (renamed from
  // Photon::power for clarity: it's the light-path analogue of RadiancePRD's
  // `attenuation`, not a physical power/flux value on its own).
  float3 throughput;
  // claudia: Lambertian-diffuse-only for Step 1 — the BDPT connection eval
  // in __closesthit__radiance treats every stored vertex as a pure Lambert
  // BRDF (albedo/pi) regardless of what material it actually came from, so a
  // MATERIAL_TEXTURED_DIFFUSE vertex's GGX specular lobe is silently dropped
  // from the connection term (only its diffuse albedo is used, here and at
  // storage time). Full glossy/dielectric light-side BSDF support is Step
  // 3's job (see the VCM plan's sub-phase breakdown) — this field only ever
  // needs to be a diffuse albedo until then.
  float3 albedo;
};

struct Params {
  unsigned int subframeIndex;
  float4 *accumBuffer; // HDR accumulation, width*height, persists across subframes
  uchar4 *frameBuffer; // tonemapped display output (CUDA-GL interop PBO pointer)
  unsigned int width;
  unsigned int height;
  unsigned int samplesPerLaunch;
  float exposure; // linear multiplier applied just before tonemapping (phase 9 UI control)

  // realism: ceiling on any single path sample's radiance, in linear scene
  // units, applied before accumulation (see clampFirefly in pathtracer.cu).
  // Trades a little energy in the extreme highlights for the absence of
  // permanent white specks. <= 0 disables. Doctrine rung 1 over rung 3.
  float fireflyClamp;

  // Phase 8: which view transform applyTonemapAndQuantize() (pathtracer.cu)
  // uses to convert linear HDR into the displayed 8-bit sRGB image. Values
  // mirror italy::TonemapOperator (optix_renderer.h) by convention, not a
  // shared type — same "device-side enums stay device-side" pattern as
  // MaterialType above.
  unsigned int tonemapOperator;

  // Phase 10 denoiser: when nonzero, __raygen__rg skips its own
  // frameBuffer write (the noisy tonemap would just be overwritten anyway)
  // and optix_renderer.cpp instead denoises accumBuffer into denoisedBuffer,
  // then runs __raygen__tonemap to read *that* into frameBuffer. denoisedBuffer
  // is null and unused when denoiserEnabled is 0.
  unsigned int denoiserEnabled;
  float4 *denoisedBuffer;

  // Scene bounding sphere — used by environment-based light-subpath emission
  // (__raygen__lightSubpath) to place an emission origin outside the
  // geometry when there's no quad light to emit from. Unrelated to the
  // camera. No default
  // member initializers: Params is a __constant__ global on the device side,
  // which CUDA requires to be trivially constructible — every field here is
  // set explicitly host-side before upload (see optix_renderer.cpp's
  // `Params params{};` then field-by-field assignment), same as every other
  // field in this struct.
  float3 sceneBoundsCenter;
  float sceneBoundsRadius;

  // Camera basis. W is the unit view axis; U and V carry the FOV scale, so
  // d.x*U + d.y*V + W is the (unnormalized) ray through NDC point d.
  float3 eye, U, V, W;

  // Thin-lens depth of field. aperture is the lens radius in world units;
  // 0 is a pinhole. focusDistance is measured along W, so the plane in focus
  // is flat rather than spherical — which is what a real lens does.
  float aperture;
  float focusDistance;

  // Rotation of the environment about +Y, radians. Folded into the single
  // direction<->uv mapping shared by lookup, pdf evaluation and sampling, so
  // the three cannot disagree (they are compared directly in the MIS weight).
  float envRotation;

  QuadLight light;
  OptixTraversableHandle handle;

  // HDRI/environment lighting (phase 6). envTex == 0 means "no environment
  // loaded" — miss shader falls back to MissData::bgColor and NEE falls back
  // to the quad light above, so scenes built before this phase render
  // identically to how they always did.
  //
  // No in-class initializers here (unlike HitGroupData below): Params is
  // declared `__constant__` in pathtracer.cu, and nvcc rejects non-trivial
  // default member initializers on `__constant__` variables ("dynamic
  // initialization is not supported"). optix_renderer.cpp's `Params
  // params{};` value-initialization zeroes these the same way it already
  // does for every other field here.
  cudaTextureObject_t envTex;
  float *envMarginalCdf;    // height+1
  float *envConditionalCdf; // height*(width+1), row-major
  int envWidth;
  int envHeight;

  // VCM Step 1 — light-subpath storage + BDPT connections (see
  // /home/joe/.claude/plans/what-is-the-next-wise-flurry.md). Replaces the
  // old phase-7 single-bounce-photon/SPPM pipeline entirely:
  // __raygen__lightSubpath emits from the light/environment, and
  // __closesthit__lightSubpath appends a LightVertex to this plain array at
  // *every* non-delta bounce (diffuse included, not just after a specular
  // one). __closesthit__radiance iterates the array directly at every
  // diffuse eye vertex and attempts a BDPT connection (traceOcclusion() +
  // BSDF*BSDF*geometry-term contribution) to each stored vertex.
  //
  // Deliberately no acceleration structure over these vertices yet, and no
  // `vertexMergeHandle`/dVCM-dVC-dVM accumulator fields — that's vertex
  // *merging*, Step 2's job. Adding that machinery now, before it does
  // anything, would leave a phantom half-finished weight term for no
  // benefit; see the VCM plan's sub-phase breakdown.
  //
  // lightVertexCount == 0 means "no light subpaths this frame" (either the
  // scene has nothing connectable, or the caller zero-weighted it for an
  // A/B comparison — see RenderSettings::lightSubpaths) — the BDPT
  // connection loop is then a no-op, same "photonHandle == 0" convention the
  // old pipeline used, so every other scene renders exactly as it did
  // before this phase.
  LightVertex *lightVertices;         // capacity lightVertexCapacity, written by the emission pass
  unsigned int *lightVertexCounter;   // atomic append index, host resets to 0 before each emission launch
  unsigned int lightVertexCapacity;
  unsigned int lightSubpathBatchSize; // light subpaths emitted per pass — the light-subpath raygen's launch width
  unsigned int totalLightPathsEmitted; // RNG-seed decorrelation counter, same role as the old totalPhotonsEmitted
  // Vertices actually valid this frame: min(*lightVertexCounter, lightVertexCapacity)
  // as read back host-side after the emission pass — the main eye-path
  // launch reads this fixed scalar rather than dereferencing the (still
  // being-appended-to, from its perspective already-finalized) counter
  // itself, so every eye thread agrees on the same bound.
  unsigned int lightVertexCount;
};

// One glTF material, device side. Texture handles are 0 when absent.
// Mirrors italy::MaterialAsset (io/mesh_asset.h) without inheriting its
// std::vector/glm dependencies, the same "device-side types stay device-side"
// split MaterialType and TonemapOperator already use.
struct GpuMaterial {
  float3 baseColorFactor;
  float metallic;
  float roughness;
  float normalScale;
  cudaTextureObject_t baseColorTex;         // sRGB-decoded in texture hardware
  cudaTextureObject_t metallicRoughnessTex; // linear; glTF packs roughness in G, metallic in B
  cudaTextureObject_t normalTex;            // linear, tangent-space

  // KHR_materials_transmission/ior/volume — per-material, unlike
  // HitGroupData::ior below (that one is scoped to the single-object
  // MATERIAL_GLASS sphere; a glTF scene can mix opaque and transmissive
  // materials on one mesh, so this has to live wherever metallic/roughness
  // already do). 0 transmission means "ignore ior/attenuation entirely,
  // this material is opaque" — see the transmission branch in
  // MATERIAL_TEXTURED_DIFFUSE's closest-hit handling in pathtracer.cu.
  float transmission;
  float ior;
  float3 attenuationColor;
  float attenuationDistance;
};

struct RayGenData {};

struct MissData {
  float3 bgColor;
};

// Same record shape for every geometry type (triangle or built-in sphere) —
// geometry itself is queried on demand in the closest-hit program via
// optixGetTriangleVertexData()/optixGetSphereData(), so no vertex/geometry
// pointers need to live here.
struct HitGroupData {
  unsigned int materialType;
  float3 albedo;
  float3 emission;
  float ior; // only used when materialType == MATERIAL_GLASS

  // Only used when materialType == MATERIAL_TEXTURED_DIFFUSE: per-triangle
  // flattened (3 entries per triangle, indexed by primitiveIndex*3+corner —
  // matches the CPU-side MeshAsset layout, no index buffer needed on device).
  float3 *normals = nullptr;
  float2 *uvs = nullptr;
  float4 *tangents = nullptr; // xyz tangent, w bitangent sign; for normal mapping
  // A whole glTF scene is one GAS, so material identity is per triangle rather
  // than per SBT record: one hit-group record cannot describe 169 meshes with
  // different materials, and splitting into one record per primitive would
  // rebuild the SBT around the asset's authoring structure for no gain.
  unsigned int *triangleMaterial = nullptr;
  GpuMaterial *materials = nullptr;

  // Only used when materialType == MATERIAL_VOXEL, indexed by
  // optixGetPrimitiveIndex(): the intersection program does its own ray/box
  // test against voxelAabbs (OptiX's custom-primitive build input only feeds
  // the BVH build, not the intersection program), and voxelColors gives that
  // voxel's baked color.
  OptixAabb *voxelAabbs = nullptr;
  float3 *voxelColors = nullptr;

  // Only used by GeometryKind::SolidSphere objects — the custom-primitive
  // stand-in for a dielectric sphere. OptiX's *built-in* sphere primitive is
  // documented hollow (Programming Guide 9.1, 9.6 "Back-face culling": "if a
  // ray starts inside a sphere primitive ... it will not hit that
  // primitive"), so a refracted ray inside one never gets an exit hit and
  // the glass is a single interface rather than a solid. Dielectrics
  // therefore get their own intersector, and a custom IS cannot call
  // optixGetSphereData(), so the geometry has to travel through the SBT.
  // sphereRadius > 0 is what marks a record as a solid sphere.
  float3 sphereCenter{};
  float sphereRadius = 0.0f;

  // Beer-Lambert extinction, per channel, in units of 1/distance: an
  // interior segment of length t is attenuated by exp(-extinction * t).
  // This is what makes glass read as *glass* rather than as a clear shell —
  // thick parts absorb more than thin ones. Zero = perfectly clear.
  float3 extinction{};

  // Only used when materialType == MATERIAL_SDF: a dense flattened
  // (z*ny+y)*nx+x grid of signed distances, sphere-traced/gradient-shaded by
  // __intersection__sdf and __closesthit__radiance directly (no OptixAabb
  // array needed — there's exactly one primitive, the grid's own bbox,
  // computed from these fields).
  float *sdfDistances = nullptr;
  float3 sdfOrigin{};
  float sdfVoxelSize = 1.0f;
  int sdfNx = 0, sdfNy = 0, sdfNz = 0;

  // Only used when materialType == MATERIAL_GSPLAT, indexed by
  // optixGetPrimitiveIndex() — one custom AABB per splat feeds the BVH build
  // (see optix_renderer.cpp's buildSplatObject, freed after build, unlike
  // voxelAabbs above: the intersection program re-derives the ellipsoid from
  // position/scale/rotation directly rather than re-reading a box). scales
  // are world-space per-local-axis standard deviations (post-exp); rotations
  // are local-to-world quaternions (x,y,z,w); opacity is post-sigmoid.
  float3 *splatPositions = nullptr;
  float3 *splatScales = nullptr;
  float4 *splatRotations = nullptr;
  float *splatOpacity = nullptr;
  float3 *splatColors = nullptr;
};
