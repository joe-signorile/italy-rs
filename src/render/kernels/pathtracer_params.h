// Shared between host (optix_renderer.cpp) and device (pathtracer.cu) code.
// Keep this CUDA/OptiX-only (no glm, no C++ STL) since it's included by nvcc
// when compiling the device program too.
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

// Threshold for treating a GGX metallic-roughness triangle as "mirror-like
// enough" to carry a caustic photon bounce (see __closesthit__photon's
// MATERIAL_TEXTURED_DIFFUSE branch in pathtracer.cu). Also used host-side by
// optix_renderer.cpp to decide up front whether a loaded mesh has anything
// that could seed a caustic at all — shared here so the two checks can't
// drift apart.
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
  // matching branch in __closesthit__photon. A mirror-like metallic/rough
  // combo (kCausticMirrorMetallic/kCausticMirrorRoughness above) gets the
  // same caustic-eligibility treatment without needing transmission at all.
  // Voxel/SDF scenes still can't (no per-primitive material to carry either
  // property) — see enablePhotonMapping's comment in optix_renderer.cpp.
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

// A single deposited caustic photon (see phase-7 comment on Params below).
// `direction` is the direction the photon was traveling when it hit the
// diffuse surface (needed by the gather step's cosine term), not a
// reflection/half-vector — matches how radiance's NEE branch uses `L`.
struct Photon {
  float3 position;
  float3 direction;
  float3 power;
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

  // Scene bounding sphere — used by environment-based photon emission
  // (__raygen__photon) to place an emission origin outside the geometry when
  // there's no quad light to emit from. Unrelated to the camera. No default
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

  // Caustics (phase 7): a global-radius progressive photon map — the
  // original Hachisuka/Ogaki/Jensen 2008 PPM formulation (one shared radius,
  // shrunk each pass via R_{i+1} = R_i * sqrt((i+alpha)/(i+1))), not the
  // later per-visible-point Stochastic PPM (2009) refinement — see
  // buildPhotonPipeline()'s comment in optix_renderer.cpp for why. Only
  // built for the fixed bring-up scene (the only one with specular objects
  // to seed a caustic from); photonHandle == 0 means "no photon map,
  // gather is a no-op," so every other scene renders exactly as it did
  // before this phase.
  OptixTraversableHandle photonHandle;
  Photon *photons;                // capacity photonCapacity; valid entries: min(*photonCounter, photonCapacity)
  unsigned int *photonCounter;    // atomic append index, host resets to 0 before each photon-emission launch
  unsigned int photonCapacity;
  unsigned int photonBatchSize;   // photons emitted per pass — the photon raygen's launch width
  float photonGatherRadius;
  unsigned int totalPhotonsEmitted;
  // The gather trace targets photonHandle directly (a bare GAS, no IAS/
  // instance wrapping — there's only one build input, photons-as-spheres),
  // so its SBT hit-group index is *only* the SBTOffset argument passed to
  // that optixTrace call (no instance.sbtOffset to add, unlike every other
  // trace call in this file which goes through params.handle's per-object
  // instances). That argument therefore has to skip past the scene's own
  // per-object hit-group records — hardcoding a literal here would silently
  // collide with whichever scene object happens to land on that index, so
  // the host computes and passes the right value (objects.size()) instead.
  unsigned int gatherHitSbtOffset;
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
};
