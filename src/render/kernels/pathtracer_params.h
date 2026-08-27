// Shared between host (optix_renderer.cpp) and device (pathtracer.cu) code.
// Keep this CUDA/OptiX-only (no glm, no C++ STL) since it's included by nvcc
// when compiling the device program too.
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

enum MaterialType : unsigned int {
  MATERIAL_DIFFUSE = 0,
  MATERIAL_MIRROR = 1,
  MATERIAL_GLASS = 2,
  MATERIAL_LIGHT = 3,
  // Diffuse BRDF (same NEE/MIS/bounce logic as MATERIAL_DIFFUSE) but the
  // albedo comes from a texture sampled with the hit triangle's interpolated
  // UV instead of a constant — used for phase-3 GLB-loaded meshes.
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

struct Params {
  unsigned int subframeIndex;
  float4 *accumBuffer; // HDR accumulation, width*height, persists across subframes
  uchar4 *frameBuffer; // tonemapped display output (CUDA-GL interop PBO pointer)
  unsigned int width;
  unsigned int height;
  unsigned int samplesPerLaunch;

  float3 eye, U, V, W; // camera basis, W points along view direction (not normalized: encodes FOV)

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
  cudaTextureObject_t baseColorTex = 0; // 0 => no texture, use albedo as a flat color

  // Only used when materialType == MATERIAL_VOXEL, indexed by
  // optixGetPrimitiveIndex(): the intersection program does its own ray/box
  // test against voxelAabbs (OptiX's custom-primitive build input only feeds
  // the BVH build, not the intersection program), and voxelColors gives that
  // voxel's baked color.
  OptixAabb *voxelAabbs = nullptr;
  float3 *voxelColors = nullptr;

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
