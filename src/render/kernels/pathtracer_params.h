// Shared between host (optix_renderer.cpp) and device (pathtracer.cu). Keep CUDA/OptiX-only (no glm, no STL) — nvcc includes it when compiling the device program.
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

inline constexpr float kCausticMirrorMetallic = 0.9f;
inline constexpr float kCausticMirrorRoughness = 0.1f;

enum MaterialType : unsigned int {
  MATERIAL_DIFFUSE = 0,
  MATERIAL_MIRROR = 1,
  MATERIAL_GLASS = 2,
  MATERIAL_LIGHT = 3,
  MATERIAL_TEXTURED_DIFFUSE = 4,
  MATERIAL_VOXEL = 5,
  MATERIAL_SDF = 6,
  MATERIAL_GSPLAT = 7,
};

struct QuadLight {
  float3 corner;
  float3 v1, v2;
  float3 normal;
  float3 emission;
};

enum SdfMaterialKindGpu : unsigned int {
  SDF_MATERIAL_OPAQUE = 0,
  SDF_MATERIAL_DIELECTRIC = 1,
};

struct SdfGpuMaterial {
  unsigned int kind;
  float3 baseColor;
  float metallic, roughness;
  float ior;
  float3 extinction;
};

struct SunLight {
  float3 direction;
  float cosAngularRadius;
  float3 radiance;
  unsigned int enabled;
};

struct LightVertex {
  float3 position;
  float3 normal;
  float3 direction;
  float3 throughput;
  float3 baseColorFactor;
  float metallic;
  float roughness;
};

struct Params {
  unsigned int subframeIndex;
  float4 *accumBuffer;
  uchar4 *frameBuffer;
  unsigned int width;
  unsigned int height;
  unsigned int samplesPerLaunch;
  float exposure;

  float fireflyClamp;

  unsigned int tonemapOperator;

  unsigned int denoiserEnabled;
  float4 *denoisedBuffer;

  float3 sceneBoundsCenter;
  float sceneBoundsRadius;

  float3 eye, U, V, W;

  float aperture;
  float focusDistance;

  float envRotation;

  QuadLight light;
  OptixTraversableHandle handle;

  cudaTextureObject_t envTex;
  float *envMarginalCdf;
  float *envConditionalCdf;
  int envWidth;
  int envHeight;

  LightVertex *lightVertices;
  unsigned int *lightVertexCounter;
  unsigned int lightVertexCapacity;
  unsigned int lightSubpathBatchSize;
  unsigned int totalLightPathsEmitted;
  unsigned int lightVertexCount;

  OptixTraversableHandle vertexMergeHandle;
  unsigned int mergeHitSbtOffset;

  float mergeRadius;

  unsigned int maxConnectionsPerVertex;

  float3 backgroundColor;
  SunLight sun;
};

struct GpuMaterial {
  float3 baseColorFactor;
  float metallic;
  float roughness;
  float normalScale;
  cudaTextureObject_t baseColorTex;
  cudaTextureObject_t metallicRoughnessTex;
  cudaTextureObject_t normalTex;

  float transmission;
  float ior;
  float3 attenuationColor;
  float attenuationDistance;
};

struct RayGenData {};

struct MissData {};

struct HitGroupData {
  unsigned int materialType;
  float3 albedo;
  float3 emission;
  float ior;

  float3 *normals = nullptr;
  float2 *uvs = nullptr;
  float4 *tangents = nullptr;
  unsigned int *triangleMaterial = nullptr;
  GpuMaterial *materials = nullptr;

  OptixAabb *voxelAabbs = nullptr;
  float3 *voxelColors = nullptr;

  float3 sphereCenter{};
  float sphereRadius = 0.0f;

  float3 extinction{};

  float *sdfDistances = nullptr;
  float3 sdfOrigin{};
  float sdfVoxelSize = 1.0f;
  int sdfNx = 0, sdfNy = 0, sdfNz = 0;
  float3 *sdfBaseColor = nullptr;
  float *sdfMetallic = nullptr;
  float *sdfRoughness = nullptr;
  unsigned char *sdfBranch = nullptr;
  SdfGpuMaterial *sdfPalette = nullptr;
  int sdfPaletteCount = 0;

  float3 *splatPositions = nullptr;
  float3 *splatScales = nullptr;
  float4 *splatRotations = nullptr;
  float *splatOpacity = nullptr;
  float3 *splatColors = nullptr;
};
