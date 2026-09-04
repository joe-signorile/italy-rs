// One-shot SDF bake: per cell, unsigned distance is the min hit distance over fixed sample directions, sign is ray-parity counting (needs a watertight mesh). Separate module/pipeline from pathtracer.cu.

#include <optix.h>

#include "sdf_bake_params.h"

#include <sutil/cuda/random.h>
#include <sutil/vec_math.h>

extern "C" {
__constant__ BakeParams params;
}

static __forceinline__ __device__ float traceHitDistance(float3 origin, float3 dir, float tmax) {
  unsigned int p0 = __float_as_uint(-1.0f);
  optixTrace(params.meshHandle, origin, dir, 1e-4f, tmax, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
             p0);
  return __uint_as_float(p0);
}

extern "C" __global__ void __closesthit__bake() { optixSetPayload_0(__float_as_uint(optixGetRayTmax())); }

extern "C" __global__ void __miss__bake() {}

static __forceinline__ __device__ float3 randomSphereDirection(unsigned int &seed) {
  const float z = 1.0f - 2.0f * sutil::rnd(seed);
  const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  const float phi = 2.0f * M_PIf * sutil::rnd(seed);
  return make_float3(r * cosf(phi), r * sinf(phi), z);
}

extern "C" __global__ void __raygen__bake_sdf() {
  const uint3 idx = optixGetLaunchIndex();
  const float3 p = params.gridOrigin +
                    make_float3(static_cast<float>(idx.x) + 0.5f, static_cast<float>(idx.y) + 0.5f,
                                static_cast<float>(idx.z) + 0.5f) *
                        params.voxelSize;

  unsigned int seed = sutil::tea<4>((idx.z * params.ny + idx.y) * params.nx + idx.x, 0);
  const float tmax = 1e16f;
  constexpr int kNumDirs = 160;
  float minDist = 1e30f;
  for (int i = 0; i < kNumDirs; ++i) {
    const float t = traceHitDistance(p, randomSphereDirection(seed), tmax);
    if (t >= 0.0f && t < minDist)
      minDist = t;
  }
  if (minDist > 1e29f)
    minDist = params.voxelSize * 4.0f;
  else
    minDist *= 0.9f;

  constexpr int kParityDirs = 5;
  int insideVotes = 0;
  for (int v = 0; v < kParityDirs; ++v) {
    const float3 dir = v == 0 ? make_float3(0.0f, 0.0f, 1.0f) : randomSphereDirection(seed);
    int hitCount = 0;
    float3 origin = p;
    for (int i = 0; i < 64; ++i) {
      const float t = traceHitDistance(origin, dir, tmax);
      if (t < 0.0f)
        break;
      ++hitCount;
      origin = origin + dir * (t + 1e-4f);
    }
    if (hitCount % 2 == 1)
      ++insideVotes;
  }
  const float sign = insideVotes * 2 > kParityDirs ? -1.0f : 1.0f;

  const unsigned int cell = (idx.z * params.ny + idx.y) * params.nx + idx.x;
  params.output[cell] = sign * minDist;
}
