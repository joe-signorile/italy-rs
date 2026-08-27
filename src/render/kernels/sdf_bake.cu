// One-shot SDF bake: for each grid cell, estimate unsigned distance to the
// mesh surface as the minimum hit distance over a fixed set of sample
// directions (OptiX has no native "closest point" query — this is the
// standard ray-tracing-hardware substitute, and it reuses the mesh's own
// BVH exactly as the design doc calls for), and sign via ray-parity
// counting along a fixed direction (requires a closed/watertight mesh).
//
// Separate module/pipeline from pathtracer.cu on purpose: baking is a 3D
// grid launch with nothing in common with the interactive path tracer's
// program shape, so sharing its Params/SBT would only add coupling.

#include <optix.h>

#include "sdf_bake_params.h"

#include <sutil/cuda/random.h>
#include <sutil/vec_math.h>

extern "C" {
__constant__ BakeParams params;
}

static __forceinline__ __device__ float traceHitDistance(float3 origin, float3 dir, float tmax) {
  unsigned int p0 = __float_as_uint(-1.0f); // sentinel: no hit
  optixTrace(params.meshHandle, origin, dir, 1e-4f, tmax, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
             p0);
  return __uint_as_float(p0);
}

extern "C" __global__ void __closesthit__bake() { optixSetPayload_0(__float_as_uint(optixGetRayTmax())); }

extern "C" __global__ void __miss__bake() { /* leave the sentinel payload untouched */ }

// Uniform-on-sphere via the standard z/phi formula, but with *independent*
// random numbers per call rather than a deterministic index — see the
// raygen's comment for why that matters here.
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

  // "Minimum hit distance over N sample directions" is a biased estimator of
  // true closest-point distance. A *fixed* direction set (e.g. a Fibonacci
  // sphere) reused identically at every cell was tried first and rejected:
  // its error is spatially correlated — worst for large flat faces, where
  // only directions landing close to perpendicular find the true nearest
  // point — and shows up as visible banding/terracing rather than grain.
  // Even jittering each fixed direction within a small cone didn't fix it,
  // because the jitter radius was smaller than the gaps *between* directions
  // in the base pattern, so it couldn't reach any direction the base pattern
  // didn't already almost cover. Independent random directions per sample
  // (no shared base pattern at all) converts the same bias into ordinary
  // noise instead, since neighboring cells no longer share any structure to
  // be correlated through.
  unsigned int seed = sutil::tea<4>((idx.z * params.ny + idx.y) * params.nx + idx.x, 0);
  const float tmax = 1e16f;
  constexpr int kNumDirs = 160;
  float minDist = 1e30f;
  for (int i = 0; i < kNumDirs; ++i) {
    const float t = traceHitDistance(p, randomSphereDirection(seed), tmax);
    if (t >= 0.0f && t < minDist)
      minDist = t;
  }
  // Only happens if every one of the kNumDirs sampled rays missed the mesh
  // entirely (e.g. a large hole, or a point far outside a small mesh) —
  // clamp to something finite rather than propagating a huge/garbage value
  // into the grid.
  if (minDist > 1e29f)
    minDist = params.voxelSize * 4.0f;
  else
    // A small safety margin: the estimator above can *overestimate* the
    // true distance (if no sampled direction, even jittered, comes close to
    // the actual nearest point), and sphere tracing needs a conservative
    // (not-too-large) distance to avoid stepping past thin/close features.
    // Shrinking slightly costs a few extra marching steps, which is cheap;
    // overshooting costs a hole in the surface, which is not.
    minDist *= 0.9f;

  // Sign via ray-parity majority vote across several directions, not just
  // one fixed axis. A single direction is the textbook description of this
  // technique, but it's fragile against imperfectly watertight/manifold
  // source geometry (e.g. an asset assembled from multiple overlapping
  // primitive pieces rather than one clean union) — a ray that grazes a
  // seam between two coincident-ish faces can flip its odd/even count from
  // what a neighboring cell's ray got, producing sharp, structured band
  // artifacts in the sign field rather than the smooth noise a distance
  // estimation error would produce. (This is what was actually happening
  // here — changing the *distance* sampling method repeatedly didn't move
  // the artifact at all, because it lives in this separate sign computation.)
  // Voting across independent directions and taking the majority is much
  // more robust to a single bad direction's seam-grazing.
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
