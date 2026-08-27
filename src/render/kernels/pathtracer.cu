// Phase 2 bring-up path tracer: unidirectional, next-event estimation toward
// a single quad light, multiple importance sampling (power heuristic)
// between light- and BSDF-sampling, Russian roulette termination, progressive
// accumulation. Diffuse/mirror/glass/light materials.
//
// Deliberately plain optixTrace (no Shader Execution Reordering) for this
// bring-up pass — SER (optixTraverse/optixReorder/optixInvoke, as NVIDIA's
// own optixPathTracer sample demonstrates) is a real performance win but is
// an optimization, not a correctness requirement; adding it is future work
// once the base pipeline is verified. See NVIDIA's OptiX SDK optixPathTracer
// sample for that pattern if/when this becomes a bottleneck.

#include <optix.h>

#include "device_onb.h"
#include "pathtracer_params.h"

#include <sutil/cuda/helpers.h>
#include <sutil/cuda/random.h>
#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void cosineSampleHemisphere(float u1, float u2, float3 &p) {
  const float r = sqrtf(u1);
  const float phi = 2.0f * M_PIf * u2;
  p.x = r * cosf(phi);
  p.y = r * sinf(phi);
  p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// Power heuristic (beta=2) for MIS between two sampling strategies.
static __forceinline__ __device__ float powerHeuristic(float pdfA, float pdfB) {
  const float a2 = pdfA * pdfA;
  const float b2 = pdfB * pdfB;
  return a2 / fmaxf(a2 + b2, 1e-8f);
}

// Standard vector refraction. `n` faces against `in` (dot(n,in) < 0). eta =
// ior_incident / ior_transmitted. Returns false on total internal reflection.
static __forceinline__ __device__ bool refractRay(const float3 &in, const float3 &n, float eta, float3 &out) {
  const float cosI = -dot(n, in);
  const float sin2T = eta * eta * (1.0f - cosI * cosI);
  if (sin2T > 1.0f)
    return false;
  const float cosT = sqrtf(1.0f - sin2T);
  out = eta * in + (eta * cosI - cosT) * n;
  return true;
}

// Schlick's approximation.
static __forceinline__ __device__ float schlickFresnel(float cosTheta, float ior) {
  float r0 = (1.0f - ior) / (1.0f + ior);
  r0 = r0 * r0;
  const float x = 1.0f - cosTheta;
  return r0 + (1.0f - r0) * x * x * x * x * x;
}

// ----------------------------------------------------------------------------
// HDRI/environment lighting (phase 6): equirectangular image, importance
// sampled via a piecewise-constant 2D distribution (PBRT-style: a marginal
// CDF over rows, a conditional CDF over columns within the sampled row).
// Direction<->(u,v) convention, used consistently by sampling, pdf
// evaluation, and radiance lookup alike:
//   theta = acos(dir.y) in [0,pi], v = theta/pi     (v=0 at +Y, v=1 at -Y)
//   phi   = atan2(dir.z, dir.x) in [-pi,pi], u = (phi+pi)/(2pi)
// Solid-angle pdf conversion: dOmega = sin(theta) dtheta dphi, and
// dtheta = pi*dv, dphi = 2pi*du, so pdf_solidangle = pdf_uv / (2 pi^2 sinTheta).
// ----------------------------------------------------------------------------

// PBRT-style binary search: cdf has n+1 entries (cdf[0]=0, cdf[n]=1); returns
// i in [0,n-1] such that cdf[i] <= u < cdf[i+1].
static __forceinline__ __device__ int findInterval(const float *cdf, int n, float u) {
  int first = 0, len = n;
  while (len > 0) {
    const int half = len >> 1;
    const int middle = first + half;
    if (cdf[middle + 1] <= u) {
      first = middle + 1;
      len -= half + 1;
    } else {
      len = half;
    }
  }
  return min(max(first, 0), n - 1);
}

static __forceinline__ __device__ float3 lookupEnvironmentRadiance(float3 dir) {
  const float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
  const float phi = atan2f(dir.z, dir.x);
  const float u = (phi + M_PIf) / (2.0f * M_PIf);
  const float v = theta / M_PIf;
  const float4 texel = tex2D<float4>(params.envTex, u, v);
  return make_float3(texel.x, texel.y, texel.z);
}

static __forceinline__ __device__ float evalEnvironmentPdf(float3 dir) {
  const float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
  const float phi = atan2f(dir.z, dir.x);
  const float u = (phi + M_PIf) / (2.0f * M_PIf);
  const float v = theta / M_PIf;
  const int h = params.envHeight, w = params.envWidth;
  const int row = min(max(static_cast<int>(v * h), 0), h - 1);
  const int col = min(max(static_cast<int>(u * w), 0), w - 1);
  const float rowPdf = (params.envMarginalCdf[row + 1] - params.envMarginalCdf[row]) * h;
  const float *rowCdf = params.envConditionalCdf + row * (w + 1);
  const float colPdf = (rowCdf[col + 1] - rowCdf[col]) * w;
  const float sinTheta = sinf(theta);
  return sinTheta > 1e-6f ? (rowPdf * colPdf) / (2.0f * M_PIf * M_PIf * sinTheta) : 0.0f;
}

// Importance-samples a direction favoring bright regions of the environment;
// pdfOut is in solid-angle measure, matching evalEnvironmentPdf's convention
// (needed since the two are compared directly in the MIS weight).
static __forceinline__ __device__ float3 sampleEnvironment(float u1, float u2, float &pdfOut) {
  const int h = params.envHeight, w = params.envWidth;
  const int row = findInterval(params.envMarginalCdf, h, u1);
  const float rowCdfLo = params.envMarginalCdf[row], rowCdfHi = params.envMarginalCdf[row + 1];
  const float rowPdf = (rowCdfHi - rowCdfLo) * h;
  const float dv = rowCdfHi > rowCdfLo ? (u1 - rowCdfLo) / (rowCdfHi - rowCdfLo) : 0.5f;
  const float v = (row + dv) / h;

  const float *rowCdf = params.envConditionalCdf + row * (w + 1);
  const int col = findInterval(rowCdf, w, u2);
  const float colCdfLo = rowCdf[col], colCdfHi = rowCdf[col + 1];
  const float colPdf = (colCdfHi - colCdfLo) * w;
  const float du = colCdfHi > colCdfLo ? (u2 - colCdfLo) / (colCdfHi - colCdfLo) : 0.5f;
  const float u = (col + du) / w;

  const float theta = v * M_PIf;
  const float phi = u * 2.0f * M_PIf - M_PIf;
  const float sinTheta = sinf(theta);
  pdfOut = sinTheta > 1e-6f ? (rowPdf * colPdf) / (2.0f * M_PIf * M_PIf * sinTheta) : 0.0f;

  return make_float3(sinTheta * cosf(phi), cosf(theta), sinTheta * sinf(phi));
}

// ----------------------------------------------------------------------------
// Payload: p0-2 attenuation, p3 seed, p4 depth, p5 prevBsdfPdf (bit-cast
// float; negative means the previous bounce was a delta/specular BSDF, so a
// direct light hit gets full weight rather than an MIS split), p6-8 emitted,
// p9-11 radiance (NEE result), p12-14 next origin, p15-17 next direction,
// p18 done.
// ----------------------------------------------------------------------------
struct RadiancePRD {
  float3 attenuation;
  unsigned int seed;
  int depth;
  float prevBsdfPdf;
  float3 emitted;
  float3 radiance;
  float3 origin;
  float3 direction;
  int done;
};

static __forceinline__ __device__ void trace(OptixTraversableHandle handle, float3 origin, float3 direction,
                                              float tmin, float tmax, RadiancePRD &prd) {
  unsigned int p[19] = {};
  p[0] = __float_as_uint(prd.attenuation.x);
  p[1] = __float_as_uint(prd.attenuation.y);
  p[2] = __float_as_uint(prd.attenuation.z);
  p[3] = prd.seed;
  p[4] = static_cast<unsigned int>(prd.depth);
  p[5] = __float_as_uint(prd.prevBsdfPdf);

  optixTrace(handle, origin, direction, tmin, tmax, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p[0],
             p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
             p[17], p[18]);

  prd.attenuation = make_float3(__uint_as_float(p[0]), __uint_as_float(p[1]), __uint_as_float(p[2]));
  prd.seed = p[3];
  prd.depth = static_cast<int>(p[4]);
  prd.prevBsdfPdf = __uint_as_float(p[5]);
  prd.emitted = make_float3(__uint_as_float(p[6]), __uint_as_float(p[7]), __uint_as_float(p[8]));
  prd.radiance = make_float3(__uint_as_float(p[9]), __uint_as_float(p[10]), __uint_as_float(p[11]));
  prd.origin = make_float3(__uint_as_float(p[12]), __uint_as_float(p[13]), __uint_as_float(p[14]));
  prd.direction = make_float3(__uint_as_float(p[15]), __uint_as_float(p[16]), __uint_as_float(p[17]));
  prd.done = static_cast<int>(p[18]);
}

// Classic OptiX shadow-ray idiom for plain optixTrace (no SER available):
// assume occluded, disable AH/CH so a hit leaves the payload untouched, and
// let a dedicated miss program (__miss__occlusion, missSBTIndex 1) clear it.
static __forceinline__ __device__ bool traceOcclusion(OptixTraversableHandle handle, float3 origin, float3 direction,
                                                        float tmin, float tmax) {
  unsigned int occluded = 1u;
  optixTrace(handle, origin, direction, tmin, tmax, 0.0f, OptixVisibilityMask(1),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             0, 1, 1, occluded);
  return occluded != 0u;
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int w = params.width;
  const unsigned int h = params.height;
  const unsigned int subframe = params.subframeIndex;

  unsigned int seed = sutil::tea<4>(idx.y * w + idx.x, subframe);

  float3 result = make_float3(0.0f);
  unsigned int spl = params.samplesPerLaunch;
  for (unsigned int s = 0; s < spl; ++s) {
    const float2 jitter = make_float2(sutil::rnd(seed), sutil::rnd(seed));
    const float2 d = 2.0f *
                         make_float2((static_cast<float>(idx.x) + jitter.x) / static_cast<float>(w),
                                     (static_cast<float>(idx.y) + jitter.y) / static_cast<float>(h)) -
                     1.0f;
    float3 origin = params.eye;
    float3 direction = normalize(d.x * params.U + d.y * params.V + params.W);

    RadiancePRD prd;
    prd.attenuation = make_float3(1.0f);
    prd.seed = seed;
    prd.depth = 0;
    prd.prevBsdfPdf = -1.0f; // primary ray: treat like a specular predecessor (full weight on direct light hit)

    for (;;) {
      trace(params.handle, origin, direction, 1e-3f, 1e16f, prd);

      // prd.attenuation here is the throughput accumulated *before* this
      // hit (the light material doesn't touch attenuation itself) — needed
      // because, unlike a simpler tracer that only ever sees emission on the
      // primary ray, MATERIAL_LIGHT's MIS weighting means `emitted` can be
      // nonzero after any number of specular bounces or a BSDF-sampled
      // diffuse ray landing on the light. Without this multiply, e.g. the
      // mirror sphere's reflection of the light ignored the mirror's own
      // tint and came out at the light's full unattenuated brightness.
      result += prd.emitted * prd.attenuation;
      result += prd.radiance * prd.attenuation;

      if (prd.done)
        break;

      // Russian roulette after a few free bounces.
      if (prd.depth >= 3) {
        const float p = fmaxf(fmaxf(prd.attenuation.x, prd.attenuation.y), prd.attenuation.z);
        if (sutil::rnd(prd.seed) > p)
          break;
        prd.attenuation /= fmaxf(p, 1e-4f);
      }
      if (prd.depth > 32)
        break;

      origin = prd.origin;
      direction = prd.direction;
      ++prd.depth;
    }
    seed = prd.seed;
  }

  const unsigned int pixel = idx.y * w + idx.x;
  float3 accum = result / static_cast<float>(spl);
  if (subframe > 0) {
    const float a = 1.0f / static_cast<float>(subframe + 1);
    const float3 prevColor = make_float3(params.accumBuffer[pixel]);
    accum = lerp(prevColor, accum, a);
  }
  params.accumBuffer[pixel] = make_float4(accum, 1.0f);
  params.frameBuffer[pixel] = sutil::make_color(accum);
}

extern "C" __global__ void __miss__radiance() {
  MissData *rt = reinterpret_cast<MissData *>(optixGetSbtDataPointer());
  float3 color;
  float weight = 1.0f;
  if (params.envTex) {
    const float3 dir = normalize(optixGetWorldRayDirection());
    color = lookupEnvironmentRadiance(dir);
    // Same MIS treatment as MATERIAL_LIGHT: full weight for the primary ray
    // or a specular predecessor (prevBsdfPdf < 0, NEE couldn't have sampled
    // that exact direction), power-heuristic split otherwise since a
    // diffuse bounce's NEE already sampled the environment directly too.
    const float prevBsdfPdf = __uint_as_float(optixGetPayload_5());
    if (prevBsdfPdf >= 0.0f)
      weight = powerHeuristic(prevBsdfPdf, evalEnvironmentPdf(dir));
  } else {
    color = rt->bgColor;
  }
  optixSetPayload_6(__float_as_uint(color.x * weight));
  optixSetPayload_7(__float_as_uint(color.y * weight));
  optixSetPayload_8(__float_as_uint(color.z * weight));
  optixSetPayload_9(__float_as_uint(0.0f));
  optixSetPayload_10(__float_as_uint(0.0f));
  optixSetPayload_11(__float_as_uint(0.0f));
  optixSetPayload_18(1u); // done
}

extern "C" __global__ void __miss__occlusion() { optixSetPayload_0(0u); }

// Shared ray/AABB slab test — used by both the voxel intersection program
// (box IS the primitive) and the SDF one (box just bounds where to start/
// stop sphere tracing). Returns false for no overlap; otherwise t0/t1 are
// the entry/exit parametric distances (clamped to the ray's own tmin/tmax)
// and enterAxis/enterUpper identify which face t0 landed on (-1 if the ray
// origin already starts inside the box, since there's no "entered" face).
static __forceinline__ __device__ bool slabTest(const float o[3], const float d[3], const float lo[3],
                                                  const float hi[3], float &t0, float &t1, int &enterAxis,
                                                  bool &enterUpper) {
  enterAxis = -1;
  enterUpper = false;
  for (int axis = 0; axis < 3; ++axis) {
    const float invD = 1.0f / d[axis];
    float tNear = (lo[axis] - o[axis]) * invD;
    float tFar = (hi[axis] - o[axis]) * invD;
    const bool upper = tNear > tFar; // ray travels in -axis direction, entering via the "hi" face
    if (upper) {
      const float tmp = tNear;
      tNear = tFar;
      tFar = tmp;
    }
    if (tNear > t0) {
      t0 = tNear;
      enterAxis = axis;
      enterUpper = upper;
    }
    if (tFar < t1)
      t1 = tFar;
    if (t0 > t1)
      return false;
  }
  return true;
}

// Custom-primitive intersection for a voxel: a plain ray/AABB slab test.
// OptiX's custom-primitive build input only consumes the AABB array to build
// the BVH — it doesn't hand the box back to us, so the intersection program
// re-reads the same AABB buffer via the SBT record. Reports which face was
// entered (0..5) as attribute_0 so the closest-hit program can derive a flat
// shading normal without a second geometry query.
extern "C" __global__ void __intersection__voxel() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const OptixAabb box = rt->voxelAabbs[optixGetPrimitiveIndex()];
  const float3 rayO = optixGetObjectRayOrigin();
  const float3 rayD = optixGetObjectRayDirection();
  const float o[3] = {rayO.x, rayO.y, rayO.z};
  const float d[3] = {rayD.x, rayD.y, rayD.z};
  const float lo[3] = {box.minX, box.minY, box.minZ};
  const float hi[3] = {box.maxX, box.maxY, box.maxZ};

  float t0 = optixGetRayTmin();
  float t1 = optixGetRayTmax();
  int enterAxis;
  bool enterUpper;
  if (!slabTest(o, d, lo, hi, t0, t1, enterAxis, enterUpper))
    return;
  if (enterAxis < 0)
    return; // ray origin starts inside the box — treat as a miss (rare, camera never starts inside a voxel here)

  const unsigned int face = static_cast<unsigned int>(enterAxis) * 2 + (enterUpper ? 1u : 0u);
  optixReportIntersection(t0, 0, face);
}

// Trilinear sample of the dense SDF grid, clamping at the border rather than
// wrapping or asserting — sphere tracing occasionally evaluates points just
// outside the grid due to floating-point slop at the bbox boundary.
static __forceinline__ __device__ float sampleSdf(const HitGroupData *rt, float3 p) {
  const float3 local = (p - rt->sdfOrigin) / rt->sdfVoxelSize - make_float3(0.5f, 0.5f, 0.5f);
  const int x0 = max(0, min(rt->sdfNx - 2, static_cast<int>(floorf(local.x))));
  const int y0 = max(0, min(rt->sdfNy - 2, static_cast<int>(floorf(local.y))));
  const int z0 = max(0, min(rt->sdfNz - 2, static_cast<int>(floorf(local.z))));
  const float fx = fminf(fmaxf(local.x - x0, 0.0f), 1.0f);
  const float fy = fminf(fmaxf(local.y - y0, 0.0f), 1.0f);
  const float fz = fminf(fmaxf(local.z - z0, 0.0f), 1.0f);

  auto at = [&](int x, int y, int z) { return rt->sdfDistances[(z * rt->sdfNy + y) * rt->sdfNx + x]; };
  const float c00 = at(x0, y0, z0) * (1 - fx) + at(x0 + 1, y0, z0) * fx;
  const float c10 = at(x0, y0 + 1, z0) * (1 - fx) + at(x0 + 1, y0 + 1, z0) * fx;
  const float c01 = at(x0, y0, z0 + 1) * (1 - fx) + at(x0 + 1, y0, z0 + 1) * fx;
  const float c11 = at(x0, y0 + 1, z0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1, z0 + 1) * fx;
  const float c0 = c00 * (1 - fy) + c10 * fy;
  const float c1 = c01 * (1 - fy) + c11 * fy;
  return c0 * (1 - fz) + c1 * fz;
}

// Sphere-traces from the SDF grid's bbox entry point to its exit point,
// reporting a hit wherever |distance| drops below a small epsilon. The
// bounding box itself is the one custom primitive; there's no per-cell AABB
// array like the voxel path since this is a single continuous field, not a
// sparse set of occupied cells.
extern "C" __global__ void __intersection__sdf() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 boundsMax = rt->sdfOrigin + make_float3(rt->sdfNx, rt->sdfNy, rt->sdfNz) * rt->sdfVoxelSize;
  const float3 rayO = optixGetObjectRayOrigin();
  const float3 rayD = optixGetObjectRayDirection();
  const float o[3] = {rayO.x, rayO.y, rayO.z};
  const float d[3] = {rayD.x, rayD.y, rayD.z};
  const float lo[3] = {rt->sdfOrigin.x, rt->sdfOrigin.y, rt->sdfOrigin.z};
  const float hi[3] = {boundsMax.x, boundsMax.y, boundsMax.z};

  float t0 = optixGetRayTmin();
  float t1 = optixGetRayTmax();
  int enterAxis;
  bool enterUpper;
  if (!slabTest(o, d, lo, hi, t0, t1, enterAxis, enterUpper))
    return;

  const float epsilon = rt->sdfVoxelSize * 0.1f;
  const float minStep = rt->sdfVoxelSize * 0.05f;
  float t = t0;
  float prevT = t0;
  float prevDist = sampleSdf(rt, rayO + t0 * rayD);
  if (prevDist < epsilon) {
    optixReportIntersection(t0, 0);
    return;
  }
  t += fmaxf(prevDist, minStep);
  for (int i = 1; i < 256 && t <= t1; ++i) {
    const float dist = sampleSdf(rt, rayO + t * rayD);
    if (dist < epsilon) {
      // Normal termination (dist still slightly positive, just under
      // epsilon) needs no correction — t is already a good estimate. Only
      // a genuine overshoot (dist went negative — the distance estimate
      // isn't a guaranteed-exact lower bound, see sdf_bake.cu) needs
      // interpolating back toward the zero-crossing between the last two
      // samples; applying that same interpolation formula unconditionally
      // was the bug here — for dist still positive it *extrapolates past*
      // the current sample instead of using it, which put shading points
      // measurably off-surface and showed up as normal-gradient noise once
      // sdfGradientNormal differentiated the field there.
      float tHit = t;
      if (dist < 0.0f) {
        const float denom = prevDist - dist;
        if (fabsf(denom) > 1e-6f)
          tHit = prevT + (t - prevT) * (prevDist / denom);
      }
      optixReportIntersection(fmaxf(tHit, t0), 0);
      return;
    }
    prevT = t;
    prevDist = dist;
    t += fmaxf(dist, minStep);
  }
}

// Central-difference gradient of the (trilinearly sampled, so C0-continuous)
// SDF at the hit point — a standard, cheap way to get a shading normal from
// an implicit surface without an analytic derivative.
static __forceinline__ __device__ float3 sdfGradientNormal(const HitGroupData *rt, float3 p) {
  const float h = rt->sdfVoxelSize * 0.5f;
  const float dx = sampleSdf(rt, p + make_float3(h, 0, 0)) - sampleSdf(rt, p - make_float3(h, 0, 0));
  const float dy = sampleSdf(rt, p + make_float3(0, h, 0)) - sampleSdf(rt, p - make_float3(0, h, 0));
  const float dz = sampleSdf(rt, p + make_float3(0, 0, h)) - sampleSdf(rt, p - make_float3(0, 0, h));
  return normalize(make_float3(dx, dy, dz));
}

// -X,+X,-Y,+Y,-Z,+Z, indexed by __intersection__voxel's face attribute.
static __forceinline__ __device__ float3 voxelFaceNormal(unsigned int face) {
  switch (face) {
  case 0:
    return make_float3(-1, 0, 0);
  case 1:
    return make_float3(1, 0, 0);
  case 2:
    return make_float3(0, -1, 0);
  case 3:
    return make_float3(0, 1, 0);
  case 4:
    return make_float3(0, 0, -1);
  default:
    return make_float3(0, 0, 1);
  }
}

// Shared normal computation for both built-in geometry types used in the
// bring-up scene: triangles (ground plane) and built-in spheres.
static __forceinline__ __device__ float3 computeShadingNormal(const float3 &rayDir) {
  float3 n;
  if (optixIsTriangleHit()) {
    float3 verts[3];
    optixGetTriangleVertexData(verts);
    n = normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
  } else {
    // Built-in sphere: query center/radius directly from the GAS rather than
    // duplicating geometry in the SBT record.
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int sbtGasIdx = optixGetSbtGASIndex();
    float4 q;
    optixGetSphereData(gas, primIdx, sbtGasIdx, 0.0f, &q);
    const float3 hitP = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    n = (hitP - make_float3(q)) / q.w;
  }
  return faceforward(n, -rayDir, n);
}

extern "C" __global__ void __closesthit__radiance() {
  HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());

  const float3 rayDir = optixGetWorldRayDirection();
  float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;

  // GLB-mesh triangles carry their own per-vertex normals/UVs (better shading
  // than the flat face normal computeShadingNormal() gives the fixed
  // bring-up scene's spheres/plane) and an optional base-color texture.
  float3 N;
  float3 albedo = rt->albedo;
  if (rt->materialType == MATERIAL_TEXTURED_DIFFUSE) {
    const unsigned int prim = optixGetPrimitiveIndex();
    const float2 bary = optixGetTriangleBarycentrics();
    const float w0 = 1.0f - bary.x - bary.y, w1 = bary.x, w2 = bary.y;
    N = normalize(w0 * rt->normals[prim * 3 + 0] + w1 * rt->normals[prim * 3 + 1] + w2 * rt->normals[prim * 3 + 2]);
    N = faceforward(N, -rayDir, N);
    if (rt->baseColorTex) {
      const float2 uv0 = rt->uvs[prim * 3 + 0], uv1 = rt->uvs[prim * 3 + 1], uv2 = rt->uvs[prim * 3 + 2];
      const float u = w0 * uv0.x + w1 * uv1.x + w2 * uv2.x;
      const float v = w0 * uv0.y + w1 * uv1.y + w2 * uv2.y;
      const float4 texel = tex2D<float4>(rt->baseColorTex, u, v);
      albedo = make_float3(texel.x, texel.y, texel.z) * rt->albedo; // rt->albedo doubles as baseColorFactor here
    }
  } else if (rt->materialType == MATERIAL_VOXEL) {
    const float3 faceN = voxelFaceNormal(optixGetAttribute_0());
    N = faceforward(faceN, -rayDir, faceN);
    albedo = rt->voxelColors[optixGetPrimitiveIndex()];
  } else if (rt->materialType == MATERIAL_SDF) {
    const float3 gradN = sdfGradientNormal(rt, P);
    N = faceforward(gradN, -rayDir, gradN);
    // albedo already defaulted to rt->albedo above — a single flat tint.
    // Sphere tracing only locates the surface to within `epsilon` (see
    // __intersection__sdf), unlike triangle/voxel geometry which is exact —
    // the shared 1e-3f NEE/bounce-ray offset below isn't reliably larger
    // than that uncertainty, so shadow/bounce rays were self-intersecting
    // the SDF surface they just came from ("shadow acne": most of the
    // surface reads as falsely self-occluded). Nudge the shading point
    // outward along the normal by a safe margin before any ray leaves it.
    P = P + N * (rt->sdfVoxelSize * 0.25f);
  } else {
    N = computeShadingNormal(rayDir);
  }

  unsigned int seed = optixGetPayload_3();
  int depth = static_cast<int>(optixGetPayload_4());
  const float prevBsdfPdf = __uint_as_float(optixGetPayload_5());
  float3 attenuation =
      make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));

  float3 emitted = make_float3(0.0f);
  float3 radiance = make_float3(0.0f);
  float3 nextOrigin = P;
  float3 nextDirection = rayDir;
  float nextPdf = -1.0f;
  int done = 0;

  if (rt->materialType == MATERIAL_LIGHT) {
    const QuadLight &light = params.light;
    const float area = length(cross(light.v1, light.v2));
    const float cosLight = dot(light.normal, -rayDir);
    if (cosLight > 1e-4f) {
      float weight = 1.0f;
      if (prevBsdfPdf >= 0.0f) {
        const float dist = optixGetRayTmax();
        const float pdfLight = (dist * dist) / (cosLight * area);
        weight = powerHeuristic(prevBsdfPdf, pdfLight);
      }
      emitted = rt->emission * weight;
    }
    done = 1;
  } else if (rt->materialType == MATERIAL_DIFFUSE || rt->materialType == MATERIAL_TEXTURED_DIFFUSE ||
             rt->materialType == MATERIAL_VOXEL || rt->materialType == MATERIAL_SDF) {
    // Next-event estimation: toward the environment if one is loaded
    // (params.envTex != 0), else toward the quad light — the two aren't
    // combined, see Params::envTex's doc comment. Neither branch multiplies
    // by albedo directly: the raygen loop computes
    // result += prd.radiance * prd.attenuation, and `attenuation` is
    // updated to include *this* bounce's albedo a few lines below, before
    // that payload is written out. Multiplying albedo in here too would
    // double it — this bit us for the entire lifetime of phases 2-4 (every
    // diffuse/textured/voxel NEE sample was too dark by an extra factor of
    // albedo) until caught in review.
    if (params.envTex) {
      float envPdf;
      const float3 dir = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), envPdf);
      const float nDl = dot(N, dir);
      if (nDl > 0.0f && envPdf > 0.0f) {
        const bool occluded = traceOcclusion(params.handle, P, dir, 1e-3f, 1e16f);
        if (!occluded) {
          const float3 envRadiance = lookupEnvironmentRadiance(dir);
          const float pdfBsdf = nDl / M_PIf;
          const float weight = powerHeuristic(envPdf, pdfBsdf);
          radiance = (envRadiance / M_PIf) * nDl * weight / fmaxf(envPdf, 1e-6f);
        }
      }
    } else {
      const QuadLight &light = params.light;
      const float z1 = sutil::rnd(seed);
      const float z2 = sutil::rnd(seed);
      const float3 lightPos = light.corner + light.v1 * z1 + light.v2 * z2;
      const float3 toLight = lightPos - P;
      const float dist = length(toLight);
      const float3 L = toLight / dist;
      const float nDl = dot(N, L);
      const float lnDl = -dot(light.normal, L);
      if (nDl > 0.0f && lnDl > 0.0f) {
        const bool occluded = traceOcclusion(params.handle, P, L, 1e-3f, dist - 2e-3f);
        if (!occluded) {
          const float area = length(cross(light.v1, light.v2));
          const float pdfLight = (dist * dist) / (lnDl * area);
          const float pdfBsdf = nDl / M_PIf;
          const float weight = powerHeuristic(pdfLight, pdfBsdf);
          radiance = (light.emission / M_PIf) * nDl * weight / fmaxf(pdfLight, 1e-6f);
        }
      }
    }

    // Sample the next bounce direction (cosine-weighted).
    float3 local;
    cosineSampleHemisphere(sutil::rnd(seed), sutil::rnd(seed), local);
    Onb onb(N);
    nextDirection = onb.toWorld(local);
    nextPdf = fmaxf(local.z, 1e-4f) / M_PIf;
    attenuation = attenuation * albedo;
  } else if (rt->materialType == MATERIAL_MIRROR) {
    nextDirection = reflect(rayDir, N);
    attenuation = attenuation * rt->albedo;
    nextPdf = -1.0f;
  } else { // MATERIAL_GLASS
    const bool entering = dot(rayDir, N) < 0.0f;
    const float3 n = entering ? N : -N;
    const float eta = entering ? (1.0f / rt->ior) : rt->ior;
    const float cosTheta = fminf(fabsf(dot(rayDir, n)), 1.0f);
    const float fresnel = schlickFresnel(cosTheta, entering ? rt->ior : (1.0f / rt->ior));

    float3 refracted;
    const bool canRefract = refractRay(rayDir, n, eta, refracted);
    if (!canRefract || sutil::rnd(seed) < fresnel) {
      nextDirection = reflect(rayDir, N);
    } else {
      nextDirection = refracted;
    }
    attenuation = attenuation * rt->albedo;
    nextPdf = -1.0f;
  }

  optixSetPayload_0(__float_as_uint(attenuation.x));
  optixSetPayload_1(__float_as_uint(attenuation.y));
  optixSetPayload_2(__float_as_uint(attenuation.z));
  optixSetPayload_3(seed);
  optixSetPayload_4(static_cast<unsigned int>(depth));
  optixSetPayload_5(__float_as_uint(nextPdf));
  optixSetPayload_6(__float_as_uint(emitted.x));
  optixSetPayload_7(__float_as_uint(emitted.y));
  optixSetPayload_8(__float_as_uint(emitted.z));
  optixSetPayload_9(__float_as_uint(radiance.x));
  optixSetPayload_10(__float_as_uint(radiance.y));
  optixSetPayload_11(__float_as_uint(radiance.z));
  optixSetPayload_12(__float_as_uint(nextOrigin.x));
  optixSetPayload_13(__float_as_uint(nextOrigin.y));
  optixSetPayload_14(__float_as_uint(nextOrigin.z));
  optixSetPayload_15(__float_as_uint(nextDirection.x));
  optixSetPayload_16(__float_as_uint(nextDirection.y));
  optixSetPayload_17(__float_as_uint(nextDirection.z));
  optixSetPayload_18(static_cast<unsigned int>(done));
}
