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

#include "pathtracer_params.h"

#include <sutil/cuda/helpers.h>
#include <sutil/cuda/random.h>
#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

// ----------------------------------------------------------------------------
// Orthonormal basis for cosine-hemisphere sampling around a shading normal.
// ----------------------------------------------------------------------------
struct Onb {
  __forceinline__ __device__ Onb(const float3 &normal) {
    m_normal = normal;
    if (fabsf(m_normal.x) > fabsf(m_normal.z)) {
      m_binormal = make_float3(-m_normal.y, m_normal.x, 0.0f);
    } else {
      m_binormal = make_float3(0.0f, -m_normal.z, m_normal.y);
    }
    m_binormal = normalize(m_binormal);
    m_tangent = cross(m_binormal, m_normal);
  }

  __forceinline__ __device__ float3 toWorld(const float3 &p) const {
    return p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
  }

  float3 m_tangent, m_binormal, m_normal;
};

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

      result += prd.emitted;
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
  optixSetPayload_6(__float_as_uint(rt->bgColor.x));
  optixSetPayload_7(__float_as_uint(rt->bgColor.y));
  optixSetPayload_8(__float_as_uint(rt->bgColor.z));
  optixSetPayload_9(__float_as_uint(0.0f));
  optixSetPayload_10(__float_as_uint(0.0f));
  optixSetPayload_11(__float_as_uint(0.0f));
  optixSetPayload_18(1u); // done
}

extern "C" __global__ void __miss__occlusion() { optixSetPayload_0(0u); }

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
  const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;

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
  } else if (rt->materialType == MATERIAL_DIFFUSE || rt->materialType == MATERIAL_TEXTURED_DIFFUSE) {
    // Next-event estimation toward the quad light.
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
        radiance = (albedo / M_PIf) * light.emission * nDl * weight / fmaxf(pdfLight, 1e-6f);
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
