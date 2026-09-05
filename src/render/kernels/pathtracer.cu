// Path tracer: unidirectional, next-event estimation to one quad light, MIS (power heuristic), Russian roulette, progressive accumulation. Diffuse/mirror/glass/light materials. Plain optixTrace, no SER.

#include <optix.h>

#include <nanovdb/NanoVDB.h>

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

static __forceinline__ __device__ float powerHeuristic(float pdfA, float pdfB) {
  const float a2 = pdfA * pdfA;
  const float b2 = pdfB * pdfB;
  return a2 / fmaxf(a2 + b2, 1e-8f);
}

static __forceinline__ __device__ bool refractRay(const float3 &in, const float3 &n, float eta, float3 &out) {
  const float cosI = -dot(n, in);
  const float sin2T = eta * eta * (1.0f - cosI * cosI);
  if (sin2T > 1.0f)
    return false;
  const float cosT = sqrtf(1.0f - sin2T);
  out = eta * in + (eta * cosI - cosT) * n;
  return true;
}

static __forceinline__ __device__ float schlickFresnel(float cosTheta, float iorFrom, float iorTo) {
  float r0 = (iorFrom - iorTo) / (iorFrom + iorTo);
  r0 = r0 * r0;
  if (r0 < 1e-8f)
    return 0.0f;
  const float x = 1.0f - cosTheta;
  return r0 + (1.0f - r0) * x * x * x * x * x;
}

static __forceinline__ __device__ float3 evalDielectricBounce(const float3 &rayDir, const float3 &N, float iorFrom,
                                                                float iorTo, const float3 &extinctionFrom,
                                                                float rayTmax, bool isLightTransport,
                                                                unsigned int &seed, float3 &throughput) {
  throughput = throughput * make_float3(expf(-extinctionFrom.x * rayTmax), expf(-extinctionFrom.y * rayTmax),
                                         expf(-extinctionFrom.z * rayTmax));
  const float eta = iorFrom / iorTo;
  const float cosTheta = fminf(fabsf(dot(rayDir, N)), 1.0f);
  const float fresnel = schlickFresnel(cosTheta, iorFrom, iorTo);

  float3 refracted;
  const bool canRefract = refractRay(rayDir, N, eta, refracted);
  const bool didRefract = canRefract && !(sutil::rnd(seed) < fresnel);
  const float3 nextDirection = didRefract ? refracted : reflect(rayDir, N);

  if (didRefract && isLightTransport) {
    const float etaCorrection = 1.0f / (eta * eta);
    throughput = throughput * etaCorrection;
  }

  return nextDirection;
}

static __forceinline__ __device__ float3 attenuationToExtinction(float3 attenuationColor, float attenuationDistance) {
  if (!(attenuationDistance > 0.0f) || isinf(attenuationDistance))
    return make_float3(0.0f);
  return make_float3(-logf(fmaxf(attenuationColor.x, 1e-6f)) / attenuationDistance,
                      -logf(fmaxf(attenuationColor.y, 1e-6f)) / attenuationDistance,
                      -logf(fmaxf(attenuationColor.z, 1e-6f)) / attenuationDistance);
}

static __forceinline__ __device__ float3 agxContrastApprox(float3 x) {
  const float3 x2 = x * x;
  const float3 x4 = x2 * x2;
  return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
}

static __forceinline__ __device__ float3 agxTonemap(float3 c) {
  const float3 r = make_float3(0.856627153315983f, 0.0951212405381588f, 0.0482516061458583f);
  const float3 g = make_float3(0.137318972929847f, 0.761241990602591f, 0.101439036467562f);
  const float3 b = make_float3(0.11189821299995f, 0.0767994186031903f, 0.811302368396859f);
  float3 v = make_float3(dot(make_float3(r.x, g.x, b.x), c), dot(make_float3(r.y, g.y, b.y), c),
                          dot(make_float3(r.z, g.z, b.z), c));

  const float minEv = -12.47393f, maxEv = 4.026069f;
  v = make_float3(fmaxf(v.x, 1e-10f), fmaxf(v.y, 1e-10f), fmaxf(v.z, 1e-10f));
  v = make_float3(log2f(v.x), log2f(v.y), log2f(v.z));
  v = clamp((v - minEv) / (maxEv - minEv), 0.0f, 1.0f);
  v = agxContrastApprox(v);

  const float3 or_ = make_float3(1.1271005818144368f, -0.11060664309660323f, -0.016493938717834573f);
  const float3 og = make_float3(-0.1413297634984383f, 1.157823702216272f, -0.016493938717834257f);
  const float3 ob = make_float3(-0.14132976349843826f, -0.11060664309660294f, 1.2519364065950405f);
  v = make_float3(dot(make_float3(or_.x, og.x, ob.x), v), dot(make_float3(or_.y, og.y, ob.y), v),
                   dot(make_float3(or_.z, og.z, ob.z), v));

  v = make_float3(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f), fmaxf(v.z, 0.0f));
  return make_float3(powf(v.x, 2.2f), powf(v.y, 2.2f), powf(v.z, 2.2f));
}

static __forceinline__ __device__ float3 reinhardTonemap(float3 c) { return c / (make_float3(1.0f) + c); }

static __forceinline__ __device__ float3 acesFilmicTonemap(float3 x) {
  const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

static __forceinline__ __device__ float3 hablePartial(float3 x) {
  const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
static __forceinline__ __device__ float3 hableTonemap(float3 c) {
  const float exposureBias = 2.0f;
  const float3 curr = hablePartial(c * exposureBias);
  const float3 whiteScale = make_float3(1.0f) / hablePartial(make_float3(11.2f));
  return curr * whiteScale;
}

static __forceinline__ __device__ float3 clampFirefly(float3 c, float maxValue) {
  if (maxValue <= 0.0f)
    return c;
  const float peak = fmaxf(fmaxf(c.x, c.y), c.z);
  return peak > maxValue ? c * (maxValue / peak) : c;
}

static __forceinline__ __device__ bool isFinite3(float3 c) {
  return isfinite(c.x) && isfinite(c.y) && isfinite(c.z);
}

static __forceinline__ __device__ uchar4 applyTonemapAndQuantize(float3 hdr, unsigned int op) {
  float3 mapped;
  switch (op) {
  case 1:
    mapped = reinhardTonemap(hdr);
    break;
  case 2:
    mapped = acesFilmicTonemap(hdr);
    break;
  case 3:
    mapped = hableTonemap(hdr);
    break;
  case 4:
    mapped = hdr;
    break;
  default:
    mapped = agxTonemap(hdr);
    break;
  }
  return sutil::make_color(mapped);
}

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

static __forceinline__ __device__ void dirToEquirectUv(float3 dir, float &u, float &v, float &theta) {
  theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
  float phi = atan2f(dir.z, dir.x) - params.envRotation;
  phi = phi - 2.0f * M_PIf * floorf((phi + M_PIf) / (2.0f * M_PIf));
  u = (phi + M_PIf) / (2.0f * M_PIf);
  v = theta / M_PIf;
}

static __forceinline__ __device__ float3 equirectUvToDir(float u, float v, float &sinTheta) {
  const float theta = v * M_PIf;
  const float phi = u * 2.0f * M_PIf - M_PIf + params.envRotation;
  sinTheta = sinf(theta);
  return make_float3(sinTheta * cosf(phi), cosf(theta), sinTheta * sinf(phi));
}

static __forceinline__ __device__ float3 lookupEnvironmentRadiance(float3 dir) {
  float u, v, theta;
  dirToEquirectUv(dir, u, v, theta);
  const float4 texel = tex2D<float4>(params.envTex, u, v);
  return make_float3(texel.x, texel.y, texel.z);
}

static __forceinline__ __device__ float evalEnvironmentPdf(float3 dir) {
  float u, v, theta;
  dirToEquirectUv(dir, u, v, theta);
  const int h = params.envHeight, w = params.envWidth;
  const int row = min(max(static_cast<int>(v * h), 0), h - 1);
  const int col = min(max(static_cast<int>(u * w), 0), w - 1);
  const float rowPdf = (params.envMarginalCdf[row + 1] - params.envMarginalCdf[row]) * h;
  const float *rowCdf = params.envConditionalCdf + row * (w + 1);
  const float colPdf = (rowCdf[col + 1] - rowCdf[col]) * w;
  const float sinTheta = sinf(theta);
  return sinTheta > 1e-6f ? (rowPdf * colPdf) / (2.0f * M_PIf * M_PIf * sinTheta) : 0.0f;
}

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

  float sinTheta;
  const float3 dir = equirectUvToDir(u, v, sinTheta);
  pdfOut = sinTheta > 1e-6f ? (rowPdf * colPdf) / (2.0f * M_PIf * M_PIf * sinTheta) : 0.0f;
  return dir;
}

static __forceinline__ __device__ float3 sampleSunCone(unsigned int &seed, float &pdfSolidAngle) {
  const float cosThetaMax = params.sun.cosAngularRadius;
  const float z = 1.0f - sutil::rnd(seed) * (1.0f - cosThetaMax);
  const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  const float phi = 2.0f * M_PIf * sutil::rnd(seed);
  const Onb onb(params.sun.direction);
  pdfSolidAngle = 1.0f / (2.0f * M_PIf * (1.0f - cosThetaMax));
  return onb.toWorld(make_float3(r * cosf(phi), r * sinf(phi), z));
}

static __forceinline__ __device__ float sunConePdf(const float3 &dir) {
  return dot(dir, params.sun.direction) > params.sun.cosAngularRadius
             ? 1.0f / (2.0f * M_PIf * (1.0f - params.sun.cosAngularRadius))
             : 0.0f;
}

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
  float3 albedo;
  float3 normal;
};

static __forceinline__ __device__ void trace(OptixTraversableHandle handle, float3 origin, float3 direction,
                                              float tmin, float tmax, RadiancePRD &prd) {
  unsigned int p[25] = {};
  p[0] = __float_as_uint(prd.attenuation.x);
  p[1] = __float_as_uint(prd.attenuation.y);
  p[2] = __float_as_uint(prd.attenuation.z);
  p[3] = prd.seed;
  p[4] = static_cast<unsigned int>(prd.depth);
  p[5] = __float_as_uint(prd.prevBsdfPdf);
  p[19] = __float_as_uint(prd.albedo.x);
  p[20] = __float_as_uint(prd.albedo.y);
  p[21] = __float_as_uint(prd.albedo.z);
  p[22] = __float_as_uint(prd.normal.x);
  p[23] = __float_as_uint(prd.normal.y);
  p[24] = __float_as_uint(prd.normal.z);

  optixTrace(handle, origin, direction, tmin, tmax, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p[0],
             p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
             p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24]);

  prd.attenuation = make_float3(__uint_as_float(p[0]), __uint_as_float(p[1]), __uint_as_float(p[2]));
  prd.seed = p[3];
  prd.depth = static_cast<int>(p[4]);
  prd.prevBsdfPdf = __uint_as_float(p[5]);
  prd.emitted = make_float3(__uint_as_float(p[6]), __uint_as_float(p[7]), __uint_as_float(p[8]));
  prd.radiance = make_float3(__uint_as_float(p[9]), __uint_as_float(p[10]), __uint_as_float(p[11]));
  prd.origin = make_float3(__uint_as_float(p[12]), __uint_as_float(p[13]), __uint_as_float(p[14]));
  prd.direction = make_float3(__uint_as_float(p[15]), __uint_as_float(p[16]), __uint_as_float(p[17]));
  prd.done = static_cast<int>(p[18]);
  prd.albedo = make_float3(__uint_as_float(p[19]), __uint_as_float(p[20]), __uint_as_float(p[21]));
  prd.normal = make_float3(__uint_as_float(p[22]), __uint_as_float(p[23]), __uint_as_float(p[24]));
}

static __forceinline__ __device__ float2 projectToPrevFrame(float3 vecFromPrevEye) {
  const float wComp = dot(vecFromPrevEye, params.prevW);
  const float wSq = dot(params.prevW, params.prevW);
  if (wComp <= 1e-6f || wSq <= 1e-12f)
    return make_float2(1e6f, 1e6f);
  const float scale = wSq / wComp;
  const float3 v = vecFromPrevEye * scale;
  const float3 vMinusW = v - params.prevW;
  const float uSq = fmaxf(dot(params.prevU, params.prevU), 1e-12f);
  const float vSq = fmaxf(dot(params.prevV, params.prevV), 1e-12f);
  const float dx = dot(vMinusW, params.prevU) / uSq;
  const float dy = dot(vMinusW, params.prevV) / vSq;
  const float px = (dx + 1.0f) * 0.5f * static_cast<float>(params.width) - 0.5f;
  const float py = (dy + 1.0f) * 0.5f * static_cast<float>(params.height) - 0.5f;
  return make_float2(px, py);
}

static __forceinline__ __device__ void writeMotionVector(const uint3 &idx, float3 vecFromPrevEye) {
  const float2 prevPixel = projectToPrevFrame(vecFromPrevEye);
  const unsigned int pixel = idx.y * params.width + idx.x;
  if (prevPixel.x > 1e5f || prevPixel.y > 1e5f) {
    params.motionVectorBuffer[pixel] = make_float2(1e6f, 1e6f);
    params.denoiserFlowBuffer[pixel] = make_float2(0.0f, 0.0f);
  } else {
    const float2 mv = make_float2(static_cast<float>(idx.x) - prevPixel.x, static_cast<float>(idx.y) - prevPixel.y);
    params.motionVectorBuffer[pixel] = mv;
    params.denoiserFlowBuffer[pixel] = mv;
  }
}

static __forceinline__ __device__ float4 sampleBuffer2D(const float4 *buf, unsigned int w, unsigned int h,
                                                          float2 p) {
  const int x0 = max(0, min(static_cast<int>(w) - 2, static_cast<int>(floorf(p.x))));
  const int y0 = max(0, min(static_cast<int>(h) - 2, static_cast<int>(floorf(p.y))));
  const float fx = fminf(fmaxf(p.x - x0, 0.0f), 1.0f);
  const float fy = fminf(fmaxf(p.y - y0, 0.0f), 1.0f);
  const float4 c00 = buf[y0 * w + x0];
  const float4 c10 = buf[y0 * w + x0 + 1];
  const float4 c01 = buf[(y0 + 1) * w + x0];
  const float4 c11 = buf[(y0 + 1) * w + x0 + 1];
  return bilerp(c00, c10, c01, c11, fx, fy);
}

static __forceinline__ __device__ bool isDisoccluded(float3 currentAlbedo, float3 currentNormal,
                                                        float3 reprojAlbedo, float3 reprojNormal) {
  const float normalDot = dot(normalize(currentNormal), normalize(reprojNormal));
  const float albedoDiff = length(currentAlbedo - reprojAlbedo);
  return !(normalDot > 0.9f && albedoDiff < 0.2f);
}

static __forceinline__ __device__ float surfaceEpsilon(const float3 &p) {
  return 1e-3f + 1e-6f * fmaxf(fmaxf(fabsf(p.x), fabsf(p.y)), fabsf(p.z));
}

static __forceinline__ __device__ bool traceOcclusion(OptixTraversableHandle handle, float3 origin, float3 direction,
                                                        float tmin, float tmax) {
  unsigned int occluded = 1u;
  optixTrace(handle, origin, direction, tmin, tmax, 0.0f, OptixVisibilityMask(1),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             0, 1, 1, occluded);
  return occluded != 0u;
}

static __forceinline__ __device__ float tentFilterWarp(float u) {
  const float x = 2.0f * u;
  return x < 1.0f ? sqrtf(x) - 1.0f : 1.0f - sqrtf(fmaxf(2.0f - x, 0.0f));
}

static __forceinline__ __device__ float2 concentricSampleDisk(float u1, float u2) {
  const float ox = 2.0f * u1 - 1.0f, oy = 2.0f * u2 - 1.0f;
  if (ox == 0.0f && oy == 0.0f)
    return make_float2(0.0f, 0.0f);
  float r, theta;
  if (fabsf(ox) > fabsf(oy)) {
    r = ox;
    theta = (M_PIf / 4.0f) * (oy / ox);
  } else {
    r = oy;
    theta = (M_PIf / 2.0f) - (M_PIf / 4.0f) * (ox / oy);
  }
  return make_float2(r * cosf(theta), r * sinf(theta));
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int w = params.width;
  const unsigned int h = params.height;
  const unsigned int subframe = params.subframeIndex;

  unsigned int seed = sutil::tea<4>(idx.y * w + idx.x, subframe);

  float3 result = make_float3(0.0f);
  float3 firstHitAlbedo = make_float3(0.0f);
  float3 firstHitNormal = make_float3(0.0f);
  unsigned int spl = params.samplesPerLaunch;
  for (unsigned int s = 0; s < spl; ++s) {
    const float2 jitter =
        make_float2(tentFilterWarp(sutil::rnd(seed)), tentFilterWarp(sutil::rnd(seed)));
    const float2 d = 2.0f *
                         make_float2((static_cast<float>(idx.x) + 0.5f + jitter.x) / static_cast<float>(w),
                                     (static_cast<float>(idx.y) + 0.5f + jitter.y) / static_cast<float>(h)) -
                     1.0f;
    float3 origin = params.eye;
    float3 direction = normalize(d.x * params.U + d.y * params.V + params.W);

    if (params.aperture > 0.0f) {
      const float cosAxis = dot(direction, params.W);
      if (cosAxis > 1e-6f) {
        const float3 focalPoint = origin + direction * (params.focusDistance / cosAxis);
        const float2 lens = concentricSampleDisk(sutil::rnd(seed), sutil::rnd(seed)) * params.aperture;
        origin = origin + normalize(params.U) * lens.x + normalize(params.V) * lens.y;
        direction = normalize(focalPoint - origin);
      }
    }

    RadiancePRD prd;
    prd.attenuation = make_float3(1.0f);
    prd.seed = seed;
    prd.depth = 0;
    prd.prevBsdfPdf = -1.0f;

    float3 sample = make_float3(0.0f);

    for (;;) {
      trace(params.handle, origin, direction, 1e-3f, 1e16f, prd);

      if (s == 0 && prd.depth == 0) {
        firstHitAlbedo = prd.albedo;
        firstHitNormal = prd.normal;
      }

      sample += prd.emitted;
      sample += prd.radiance;

      if (prd.done)
        break;

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

    if (isFinite3(sample))
      result += clampFirefly(sample, params.fireflyClamp);
  }

  const unsigned int pixel = idx.y * w + idx.x;
  float3 accum = result / static_cast<float>(spl);

  float oldHistory = 0.0f;
  float3 prevColorForBlend = accum;
  if (subframe > 0) {
    if (params.cameraMoved) {
      const float2 mv = params.motionVectorBuffer[pixel];
      if (fabsf(mv.x) < 1e5f && fabsf(mv.y) < 1e5f) {
        const float2 prevPixel =
            make_float2(static_cast<float>(idx.x), static_cast<float>(idx.y)) - mv;
        if (prevPixel.x >= 0.0f && prevPixel.x <= static_cast<float>(w - 1) && prevPixel.y >= 0.0f &&
            prevPixel.y <= static_cast<float>(h - 1)) {
          const float4 reprojColor = sampleBuffer2D(params.prevAccumBuffer, w, h, prevPixel);
          const float3 reprojAlbedo = make_float3(sampleBuffer2D(params.prevAccumAlbedoBuffer, w, h, prevPixel));
          const float3 reprojNormal = make_float3(sampleBuffer2D(params.prevAccumNormalBuffer, w, h, prevPixel));
          if (!isDisoccluded(firstHitAlbedo, firstHitNormal, reprojAlbedo, reprojNormal)) {
            oldHistory = reprojColor.w;
            prevColorForBlend = make_float3(reprojColor);
          }
        }
      }
    } else {
      const float4 prevSelf = params.prevAccumBuffer[pixel];
      oldHistory = prevSelf.w;
      prevColorForBlend = make_float3(prevSelf);
    }
  }
  oldHistory = fminf(oldHistory, 127.0f);
  const float a = 1.0f / (oldHistory + 1.0f);
  accum = lerp(prevColorForBlend, accum, a);

  params.accumBuffer[pixel] = make_float4(accum, oldHistory + 1.0f);
  params.accumAlbedoBuffer[pixel] = make_float4(firstHitAlbedo, 1.0f);
  params.accumNormalBuffer[pixel] = make_float4(firstHitNormal, 0.0f);
  if (!params.denoiserEnabled)
    params.frameBuffer[pixel] = applyTonemapAndQuantize(accum * params.exposure, params.tonemapOperator);
}

extern "C" __global__ void __raygen__reservoirBuild() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int w = params.width;
  const unsigned int h = params.height;
  const unsigned int pixel = idx.y * w + idx.x;
  params.reservoirBuffer[pixel] = Reservoir{};

  if (!params.reservoirNEE)
    return;

  unsigned int seed = sutil::tea<4>(pixel, params.subframeIndex ^ 0x9e3779b9u);
  const float2 jitter = make_float2(tentFilterWarp(sutil::rnd(seed)), tentFilterWarp(sutil::rnd(seed)));
  const float2 d = 2.0f *
                       make_float2((static_cast<float>(idx.x) + 0.5f + jitter.x) / static_cast<float>(w),
                                   (static_cast<float>(idx.y) + 0.5f + jitter.y) / static_cast<float>(h)) -
                   1.0f;
  float3 origin = params.eye;
  float3 direction = normalize(d.x * params.U + d.y * params.V + params.W);

  if (params.aperture > 0.0f) {
    const float cosAxis = dot(direction, params.W);
    if (cosAxis > 1e-6f) {
      const float3 focalPoint = origin + direction * (params.focusDistance / cosAxis);
      const float2 lens = concentricSampleDisk(sutil::rnd(seed), sutil::rnd(seed)) * params.aperture;
      origin = origin + normalize(params.U) * lens.x + normalize(params.V) * lens.y;
      direction = normalize(focalPoint - origin);
    }
  }

  RadiancePRD prd;
  prd.attenuation = make_float3(1.0f);
  prd.seed = seed;
  prd.depth = 0;
  prd.prevBsdfPdf = -1.0f;
  trace(params.handle, origin, direction, 1e-3f, 1e16f, prd);
}

extern "C" __global__ void __raygen__tonemap() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int pixel = idx.y * params.width + idx.x;
  const float3 color = make_float3(params.denoisedBuffer[pixel]);
  params.frameBuffer[pixel] = applyTonemapAndQuantize(color * params.exposure, params.tonemapOperator);
}

extern "C" __global__ void __miss__radiance() {
  const int depth = static_cast<int>(optixGetPayload_4());
  const float3 dir = normalize(optixGetWorldRayDirection());
  const float prevBsdfPdf = __uint_as_float(optixGetPayload_5());

  float3 color;
  float weight = 1.0f;
  if (depth == 0) {
    color = params.backgroundColor;
  } else if (params.envTex) {
    color = lookupEnvironmentRadiance(dir);
    if (prevBsdfPdf >= 0.0f) {
      const float otherPdfScale = params.sun.enabled ? 0.5f : 1.0f;
      weight = powerHeuristic(prevBsdfPdf, evalEnvironmentPdf(dir) * otherPdfScale);
    }
  } else {
    color = params.backgroundColor;
  }
  const float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()),
                                          __uint_as_float(optixGetPayload_2()));
  color = color * weight * throughput;

  if (depth > 0 && params.sun.enabled && dot(dir, params.sun.direction) > params.sun.cosAngularRadius) {
    float sunWeight = 1.0f;
    if (prevBsdfPdf >= 0.0f)
      sunWeight = powerHeuristic(prevBsdfPdf, sunConePdf(dir) * 0.5f);
    color = color + params.sun.radiance * sunWeight * throughput;
  }

  optixSetPayload_6(__float_as_uint(color.x));
  optixSetPayload_7(__float_as_uint(color.y));
  optixSetPayload_8(__float_as_uint(color.z));
  optixSetPayload_9(__float_as_uint(0.0f));
  optixSetPayload_10(__float_as_uint(0.0f));
  optixSetPayload_11(__float_as_uint(0.0f));
  optixSetPayload_18(1u);

  if (depth == 0) {
    optixSetPayload_19(__float_as_uint(params.backgroundColor.x));
    optixSetPayload_20(__float_as_uint(params.backgroundColor.y));
    optixSetPayload_21(__float_as_uint(params.backgroundColor.z));
    optixSetPayload_22(__float_as_uint(-dir.x));
    optixSetPayload_23(__float_as_uint(-dir.y));
    optixSetPayload_24(__float_as_uint(-dir.z));
    writeMotionVector(optixGetLaunchIndex(), dir);
  }
}

extern "C" __global__ void __miss__occlusion() { optixSetPayload_0(0u); }

extern "C" __global__ void __miss__merge() {}

static __forceinline__ __device__ bool slabTest(const float o[3], const float d[3], const float lo[3],
                                                  const float hi[3], float &t0, float &t1, int &enterAxis,
                                                  bool &enterUpper) {
  enterAxis = -1;
  enterUpper = false;
  for (int axis = 0; axis < 3; ++axis) {
    const float invD = 1.0f / d[axis];
    float tNear = (lo[axis] - o[axis]) * invD;
    float tFar = (hi[axis] - o[axis]) * invD;
    const bool upper = tNear > tFar;
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
    return;

  const unsigned int face = static_cast<unsigned int>(enterAxis) * 2 + (enterUpper ? 1u : 0u);
  optixReportIntersection(t0, 0, face);
}

extern "C" __global__ void __intersection__nvdb() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 rayO = optixGetObjectRayOrigin();
  const float3 rayD = optixGetObjectRayDirection();
  const float o[3] = {rayO.x, rayO.y, rayO.z};
  const float d[3] = {rayD.x, rayD.y, rayD.z};
  const float lo[3] = {rt->nvdbBoundsMin.x, rt->nvdbBoundsMin.y, rt->nvdbBoundsMin.z};
  const float hi[3] = {rt->nvdbBoundsMax.x, rt->nvdbBoundsMax.y, rt->nvdbBoundsMax.z};

  float t0 = optixGetRayTmin();
  float t1 = optixGetRayTmax();
  int enterAxis;
  bool enterUpper;
  if (!slabTest(o, d, lo, hi, t0, t1, enterAxis, enterUpper))
    return;

  optixReportIntersection(fmaxf(t0, optixGetRayTmin()), 0, __float_as_uint(t1));
}

#define SPHERE_HIT_FROM_OUTSIDE 0u
#define SPHERE_HIT_FROM_INSIDE 1u

extern "C" __global__ void __intersection__sphere_solid() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 o = optixGetObjectRayOrigin() - rt->sphereCenter;
  const float3 d = optixGetObjectRayDirection();

  const float a = dot(d, d);
  const float b = dot(o, d);
  const float c = dot(o, o) - rt->sphereRadius * rt->sphereRadius;
  const float disc = b * b - a * c;
  if (disc < 0.0f)
    return;

  const float sqrtDisc = sqrtf(disc);
  const float tNear = (-b - sqrtDisc) / a;
  const float tFar = (-b + sqrtDisc) / a;
  const float tmin = optixGetRayTmin();
  const float tmax = optixGetRayTmax();

  if (tNear > tmin && tNear < tmax)
    optixReportIntersection(tNear, SPHERE_HIT_FROM_OUTSIDE);
  else if (tFar > tmin && tFar < tmax)
    optixReportIntersection(tFar, SPHERE_HIT_FROM_INSIDE);
}

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

static __forceinline__ __device__ int sdfNearestCellIndex(const HitGroupData *rt, float3 p) {
  const float3 local = (p - rt->sdfOrigin) / rt->sdfVoxelSize - make_float3(0.5f, 0.5f, 0.5f);
  const int x = max(0, min(rt->sdfNx - 1, static_cast<int>(lroundf(local.x))));
  const int y = max(0, min(rt->sdfNy - 1, static_cast<int>(lroundf(local.y))));
  const int z = max(0, min(rt->sdfNz - 1, static_cast<int>(lroundf(local.z))));
  return (z * rt->sdfNy + y) * rt->sdfNx + x;
}

static __forceinline__ __device__ void sdfNearestCellMaterial(const HitGroupData *rt, float3 p, float3 &baseColor,
                                                                float &metallic, float &roughness) {
  const int idx = sdfNearestCellIndex(rt, p);
  baseColor = rt->sdfBaseColor[idx];
  metallic = rt->sdfMetallic[idx];
  roughness = rt->sdfRoughness[idx];
}

static __forceinline__ __device__ int sdfBranchAt(const HitGroupData *rt, float3 p) {
  if (!rt->sdfBranch)
    return -1;
  if (sampleSdf(rt, p) >= 0.0f)
    return -1;
  return static_cast<int>(rt->sdfBranch[sdfNearestCellIndex(rt, p)]);
}

static __forceinline__ __device__ SdfGpuMaterial sdfPaletteOrAir(const HitGroupData *rt, int branch) {
  if (branch < 0 || branch >= rt->sdfPaletteCount)
    return SdfGpuMaterial{SDF_MATERIAL_OPAQUE, make_float3(1.0f), 0.0f, 1.0f, 1.0f, make_float3(0.0f)};
  return rt->sdfPalette[branch];
}

static __forceinline__ __device__ void sdfProbeMaterials(const HitGroupData *rt, float3 p, float3 n,
                                                        SdfGpuMaterial &behind, SdfGpuMaterial &ahead) {
  const float bias = rt->sdfVoxelSize * 0.75f;
  behind = sdfPaletteOrAir(rt, sdfBranchAt(rt, p + n * bias));
  ahead = sdfPaletteOrAir(rt, sdfBranchAt(rt, p - n * bias));
}

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

  const float epsilon = rt->sdfVoxelSize * 0.02f;
  const float minStep = rt->sdfVoxelSize * 0.05f;
  const float minAdvance = rt->sdfVoxelSize * 0.3f;
  float t = t0;
  float prevT = t0;
  float prevDist = sampleSdf(rt, rayO + t0 * rayD);
  t += fmaxf(fabsf(prevDist), minStep);
  for (int i = 1; i < 512 && t <= t1; ++i) {
    const float dist = sampleSdf(rt, rayO + t * rayD);
    const bool crossed = (prevDist > 0.0f) != (dist > 0.0f);
    if ((crossed || fabsf(dist) < epsilon) && (t - t0) > minAdvance) {
      float tHit = t;
      if (crossed) {
        float a = prevT, b = t, fa = prevDist, fb = dist;
        for (int k = 0; k < 8; ++k) {
          const float m = 0.5f * (a + b);
          const float fm = sampleSdf(rt, rayO + m * rayD);
          if ((fa > 0.0f) != (fm > 0.0f)) {
            b = m;
            fb = fm;
          } else {
            a = m;
            fa = fm;
          }
        }
        const float denom = fa - fb;
        tHit = fabsf(denom) > 1e-9f ? a + (b - a) * (fa / denom) : 0.5f * (a + b);
      }
      optixReportIntersection(fmaxf(tHit, t0), 0);
      return;
    }
    prevT = t;
    prevDist = dist;
    t += fmaxf(fabsf(dist) * 0.85f, minStep);
  }
}

static __forceinline__ __device__ void cubicBsplineWeights(float t, float w[4], float dw[4]) {
  const float t2 = t * t, t3 = t2 * t;
  w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
  w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
  w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
  w[3] = t3 * (1.0f / 6.0f);
  dw[0] = (-1.0f + 2.0f * t - t2) * 0.5f;
  dw[1] = (-4.0f * t + 3.0f * t2) * 0.5f;
  dw[2] = (1.0f + 2.0f * t - 3.0f * t2) * 0.5f;
  dw[3] = t2 * 0.5f;
}

static __forceinline__ __device__ float3 sdfGradientNormal(const HitGroupData *rt, float3 p) {
  const float3 local = (p - rt->sdfOrigin) / rt->sdfVoxelSize - make_float3(0.5f, 0.5f, 0.5f);
  const int bx = static_cast<int>(floorf(local.x));
  const int by = static_cast<int>(floorf(local.y));
  const int bz = static_cast<int>(floorf(local.z));

  float wx[4], dwx[4], wy[4], dwy[4], wz[4], dwz[4];
  cubicBsplineWeights(local.x - bx, wx, dwx);
  cubicBsplineWeights(local.y - by, wy, dwy);
  cubicBsplineWeights(local.z - bz, wz, dwz);

  int ix[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    ix[i] = max(0, min(rt->sdfNx - 1, bx - 1 + i));

  float3 g = make_float3(0.0f);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int zRow = max(0, min(rt->sdfNz - 1, bz - 1 + k)) * rt->sdfNy;
    float3 plane = make_float3(0.0f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = (zRow + max(0, min(rt->sdfNy - 1, by - 1 + j))) * rt->sdfNx;
      float v = 0.0f, dv = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float f = rt->sdfDistances[row + ix[i]];
        v += wx[i] * f;
        dv += dwx[i] * f;
      }
      plane.x += dv * wy[j];
      plane.y += v * dwy[j];
      plane.z += v * wy[j];
    }
    g.x += plane.x * wz[k];
    g.y += plane.y * wz[k];
    g.z += plane.z * dwz[k];
  }
  return normalize(g);
}

#define GSPLAT_SIGMA_EXTENT 3.0f

static __forceinline__ __device__ float4 quatConjugate(float4 q) { return make_float4(-q.x, -q.y, -q.z, q.w); }

static __forceinline__ __device__ float3 quatRotate(float4 q, float3 v) {
  const float3 qv = make_float3(q.x, q.y, q.z);
  const float3 t = 2.0f * cross(qv, v);
  return v + q.w * t + cross(qv, t);
}

static __forceinline__ __device__ float3 splatLocalSigma(float4 rotation, float3 scale, float3 worldOffset) {
  return quatRotate(quatConjugate(rotation), worldOffset) / scale;
}

static __forceinline__ __device__ void reportEllipsoidIntersection(float3 so, float3 sd, float sigmaExtent2) {
  const float a = dot(sd, sd);
  const float b = dot(so, sd);
  const float c = dot(so, so) - sigmaExtent2;
  const float disc = b * b - a * c;
  if (disc < 0.0f)
    return;

  const float sqrtDisc = sqrtf(disc);
  const float tNear = (-b - sqrtDisc) / a;
  const float tFar = (-b + sqrtDisc) / a;
  const float tmin = optixGetRayTmin();
  const float tmax = optixGetRayTmax();

  if (tNear > tmin && tNear < tmax)
    optixReportIntersection(tNear, 0);
  else if (tFar > tmin && tFar < tmax)
    optixReportIntersection(tFar, 0);
}

extern "C" __global__ void __intersection__gsplat() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const unsigned int i = optixGetPrimitiveIndex();
  const float3 center = rt->splatPositions[i];
  const float3 scale = rt->splatScales[i];
  const float4 rotation = rt->splatRotations[i];

  const float3 rayO = optixGetObjectRayOrigin() - center;
  const float3 rayD = optixGetObjectRayDirection();
  const float3 so = splatLocalSigma(rotation, scale, rayO);
  const float3 sd = quatRotate(quatConjugate(rotation), rayD) / scale;

  reportEllipsoidIntersection(so, sd, GSPLAT_SIGMA_EXTENT * GSPLAT_SIGMA_EXTENT);
}

extern "C" __global__ void __anyhit__gsplat() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const unsigned int i = optixGetPrimitiveIndex();
  const float3 center = rt->splatPositions[i];
  const float3 scale = rt->splatScales[i];
  const float4 rotation = rt->splatRotations[i];

  const float3 hitP = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
  const float3 sigma = splatLocalSigma(rotation, scale, hitP - center);
  const float density = expf(-0.5f * dot(sigma, sigma));
  const float alpha = rt->splatOpacity[i] * density;

  unsigned int seed = optixGetPayload_3();
  const bool accept = sutil::rnd(seed) < alpha;
  optixSetPayload_3(seed);
  if (!accept)
    optixIgnoreIntersection();
}

static __forceinline__ __device__ void causticSplatFrame(float3 tangent, float3 normal, float stretchRatio,
                                                           float mergeRadius, float3 &bitangent, float3 &axisScale) {
  bitangent = cross(normal, tangent);
  const float stretchSqrt = sqrtf(fmaxf(stretchRatio, 1.0f));
  const float3 extent = make_float3(mergeRadius * stretchSqrt, mergeRadius / stretchSqrt, mergeRadius);
  axisScale = extent / GSPLAT_SIGMA_EXTENT;
}

static __forceinline__ __device__ void causticTangentAndStretch(float3 rayDir, float3 N, float3 &tangent,
                                                                  float &stretchRatio) {
  const float cosTheta = fabsf(dot(rayDir, N));
  const float3 tangentRaw = rayDir - N * dot(rayDir, N);
  const float tangentLenSq = dot(tangentRaw, tangentRaw);
  if (tangentLenSq > 1e-12f) {
    tangent = tangentRaw * rsqrtf(tangentLenSq);
  } else {
    const Onb onb(N);
    tangent = onb.m_tangent;
  }
  stretchRatio = clamp(1.0f / fmaxf(cosTheta, 1e-4f), 1.0f, 3.0f);
}

extern "C" __global__ void __intersection__causticSplat() {
  const unsigned int i = optixGetPrimitiveIndex();
  const LightVertex &lv = params.lightVertices[i];

  float3 bitangent, axisScale;
  causticSplatFrame(lv.tangent, lv.normal, lv.stretchRatio, params.mergeRadius, bitangent, axisScale);

  const float3 rayO = optixGetWorldRayOrigin() - lv.position;
  const float3 rayD = optixGetWorldRayDirection();
  const float3 so = make_float3(dot(rayO, lv.tangent), dot(rayO, bitangent), dot(rayO, lv.normal)) / axisScale;
  const float3 sd = make_float3(dot(rayD, lv.tangent), dot(rayD, bitangent), dot(rayD, lv.normal)) / axisScale;

  reportEllipsoidIntersection(so, sd, GSPLAT_SIGMA_EXTENT * GSPLAT_SIGMA_EXTENT);
}

static __forceinline__ __device__ float3 splatEllipsoidNormal(float4 rotation, float3 scale, float3 sigma) {
  const float3 gradLocal = sigma / scale;
  return normalize(quatRotate(rotation, gradLocal));
}

struct ShadingMaterial {
  float3 diffuse;
  float3 f0;
  float alpha;
  bool hasSpecular;
};

static __forceinline__ __device__ float luminance3(float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static __forceinline__ __device__ float ggxD(float NdotH, float alpha) {
  const float a2 = alpha * alpha;
  const float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
  return a2 / fmaxf(M_PIf * d * d, 1e-9f);
}

static __forceinline__ __device__ float smithG1(float NdotX, float alpha) {
  if (NdotX <= 0.0f)
    return 0.0f;
  const float a2 = alpha * alpha;
  return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * NdotX * NdotX));
}

static __forceinline__ __device__ float3 fresnelSchlick3(float3 f0, float cosTheta) {
  const float m = fminf(fmaxf(1.0f - cosTheta, 0.0f), 1.0f);
  const float m5 = m * m * m * m * m;
  return f0 + (make_float3(1.0f) - f0) * m5;
}

static __forceinline__ __device__ float specularLobeProbability(const ShadingMaterial &m) {
  if (!m.hasSpecular)
    return 0.0f;
  const float d = luminance3(m.diffuse);
  const float s = luminance3(m.f0);
  const float total = d + s;
  return total > 0.0f ? fminf(fmaxf(s / total, 0.1f), 0.9f) : 0.5f;
}

static __forceinline__ __device__ float3 sampleGgxVndf(float3 Ve, float alpha, float u1, float u2) {
  const float3 Vh = normalize(make_float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
  const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  const float3 T1 =
      lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
  const float3 T2 = cross(Vh, T1);
  const float r = sqrtf(u1);
  const float phi = 2.0f * M_PIf * u2;
  const float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  const float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
  const float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(1e-6f, Nh.z)));
}

static __forceinline__ __device__ void evalBsdf(const ShadingMaterial &m, float3 N, float3 V, float3 L,
                                                  float3 &fCos, float &pdf) {
  fCos = make_float3(0.0f);
  pdf = 0.0f;
  const float NdotL = dot(N, L);
  const float NdotV = dot(N, V);
  if (NdotL <= 0.0f || NdotV <= 0.0f)
    return;

  const float pSpec = specularLobeProbability(m);
  fCos = m.diffuse * (NdotL / M_PIf);
  pdf = (1.0f - pSpec) * (NdotL / M_PIf);

  if (m.hasSpecular) {
    const float3 H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float VdotH = dot(V, H);
    if (NdotH > 0.0f && VdotH > 0.0f) {
      const float D = ggxD(NdotH, m.alpha);
      const float G1v = smithG1(NdotV, m.alpha);
      const float G = G1v * smithG1(NdotL, m.alpha);
      const float3 F = fresnelSchlick3(m.f0, VdotH);
      fCos = fCos + F * (D * G / (4.0f * NdotV));
      pdf += pSpec * (G1v * D / (4.0f * NdotV));
    }
  }
}

static __forceinline__ __device__ bool sampleBsdf(const ShadingMaterial &m, float3 N, float3 V,
                                                    unsigned int &seed, float3 &L, float3 &weight,
                                                    float &pdf) {
  const Onb onb(N);
  if (sutil::rnd(seed) < specularLobeProbability(m)) {
    const float3 Vl = onb.toLocal(V);
    if (Vl.z <= 0.0f)
      return false;
    const float3 Hl = sampleGgxVndf(Vl, m.alpha, sutil::rnd(seed), sutil::rnd(seed));
    const float3 Ll = reflect(-Vl, Hl);
    if (Ll.z <= 0.0f)
      return false;
    L = onb.toWorld(Ll);
  } else {
    float3 local;
    cosineSampleHemisphere(sutil::rnd(seed), sutil::rnd(seed), local);
    L = onb.toWorld(local);
  }
  float3 fCos;
  evalBsdf(m, N, V, L, fCos, pdf);
  if (!(pdf > 1e-9f))
    return false;
  weight = fCos / pdf;
  return true;
}

static __forceinline__ __device__ ShadingMaterial materialFromMetallicRoughness(float3 baseColorFactor, float metallic,
                                                                                float roughness) {
  ShadingMaterial m;
  m.diffuse = baseColorFactor * (1.0f - metallic);
  m.f0 = lerp(make_float3(0.04f), baseColorFactor, metallic);
  const float r = fminf(fmaxf(roughness, 0.03f), 1.0f);
  m.alpha = r * r;
  m.hasSpecular = true;
  return m;
}

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

static __forceinline__ __device__ float3 computeGeometricNormal(const float3 &rayDir) {
  float3 n;
  if (optixIsTriangleHit()) {
    float3 verts[3];
    optixGetTriangleVertexData(verts);
    n = normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
  } else {
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int sbtGasIdx = optixGetSbtGASIndex();
    float4 q;
    optixGetSphereData(gas, primIdx, sbtGasIdx, 0.0f, &q);
    const float3 hitP = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    n = (hitP - make_float3(q)) / q.w;
  }
  return n;
}

static __forceinline__ __device__ float3 geometricNormalFor(const HitGroupData *rt, const float3 &P,
                                                              const float3 &rayDir) {
  if (rt->sphereRadius > 0.0f)
    return (P - rt->sphereCenter) / rt->sphereRadius;
  if (rt->sdfDistances)
    return sdfGradientNormal(rt, P);
  return computeGeometricNormal(rayDir);
}

static __forceinline__ __device__ float sampleNvdbDensity(const void *gridPtr, float3 worldPos) {
  const nanovdb::FloatGrid *grid = reinterpret_cast<const nanovdb::FloatGrid *>(gridPtr);
  const nanovdb::Vec3f ijkf = grid->worldToIndexF(nanovdb::Vec3f(worldPos.x, worldPos.y, worldPos.z));
  const nanovdb::Coord ijk = nanovdb::Coord::Floor(ijkf);
  auto acc = grid->tree().getAccessor();
  return acc.getValue(ijk);
}

static __forceinline__ __device__ bool rayVolumeInterval(const NvdbMedium &vol, float3 rayO, float3 rayD, float tmin,
                                                            float tmax, float &tEnter, float &tExit) {
  const float o[3] = {rayO.x, rayO.y, rayO.z};
  const float d[3] = {rayD.x, rayD.y, rayD.z};
  const float lo[3] = {vol.boundsMin.x, vol.boundsMin.y, vol.boundsMin.z};
  const float hi[3] = {vol.boundsMax.x, vol.boundsMax.y, vol.boundsMax.z};
  tEnter = tmin;
  tExit = tmax;
  int enterAxis;
  bool enterUpper;
  return slabTest(o, d, lo, hi, tEnter, tExit, enterAxis, enterUpper);
}

static __forceinline__ __device__ float evalHenyeyGreenstein(float cosTheta, float g) {
  const float g2 = g * g;
  const float denom = fmaxf(1.0f + g2 - 2.0f * g * cosTheta, 1e-6f);
  return (1.0f - g2) / (4.0f * M_PIf * denom * sqrtf(denom));
}

static __forceinline__ __device__ float3 sampleHenyeyGreenstein(float3 forward, float g, unsigned int &seed,
                                                                   float &pdfOut) {
  const float u1 = sutil::rnd(seed);
  const float u2 = sutil::rnd(seed);
  float cosTheta;
  if (fabsf(g) < 1e-3f) {
    cosTheta = 1.0f - 2.0f * u1;
  } else {
    const float sqrTerm = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
    cosTheta = -(1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
  }
  const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
  const float phi = 2.0f * M_PIf * u2;
  const float3 local = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
  const Onb onb(forward);
  pdfOut = evalHenyeyGreenstein(cosTheta, g);
  return onb.toWorld(local);
}

static __forceinline__ __device__ float3 mediumTransmittance(float3 rayO, float3 rayD, float tmin, float tmax,
                                                                unsigned int &seed) {
  if (!params.volume.enabled)
    return make_float3(1.0f);
  float tEnter, tExit;
  if (!rayVolumeInterval(params.volume, rayO, rayD, tmin, tmax, tEnter, tExit))
    return make_float3(1.0f);
  const float sigmaBar = params.volume.majorant;
  if (!(sigmaBar > 0.0f))
    return make_float3(1.0f);
  const float sigmaScalar =
      (params.volume.sigmaT.x + params.volume.sigmaT.y + params.volume.sigmaT.z) / 3.0f;
  float t = tEnter;
  float transmittance = 1.0f;
  for (int i = 0; i < 256; ++i) {
    t -= logf(1.0f - sutil::rnd(seed)) / sigmaBar;
    if (t >= tExit)
      break;
    const float3 pos = rayO + t * rayD;
    const float density = sampleNvdbDensity(params.volume.grid, pos) * params.volume.densityScale;
    const float sigmaLocal = sigmaScalar * density;
    transmittance *= 1.0f - sigmaLocal / sigmaBar;
  }
  return make_float3(transmittance);
}

static constexpr int kReservoirCandidates = 6;
static constexpr int kReservoirSpatialNeighbors = 4;
static constexpr int kReservoirSpatialRadius = 4;
static constexpr float kReservoirTemporalMaxM = 30.0f;

static __forceinline__ __device__ void pickQuadLightUniform(unsigned int &seed, float3 &pos, float3 &normal,
                                                              float3 &emission, float &pdfArea) {
  const unsigned int n = 1u + params.extraLightCount;
  unsigned int pick = static_cast<unsigned int>(sutil::rnd(seed) * static_cast<float>(n));
  if (pick >= n)
    pick = n - 1u;
  const QuadLight &light = pick == 0u ? params.light : params.extraLights[pick - 1u];
  const float z1 = sutil::rnd(seed), z2 = sutil::rnd(seed);
  pos = light.corner + light.v1 * z1 + light.v2 * z2;
  normal = light.normal;
  emission = light.emission;
  const float area = length(cross(light.v1, light.v2));
  pdfArea = 1.0f / (fmaxf(area, 1e-8f) * static_cast<float>(n));
}

static __forceinline__ __device__ void initReservoir(Reservoir &r) {
  r.sample = LightSample{};
  r.weightSum = 0.0f;
  r.M = 0.0f;
  r.W = 0.0f;
}

static __forceinline__ __device__ bool updateReservoir(Reservoir &r, const LightSample &s, float weight, float rnd) {
  r.weightSum += weight;
  r.M += 1.0f;
  if (weight > 0.0f && rnd < weight / r.weightSum) {
    r.sample = s;
    return true;
  }
  return false;
}

static __forceinline__ __device__ void combineReservoirs(Reservoir &r, const Reservoir &other, float targetPdfAtR,
                                                           float rnd) {
  if (other.M <= 0.0f)
    return;
  const float weight = targetPdfAtR * other.W * other.M;
  r.weightSum += weight;
  r.M += other.M;
  if (weight > 0.0f && rnd < weight / fmaxf(r.weightSum, 1e-12f))
    r.sample = other.sample;
}

static __forceinline__ __device__ void drawLightCandidate(unsigned int &seed, LightSample &ls) {
  const float pSun = params.sun.enabled ? 0.5f : 0.0f;
  if (params.sun.enabled && sutil::rnd(seed) < pSun) {
    float sunPdf;
    const float3 dir = sampleSunCone(seed, sunPdf);
    ls.lightType = LIGHT_SAMPLE_SUN;
    ls.dirOrPos = dir;
    ls.normal = make_float3(0.0f);
    ls.radiance = params.sun.radiance;
    ls.pdf = sunPdf * pSun;
  } else {
    const float otherPdfScale = params.sun.enabled ? (1.0f - pSun) : 1.0f;
    if (params.envTex) {
      float envPdf;
      const float3 dir = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), envPdf);
      ls.lightType = LIGHT_SAMPLE_ENV;
      ls.dirOrPos = dir;
      ls.normal = make_float3(0.0f);
      ls.radiance = lookupEnvironmentRadiance(dir);
      ls.pdf = envPdf * otherPdfScale;
    } else {
      float3 pos, normal, emission;
      float pdfArea;
      pickQuadLightUniform(seed, pos, normal, emission, pdfArea);
      ls.lightType = LIGHT_SAMPLE_QUAD;
      ls.dirOrPos = pos;
      ls.normal = normal;
      ls.radiance = emission;
      ls.pdf = pdfArea * otherPdfScale;
    }
  }
}

static __forceinline__ __device__ bool evalLightSampleAtPoint(const LightSample &ls, float3 P, float3 N, float3 V,
                                                                const ShadingMaterial &shading, float3 &dirOut,
                                                                float3 &fCosOut, float &pdfBsdfOut,
                                                                float &mixedPdfOut, float3 &radianceOut,
                                                                float &distOut) {
  if (ls.lightType == LIGHT_SAMPLE_QUAD) {
    const float3 toLight = ls.dirOrPos - P;
    const float dist = length(toLight);
    if (dist < 1e-6f)
      return false;
    dirOut = toLight / dist;
    const float lnDl = -dot(ls.normal, dirOut);
    if (lnDl <= 0.0f)
      return false;
    mixedPdfOut = ls.pdf * (dist * dist) / fmaxf(lnDl, 1e-6f);
    radianceOut = ls.radiance;
    distOut = dist;
  } else {
    dirOut = ls.dirOrPos;
    mixedPdfOut = ls.pdf;
    radianceOut = ls.radiance;
    distOut = 1e16f;
  }
  evalBsdf(shading, N, V, dirOut, fCosOut, pdfBsdfOut);
  return (fCosOut.x + fCosOut.y + fCosOut.z) > 0.0f && mixedPdfOut > 1e-8f;
}

static __forceinline__ __device__ float lightSampleTargetPdf(const LightSample &ls, float3 P, float3 N, float3 V,
                                                               const ShadingMaterial &shading) {
  float3 dir, fCos, rad;
  float pdfBsdf, mixedPdf, dist;
  if (!evalLightSampleAtPoint(ls, P, N, V, shading, dir, fCos, pdfBsdf, mixedPdf, rad, dist))
    return 0.0f;
  return luminance3(fCos * rad);
}

static __forceinline__ __device__ bool lightSampleTargetAndMixedPdf(const LightSample &ls, float3 P, float3 N,
                                                                      float3 V, const ShadingMaterial &shading,
                                                                      float &targetPdfOut, float &mixedPdfOut) {
  float3 dir, fCos, rad;
  float pdfBsdf, dist;
  if (!evalLightSampleAtPoint(ls, P, N, V, shading, dir, fCos, pdfBsdf, mixedPdfOut, rad, dist)) {
    targetPdfOut = 0.0f;
    return false;
  }
  targetPdfOut = luminance3(fCos * rad);
  return true;
}

static __forceinline__ __device__ void buildAndStoreReservoir(float3 P, float3 N, float3 V, float3 albedo,
                                                                const ShadingMaterial &shading, unsigned int &seed) {
  Reservoir r;
  initReservoir(r);
  for (int i = 0; i < kReservoirCandidates; ++i) {
    LightSample ls;
    drawLightCandidate(seed, ls);
    float targetPdf, mixedPdf;
    lightSampleTargetAndMixedPdf(ls, P, N, V, shading, targetPdf, mixedPdf);
    const float w = mixedPdf > 1e-8f ? targetPdf / mixedPdf : 0.0f;
    updateReservoir(r, ls, w, sutil::rnd(seed));
  }

  if (params.reservoirTemporal) {
    const uint3 idxT = optixGetLaunchIndex();
    unsigned int prevIdx = idxT.y * params.width + idxT.x;
    bool valid = true;
    if (params.cameraMoved) {
      valid = false;
      const float2 prevPixel = projectToPrevFrame(P - params.prevEye);
      if (prevPixel.x >= 0.0f && prevPixel.x <= static_cast<float>(params.width - 1) && prevPixel.y >= 0.0f &&
          prevPixel.y <= static_cast<float>(params.height - 1)) {
        const float3 reprojAlbedo =
            make_float3(sampleBuffer2D(params.prevAccumAlbedoBuffer, params.width, params.height, prevPixel));
        const float3 reprojNormal =
            make_float3(sampleBuffer2D(params.prevAccumNormalBuffer, params.width, params.height, prevPixel));
        if (!isDisoccluded(albedo, N, reprojAlbedo, reprojNormal)) {
          const int px = min(max(static_cast<int>(roundf(prevPixel.x)), 0), static_cast<int>(params.width) - 1);
          const int py = min(max(static_cast<int>(roundf(prevPixel.y)), 0), static_cast<int>(params.height) - 1);
          prevIdx = static_cast<unsigned int>(py) * params.width + static_cast<unsigned int>(px);
          valid = true;
        }
      }
    }
    if (valid) {
      Reservoir prevR = params.prevReservoirBuffer[prevIdx];
      prevR.M = fminf(prevR.M, kReservoirTemporalMaxM);
      if (prevR.M > 0.0f) {
        const float prevTargetAtSelf = lightSampleTargetPdf(prevR.sample, P, N, V, shading);
        combineReservoirs(r, prevR, prevTargetAtSelf, sutil::rnd(seed));
      }
    }
  }

  const float finalTarget = lightSampleTargetPdf(r.sample, P, N, V, shading);
  r.W = (r.weightSum > 0.0f && finalTarget > 0.0f) ? r.weightSum / (r.M * finalTarget) : 0.0f;
  const uint3 idx = optixGetLaunchIndex();
  params.reservoirBuffer[idx.y * params.width + idx.x] = r;
}

extern "C" __global__ void __closesthit__radiance() {
  HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 rayDir = optixGetWorldRayDirection();

  if (rt->materialType == MATERIAL_NVDB) {
    const float3 rayO = optixGetWorldRayOrigin();
    const float tEnter = optixGetRayTmax();
    const float tExit = __uint_as_float(optixGetAttribute_0());

    unsigned int seed = optixGetPayload_3();
    const int depth = static_cast<int>(optixGetPayload_4());
    float3 attenuation = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()),
                                      __uint_as_float(optixGetPayload_2()));
    const float3 throughput = attenuation;

    const float sigmaScalar = (rt->nvdbSigmaT.x + rt->nvdbSigmaT.y + rt->nvdbSigmaT.z) / 3.0f;
    const float sigmaBar = rt->nvdbMajorant;

    float3 emitted = make_float3(0.0f);
    float3 radiance = make_float3(0.0f);
    float3 nextOrigin = rayO + tExit * rayDir;
    float3 nextDirection = rayDir;
    float nextPdf = -1.0f;
    int done = 0;

    if (sigmaBar > 0.0f) {
      float t = tEnter;
      bool scattered = false;
      float3 scatterPos = make_float3(0.0f);
      for (int i = 0; i < 512; ++i) {
        t -= logf(1.0f - sutil::rnd(seed)) / sigmaBar;
        if (t >= tExit)
          break;
        const float3 pos = rayO + t * rayDir;
        const float density = sampleNvdbDensity(rt->nvdbGrid, pos) * rt->nvdbDensityScale;
        const float sigmaLocal = sigmaScalar * density;
        if (sutil::rnd(seed) < sigmaLocal / sigmaBar) {
          scatterPos = pos;
          scattered = true;
          break;
        }
      }

      if (scattered) {
        const float3 albedo = rt->nvdbScatterAlbedo;
        const float albedoAvg = fmaxf((albedo.x + albedo.y + albedo.z) / 3.0f, 1e-4f);
        if (sutil::rnd(seed) < albedoAvg) {
          const float3 albedoRatio = albedo / albedoAvg;
          attenuation = attenuation * albedoRatio;

          const float pSun = params.sun.enabled ? 0.5f : 0.0f;
          if (params.sun.enabled && sutil::rnd(seed) < pSun) {
            float sunPdf;
            const float3 dir = sampleSunCone(seed, sunPdf);
            const float phaseVal = evalHenyeyGreenstein(dot(dir, rayDir), rt->nvdbG);
            if (phaseVal > 0.0f && !traceOcclusion(params.handle, scatterPos, dir, 1e-3f, 1e16f)) {
              const float3 mediumT = mediumTransmittance(scatterPos, dir, 1e-3f, 1e16f, seed);
              const float mixedPdf = sunPdf * pSun;
              const float weight = powerHeuristic(mixedPdf, phaseVal);
              radiance = params.sun.radiance * phaseVal * mediumT * albedoRatio * (weight / fmaxf(mixedPdf, 1e-6f));
            }
          } else if (params.envTex) {
            float envPdf;
            const float3 dir = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), envPdf);
            const float otherPdfScale = params.sun.enabled ? (1.0f - pSun) : 1.0f;
            const float phaseVal = evalHenyeyGreenstein(dot(dir, rayDir), rt->nvdbG);
            if (envPdf > 0.0f && phaseVal > 0.0f && !traceOcclusion(params.handle, scatterPos, dir, 1e-3f, 1e16f)) {
              const float3 mediumT = mediumTransmittance(scatterPos, dir, 1e-3f, 1e16f, seed);
              const float3 envRadiance = lookupEnvironmentRadiance(dir);
              const float mixedPdf = envPdf * otherPdfScale;
              const float weight = powerHeuristic(mixedPdf, phaseVal);
              radiance = envRadiance * phaseVal * mediumT * albedoRatio * (weight / mixedPdf);
            }
          } else {
            float3 lightPos, lightNormal, lightEmission;
            float pdfArea;
            pickQuadLightUniform(seed, lightPos, lightNormal, lightEmission, pdfArea);
            const float3 toLight = lightPos - scatterPos;
            const float dist = length(toLight);
            const float3 dir = toLight / dist;
            const float lnDl = -dot(lightNormal, dir);
            const float phaseVal = evalHenyeyGreenstein(dot(dir, rayDir), rt->nvdbG);
            if (lnDl > 0.0f && phaseVal > 0.0f) {
              const float eps = 1e-3f;
              if (!traceOcclusion(params.handle, scatterPos, dir, eps, dist - 2.0f * eps)) {
                const float3 mediumT = mediumTransmittance(scatterPos, dir, eps, dist - 2.0f * eps, seed);
                const float pdfLight = (dist * dist) / fmaxf(lnDl, 1e-6f) * pdfArea;
                const float weight = powerHeuristic(pdfLight, phaseVal);
                radiance = lightEmission * phaseVal * mediumT * albedoRatio * (weight / fmaxf(pdfLight, 1e-6f));
              }
            }
          }

          float phasePdf;
          nextDirection = sampleHenyeyGreenstein(rayDir, rt->nvdbG, seed, phasePdf);
          nextOrigin = scatterPos;
          nextPdf = phasePdf;
        } else {
          done = 1;
        }
      }
    }

    emitted = emitted * throughput;
    radiance = radiance * throughput;

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
    if (depth == 0) {
      optixSetPayload_19(__float_as_uint(rt->nvdbScatterAlbedo.x));
      optixSetPayload_20(__float_as_uint(rt->nvdbScatterAlbedo.y));
      optixSetPayload_21(__float_as_uint(rt->nvdbScatterAlbedo.z));
      optixSetPayload_22(__float_as_uint(-rayDir.x));
      optixSetPayload_23(__float_as_uint(-rayDir.y));
      optixSetPayload_24(__float_as_uint(-rayDir.z));
      const uint3 idx = optixGetLaunchIndex();
      params.motionVectorBuffer[idx.y * params.width + idx.x] = make_float2(1e6f, 1e6f);
      params.denoiserFlowBuffer[idx.y * params.width + idx.x] = make_float2(0.0f, 0.0f);
    }
    return;
  }

  float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;

  float3 N;
  float3 Ng = make_float3(0.0f, 1.0f, 0.0f);
  float3 albedo = rt->albedo;
  ShadingMaterial shading;
  shading.diffuse = albedo;
  shading.f0 = make_float3(0.0f);
  shading.alpha = 1.0f;
  shading.hasSpecular = false;

  float transmission = 0.0f;
  float dielectricIor = rt->ior;
  float3 dielectricExtinction = rt->extinction;
  float sdfIorFrom = 1.0f, sdfIorTo = 1.0f;
  float3 sdfExtinctionFrom = make_float3(0.0f);

  if (rt->materialType == MATERIAL_TEXTURED_DIFFUSE) {
    const unsigned int prim = optixGetPrimitiveIndex();
    const float2 bary = optixGetTriangleBarycentrics();
    const float w0 = 1.0f - bary.x - bary.y, w1 = bary.x, w2 = bary.y;
    N = normalize(w0 * rt->normals[prim * 3 + 0] + w1 * rt->normals[prim * 3 + 1] + w2 * rt->normals[prim * 3 + 2]);
    N = faceforward(N, -rayDir, N);

    const unsigned int matIdx = rt->triangleMaterial ? rt->triangleMaterial[prim] : 0u;
    const GpuMaterial &mat = rt->materials[matIdx];

    const float2 uv0 = rt->uvs[prim * 3 + 0], uv1 = rt->uvs[prim * 3 + 1], uv2 = rt->uvs[prim * 3 + 2];
    const float u = w0 * uv0.x + w1 * uv1.x + w2 * uv2.x;
    const float v = w0 * uv0.y + w1 * uv1.y + w2 * uv2.y;

    float3 baseColor = mat.baseColorFactor;
    if (mat.baseColorTex) {
      const float4 texel = tex2D<float4>(mat.baseColorTex, u, v);
      baseColor = baseColor * make_float3(texel.x, texel.y, texel.z);
    }
    float metallic = mat.metallic;
    float roughness = mat.roughness;
    if (mat.metallicRoughnessTex) {
      const float4 mr = tex2D<float4>(mat.metallicRoughnessTex, u, v);
      roughness *= mr.y;
      metallic *= mr.z;
    }

    if (mat.normalTex && rt->tangents) {
      const float4 t0 = rt->tangents[prim * 3 + 0], t1 = rt->tangents[prim * 3 + 1],
                   t2 = rt->tangents[prim * 3 + 2];
      float3 T = w0 * make_float3(t0.x, t0.y, t0.z) + w1 * make_float3(t1.x, t1.y, t1.z) +
                 w2 * make_float3(t2.x, t2.y, t2.z);
      T = T - N * dot(N, T);
      const float tlen = length(T);
      if (tlen > 1e-6f) {
        T = T / tlen;
        const float3 B = cross(N, T) * t0.w;
        const float4 nt = tex2D<float4>(mat.normalTex, u, v);
        float3 tn = make_float3(nt.x * 2.0f - 1.0f, nt.y * 2.0f - 1.0f, nt.z * 2.0f - 1.0f);
        tn.x *= mat.normalScale;
        tn.y *= mat.normalScale;
        const float3 mapped = normalize(tn.x * T + tn.y * B + tn.z * N);
        if (dot(mapped, -rayDir) > 0.0f)
          N = mapped;
      }
    }

    albedo = baseColor;
    shading.diffuse = baseColor * (1.0f - metallic);
    shading.f0 = lerp(make_float3(0.04f), baseColor, metallic);
    const float r = fminf(fmaxf(roughness, 0.03f), 1.0f);
    shading.alpha = r * r;
    shading.hasSpecular = true;

    transmission = mat.transmission;
    if (transmission > 0.0f) {
      dielectricIor = mat.ior;
      dielectricExtinction = attenuationToExtinction(mat.attenuationColor, mat.attenuationDistance);
      Ng = computeGeometricNormal(rayDir);
    }
  } else if (rt->materialType == MATERIAL_VOXEL) {
    const float3 faceN = voxelFaceNormal(optixGetAttribute_0());
    N = faceforward(faceN, -rayDir, faceN);
    albedo = rt->voxelColors[optixGetPrimitiveIndex()];
    shading.diffuse = albedo;
  } else if (rt->materialType == MATERIAL_SDF) {
    const float3 gradN = sdfGradientNormal(rt, P);
    N = faceforward(gradN, -rayDir, gradN);
    Ng = gradN;
    SdfGpuMaterial matBehind, matAhead;
    sdfProbeMaterials(rt, P, N, matBehind, matAhead);
    if (matBehind.kind == SDF_MATERIAL_DIELECTRIC || matAhead.kind == SDF_MATERIAL_DIELECTRIC) {
      transmission = 1.0f;
      sdfIorFrom = matBehind.kind == SDF_MATERIAL_DIELECTRIC ? matBehind.ior : 1.0f;
      sdfIorTo = matAhead.kind == SDF_MATERIAL_DIELECTRIC ? matAhead.ior : 1.0f;
      sdfExtinctionFrom = matBehind.kind == SDF_MATERIAL_DIELECTRIC ? matBehind.extinction : make_float3(0.0f);
    } else {
      float sdfMetallic = matAhead.metallic;
      float sdfRoughness = matAhead.roughness;
      if (rt->sdfBranch) {
        albedo = matAhead.baseColor;
      } else {
        sdfNearestCellMaterial(rt, P, albedo, sdfMetallic, sdfRoughness);
      }
      shading = materialFromMetallicRoughness(albedo, sdfMetallic, sdfRoughness);
      P = P + N * (rt->sdfVoxelSize * 0.25f);
    }
  } else if (rt->materialType == MATERIAL_GSPLAT) {
    const unsigned int prim = optixGetPrimitiveIndex();
    const float3 center = rt->splatPositions[prim];
    const float3 scale = rt->splatScales[prim];
    const float4 rotation = rt->splatRotations[prim];
    const float3 sigma = splatLocalSigma(rotation, scale, P - center);
    const float3 gradN = splatEllipsoidNormal(rotation, scale, sigma);
    N = faceforward(gradN, -rayDir, gradN);
    albedo = rt->splatColors[prim];
    shading.diffuse = albedo;
  } else {
    Ng = geometricNormalFor(rt, P, rayDir);
    N = faceforward(Ng, -rayDir, Ng);
  }

  unsigned int seed = optixGetPayload_3();
  int depth = static_cast<int>(optixGetPayload_4());
  const float prevBsdfPdf = __uint_as_float(optixGetPayload_5());
  float3 attenuation =
      make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));
  const float3 throughput = attenuation;

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
        const float otherPdfScale = params.sun.enabled ? 0.5f : 1.0f;
        const float pdfLight = (dist * dist) / (cosLight * area) * otherPdfScale;
        weight = powerHeuristic(prevBsdfPdf, pdfLight);
      }
      emitted = rt->emission * weight;
    }
    done = 1;
  } else if ((rt->materialType == MATERIAL_DIFFUSE || rt->materialType == MATERIAL_TEXTURED_DIFFUSE ||
              rt->materialType == MATERIAL_VOXEL || rt->materialType == MATERIAL_SDF ||
              rt->materialType == MATERIAL_GSPLAT) &&
             !(transmission > 0.0f && sutil::rnd(seed) < transmission)) {
    const float3 V = -rayDir;

    if (params.reservoirBuildPass) {
      if (params.reservoirNEE)
        buildAndStoreReservoir(P, N, V, albedo, shading, seed);
      done = 1;
    } else {
    const bool useReservoir = params.reservoirNEE && depth == 0;
    if (useReservoir) {
      const uint3 idx2 = optixGetLaunchIndex();
      const unsigned int pixel2 = idx2.y * params.width + idx2.x;
      Reservoir combined;
      initReservoir(combined);
      const Reservoir &own = params.reservoirBuffer[pixel2];
      if (own.M > 0.0f) {
        const float targetOwnAtSelf = lightSampleTargetPdf(own.sample, P, N, V, shading);
        combineReservoirs(combined, own, targetOwnAtSelf, sutil::rnd(seed));
      }
      for (int k = 0; k < kReservoirSpatialNeighbors; ++k) {
        const int ox = static_cast<int>(sutil::rnd(seed) * static_cast<float>(2 * kReservoirSpatialRadius + 1)) -
                       kReservoirSpatialRadius;
        const int oy = static_cast<int>(sutil::rnd(seed) * static_cast<float>(2 * kReservoirSpatialRadius + 1)) -
                       kReservoirSpatialRadius;
        const int nx = min(max(static_cast<int>(idx2.x) + ox, 0), static_cast<int>(params.width) - 1);
        const int ny = min(max(static_cast<int>(idx2.y) + oy, 0), static_cast<int>(params.height) - 1);
        const Reservoir &nb = params.reservoirBuffer[ny * params.width + nx];
        if (nb.M <= 0.0f)
          continue;
        const float targetAtSelf = lightSampleTargetPdf(nb.sample, P, N, V, shading);
        combineReservoirs(combined, nb, targetAtSelf, sutil::rnd(seed));
      }
      const float finalTarget = lightSampleTargetPdf(combined.sample, P, N, V, shading);
      combined.W =
          (combined.weightSum > 0.0f && finalTarget > 0.0f) ? combined.weightSum / (combined.M * finalTarget) : 0.0f;

      if (combined.W > 0.0f) {
        float3 dir, fCos, rad;
        float pdfBsdf, mixedPdf, dist;
        if (evalLightSampleAtPoint(combined.sample, P, N, V, shading, dir, fCos, pdfBsdf, mixedPdf, rad, dist)) {
          const float eps = surfaceEpsilon(P);
          const float tmax = dist < 1e15f ? dist - 2.0f * eps : 1e16f;
          const bool occluded = traceOcclusion(params.handle, P, dir, eps, tmax);
          if (!occluded) {
            const float3 mediumT = mediumTransmittance(P, dir, eps, tmax, seed);
            const float weight = powerHeuristic(mixedPdf, pdfBsdf);
            radiance = rad * fCos * mediumT * weight * combined.W;
          }
        }
      }
    } else {
    const float pSun = params.sun.enabled ? 0.5f : 0.0f;
    if (params.sun.enabled && sutil::rnd(seed) < pSun) {
      float sunPdf;
      const float3 dir = sampleSunCone(seed, sunPdf);
      float3 fCos;
      float pdfBsdf;
      evalBsdf(shading, N, V, dir, fCos, pdfBsdf);
      if (fCos.x + fCos.y + fCos.z > 0.0f) {
        const bool occluded = traceOcclusion(params.handle, P, dir, surfaceEpsilon(P), 1e16f);
        if (!occluded) {
          const float3 mediumT = mediumTransmittance(P, dir, surfaceEpsilon(P), 1e16f, seed);
          const float mixedPdf = sunPdf * pSun;
          const float weight = powerHeuristic(mixedPdf, pdfBsdf);
          radiance = params.sun.radiance * fCos * mediumT * (weight / fmaxf(mixedPdf, 1e-6f));
        }
      }
    } else {
      const float otherPdfScale = params.sun.enabled ? (1.0f - pSun) : 1.0f;
      if (params.envTex) {
        float envPdf;
        const float3 dir = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), envPdf);
        if (envPdf > 0.0f) {
          float3 fCos;
          float pdfBsdf;
          evalBsdf(shading, N, V, dir, fCos, pdfBsdf);
          if (fCos.x + fCos.y + fCos.z > 0.0f) {
            const bool occluded = traceOcclusion(params.handle, P, dir, surfaceEpsilon(P), 1e16f);
            if (!occluded) {
              const float3 mediumT = mediumTransmittance(P, dir, surfaceEpsilon(P), 1e16f, seed);
              const float3 envRadiance = lookupEnvironmentRadiance(dir);
              const float mixedPdf = envPdf * otherPdfScale;
              const float weight = powerHeuristic(mixedPdf, pdfBsdf);
              radiance = envRadiance * fCos * mediumT * (weight / mixedPdf);
            }
          }
        }
      } else {
        float3 lightPos, lightNormal, lightEmission;
        float pdfArea;
        pickQuadLightUniform(seed, lightPos, lightNormal, lightEmission, pdfArea);
        const float3 toLight = lightPos - P;
        const float dist = length(toLight);
        const float3 L = toLight / dist;
        const float lnDl = -dot(lightNormal, L);
        if (lnDl > 0.0f) {
          float3 fCos;
          float pdfBsdf;
          evalBsdf(shading, N, V, L, fCos, pdfBsdf);
          if (fCos.x + fCos.y + fCos.z > 0.0f) {
            const float eps = surfaceEpsilon(P);
            const bool occluded = traceOcclusion(params.handle, P, L, eps, dist - 2.0f * eps);
            if (!occluded) {
              const float3 mediumT = mediumTransmittance(P, L, eps, dist - 2.0f * eps, seed);
              const float pdfLight = (dist * dist) / fmaxf(lnDl, 1e-6f) * pdfArea * otherPdfScale;
              const float weight = powerHeuristic(pdfLight, pdfBsdf);
              radiance = lightEmission * fCos * mediumT * (weight / fmaxf(pdfLight, 1e-6f));
            }
          }
        }
      }
    }
    }

    if (params.lightVertexCount > 0 &&
        (rt->materialType == MATERIAL_DIFFUSE || rt->materialType == MATERIAL_TEXTURED_DIFFUSE)) {
      const float mergeRadius = params.mergeRadius;
      const float mergeRadius2 = mergeRadius * mergeRadius;
      const float mergeDiskArea = M_PIf * fmaxf(mergeRadius2, 1e-12f);

      float3 merged = make_float3(0.0f);
      if (params.vertexMergeHandle && mergeRadius > 0.0f) {
        unsigned int g0 = __float_as_uint(N.x), g1 = __float_as_uint(N.y), g2 = __float_as_uint(N.z);
        unsigned int g3 = __float_as_uint(V.x), g4 = __float_as_uint(V.y), g5 = __float_as_uint(V.z);
        unsigned int g6 = __float_as_uint(shading.diffuse.x), g7 = __float_as_uint(shading.diffuse.y),
                     g8 = __float_as_uint(shading.diffuse.z);
        unsigned int g9 = __float_as_uint(shading.f0.x), g10 = __float_as_uint(shading.f0.y),
                     g11 = __float_as_uint(shading.f0.z);
        unsigned int g12 = __float_as_uint(shading.alpha);
        unsigned int g13 = shading.hasSpecular ? 1u : 0u;
        unsigned int g14 = 0u, g15 = 0u, g16 = 0u;
        const float tmax = 2.02f * mergeRadius;
        const float3 mergeOrigin = P - N * (1.01f * mergeRadius);
        optixTrace(params.vertexMergeHandle, mergeOrigin, N, 0.0f, tmax, 0.0f, OptixVisibilityMask(1),
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, params.mergeHitSbtOffset, 1, 2, g0, g1, g2, g3, g4, g5, g6, g7,
                   g8, g9, g10, g11, g12, g13, g14, g15, g16);
        merged = make_float3(__uint_as_float(g14), __uint_as_float(g15), __uint_as_float(g16)) / mergeDiskArea;
      }

      const bool subsample =
          params.maxConnectionsPerVertex > 0 && params.lightVertexCount > params.maxConnectionsPerVertex;
      const unsigned int iterCount = subsample ? params.maxConnectionsPerVertex : params.lightVertexCount;
      const float reweight =
          subsample ? static_cast<float>(params.lightVertexCount) / static_cast<float>(params.maxConnectionsPerVertex)
                    : 1.0f;

      float3 connected = make_float3(0.0f);
      for (unsigned int k = 0; k < iterCount; ++k) {
        unsigned int i = k;
        if (subsample) {
          i = static_cast<unsigned int>(sutil::rnd(seed) * static_cast<float>(params.lightVertexCount));
          if (i >= params.lightVertexCount)
            i = params.lightVertexCount - 1;
        }
        const LightVertex &lv = params.lightVertices[i];
        const float3 toVertex = lv.position - P;
        const float dist2 = dot(toVertex, toVertex);

        if (mergeRadius > 0.0f && dist2 <= mergeRadius2 && dot(N, lv.normal) > 0.0f)
          continue;

        if (dist2 < 1e-8f)
          continue;
        const float dist = sqrtf(dist2);
        const float3 dir = toVertex / dist;
        const float cosLight = dot(lv.normal, -dir);
        if (cosLight <= 0.0f)
          continue;
        float3 fCosEye;
        float pdfEyeUnused;
        evalBsdf(shading, N, V, dir, fCosEye, pdfEyeUnused);
        if (fCosEye.x + fCosEye.y + fCosEye.z <= 0.0f)
          continue;
        const float eps = surfaceEpsilon(P);
        if (traceOcclusion(params.handle, P, dir, eps, dist - 2.0f * eps))
          continue;
        const ShadingMaterial lightMat = materialFromMetallicRoughness(lv.baseColorFactor, lv.metallic, lv.roughness);
        float3 fLight;
        float pdfLightUnused;
        evalBsdf(lightMat, lv.normal, -lv.direction, -dir, fLight, pdfLightUnused);
        if (fLight.x + fLight.y + fLight.z <= 0.0f)
          continue;
        connected += fCosEye * fLight * (lv.throughput / dist2);
      }
      radiance += connected * reweight + merged;
    }

    float3 bounceDir, bounceWeight;
    float bouncePdf;
    if (sampleBsdf(shading, N, V, seed, bounceDir, bounceWeight, bouncePdf)) {
      nextDirection = bounceDir;
      nextPdf = bouncePdf;
      attenuation = attenuation * bounceWeight;
    } else {
      done = 1;
    }
    }
  } else if (rt->materialType == MATERIAL_MIRROR) {
    nextDirection = reflect(rayDir, N);
    attenuation = attenuation * rt->albedo;
    nextPdf = -1.0f;
  } else {
    float iorFrom, iorTo;
    float3 extinctionFrom;
    if (rt->materialType == MATERIAL_SDF) {
      iorFrom = sdfIorFrom;
      iorTo = sdfIorTo;
      extinctionFrom = sdfExtinctionFrom;
    } else {
      const bool entering = rt->sphereRadius > 0.0f ? (optixGetHitKind() == SPHERE_HIT_FROM_OUTSIDE)
                                                     : (dot(rayDir, Ng) < 0.0f);
      iorFrom = entering ? 1.0f : dielectricIor;
      iorTo = entering ? dielectricIor : 1.0f;
      extinctionFrom = entering ? make_float3(0.0f) : dielectricExtinction;
    }

    nextDirection = evalDielectricBounce(rayDir, N, iorFrom, iorTo, extinctionFrom, optixGetRayTmax(),
                                          /*isLightTransport=*/false, seed, attenuation);
    nextPdf = -1.0f;
  }

  emitted = emitted * throughput;
  radiance = radiance * throughput;

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

  if (depth == 0) {
    optixSetPayload_19(__float_as_uint(albedo.x));
    optixSetPayload_20(__float_as_uint(albedo.y));
    optixSetPayload_21(__float_as_uint(albedo.z));
    optixSetPayload_22(__float_as_uint(N.x));
    optixSetPayload_23(__float_as_uint(N.y));
    optixSetPayload_24(__float_as_uint(N.z));
    writeMotionVector(optixGetLaunchIndex(), P - params.prevEye);
  }
}

extern "C" __global__ void __anyhit__merge() {
  const LightVertex &lv = params.lightVertices[optixGetPrimitiveIndex()];

  const float3 N = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()),
                                __uint_as_float(optixGetPayload_2()));

  if (dot(N, lv.normal) <= 0.0f) {
    optixIgnoreIntersection();
    return;
  }

  const float3 P = optixGetWorldRayOrigin() + N * (1.01f * params.mergeRadius);
  const float3 offset = lv.position - P;

  float3 bitangent, axisScale;
  causticSplatFrame(lv.tangent, lv.normal, lv.stretchRatio, params.mergeRadius, bitangent, axisScale);
  const float3 sigma =
      make_float3(dot(offset, lv.tangent), dot(offset, bitangent), dot(offset, lv.normal)) / axisScale;
  const float d2 = dot(sigma, sigma);
  if (d2 > GSPLAT_SIGMA_EXTENT * GSPLAT_SIGMA_EXTENT) {
    optixIgnoreIntersection();
    return;
  }
  const float weight = expf(-0.5f * d2);

  ShadingMaterial m;
  m.diffuse = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()),
                           __uint_as_float(optixGetPayload_8()));
  m.f0 = make_float3(__uint_as_float(optixGetPayload_9()), __uint_as_float(optixGetPayload_10()),
                      __uint_as_float(optixGetPayload_11()));
  m.alpha = __uint_as_float(optixGetPayload_12());
  m.hasSpecular = optixGetPayload_13() != 0u;

  const float3 V = make_float3(__uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()),
                                __uint_as_float(optixGetPayload_5()));

  float3 fCosEye;
  float pdfUnused;
  evalBsdf(m, N, V, -lv.direction, fCosEye, pdfUnused);
  if (fCosEye.x + fCosEye.y + fCosEye.z > 0.0f) {
    const float3 sum = make_float3(__uint_as_float(optixGetPayload_14()), __uint_as_float(optixGetPayload_15()),
                                    __uint_as_float(optixGetPayload_16())) +
                        fCosEye * lv.throughput * weight;
    optixSetPayload_14(__float_as_uint(sum.x));
    optixSetPayload_15(__float_as_uint(sum.y));
    optixSetPayload_16(__float_as_uint(sum.z));
  }
  optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__causticAabb() {
  const unsigned int i = optixGetLaunchIndex().x;
  if (i >= params.lightVertexCount)
    return;
  const LightVertex &lv = params.lightVertices[i];

  float3 bitangent, axisScale;
  causticSplatFrame(lv.tangent, lv.normal, lv.stretchRatio, params.mergeRadius, bitangent, axisScale);
  const float3 extent = axisScale * GSPLAT_SIGMA_EXTENT;

  const float3 absTangent = make_float3(fabsf(lv.tangent.x), fabsf(lv.tangent.y), fabsf(lv.tangent.z));
  const float3 absBitangent = make_float3(fabsf(bitangent.x), fabsf(bitangent.y), fabsf(bitangent.z));
  const float3 absNormal = make_float3(fabsf(lv.normal.x), fabsf(lv.normal.y), fabsf(lv.normal.z));
  const float3 worldHalf = absTangent * extent.x + absBitangent * extent.y + absNormal * extent.z;

  params.causticAabbs[i] = OptixAabb{lv.position.x - worldHalf.x, lv.position.y - worldHalf.y,
                                      lv.position.z - worldHalf.z, lv.position.x + worldHalf.x,
                                      lv.position.y + worldHalf.y, lv.position.z + worldHalf.z};
}

extern "C" __global__ void __closesthit__lightSubpath() {
  HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 rayDir = optixGetWorldRayDirection();
  const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
  const float3 Ng = geometricNormalFor(rt, P, rayDir);
  const float3 N = faceforward(Ng, -rayDir, Ng);

  float3 causticTangent;
  float causticStretchRatio;
  causticTangentAndStretch(rayDir, N, causticTangent, causticStretchRatio);

  unsigned int seed = optixGetPayload_0();
  float3 throughput = make_float3(__uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()),
                                    __uint_as_float(optixGetPayload_3()));

  float3 nextDirection = rayDir;
  unsigned int done = 1u;
  unsigned int sawSpecular = optixGetPayload_11();

  if (rt->materialType == MATERIAL_MIRROR) {
    nextDirection = reflect(rayDir, N);
    throughput = throughput * rt->albedo;
    sawSpecular = 1u;
    done = 0u;
  } else if (rt->materialType == MATERIAL_GLASS) {
    const bool entering = rt->sphereRadius > 0.0f ? (optixGetHitKind() == SPHERE_HIT_FROM_OUTSIDE)
                                                   : (dot(rayDir, Ng) < 0.0f);
    const float iorFrom = entering ? 1.0f : rt->ior;
    const float iorTo = entering ? rt->ior : 1.0f;
    const float3 extinctionFrom = entering ? make_float3(0.0f) : rt->extinction;
    nextDirection = evalDielectricBounce(rayDir, N, iorFrom, iorTo, extinctionFrom, optixGetRayTmax(),
                                          /*isLightTransport=*/true, seed, throughput);
    sawSpecular = 1u;
    done = 0u;
  } else if (rt->materialType == MATERIAL_SDF) {
    SdfGpuMaterial matBehind, matAhead;
    sdfProbeMaterials(rt, P, N, matBehind, matAhead);
    if (matBehind.kind == SDF_MATERIAL_DIELECTRIC || matAhead.kind == SDF_MATERIAL_DIELECTRIC) {
      const float iorFrom = matBehind.kind == SDF_MATERIAL_DIELECTRIC ? matBehind.ior : 1.0f;
      const float iorTo = matAhead.kind == SDF_MATERIAL_DIELECTRIC ? matAhead.ior : 1.0f;
      const float3 extinctionFrom =
          matBehind.kind == SDF_MATERIAL_DIELECTRIC ? matBehind.extinction : make_float3(0.0f);
      nextDirection = evalDielectricBounce(rayDir, N, iorFrom, iorTo, extinctionFrom, optixGetRayTmax(),
                                            /*isLightTransport=*/true, seed, throughput);
      sawSpecular = 1u;
      done = 0u;
    }
  } else if (rt->materialType == MATERIAL_TEXTURED_DIFFUSE) {
    const unsigned int prim = optixGetPrimitiveIndex();
    const unsigned int matIdx = rt->triangleMaterial ? rt->triangleMaterial[prim] : 0u;
    const GpuMaterial &mat = rt->materials[matIdx];

    if (mat.transmission > 0.0f) {
      const bool entering = dot(rayDir, Ng) < 0.0f;
      const float3 extinction = attenuationToExtinction(mat.attenuationColor, mat.attenuationDistance);
      const float iorFrom = entering ? 1.0f : mat.ior;
      const float iorTo = entering ? mat.ior : 1.0f;
      const float3 extinctionFrom = entering ? make_float3(0.0f) : extinction;
      nextDirection = evalDielectricBounce(rayDir, N, iorFrom, iorTo, extinctionFrom, optixGetRayTmax(),
                                            /*isLightTransport=*/true, seed, throughput);
      sawSpecular = 1u;
      done = 0u;
    } else if (mat.metallic > kCausticMirrorMetallic && mat.roughness < kCausticMirrorRoughness) {
      nextDirection = reflect(rayDir, N);
      throughput = throughput * mat.baseColorFactor;
      sawSpecular = 1u;
      done = 0u;
    } else {
      const unsigned int idx = sawSpecular ? atomicAdd(params.lightVertexCounter, 1u) : params.lightVertexCapacity;
      if (idx < params.lightVertexCapacity) {
        params.lightVertices[idx].position = P;
        params.lightVertices[idx].normal = N;
        params.lightVertices[idx].direction = rayDir;
        params.lightVertices[idx].throughput = throughput;
        params.lightVertices[idx].baseColorFactor = mat.baseColorFactor;
        params.lightVertices[idx].metallic = mat.metallic;
        params.lightVertices[idx].roughness = mat.roughness;
        params.lightVertices[idx].tangent = causticTangent;
        params.lightVertices[idx].stretchRatio = causticStretchRatio;
      }
      const ShadingMaterial lightMat = materialFromMetallicRoughness(mat.baseColorFactor, mat.metallic, mat.roughness);
      float3 bounceDir, bounceWeight;
      float bouncePdf;
      if (sampleBsdf(lightMat, N, -rayDir, seed, bounceDir, bounceWeight, bouncePdf)) {
        nextDirection = bounceDir;
        throughput = throughput * bounceWeight;
        done = 0u;
      } else {
        done = 1u;
      }
    }
  } else if (rt->materialType == MATERIAL_DIFFUSE) {
    const unsigned int idx = sawSpecular ? atomicAdd(params.lightVertexCounter, 1u) : params.lightVertexCapacity;
    if (idx < params.lightVertexCapacity) {
      params.lightVertices[idx].position = P;
      params.lightVertices[idx].normal = N;
      params.lightVertices[idx].direction = rayDir;
      params.lightVertices[idx].throughput = throughput;
      params.lightVertices[idx].baseColorFactor = rt->albedo;
      params.lightVertices[idx].metallic = 0.0f;
      params.lightVertices[idx].roughness = 1.0f;
      params.lightVertices[idx].tangent = causticTangent;
      params.lightVertices[idx].stretchRatio = causticStretchRatio;
    }
    const ShadingMaterial lightMat = materialFromMetallicRoughness(rt->albedo, 0.0f, 1.0f);
    float3 bounceDir, bounceWeight;
    float bouncePdf;
    if (sampleBsdf(lightMat, N, -rayDir, seed, bounceDir, bounceWeight, bouncePdf)) {
      nextDirection = bounceDir;
      throughput = throughput * bounceWeight;
      done = 0u;
    } else {
      done = 1u;
    }
  }

  optixSetPayload_11(sawSpecular);
  optixSetPayload_0(seed);
  optixSetPayload_1(__float_as_uint(throughput.x));
  optixSetPayload_2(__float_as_uint(throughput.y));
  optixSetPayload_3(__float_as_uint(throughput.z));
  optixSetPayload_4(__float_as_uint(nextDirection.x));
  optixSetPayload_5(__float_as_uint(nextDirection.y));
  optixSetPayload_6(__float_as_uint(nextDirection.z));
  optixSetPayload_7(done);
  optixSetPayload_8(__float_as_uint(P.x));
  optixSetPayload_9(__float_as_uint(P.y));
  optixSetPayload_10(__float_as_uint(P.z));
}

extern "C" __global__ void __miss__lightSubpath() { optixSetPayload_7(1u); }

static __forceinline__ __device__ void emitFromQuadLight(unsigned int &seed, float3 &origin, float3 &direction,
                                                           float3 &power) {
  const QuadLight &light = params.light;
  const float z1 = sutil::rnd(seed), z2 = sutil::rnd(seed);
  origin = light.corner + light.v1 * z1 + light.v2 * z2;
  float3 local;
  cosineSampleHemisphere(sutil::rnd(seed), sutil::rnd(seed), local);
  const Onb onb(light.normal);
  direction = onb.toWorld(local);
  const float area = length(cross(light.v1, light.v2));
  power = (light.emission * area * M_PIf) / fmaxf(static_cast<float>(params.lightSubpathBatchSize), 1.0f);
}

static __forceinline__ __device__ void emitFromEnvironment(unsigned int &seed, float3 &origin, float3 &direction,
                                                             float3 &power) {
  float pdfDir;
  const float3 toLight = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), pdfDir);
  direction = -toLight;
  const float3 radiance = lookupEnvironmentRadiance(toLight);

  const Onb onb(direction);
  const float2 disk = concentricSampleDisk(sutil::rnd(seed), sutil::rnd(seed));
  const float r = params.sceneBoundsRadius;
  origin = params.sceneBoundsCenter - direction * r + onb.m_tangent * (disk.x * r) + onb.m_binormal * (disk.y * r);

  const float diskArea = M_PIf * r * r;
  power = pdfDir > 1e-8f
              ? (radiance * diskArea) / (pdfDir * fmaxf(static_cast<float>(params.lightSubpathBatchSize), 1.0f))
              : make_float3(0.0f);
}

static __forceinline__ __device__ void emitFromSun(unsigned int &seed, float3 &origin, float3 &direction,
                                                     float3 &power) {
  float pdfDir;
  const float3 toSun = sampleSunCone(seed, pdfDir);
  direction = -toSun;

  const Onb onb(direction);
  const float2 disk = concentricSampleDisk(sutil::rnd(seed), sutil::rnd(seed));
  const float r = params.sceneBoundsRadius;
  origin = params.sceneBoundsCenter - direction * r + onb.m_tangent * (disk.x * r) + onb.m_binormal * (disk.y * r);

  const float diskArea = M_PIf * r * r;
  power = pdfDir > 1e-8f ? (params.sun.radiance * diskArea) /
                               (pdfDir * fmaxf(static_cast<float>(params.lightSubpathBatchSize), 1.0f))
                         : make_float3(0.0f);
}

extern "C" __global__ void __raygen__lightSubpath() {
  const unsigned int idx = optixGetLaunchIndex().x;
  unsigned int seed = sutil::tea<4>(idx, params.totalLightPathsEmitted + 0x9e3779b9u);

  float3 origin, direction, power;
  unsigned int sawSpecular = 0u;
  const float pSun = params.sun.enabled ? 0.5f : 0.0f;
  if (params.sun.enabled && sutil::rnd(seed) < pSun) {
    emitFromSun(seed, origin, direction, power);
    power = power / pSun;
  } else {
    if (params.envTex)
      emitFromEnvironment(seed, origin, direction, power);
    else
      emitFromQuadLight(seed, origin, direction, power);
    if (params.sun.enabled)
      power = power / (1.0f - pSun);
  }

  for (int depth = 0; depth < 12; ++depth) {
    unsigned int p0 = seed, p1 = __float_as_uint(power.x), p2 = __float_as_uint(power.y),
                 p3 = __float_as_uint(power.z), p4 = __float_as_uint(direction.x), p5 = __float_as_uint(direction.y),
                 p6 = __float_as_uint(direction.z), p7 = 0u, p8 = __float_as_uint(origin.x),
                 p9 = __float_as_uint(origin.y), p10 = __float_as_uint(origin.z), p11 = sawSpecular;
    optixTrace(params.handle, origin, direction, 1e-3f, 1e16f, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1,
               0, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
    sawSpecular = p11;
    seed = p0;
    power = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
    direction = make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6));
    const unsigned int done = p7;
    origin = make_float3(__uint_as_float(p8), __uint_as_float(p9), __uint_as_float(p10));
    if (done)
      break;

    if (depth >= 3) {
      const float rrP = fmaxf(fmaxf(power.x, power.y), power.z);
      if (sutil::rnd(seed) > rrP)
        break;
      power /= fmaxf(rrP, 1e-4f);
    }
  }
}
