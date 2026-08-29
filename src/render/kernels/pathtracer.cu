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
// Tonemapping (phase 8): converts linear HDR radiance to the [0,1] range
// sutil::toSRGB()/quantizeUnsigned8Bits() (sutil/cuda/helpers.h) then encode
// to 8-bit display output — same final encode step every operator shares,
// only what happens before it differs. AgX is the default; the rest are
// alternates/debugging aids (see optix_renderer.h's TonemapOperator).
// ----------------------------------------------------------------------------

// AgX: a widely-circulated community GLSL approximation of Blender's AgX
// view transform (the same one Godot 4.3+ and Bevy ship instead of pulling
// in OpenColorIO for a single transform) — inset matrix into a
// log2-encoded working space, a 6th-order polynomial fit of AgX's "base
// contrast" sigmoid, then an outset matrix back to linear. The matrix and
// polynomial constants below are reproduced from memory of that
// community fit, not diffed against Blender's reference OCIO config
// byte-for-byte in this environment (no Blender install/reference render
// available here to diff against) — verification is behavioral: does it
// visibly preserve highlight gradation instead of clipping to flat white,
// compared side by side against TONEMAP_CLAMP (see render()'s comparison
// dumps).
//
// The polynomial's output is *display-encoded*, not linear — that is what
// the canonical chain's closing pow(2.2) is for (it appears as `agxEotf` in
// the Godot/Filament/three.js ports). Every operator here shares one final
// encode step, sutil::make_color()'s toSRGB(), so AgX has to hand back
// linear like the others do. Omitting the pow ran the sRGB OETF on
// already-encoded values: a double encode, which lifts shadows, washes out
// midtones and flattens contrast. That milky look was mistaken for AgX's
// real signature (lifted shadows, soft rolloff) and recorded as verified in
// humans.md — the two are easy to confuse by eye, which is why the
// background of the fixed test scene is the tell: bgColor is linear
// (0.05, 0.06, 0.08), near-black, and it was rendering as light grey.
static __forceinline__ __device__ float3 agxContrastApprox(float3 x) {
  const float3 x2 = x * x;
  const float3 x4 = x2 * x2;
  return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
}

static __forceinline__ __device__ float3 agxTonemap(float3 c) {
  // AgX Inset matrix (working linear -> AgX log2 space).
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

  // AgX Outset matrix (AgX space -> display-referred linear).
  const float3 or_ = make_float3(1.1271005818144368f, -0.11060664309660323f, -0.016493938717834573f);
  const float3 og = make_float3(-0.1413297634984383f, 1.157823702216272f, -0.016493938717834257f);
  const float3 ob = make_float3(-0.14132976349843826f, -0.11060664309660294f, 1.2519364065950405f);
  v = make_float3(dot(make_float3(or_.x, og.x, ob.x), v), dot(make_float3(or_.y, og.y, ob.y), v),
                   dot(make_float3(or_.z, og.z, ob.z), v));

  // Back to linear, so the shared toSRGB() encode in make_color() is the one
  // and only OETF applied. See the block comment above.
  v = make_float3(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f), fmaxf(v.z, 0.0f));
  return make_float3(powf(v.x, 2.2f), powf(v.y, 2.2f), powf(v.z, 2.2f));
}

// Simple Reinhard (c / (1+c)) — the classic "everything gently compresses
// toward white" operator, kept mainly as a contrast baseline against AgX.
static __forceinline__ __device__ float3 reinhardTonemap(float3 c) { return c / (make_float3(1.0f) + c); }

// Narkowicz 2015 ACES filmic curve fit.
static __forceinline__ __device__ float3 acesFilmicTonemap(float3 x) {
  const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// Uncharted2/Hable filmic curve.
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

// realism: caps a single sample's radiance — an unbiased estimator is free to
// return an arbitrarily bright sample, and the rare ones that do show up as
// permanent white specks ("fireflies") that no amount of further averaging
// removes at interactive sample counts. Scaling by max-component rather than
// clamping per channel keeps the hue exactly. Slight energy loss in the
// brightest highlights, in exchange for a clean image — doctrine rung 1 over
// rung 3. params.fireflyClamp <= 0 disables it.
static __forceinline__ __device__ float3 clampFirefly(float3 c, float maxValue) {
  if (maxValue <= 0.0f)
    return c;
  const float peak = fmaxf(fmaxf(c.x, c.y), c.z);
  return peak > maxValue ? c * (maxValue / peak) : c;
}

// A NaN/Inf reaching the accumulator is unrecoverable: every later subframe
// lerps against it, so one bad sample kills that pixel for the rest of the
// session. Cheaper to drop the sample than to explain the dead pixel.
static __forceinline__ __device__ bool isFinite3(float3 c) {
  return isfinite(c.x) && isfinite(c.y) && isfinite(c.z);
}

static __forceinline__ __device__ uchar4 applyTonemapAndQuantize(float3 hdr, unsigned int op) {
  float3 mapped;
  switch (op) {
  case 1: // Reinhard
    mapped = reinhardTonemap(hdr);
    break;
  case 2: // ACES
    mapped = acesFilmicTonemap(hdr);
    break;
  case 3: // Hable
    mapped = hableTonemap(hdr);
    break;
  case 4: // Clamp — the bare behavior every render used before this phase
    mapped = hdr;
    break;
  default: // AgX
    mapped = agxTonemap(hdr);
    break;
  }
  return sutil::make_color(mapped);
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

// The single source of truth for direction <-> (u,v). Radiance lookup, pdf
// evaluation and importance sampling all route through this pair, because
// their results are compared directly in the MIS weight — if the environment
// rotation reached only some of them, the weights would silently stop
// matching the samples they weight.
static __forceinline__ __device__ void dirToEquirectUv(float3 dir, float &u, float &v, float &theta) {
  theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
  float phi = atan2f(dir.z, dir.x) - params.envRotation;
  // atan2 returns [-pi,pi]; subtracting the rotation can leave that range, so
  // wrap back before mapping to [0,1] rather than relying on texture wrap
  // (the CDF lookups below index a plain buffer and would go out of bounds).
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

  float sinTheta;
  const float3 dir = equirectUvToDir(u, v, sinTheta);
  pdfOut = sinTheta > 1e-6f ? (rowPdf * colPdf) / (2.0f * M_PIf * M_PIf * sinTheta) : 0.0f;
  return dir;
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

// Tent (Bartlett) reconstruction filter of radius 1 pixel, sampled by
// inverting its triangular CDF. The previous uniform jitter was a box filter
// over the pixel's own area: every position inside the pixel contributed
// equally and nothing outside it contributed at all, which is the worst
// reconstruction filter there is for the same sample count — it both aliases
// harder and looks softer. A tent weights the pixel centre more and lets
// neighbours bleed in slightly. Returns [-1, 1].
static __forceinline__ __device__ float tentFilterWarp(float u) {
  const float x = 2.0f * u;
  return x < 1.0f ? sqrtf(x) - 1.0f : 1.0f - sqrtf(fmaxf(2.0f - x, 0.0f));
}

// Shirley's concentric mapping: squares to disk with far less distortion than
// the naive (r = sqrt(u), theta = 2*pi*v) polar mapping, which clumps samples
// toward the centre and shows up as a dirty-looking bokeh disc.
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

    // Thin-lens depth of field. The pinhole ray above already points at the
    // right spot on the focal plane; displacing the origin across the lens
    // and re-aiming at that same point is the whole model. dot(direction, W)
    // converts the focus distance from "along the view axis" to "along this
    // ray", which keeps the focal surface a plane instead of a sphere.
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
    prd.prevBsdfPdf = -1.0f; // primary ray: treat like a specular predecessor (full weight on direct light hit)

    // Per-sample, not per-launch: the firefly clamp below has to see one
    // path's radiance to know whether *that path* spiked.
    float3 sample = make_float3(0.0f);

    for (;;) {
      trace(params.handle, origin, direction, 1e-3f, 1e16f, prd);

      // Both already include the path throughput that led to them — the hit
      // and miss programs apply it themselves, since only they know whether
      // a given term belongs before or after this surface's own BSDF weight.
      sample += prd.emitted;
      sample += prd.radiance;

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

    if (isFinite3(sample))
      result += clampFirefly(sample, params.fireflyClamp);
  }

  const unsigned int pixel = idx.y * w + idx.x;
  float3 accum = result / static_cast<float>(spl);
  if (subframe > 0) {
    const float a = 1.0f / static_cast<float>(subframe + 1);
    const float3 prevColor = make_float3(params.accumBuffer[pixel]);
    accum = lerp(prevColor, accum, a);
  }
  params.accumBuffer[pixel] = make_float4(accum, 1.0f);
  // Exposure is applied only to the display output, not the stored
  // accumulator — so dragging the exposure slider doesn't need an
  // accumulation reset, it just changes how the same HDR average is
  // displayed this frame. Skipped entirely when the denoiser is active:
  // optix_renderer.cpp denoises accumBuffer and runs __raygen__tonemap on
  // the result instead, so writing a tonemap of the *noisy* accum here
  // would just be wasted work, immediately overwritten.
  if (!params.denoiserEnabled)
    params.frameBuffer[pixel] = applyTonemapAndQuantize(accum * params.exposure, params.tonemapOperator);
}

// Phase 10: reads the denoiser's output (already-converged-looking HDR)
// instead of the raw progressive accumulator, and does nothing else — no
// ray tracing, just the same tonemap step __raygen__rg applies inline when
// the denoiser is off. A raygen program rather than a plain CUDA kernel so
// it can reuse the pipeline/module/SBT-launch machinery already built for
// everything else in this file instead of standing up a separate CUDA
// compilation path for one trivial per-pixel op.
extern "C" __global__ void __raygen__tonemap() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int pixel = idx.y * params.width + idx.x;
  const float3 color = make_float3(params.denoisedBuffer[pixel]);
  params.frameBuffer[pixel] = applyTonemapAndQuantize(color * params.exposure, params.tonemapOperator);
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
  // Scale by the incoming throughput here for the same reason
  // __closesthit__radiance does: the raygen loop adds `emitted` as-is now.
  const float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()),
                                          __uint_as_float(optixGetPayload_2()));
  color = color * weight * throughput;
  optixSetPayload_6(__float_as_uint(color.x));
  optixSetPayload_7(__float_as_uint(color.y));
  optixSetPayload_8(__float_as_uint(color.z));
  optixSetPayload_9(__float_as_uint(0.0f));
  optixSetPayload_10(__float_as_uint(0.0f));
  optixSetPayload_11(__float_as_uint(0.0f));
  optixSetPayload_18(1u); // done
}

extern "C" __global__ void __miss__occlusion() { optixSetPayload_0(0u); }

// Gather queries (missSBTIndex 2, see the gather trace in
// __closesthit__radiance) intentionally do nothing on a miss: "no photons
// within radius" needs no payload change, since the accumulator (payload
// registers 6-8) was already zero-initialized by the caller.
extern "C" __global__ void __miss__gather() {}

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

// Hit kinds reported by __intersection__sphere_solid. Which root the
// intersector picked is an unambiguous fact about the geometry, whereas
// re-deriving it in the closest-hit from sign(dot(rayDir, Ng)) is
// ill-conditioned for grazing rays where that dot product is ~0.
#define SPHERE_HIT_FROM_OUTSIDE 0u
#define SPHERE_HIT_FROM_INSIDE 1u

// Custom-primitive intersection for a *solid* sphere — the dielectric
// stand-in for OptiX's built-in sphere, which is documented hollow and
// back-face culled (Programming Guide 9.1, 9.6), so a ray refracted into a
// built-in sphere never receives an exit intersection: the glass ends up a
// single refracting interface with no interior, no total internal
// reflection, and nothing for Beer-Lambert to attenuate over.
//
// The whole point is the `else` clause below: when the near root is behind
// the ray (origin inside, or exactly on the surface heading in), report the
// *far* root instead of giving up. NVIDIA's own dielectric sample does the
// same thing for the same reason — see optixWhitted's
// __intersection__sphere_shell.
extern "C" __global__ void __intersection__sphere_solid() {
  const HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 o = optixGetObjectRayOrigin() - rt->sphereCenter;
  const float3 d = optixGetObjectRayDirection();

  // Quadratic in t with the half-b form: a t^2 + 2b t + c = 0.
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

// ----------------------------------------------------------------------------
// Metallic-roughness BSDF: Lambert diffuse + GGX microfacet specular, the
// glTF material model. Everything before this phase was pure Lambert, so
// loaded assets rendered as matte clay no matter what their material said.
//
// One surface, two lobes, sampled with one ray: pick a lobe by probability,
// then weight by the *combined* pdf of both. That combination is what keeps
// the estimator unbiased, and it is why every pdf below has to agree with the
// sampling routine exactly — a mismatch here does not look like a bug, it
// looks like the wrong material.
// ----------------------------------------------------------------------------

struct ShadingMaterial {
  float3 diffuse;   // base colour, already scaled by (1 - metallic)
  float3 f0;        // normal-incidence specular reflectance
  float alpha;      // GGX roughness^2
  bool hasSpecular; // false for the fixed scene's plain-diffuse/voxel/SDF surfaces
};

static __forceinline__ __device__ float luminance3(float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static __forceinline__ __device__ float ggxD(float NdotH, float alpha) {
  const float a2 = alpha * alpha;
  const float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
  return a2 / fmaxf(M_PIf * d * d, 1e-9f);
}

// Exact Smith masking-shadowing G1 for GGX (not the Schlick approximation —
// the exact form is barely more arithmetic and avoids darkening at grazing
// angles, which is precisely where a car body or a tank panel is most visible).
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

// How often to sample the specular lobe. Clamped away from 0 and 1: a lobe
// that still contributes to the combined pdf but is almost never sampled
// produces enormous variance on the rare occasions it is picked, which reads
// as fireflies rather than as noise.
static __forceinline__ __device__ float specularLobeProbability(const ShadingMaterial &m) {
  if (!m.hasSpecular)
    return 0.0f;
  const float d = luminance3(m.diffuse);
  const float s = luminance3(m.f0);
  const float total = d + s;
  return total > 0.0f ? fminf(fmaxf(s / total, 0.1f), 0.9f) : 0.5f;
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals". Isotropic.
// Local frame with the shading normal along +Z. Sampling visible normals
// rather than the raw NDF is what makes rough metal converge in a sane number
// of samples — the naive NDF sampling generates half-vectors facing away from
// the viewer that are then thrown away.
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

// f(V,L) * cos(theta_L), plus the pdf the sampler below would have assigned to
// this same L. Used by NEE (which needs both) and by the sampler (which uses
// it to form its weight, so the two can never disagree).
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
      // f_spec * NdotL = D*G*F / (4*NdotV) — the NdotL cancels.
      fCos = fCos + F * (D * G / (4.0f * NdotV));
      // VNDF pdf over half-vectors, reparameterised to the L measure by the
      // 1/(4*VdotH) reflection Jacobian: G1(V)*VdotH*D/NdotV / (4*VdotH).
      pdf += pSpec * (G1v * D / (4.0f * NdotV));
    }
  }
}

// Picks a lobe, samples it, and returns throughput = f*cos/pdf using the
// combined pdf. Returns false for a degenerate sample the caller should drop.
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
//
// Returns the *raw geometric* normal — deliberately NOT faceforwarded.
// Callers that only shade should take faceforward(Ng, -rayDir, Ng)
// themselves; callers that need to know which side of the surface the ray
// arrived on (i.e. dielectrics) must read that off this raw normal, because
// faceforward destroys exactly that information. faceforward(n, -d, n)
// returns n * copysign(1, dot(-d, n)), so its result *always* satisfies
// dot(d, N) < 0 — which made the glass branch's
// `bool entering = dot(rayDir, N) < 0` unconditionally true, so entry and
// exit both used eta = 1/ior. Rays leaving the glass bent the wrong way and
// total internal reflection was never detected from the inside.
static __forceinline__ __device__ float3 computeGeometricNormal(const float3 &rayDir) {
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
  return n;
}

// Raw geometric normal for whichever geometry this record describes. Solid
// spheres are custom primitives, so optixGetSphereData() (which
// computeGeometricNormal falls through to) would read a built-in sphere that
// isn't there — the centre/radius come from the SBT record instead.
static __forceinline__ __device__ float3 geometricNormalFor(const HitGroupData *rt, const float3 &P,
                                                              const float3 &rayDir) {
  if (rt->sphereRadius > 0.0f)
    return (P - rt->sphereCenter) / rt->sphereRadius;
  return computeGeometricNormal(rayDir);
}

extern "C" __global__ void __closesthit__radiance() {
  HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());

  const float3 rayDir = optixGetWorldRayDirection();
  float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;

  // GLB-mesh triangles carry their own per-vertex normals/UVs (better shading
  // than the flat face normal computeGeometricNormal() gives the fixed
  // bring-up scene's spheres/plane) and an optional base-color texture.
  // N is the shading normal (always facing against the incident ray); Ng is
  // the raw geometric normal, whose sign still encodes which side we hit
  // from. Only the dielectric branch needs Ng — see computeGeometricNormal.
  float3 N;
  float3 Ng = make_float3(0.0f, 1.0f, 0.0f);
  float3 albedo = rt->albedo;
  // Defaults give the pre-GGX behaviour exactly: pure Lambert, no specular
  // lobe. Only glTF meshes, which actually carry metallic/roughness data,
  // turn the specular lobe on.
  ShadingMaterial shading;
  shading.diffuse = albedo;
  shading.f0 = make_float3(0.0f);
  shading.alpha = 1.0f;
  shading.hasSpecular = false;

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
      // glTF packs roughness in G and metallic in B, and the texture is
      // linear data — not a colour — so it must not be sRGB-decoded.
      const float4 mr = tex2D<float4>(mat.metallicRoughnessTex, u, v);
      roughness *= mr.y;
      metallic *= mr.z;
    }

    // Tangent-space normal mapping. Most of the fine surface detail on a
    // photogrammetry or generated asset lives here rather than in geometry,
    // so ignoring it throws away most of the model's apparent resolution.
    if (mat.normalTex && rt->tangents) {
      const float4 t0 = rt->tangents[prim * 3 + 0], t1 = rt->tangents[prim * 3 + 1],
                   t2 = rt->tangents[prim * 3 + 2];
      float3 T = w0 * make_float3(t0.x, t0.y, t0.z) + w1 * make_float3(t1.x, t1.y, t1.z) +
                 w2 * make_float3(t2.x, t2.y, t2.z);
      // Re-orthogonalise: interpolating tangents across a triangle does not
      // preserve perpendicularity to the interpolated normal.
      T = T - N * dot(N, T);
      const float tlen = length(T);
      if (tlen > 1e-6f) {
        T = T / tlen;
        const float3 B = cross(N, T) * t0.w; // handedness is per-vertex but constant across a triangle
        const float4 nt = tex2D<float4>(mat.normalTex, u, v);
        float3 tn = make_float3(nt.x * 2.0f - 1.0f, nt.y * 2.0f - 1.0f, nt.z * 2.0f - 1.0f);
        tn.x *= mat.normalScale;
        tn.y *= mat.normalScale;
        const float3 mapped = normalize(tn.x * T + tn.y * B + tn.z * N);
        // Only accept the perturbed normal if it stays on the visible side.
        // A normal map can otherwise tip the shading normal below the
        // horizon, which makes the surface self-shadow into black speckle.
        if (dot(mapped, -rayDir) > 0.0f)
          N = mapped;
      }
    }

    albedo = baseColor;
    // Metals have no diffuse lobe and tint their specular reflection;
    // dielectrics reflect ~4% achromatically and keep their colour in the
    // diffuse term. This is the standard glTF metallic-roughness split.
    shading.diffuse = baseColor * (1.0f - metallic);
    shading.f0 = lerp(make_float3(0.04f), baseColor, metallic);
    const float r = fminf(fmaxf(roughness, 0.03f), 1.0f); // floor keeps the GGX lobe numerically sane
    shading.alpha = r * r;
    shading.hasSpecular = true;
  } else if (rt->materialType == MATERIAL_VOXEL) {
    const float3 faceN = voxelFaceNormal(optixGetAttribute_0());
    N = faceforward(faceN, -rayDir, faceN);
    albedo = rt->voxelColors[optixGetPrimitiveIndex()];
    shading.diffuse = albedo;
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
    Ng = geometricNormalFor(rt, P, rayDir);
    N = faceforward(Ng, -rayDir, Ng);
  }

  unsigned int seed = optixGetPayload_3();
  int depth = static_cast<int>(optixGetPayload_4());
  const float prevBsdfPdf = __uint_as_float(optixGetPayload_5());
  float3 attenuation =
      make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));
  // Path throughput as it stood *arriving* at this hit. Emission seen here,
  // and any NEE contribution gathered here, are both scaled by this — not by
  // the outgoing attenuation, which already includes this surface's own BSDF
  // weight and would double-count it. Doing that scaling here rather than in
  // the raygen loop is the point: the previous split (raygen multiplied by
  // whatever attenuation came back, so the NEE term had to deliberately omit
  // albedo to compensate) was the shape that produced two separate
  // double-counting bugs, and it stops being expressible once the two are
  // resolved in the same place.
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
    // combined, see Params::envTex's doc comment.
    //
    // Both branches now use the full BSDF via evalBsdf(), which already
    // includes the surface colour, and the MIS partner pdf is that same
    // function's combined two-lobe pdf rather than the bare cosine pdf a
    // Lambert-only tracer could assume. Getting the second wrong is
    // invisible on a rough surface and catastrophic on a smooth one.
    const float3 V = -rayDir;
    if (params.envTex) {
      float envPdf;
      const float3 dir = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), envPdf);
      if (envPdf > 0.0f) {
        float3 fCos;
        float pdfBsdf;
        evalBsdf(shading, N, V, dir, fCos, pdfBsdf);
        if (fCos.x + fCos.y + fCos.z > 0.0f) {
          const bool occluded = traceOcclusion(params.handle, P, dir, 1e-3f, 1e16f);
          if (!occluded) {
            const float3 envRadiance = lookupEnvironmentRadiance(dir);
            const float weight = powerHeuristic(envPdf, pdfBsdf);
            radiance = envRadiance * fCos * (weight / envPdf);
          }
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
      const float lnDl = -dot(light.normal, L);
      if (lnDl > 0.0f) {
        float3 fCos;
        float pdfBsdf;
        evalBsdf(shading, N, V, L, fCos, pdfBsdf);
        if (fCos.x + fCos.y + fCos.z > 0.0f) {
          const bool occluded = traceOcclusion(params.handle, P, L, 1e-3f, dist - 2e-3f);
          if (!occluded) {
            const float area = length(cross(light.v1, light.v2));
            const float pdfLight = (dist * dist) / (lnDl * area);
            const float weight = powerHeuristic(pdfLight, pdfBsdf);
            radiance = light.emission * fCos * (weight / fmaxf(pdfLight, 1e-6f));
          }
        }
      }
    }

    // Caustics: gather nearby deposited photons (see phase-7 comment on
    // Params::photonHandle) and add their contribution alongside NEE — same
    // "radiance" channel, so it gets the same attenuation multiply in the
    // raygen loop. No-op when there's no photon map for this scene.
    if (params.photonHandle) {
      unsigned int g0 = __float_as_uint(N.x), g1 = __float_as_uint(N.y), g2 = __float_as_uint(N.z);
      // The gather reconstructs a diffuse-lobe estimate, so it wants the
      // diffuse albedo, not the base colour: a metal has no diffuse lobe and
      // must not pick up caustic photons through one.
      unsigned int g3 = __float_as_uint(shading.diffuse.x), g4 = __float_as_uint(shading.diffuse.y),
                   g5 = __float_as_uint(shading.diffuse.z);
      unsigned int g6 = 0u, g7 = 0u, g8 = 0u;
      // Any fixed direction correctly finds every sphere containing P, as
      // long as tmax covers the largest possible chord through one of
      // them (the diameter) — see optix_renderer.cpp's photon-BVH comment
      // for the derivation. N is a convenient already-unit-length choice.
      const float tmax = 2.02f * params.photonGatherRadius;
      optixTrace(params.photonHandle, P, N, 0.0f, tmax, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                 params.gatherHitSbtOffset, 1, 2, g0, g1, g2, g3, g4, g5, g6, g7, g8);
      const float3 gathered = make_float3(__uint_as_float(g6), __uint_as_float(g7), __uint_as_float(g8));
      const float diskArea = M_PIf * params.photonGatherRadius * params.photonGatherRadius;
      // Divide by the gather disk's area and nothing else. Each photon's
      // `power` was *already* divided by photonBatchSize at emission time
      // (see __raygen__photon), which is where the "flux per photon"
      // normalization belongs; dividing by it a second time here scaled the
      // whole caustic term down by another factor of 65536, making it
      // numerically present but invisible. That is what the 200x
      // ITALY_DEBUG_CAUSTICS_ONLY boost below was compensating for.
      // Bias from the shrinking radius, and noise from a single pass's
      // photon count, both average out via the same subframe accumulation
      // that already smooths NEE noise.
      radiance += gathered / diskArea;
      // Debug aid: build with `-DITALY_DEBUG_CAUSTICS_ONLY` (not part of the
      // normal CMake build) to replace radiance with *only* the caustic
      // term, to check the gather finds a spatially coherent caustic
      // independent of how it reads against direct lighting. No brightness
      // boost any more — with the double-divide above fixed, the term is
      // now at its true scale and legible on its own.
#ifdef ITALY_DEBUG_CAUSTICS_ONLY
      radiance = gathered / diskArea;
#endif
    }

    // Sample the next bounce from the same BSDF the NEE above evaluated, so
    // the pdf recorded for the next hit's MIS weight is the real one.
    float3 bounceDir, bounceWeight;
    float bouncePdf;
    if (sampleBsdf(shading, N, V, seed, bounceDir, bounceWeight, bouncePdf)) {
      nextDirection = bounceDir;
      nextPdf = bouncePdf;
      attenuation = attenuation * bounceWeight;
    } else {
      done = 1; // degenerate sample (below the horizon); kill the path rather than bias it
    }
  } else if (rt->materialType == MATERIAL_MIRROR) {
    nextDirection = reflect(rayDir, N);
    attenuation = attenuation * rt->albedo;
    nextPdf = -1.0f;
  } else { // MATERIAL_GLASS
    // Sidedness never comes from N: it has already been faceforwarded and so
    // always reports "entering". Prefer the intersector's own report where
    // there is one (solid spheres), and fall back to the *geometric* normal
    // otherwise, which is what a future triangle-mesh dielectric would use.
    // N itself is still the right normal to refract/reflect about either
    // way, since refractRay wants one facing against the incident ray.
    const bool entering = rt->sphereRadius > 0.0f ? (optixGetHitKind() == SPHERE_HIT_FROM_OUTSIDE)
                                                   : (dot(rayDir, Ng) < 0.0f);

    // Beer-Lambert. On an exit hit the ray has just crossed the interior, and
    // optixGetRayTmax() is exactly that segment's length, so the absorption
    // depends on how much glass was actually traversed — thick centre darker
    // than thin rim. Applies to segments ended by total internal reflection
    // too, since those are exit-side hits as well.
    if (!entering)
      attenuation = attenuation * make_float3(expf(-rt->extinction.x * optixGetRayTmax()),
                                               expf(-rt->extinction.y * optixGetRayTmax()),
                                               expf(-rt->extinction.z * optixGetRayTmax()));

    const float eta = entering ? (1.0f / rt->ior) : rt->ior;
    const float cosTheta = fminf(fabsf(dot(rayDir, N)), 1.0f);
    // Schlick's r0 = ((1-ior)/(1+ior))^2 is invariant under ior <-> 1/ior, so
    // one argument covers both sides; TIR on the way out is handled by
    // refractRay returning false, not by the Fresnel term.
    const float fresnel = schlickFresnel(cosTheta, rt->ior);

    float3 refracted;
    const bool canRefract = refractRay(rayDir, N, eta, refracted);
    if (!canRefract || sutil::rnd(seed) < fresnel) {
      nextDirection = reflect(rayDir, N);
    } else {
      nextDirection = refracted;
    }
    // No albedo multiply here: a dielectric's colour is absorption through
    // its volume (the Beer-Lambert term above), not a per-interface tint.
    // Multiplying a flat albedo at every crossing was thickness-independent,
    // so it could never make thick glass read differently from thin.
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
}

// Any-hit half of the "ray-traced range query" trick used to gather nearby
// photons: any-hit runs once per candidate primitive along the ray *without*
// stopping traversal (optixIgnoreIntersection keeps it going), so as long as
// the ray is long enough to guarantee crossing every candidate photon-sphere
// it starts inside (see __closesthit__radiance's tmax comment), this visits
// every photon within photonGatherRadius of the query point exactly once.
// Payload: p0-2 query shading normal, p3-5 query albedo (both set by the
// caller before tracing), p6-8 running sum (read-modify-write here).
extern "C" __global__ void __anyhit__gather() {
  const Photon &p = params.photons[optixGetPrimitiveIndex()];
  const float3 N = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()),
                                __uint_as_float(optixGetPayload_2()));
  const float cosTheta = dot(N, -p.direction);
  if (cosTheta > 0.0f) {
    const float3 albedo = make_float3(__uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()),
                                       __uint_as_float(optixGetPayload_5()));
    const float3 sum = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()),
                                    __uint_as_float(optixGetPayload_8())) +
                        (albedo / M_PIf) * p.power * cosTheta;
    optixSetPayload_6(__float_as_uint(sum.x));
    optixSetPayload_7(__float_as_uint(sum.y));
    optixSetPayload_8(__float_as_uint(sum.z));
  }
  optixIgnoreIntersection();
}

// ----------------------------------------------------------------------------
// Photon emission/tracing (phase 7 caustics). Iterative bounce loop in
// raygen, same shape as __raygen__rg/trace() above, but with a much smaller
// dedicated payload — this is a separate, simpler walk (emit from the light,
// follow specular bounces, deposit at the first diffuse hit, done) rather
// than a variant of the camera path. Scoped to MATERIAL_DIFFUSE/MIRROR/
// GLASS/LIGHT only (the fixed bring-up scene) — see optix_renderer.cpp's
// buildPhotonPipeline() for why mesh/voxel/sdf scenes don't get a photon map
// at all.
//
// Payload: p0 seed, p1-3 power, p4-6 next direction, p7 causticEligible
// (0/1 — becomes 1 after the first specular bounce; only photons that pass
// through at least one specular surface get deposited, since direct
// light->diffuse paths are already handled by NEE), p8 done, p9-11 next
// origin.
// ----------------------------------------------------------------------------

extern "C" __global__ void __closesthit__photon() {
  HitGroupData *rt = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
  const float3 rayDir = optixGetWorldRayDirection();
  const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
  const float3 Ng = geometricNormalFor(rt, P, rayDir);
  const float3 N = faceforward(Ng, -rayDir, Ng);

  unsigned int seed = optixGetPayload_0();
  float3 power = make_float3(__uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()),
                              __uint_as_float(optixGetPayload_3()));
  unsigned int causticEligible = optixGetPayload_7();

  float3 nextDirection = rayDir;
  unsigned int done = 1u;

  if (rt->materialType == MATERIAL_MIRROR) {
    nextDirection = reflect(rayDir, N);
    power = power * rt->albedo;
    causticEligible = 1u;
    done = 0u;
  } else if (rt->materialType == MATERIAL_GLASS) {
    // Same sidedness and absorption as __closesthit__radiance's glass branch
    // — photons have to refract out of the sphere correctly, and lose the
    // same energy on the way through, or the caustic they focus lands in the
    // wrong place with the wrong colour.
    const bool entering = rt->sphereRadius > 0.0f ? (optixGetHitKind() == SPHERE_HIT_FROM_OUTSIDE)
                                                   : (dot(rayDir, Ng) < 0.0f);
    if (!entering)
      power = power * make_float3(expf(-rt->extinction.x * optixGetRayTmax()),
                                   expf(-rt->extinction.y * optixGetRayTmax()),
                                   expf(-rt->extinction.z * optixGetRayTmax()));
    const float eta = entering ? (1.0f / rt->ior) : rt->ior;
    const float cosTheta = fminf(fabsf(dot(rayDir, N)), 1.0f);
    const float fresnel = schlickFresnel(cosTheta, rt->ior);
    float3 refracted;
    const bool canRefract = refractRay(rayDir, N, eta, refracted);
    nextDirection = (!canRefract || sutil::rnd(seed) < fresnel) ? reflect(rayDir, N) : refracted;
    causticEligible = 1u;
    done = 0u;
  } else if (rt->materialType == MATERIAL_DIFFUSE && causticEligible) {
    // Deposit and terminate — single-bounce caustics only (matches the
    // "light through glass onto a table" verification scenario exactly;
    // multi-bounce caustic chains are a natural but unneeded-for-MVP
    // extension, same "obvious next step, not required now" spirit as the
    // rest of this file's scope choices).
    const unsigned int idx = atomicAdd(params.photonCounter, 1u);
    if (idx < params.photonCapacity) {
      params.photons[idx].position = P;
      params.photons[idx].direction = rayDir;
      params.photons[idx].power = power;
    }
    done = 1u;
  }
  // MATERIAL_DIFFUSE-without-causticEligible (direct light->diffuse, already
  // covered by NEE) and MATERIAL_LIGHT (photon hit another light) both fall
  // through to the done=1u/no-deposit default above.

  optixSetPayload_0(seed);
  optixSetPayload_1(__float_as_uint(power.x));
  optixSetPayload_2(__float_as_uint(power.y));
  optixSetPayload_3(__float_as_uint(power.z));
  optixSetPayload_4(__float_as_uint(nextDirection.x));
  optixSetPayload_5(__float_as_uint(nextDirection.y));
  optixSetPayload_6(__float_as_uint(nextDirection.z));
  optixSetPayload_7(causticEligible);
  optixSetPayload_8(done);
  optixSetPayload_9(__float_as_uint(P.x));
  optixSetPayload_10(__float_as_uint(P.y));
  optixSetPayload_11(__float_as_uint(P.z));
}

extern "C" __global__ void __miss__photon() { optixSetPayload_8(1u); /* done — escaped the scene */ }

// Emits one photon from the quad light: uniform over its area, cosine-
// weighted into the hemisphere above it. Total flux for a Lambertian area
// emitter is Le * A * pi; spread evenly across this pass's photons.
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
  power = (light.emission * area * M_PIf) / fmaxf(static_cast<float>(params.photonBatchSize), 1.0f);
}

// Emits one photon from the environment: importance-sample a direction the
// same way NEE does (bright regions of the sky preferentially chosen), then
// place the origin on a disk just outside the scene's bounding sphere,
// facing into the scene along that direction — PBRT's standard
// InfiniteAreaLight::Sample_Le construction. A photon "from the environment"
// has no single point of origin, so the disk stands in for "everywhere
// outside the scene that direction could have come from."
static __forceinline__ __device__ void emitFromEnvironment(unsigned int &seed, float3 &origin, float3 &direction,
                                                             float3 &power) {
  float pdfDir;
  // sampleEnvironment() returns the direction *toward* the light (the
  // convention NEE uses); a photon needs to travel the other way.
  const float3 toLight = sampleEnvironment(sutil::rnd(seed), sutil::rnd(seed), pdfDir);
  direction = -toLight;
  const float3 radiance = lookupEnvironmentRadiance(toLight);

  const Onb onb(direction);
  const float2 disk = concentricSampleDisk(sutil::rnd(seed), sutil::rnd(seed));
  const float r = params.sceneBoundsRadius;
  origin = params.sceneBoundsCenter - direction * r + onb.m_tangent * (disk.x * r) + onb.m_binormal * (disk.y * r);

  // Sample_Le's area pdf is 1/(pi*r^2) over the disk; combined with the
  // direction pdf (already in solid-angle measure), power = Le * diskArea *
  // (no cosine term — the disk is oriented perpendicular to the ray by
  // construction, unlike the quad light's cosine-weighted hemisphere) /
  // pdfDir, spread over this pass's photon count the same way the quad
  // light's flux is.
  const float diskArea = M_PIf * r * r;
  power = pdfDir > 1e-8f ? (radiance * diskArea) / (pdfDir * fmaxf(static_cast<float>(params.photonBatchSize), 1.0f))
                         : make_float3(0.0f);
}

extern "C" __global__ void __raygen__photon() {
  const unsigned int idx = optixGetLaunchIndex().x;
  // Distinct seed stream from camera rays: same idx values are used by both
  // (photon launch width vs. pixel count don't correspond to anything), and
  // params.subframeIndex vs a would-be "photon pass index" would otherwise
  // coincide too, so tea<> alone isn't enough — fold in a large odd
  // constant to decorrelate the two RNG streams.
  unsigned int seed = sutil::tea<4>(idx, params.totalPhotonsEmitted + 0x9e3779b9u);

  // The two lighting modes aren't blended (see Params::envTex) — photon
  // emission follows whichever one NEE is using, so caustics are lit
  // consistently with everything else in the scene.
  float3 origin, direction, power;
  if (params.envTex)
    emitFromEnvironment(seed, origin, direction, power);
  else
    emitFromQuadLight(seed, origin, direction, power);

  unsigned int causticEligible = 0u;
  for (int depth = 0; depth < 8; ++depth) {
    unsigned int p0 = seed, p1 = __float_as_uint(power.x), p2 = __float_as_uint(power.y),
                 p3 = __float_as_uint(power.z), p4 = __float_as_uint(direction.x), p5 = __float_as_uint(direction.y),
                 p6 = __float_as_uint(direction.z), p7 = causticEligible, p8 = 0u, p9 = __float_as_uint(origin.x),
                 p10 = __float_as_uint(origin.y), p11 = __float_as_uint(origin.z);
    optixTrace(params.handle, origin, direction, 1e-3f, 1e16f, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1,
               0, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
    seed = p0;
    power = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
    direction = make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6));
    causticEligible = p7;
    const unsigned int done = p8;
    origin = make_float3(__uint_as_float(p9), __uint_as_float(p10), __uint_as_float(p11));
    if (done)
      break;
  }
}
