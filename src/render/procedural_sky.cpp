#include "render/procedural_sky.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

namespace italy {
namespace {

constexpr float kPi = 3.14159265358979323846f;

// Preetham/Perez daylight model (Preetham, Shirley, Smits 1999, "A Practical
// Analytic Model for Daylight"), chosen over the higher-fidelity Hosek-Wilkie
// model specifically because it's fully closed-form: a handful of published
// polynomial coefficients rather than Hosek-Wilkie's large fitted dataset,
// which would have to be transcribed from the reference implementation
// without a way to diff against it here. This project already shipped one
// bug from "reproduced from memory, not verified against reference" (AgX's
// missing pow(2.2), see humans.md) — smaller surface area for the same
// mistake is worth the fidelity this gives up. The coefficients below are
// transcribed from the paper's published tables; procedural_sky_test.cpp
// checks the *behavioral* properties that actually matter for lighting
// (brighter toward the sun, no negative/NaN radiance) rather than trusting
// the transcription blind.
struct PerezCoeffs {
  float A, B, C, D, E;
};

PerezCoeffs perezY(float t) { return {0.1787f * t - 1.4630f, -0.3554f * t + 0.4275f, -0.0227f * t + 5.3251f,
                                       0.1206f * t - 2.5771f, -0.0670f * t + 0.3703f}; }
PerezCoeffs perezX(float t) { return {-0.0193f * t - 0.2592f, -0.0665f * t + 0.0008f, -0.0004f * t + 0.2125f,
                                       -0.0641f * t - 0.8989f, -0.0033f * t + 0.0452f}; }
PerezCoeffs perezYc(float t) { return {-0.0167f * t - 0.2608f, -0.0950f * t + 0.0092f, -0.0079f * t + 0.2102f,
                                        -0.0441f * t - 1.6537f, -0.0109f * t + 0.0529f}; }

float perezF(float theta, float gamma, const PerezCoeffs &c) {
  // cosTheta is floored away from 0 rather than at 0: theta approaches pi/2
  // at and below the horizon, where 1/cosTheta blows up. Callers already
  // clamp theta before this point (see skyTheta below) — the floor here is
  // just a second line of defense against exactly-grazing directions.
  const float cosTheta = std::max(std::cos(theta), 1e-3f);
  const float cosGamma = std::cos(gamma);
  return (1.0f + c.A * std::exp(c.B / cosTheta)) * (1.0f + c.C * std::exp(c.D * gamma) + c.E * cosGamma * cosGamma);
}

// Zenith luminance (kcd/m^2) and chromaticity, as functions of turbidity and
// the sun's zenith angle — same paper, same "transcribed, verify behaviorally
// not byte-for-byte" caveat as the Perez coefficients above.
float zenithLuminance(float t, float thetaS) {
  const float chi = (4.0f / 9.0f - t / 120.0f) * (kPi - 2.0f * thetaS);
  return (4.0453f * t - 4.9710f) * std::tan(chi) - 0.2155f * t + 2.4192f;
}

// Published Preetham zenith-chromaticity polynomials, grouped by power of
// turbidity T (T^2, T^1, T^0), each of those a cubic in thetaS — verified
// digit-for-digit against two independent open-source implementations
// (github.com/ebruneton/clear-sky-models and github.com/diharaw/sky-models,
// both transcribing the same published paper table) after an earlier draft
// of this function had the matrix transposed: grouping by power of thetaS
// instead of T looks superficially similar but silently permutes which
// coefficient multiplies which term, and produced a chromaticity so wrong
// (olive-green instead of blue sky) that it was caught by eye rather than
// by the unit test — the test only checked brightness, not hue. This is
// exactly the transcription-risk class Preetham was chosen over
// Hosek-Wilkie to minimize, and it still needed checking against a real
// reference rather than trusting the first plausible-looking recollection.
float zenithChromaticityX(float t, float thetaS) {
  const float t2 = t * t;
  const float th = thetaS, th2 = th * th, th3 = th2 * th;
  return (0.00166f * th3 - 0.00375f * th2 + 0.00209f * th + 0.0f) * t2 +
         (-0.02903f * th3 + 0.06377f * th2 - 0.03202f * th + 0.00394f) * t +
         (0.11693f * th3 - 0.21196f * th2 + 0.06052f * th + 0.25886f);
}

float zenithChromaticityY(float t, float thetaS) {
  const float t2 = t * t;
  const float th = thetaS, th2 = th * th, th3 = th2 * th;
  return (0.00275f * th3 - 0.00610f * th2 + 0.00317f * th + 0.0f) * t2 +
         (-0.04214f * th3 + 0.08970f * th2 - 0.04153f * th + 0.00516f) * t +
         (0.15346f * th3 - 0.26756f * th2 + 0.06670f * th + 0.26688f);
}

// Same host/device equirect convention pathtracer.cu's dirToEquirectUv uses
// (theta measured from +Y, v=0 at the top row) — duplicated here in plain
// C++ rather than shared, since that one is a CUDA __device__ function
// reading params.envRotation. Rotation is deliberately not baked in here,
// same as a loaded HDRI: it's applied uniformly at render time regardless of
// where the environment image came from.
glm::vec3 equirectUvToDir(float u, float v) {
  const float theta = v * kPi;
  const float phi = u * 2.0f * kPi - kPi;
  const float sinTheta = std::sin(theta);
  return glm::vec3(sinTheta * std::cos(phi), std::cos(theta), sinTheta * std::sin(phi));
}

glm::vec3 xyYToLinearSRGB(float x, float y, float Y) {
  y = std::max(y, 1e-4f); // guard the x/y, (1-x-y)/y divisions below
  const float X = (x / y) * Y;
  const float Z = ((1.0f - x - y) / y) * Y;
  // Standard D65 XYZ -> linear sRGB matrix.
  glm::vec3 rgb(3.2406f * X - 1.5372f * Y - 0.4986f * Z, -0.9689f * X + 1.8758f * Y + 0.0415f * Z,
                0.0557f * X - 0.2040f * Y + 1.0570f * Z);
  // Perez chromaticity strays outside the sRGB gamut near the horizon at
  // high turbidity — clamp rather than let it go negative, same "clamp, log,
  // move on" spirit as this project's other float-precision edge guards.
  return glm::max(rgb, glm::vec3(0.0f));
}

// realism: the real sun subtends ~0.25 degrees — at the 1k-ish equirect
// resolutions this generator targets (matching the bundled HDRI presets)
// that's a handful of texels, too small to importance-sample without heavy
// fireflying and too small to read as a specular highlight at all. Widened
// to something a mirror/glass reflection can actually resolve. Brightness
// is a flat artist-tuned boost over the local sky colour, not a physically
// derived solar irradiance — there's no principled "correct" answer once
// the disk's own size has already been fudged for beauty over realism.
constexpr float kSunAngularRadiusDeg = 2.0f;
constexpr float kSunDiskBoost = 60.0f;

// realism: Preetham's zenith luminance is in kcd/m^2, tuned for a display
// tone-mapping pipeline the paper assumes; this renderer's exposure/AgX
// chain instead expects roughly the same scene-linear scale the bundled
// Poly Haven HDRIs already sit at. This constant is a by-eye match to that
// scale (see procedural_sky_test.cpp/humans.md verification), not a
// radiometric conversion factor — same spirit as MATERIAL_GLASS's
// by-eye-tuned extinction color.
constexpr float kIntensityScale = 0.075f;

} // namespace

bool buildProceduralSky(int width, int height, float sunElevationDeg, float sunAzimuthDeg, float turbidity,
                         EnvironmentMap &out, std::string &err) {
  if (width <= 0 || height <= 0) {
    err = "procedural sky: width/height must be positive";
    return false;
  }
  const float t = std::clamp(turbidity, 1.9f, 10.0f);

  const float sunElevationRad = glm::radians(sunElevationDeg);
  const float sunAzimuthRad = glm::radians(sunAzimuthDeg);
  const glm::vec3 sunDir = glm::normalize(
      glm::vec3(std::cos(sunElevationRad) * std::cos(sunAzimuthRad), std::sin(sunElevationRad),
                std::cos(sunElevationRad) * std::sin(sunAzimuthRad)));
  // thetaS is the sun's zenith angle. Below-horizon suns (elevation < 0)
  // give thetaS > pi/2, which the Perez zenith-luminance formula isn't fit
  // for — clamp to just above the horizon rather than extrapolate into
  // nonsense (a large negative or NaN zenith luminance).
  const float thetaS = std::min(std::acos(std::clamp(sunDir.y, -1.0f, 1.0f)), glm::radians(89.0f));

  const PerezCoeffs cY = perezY(t), cX = perezX(t), cYc = perezYc(t);
  const float Yz = std::max(zenithLuminance(t, thetaS), 0.0f);
  const float xz = zenithChromaticityX(t, thetaS);
  const float yz = zenithChromaticityY(t, thetaS);

  const float perezYzenith = perezF(0.0f, thetaS, cY);
  const float perezXzenith = perezF(0.0f, thetaS, cX);
  const float perezYczenith = perezF(0.0f, thetaS, cYc);
  const float sunAngularRadiusRad = glm::radians(kSunAngularRadiusDeg);

  std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
  for (int py = 0; py < height; ++py) {
    const float v = (py + 0.5f) / height;
    for (int px = 0; px < width; ++px) {
      const float u = (px + 0.5f) / width;
      const glm::vec3 dir = equirectUvToDir(u, v);

      // View zenith angle, clamped just above the horizon: below-horizon
      // directions get the horizon's own colour rather than extrapolating
      // the Perez formula past where it's valid (it isn't singular exactly
      // at pi/2, but it isn't meaningful past it either — there's no "sky"
      // to look at underground). Same clamp-not-wrap spirit as the loaded-
      // HDRI texture's CLAMP addressing at the poles.
      const float theta = std::min(std::acos(std::clamp(dir.y, -1.0f, 1.0f)), glm::radians(89.0f));
      const float gamma = std::acos(std::clamp(glm::dot(dir, sunDir), -1.0f, 1.0f));

      const float Y = Yz * perezF(theta, gamma, cY) / std::max(perezYzenith, 1e-6f);
      const float x = xz * perezF(theta, gamma, cX) / std::max(perezXzenith, 1e-6f);
      const float y = yz * perezF(theta, gamma, cYc) / std::max(perezYczenith, 1e-6f);

      glm::vec3 rgb = xyYToLinearSRGB(x, y, std::max(Y, 0.0f)) * kIntensityScale;

      if (gamma < sunAngularRadiusRad) {
        // Soft-edged disk (smoothstep over the outer ~20% of its radius)
        // rather than a hard cutoff, so a handful-of-texels disk doesn't
        // alias into a jagged square at typical resolutions.
        const float edge = glm::smoothstep(sunAngularRadiusRad, sunAngularRadiusRad * 0.8f, gamma);
        const glm::vec3 zenithRgb = xyYToLinearSRGB(xz, yz, Yz) * kIntensityScale;
        rgb += zenithRgb * kSunDiskBoost * edge;
      }

      float *p = &pixels[(static_cast<size_t>(py) * width + px) * 3];
      p[0] = rgb.x;
      p[1] = rgb.y;
      p[2] = rgb.z;
    }
  }

  return buildEnvironmentCdf(width, height, pixels, out, err);
}

} // namespace italy
