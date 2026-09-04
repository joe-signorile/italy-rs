#include "render/procedural_sky.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

namespace italy {
namespace {

constexpr float kPi = 3.14159265358979323846f;

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
  const float cosTheta = std::max(std::cos(theta), 1e-3f);
  const float cosGamma = std::cos(gamma);
  return (1.0f + c.A * std::exp(c.B / cosTheta)) * (1.0f + c.C * std::exp(c.D * gamma) + c.E * cosGamma * cosGamma);
}

float zenithLuminance(float t, float thetaS) {
  const float chi = (4.0f / 9.0f - t / 120.0f) * (kPi - 2.0f * thetaS);
  return (4.0453f * t - 4.9710f) * std::tan(chi) - 0.2155f * t + 2.4192f;
}

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

glm::vec3 equirectUvToDir(float u, float v) {
  const float theta = v * kPi;
  const float phi = u * 2.0f * kPi - kPi;
  const float sinTheta = std::sin(theta);
  return glm::vec3(sinTheta * std::cos(phi), std::cos(theta), sinTheta * std::sin(phi));
}

glm::vec3 xyYToLinearSRGB(float x, float y, float Y) {
  y = std::max(y, 1e-4f);
  const float X = (x / y) * Y;
  const float Z = ((1.0f - x - y) / y) * Y;
  glm::vec3 rgb(3.2406f * X - 1.5372f * Y - 0.4986f * Z, -0.9689f * X + 1.8758f * Y + 0.0415f * Z,
                0.0557f * X - 0.2040f * Y + 1.0570f * Z);
  return glm::max(rgb, glm::vec3(0.0f));
}

constexpr float kSunAngularRadiusDeg = 2.0f;
constexpr float kSunDiskBoost = 60.0f;

constexpr float kIntensityScale = 0.075f;

} // namespace

bool buildProceduralSky(int width, int height, float sunElevationDeg, float sunAzimuthDeg, float turbidity,
                         EnvironmentMap &out, std::string &err, bool bakeSunDisk) {
  if (width <= 0 || height <= 0) {
    err = "procedural sky: width/height must be positive";
    return false;
  }
  const float t = std::max(turbidity, 1.9f);

  const float sunElevationRad = glm::radians(sunElevationDeg);
  const float sunAzimuthRad = glm::radians(sunAzimuthDeg);
  const glm::vec3 sunDir = glm::normalize(
      glm::vec3(std::cos(sunElevationRad) * std::cos(sunAzimuthRad), std::sin(sunElevationRad),
                std::cos(sunElevationRad) * std::sin(sunAzimuthRad)));
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

      const float theta = std::min(std::acos(std::clamp(dir.y, -1.0f, 1.0f)), glm::radians(89.0f));
      const float gamma = std::acos(std::clamp(glm::dot(dir, sunDir), -1.0f, 1.0f));

      const float Y = Yz * perezF(theta, gamma, cY) / std::max(perezYzenith, 1e-6f);
      const float x = xz * perezF(theta, gamma, cX) / std::max(perezXzenith, 1e-6f);
      const float y = yz * perezF(theta, gamma, cYc) / std::max(perezYczenith, 1e-6f);

      glm::vec3 rgb = xyYToLinearSRGB(x, y, std::max(Y, 0.0f)) * kIntensityScale;

      if (bakeSunDisk && gamma < sunAngularRadiusRad) {
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
