// Standalone sanity check for the Preetham procedural sky generator (render/procedural_sky.cpp): host-side only, checks behavioral lighting properties, not coefficients.
#include <cassert>
#include <cmath>
#include <cstdio>

#include "render/environment.h"
#include "render/procedural_sky.h"

using italy::buildProceduralSky;
using italy::EnvironmentMap;

namespace {

float luminanceAt(const EnvironmentMap &env, int px, int py) {
  const float *p = &env.pixels[(static_cast<size_t>(py) * env.width + px) * 3];
  return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
}

} // namespace

int main() {
  const int width = 256, height = 128;

  EnvironmentMap sky;
  std::string err;
  const float sunElevationDeg = 45.0f, sunAzimuthDeg = 0.0f, turbidity = 3.0f;
  assert(buildProceduralSky(width, height, sunElevationDeg, sunAzimuthDeg, turbidity, sky, err));
  assert(sky.width == width && sky.height == height);
  for (float v : sky.pixels) {
    assert(std::isfinite(v));
    assert(v >= 0.0f);
  }

  const float sunElevationRad = sunElevationDeg * static_cast<float>(M_PI) / 180.0f;
  const float sunAzimuthRad = sunAzimuthDeg * static_cast<float>(M_PI) / 180.0f;
  auto pixelForDir = [&](float elevationRad, float azimuthRad) {
    const float dirY = std::sin(elevationRad);
    const float theta = std::acos(std::max(-1.0f, std::min(1.0f, dirY)));
    float u = (azimuthRad + static_cast<float>(M_PI)) / (2.0f * static_cast<float>(M_PI));
    u -= std::floor(u);
    const float v = theta / static_cast<float>(M_PI);
    int px = std::min(width - 1, static_cast<int>(u * width));
    int py = std::min(height - 1, static_cast<int>(v * height));
    return std::pair<int, int>(px, py);
  };
  const auto [sunPx, sunPy] = pixelForDir(sunElevationRad, sunAzimuthRad);
  const auto [antiPx, antiPy] = pixelForDir(-sunElevationRad, sunAzimuthRad + static_cast<float>(M_PI));
  const float sunLum = luminanceAt(sky, sunPx, sunPy);
  const float antiLum = luminanceAt(sky, antiPx, antiPy);
  std::printf("procedural_sky_test: sun-direction luminance %.4f vs anti-sun %.4f\n", sunLum, antiLum);
  assert(sunLum > antiLum * 2.0f && "sky should read distinctly brighter toward the sun than away from it");

  EnvironmentMap sunset;
  assert(buildProceduralSky(width, height, 5.0f, 90.0f, 4.0f, sunset, err));

  std::printf("procedural_sky_test: OK\n");
  return 0;
}
