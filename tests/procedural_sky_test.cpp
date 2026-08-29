// Standalone sanity check for the Preetham procedural sky generator
// (render/procedural_sky.cpp) — host-side only, no GPU/OptiX needed, same
// as environment_cdf_test.cpp. Preetham's coefficients were transcribed by
// hand from the published paper (see procedural_sky.cpp's own caveat about
// why Preetham was chosen over Hosek-Wilkie in the first place: smaller,
// more checkable surface area for exactly this kind of mistake) — this
// checks the *behavioral* properties that actually matter for lighting a
// scene, not the coefficients themselves term-by-term.
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

  // 1) Ordinary midday sky: succeeds, every pixel finite and non-negative
  //    (a bad transcription is far more likely to produce NaN/negative
  //    radiance from an out-of-domain exp()/division than a subtly wrong
  //    but still well-formed number).
  EnvironmentMap sky;
  std::string err;
  const float sunElevationDeg = 45.0f, sunAzimuthDeg = 0.0f, turbidity = 3.0f;
  assert(buildProceduralSky(width, height, sunElevationDeg, sunAzimuthDeg, turbidity, sky, err));
  assert(sky.width == width && sky.height == height);
  for (float v : sky.pixels) {
    assert(std::isfinite(v));
    assert(v >= 0.0f);
  }

  // 2) Brighter looking at the sun than looking away from it. Convert the
  //    sun direction to the same equirect (u,v) buildProceduralSky uses
  //    (theta from +Y, phi via atan2(z,x)) to find its pixel, and compare
  //    against the antipodal direction's pixel.
  const float sunElevationRad = sunElevationDeg * static_cast<float>(M_PI) / 180.0f;
  const float sunAzimuthRad = sunAzimuthDeg * static_cast<float>(M_PI) / 180.0f;
  auto pixelForDir = [&](float elevationRad, float azimuthRad) {
    const float dirY = std::sin(elevationRad);
    const float theta = std::acos(std::max(-1.0f, std::min(1.0f, dirY)));
    // phi convention matches equirectUvToDir: dir = (sinTheta*cos(phi), cosTheta, sinTheta*sin(phi));
    // azimuthRad here is exactly that phi (both measured the same way buildProceduralSky derives sunDir).
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

  // 3) A low-but-above-horizon sun (sunset-adjacent, not the "below horizon"
  //    case procedural_sky.h documents as producing a dim/degenerate sky)
  //    still succeeds rather than hitting the all-black CDF rejection path.
  EnvironmentMap sunset;
  assert(buildProceduralSky(width, height, /*sunElevationDeg=*/5.0f, 90.0f, 4.0f, sunset, err));

  std::printf("procedural_sky_test: OK\n");
  return 0;
}
