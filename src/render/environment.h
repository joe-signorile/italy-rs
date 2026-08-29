#pragma once

#include <string>
#include <vector>

namespace italy {

// Equirectangular HDRI environment map with a precomputed piecewise-constant
// 2D distribution for importance sampling (luminance-weighted, PBRT-style)
// — see environment.cpp for the sampling math and pathtracer.cu for how the
// device side uses it (NEE toward bright regions of the sky instead of
// uniform hemisphere sampling).
struct EnvironmentMap {
  int width = 0, height = 0;
  std::vector<float> pixels; // width*height*3, linear radiance (HDR files are already linear — no sRGB decode)

  std::vector<float> marginalCdf;    // height+1
  std::vector<float> conditionalCdf; // height*(width+1), row-major
};

bool loadEnvironmentMap(const std::string &path, EnvironmentMap &out, std::string &err);

// Builds the marginal/conditional CDFs from already-loaded pixels (linear
// radiance, width*height*3). Split out of loadEnvironmentMap so the
// importance-sampling math is testable against a synthetic image without
// needing a real .hdr file on disk — see tests/environment_cdf_test.cpp.
// Returns false (with err set) if the image is entirely black, same as
// loadEnvironmentMap.
bool buildEnvironmentCdf(int width, int height, const std::vector<float> &pixels, EnvironmentMap &out,
                          std::string &err);

} // namespace italy
