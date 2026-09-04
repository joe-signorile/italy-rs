#pragma once

// Equirectangular HDRI environment map with a precomputed piecewise-constant 2D distribution for luminance-weighted importance sampling (PBRT-style).

#include <string>
#include <vector>

namespace italy {

struct EnvironmentMap {
  int width = 0, height = 0;
  std::vector<float> pixels;

  std::vector<float> marginalCdf;
  std::vector<float> conditionalCdf;
};

bool loadEnvironmentMap(const std::string &path, EnvironmentMap &out, std::string &err);

bool buildEnvironmentCdf(int width, int height, const std::vector<float> &pixels, EnvironmentMap &out,
                          std::string &err);

} // namespace italy
