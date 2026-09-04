// Standalone sanity check for the HDRI importance-sampling CDF build (render/environment.cpp): no framework, just asserts.
#include <cassert>
#include <cstdio>
#include <vector>

#include "render/environment.h"

using italy::buildEnvironmentCdf;
using italy::EnvironmentMap;

namespace {

std::vector<float> makeSinglePixelImage(int w, int h, int litX, int litY, float radiance) {
  std::vector<float> pixels(static_cast<size_t>(w) * h * 3, 0.0f);
  float *p = &pixels[(static_cast<size_t>(litY) * w + litX) * 3];
  p[0] = p[1] = p[2] = radiance;
  return pixels;
}

} // namespace

int main() {
  const int w = 8, h = 4;

  {
    EnvironmentMap env;
    std::string err;
    const std::vector<float> black(static_cast<size_t>(w) * h * 3, 0.0f);
    assert(!buildEnvironmentCdf(w, h, black, env, err));
    assert(!err.empty());
  }

  {
    const int litX = 5, litY = 2;
    EnvironmentMap env;
    std::string err;
    assert(buildEnvironmentCdf(w, h, makeSinglePixelImage(w, h, litX, litY, 100.0f), env, err));

    assert(env.marginalCdf.size() == static_cast<size_t>(h) + 1);
    const float rowMassBefore = env.marginalCdf[litY];
    const float rowMassAfter = env.marginalCdf[litY + 1];
    assert(rowMassBefore < 1e-6f);
    assert(rowMassAfter > 1.0f - 1e-6f);

    const float *rowCdf = &env.conditionalCdf[static_cast<size_t>(litY) * (w + 1)];
    assert(rowCdf[litX] < 1e-6f);
    assert(rowCdf[litX + 1] > 1.0f - 1e-6f);

    for (int y = 0; y < h; ++y) {
      if (y == litY)
        continue;
      const float *r = &env.conditionalCdf[static_cast<size_t>(y) * (w + 1)];
      for (int x = 0; x <= w; ++x)
        assert(r[x] == r[x]);
    }
  }

  std::printf("environment_cdf_test: OK\n");
  return 0;
}
