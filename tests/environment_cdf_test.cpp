// Standalone sanity check for the HDRI importance-sampling CDF build
// (render/environment.cpp), matching voxelize_test.cpp's tool-scale: no
// framework, just asserts, run via the `environment_cdf_test` CMake target.
//
// Builds a synthetic known-luminance image directly in memory via
// buildEnvironmentCdf (the same code loadEnvironmentMap calls after
// decoding a real .hdr file) rather than round-tripping through a
// hand-written Radiance HDR file on disk.
#include <cassert>
#include <cstdio>
#include <vector>

#include "render/environment.h"

using italy::buildEnvironmentCdf;
using italy::EnvironmentMap;

namespace {

// width*height*3 linear radiance, single pixel lit, everything else black.
std::vector<float> makeSinglePixelImage(int w, int h, int litX, int litY, float radiance) {
  std::vector<float> pixels(static_cast<size_t>(w) * h * 3, 0.0f);
  float *p = &pixels[(static_cast<size_t>(litY) * w + litX) * 3];
  p[0] = p[1] = p[2] = radiance;
  return pixels;
}

} // namespace

int main() {
  const int w = 8, h = 4;

  // 1) All-black image must fail cleanly (marginal total is zero) rather
  //    than produce a distribution that silently samples garbage.
  {
    EnvironmentMap env;
    std::string err;
    const std::vector<float> black(static_cast<size_t>(w) * h * 3, 0.0f);
    assert(!buildEnvironmentCdf(w, h, black, env, err));
    assert(!err.empty());
  }

  // 2) A single bright texel should concentrate essentially all sampling
  //    probability at its row (marginal CDF) and its column within that
  //    row (conditional CDF) — the actual property that distinguishes a
  //    working importance sampler from a uniform one, and the thing that
  //    lets the noon-preset sun disk converge quickly instead of fireflying
  //    (see humans.md's phase 6 note).
  {
    const int litX = 5, litY = 2;
    EnvironmentMap env;
    std::string err;
    assert(buildEnvironmentCdf(w, h, makeSinglePixelImage(w, h, litX, litY, 100.0f), env, err));

    // Marginal CDF: row litY should absorb ~all the probability mass, i.e.
    // marginalCdf[litY] ~ 0 and marginalCdf[litY+1] ~ 1.
    assert(env.marginalCdf.size() == static_cast<size_t>(h) + 1);
    const float rowMassBefore = env.marginalCdf[litY];
    const float rowMassAfter = env.marginalCdf[litY + 1];
    assert(rowMassBefore < 1e-6f);
    assert(rowMassAfter > 1.0f - 1e-6f);

    // Conditional CDF within the lit row: column litX should absorb ~all
    // the mass in that row too.
    const float *rowCdf = &env.conditionalCdf[static_cast<size_t>(litY) * (w + 1)];
    assert(rowCdf[litX] < 1e-6f);
    assert(rowCdf[litX + 1] > 1.0f - 1e-6f);

    // Every other row's conditional CDF is the degenerate all-black
    // fallback (uniform), never selected because its marginal weight is
    // zero, but it must still be a well-formed monotonic CDF (no NaN/div0).
    for (int y = 0; y < h; ++y) {
      if (y == litY)
        continue;
      const float *r = &env.conditionalCdf[static_cast<size_t>(y) * (w + 1)];
      for (int x = 0; x <= w; ++x)
        assert(r[x] == r[x]); // not NaN
    }
  }

  std::printf("environment_cdf_test: OK\n");
  return 0;
}
