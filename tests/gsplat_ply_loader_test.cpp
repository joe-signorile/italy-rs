// Standalone sanity check for the gsplat .ply loader (io/gsplat_ply_loader.cpp)
// and its AABB helper (convert/gsplat_bounds.h), matching this repo's other
// tests' tool-scale: no framework, just asserts.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

#include "convert/gsplat_bounds.h"
#include "io/gsplat_ply_loader.h"

using italy::computeSplatAabb;
using italy::GsplatAsset;
using italy::loadGsplatPly;

namespace {

constexpr float kShC0 = 0.28209479177387814f;

// Two splats, deliberately different scale/rotation/opacity/color so field
// mixups (e.g. swapped x/y, wrong SH constant) would show up as a wrong
// value rather than accidentally passing.
const char *kAsciiPly =
    "ply\n"
    "format ascii 1.0\n"
    "element vertex 2\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property float f_dc_0\n"
    "property float f_dc_1\n"
    "property float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\n"
    "property float scale_1\n"
    "property float scale_2\n"
    "property float rot_0\n"
    "property float rot_1\n"
    "property float rot_2\n"
    "property float rot_3\n"
    "end_header\n"
    "1 2 3 1 0 -1 0 0 0.6931472 1.0986123 1 0 0 0\n"
    "-1 0 0.5 0 0 0 -2 0 0 0 0 0 0 1\n";

void writeFile(const std::string &path, const char *contents) {
  std::ofstream f(path, std::ios::binary);
  f << contents;
}

} // namespace

int main() {
  const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                            "/italy_gsplat_test.ply";
  writeFile(path, kAsciiPly);

  GsplatAsset splats;
  std::string err;
  const bool ok = loadGsplatPly(path, splats, err);
  assert(ok);
  assert(err.empty());
  assert(splats.count() == 2);

  // Vertex 0: position passes through unchanged.
  assert(std::abs(splats.positions[0].x - 1.0f) < 1e-5f);
  assert(std::abs(splats.positions[0].y - 2.0f) < 1e-5f);
  assert(std::abs(splats.positions[0].z - 3.0f) < 1e-5f);

  // scale_0/1/2 = 0, 0.6931472 (=ln2), 1.0986123 (=ln3) -> exp() = 1, 2, 3.
  assert(std::abs(splats.scales[0].x - 1.0f) < 1e-4f);
  assert(std::abs(splats.scales[0].y - 2.0f) < 1e-3f);
  assert(std::abs(splats.scales[0].z - 3.0f) < 1e-3f);

  // opacity logit 0 -> sigmoid(0) = 0.5.
  assert(std::abs(splats.opacity[0] - 0.5f) < 1e-5f);

  // f_dc = (1, 0, -1) -> color = clamp(0.5 + SH_C0*f_dc, 0, 1).
  const float expectedR = std::min(1.0f, 0.5f + kShC0 * 1.0f);
  const float expectedG = 0.5f;
  const float expectedB = std::max(0.0f, 0.5f + kShC0 * -1.0f);
  assert(std::abs(splats.colorDC[0].r - expectedR) < 1e-5f);
  assert(std::abs(splats.colorDC[0].g - expectedG) < 1e-5f);
  assert(std::abs(splats.colorDC[0].b - expectedB) < 1e-5f);

  // Vertex 1: opacity logit -2 -> sigmoid(-2) ~= 0.11920292.
  assert(std::abs(splats.opacity[1] - 0.11920292f) < 1e-5f);

  // Bounds cover both positions.
  assert(splats.boundsMin.x <= -1.0f && splats.boundsMax.x >= 1.0f);
  assert(splats.boundsMin.y <= 0.0f && splats.boundsMax.y >= 2.0f);

  // A malformed file (missing required properties) is rejected, not
  // silently accepted with garbage fields.
  {
    const char *kBadPly =
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n1 2 3\n";
    writeFile(path, kBadPly);
    GsplatAsset bad;
    std::string badErr;
    assert(!loadGsplatPly(path, bad, badErr));
    assert(!badErr.empty());
  }

  // computeSplatAabb: identity rotation, isotropic scale 1 -> AABB is
  // exactly [pos - 3, pos + 3] on every axis.
  {
    const glm::vec3 pos(5.0f, -1.0f, 2.0f);
    const glm::vec3 scale(1.0f, 1.0f, 1.0f);
    const glm::vec4 identityRot(0.0f, 0.0f, 0.0f, 1.0f); // (x,y,z,w)
    auto [lo, hi] = computeSplatAabb(pos, scale, identityRot);
    assert(std::abs(lo.x - 2.0f) < 1e-4f && std::abs(hi.x - 8.0f) < 1e-4f);
    assert(std::abs(lo.y - -4.0f) < 1e-4f && std::abs(hi.y - 2.0f) < 1e-4f);
    assert(std::abs(lo.z - -1.0f) < 1e-4f && std::abs(hi.z - 5.0f) < 1e-4f);
  }

  // computeSplatAabb: a 90-degree rotation about Z swaps which local axis
  // (x vs y) contributes to which world axis — an anisotropic scale should
  // show up rotated, not axis-aligned to the local frame.
  {
    const glm::vec3 pos(0.0f);
    const glm::vec3 scale(1.0f, 4.0f, 1.0f); // local Y is the long axis
    const float half = std::sqrt(0.5f);
    const glm::vec4 rot90AboutZ(0.0f, 0.0f, half, half); // (x,y,z,w)
    auto [lo, hi] = computeSplatAabb(pos, scale, rot90AboutZ);
    // Local Y (long axis, half-extent 12) rotates onto world -X, so world X
    // half-extent should be ~12, not ~3 (which it would be if rotation were
    // ignored and the local box were used axis-aligned).
    assert(hi.x > 10.0f);
    assert(hi.y < 4.0f);
  }

  std::printf("gsplat_ply_loader_test: OK\n");
  return 0;
}
