// Sanity check for sdf_baker.cpp's sign/magnitude correctness against a
// primitive with a known analytic SDF (a unit sphere), per the original
// plan doc's verification ask. Unlike voxelize_test.cpp this exercises the
// real OptiX bake pipeline (sdf_baker.cpp builds its own short-lived
// OptixDeviceContext — see that file's header comment), so it needs an
// actual GPU/OptiX SDK at run time, same as italy-rs itself.
//
// optix_function_table_definition.h defines a process-wide global and must
// appear in exactly one translation unit in this binary (see the comment
// at the top of sdf_baker.cpp) — this test binary doesn't link
// optix_renderer.cpp, so it provides that definition itself.
#include <optix_function_table_definition.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "convert/sdf_baker.h"
#include "io/mesh_asset.h"

using italy::MeshAsset;
using italy::SdfGrid;
using italy::bakeSdf;

namespace {

// Closed triangle-soup UV sphere, radius r centered at the origin —
// watertight (shared latitude/longitude grid, no gaps), which is what
// bakeSdf's ray-parity sign vote requires to be reliable.
MeshAsset makeUvSphere(float r, int lonSegs, int latSegs) {
  MeshAsset mesh;
  auto vertex = [&](int lat, int lon) {
    const float theta = static_cast<float>(lat) / latSegs * static_cast<float>(M_PI);       // 0..pi
    const float phi = static_cast<float>(lon) / lonSegs * 2.0f * static_cast<float>(M_PI); // 0..2pi
    return glm::vec3(r * std::sin(theta) * std::cos(phi), r * std::cos(theta), r * std::sin(theta) * std::sin(phi));
  };
  for (int lat = 0; lat < latSegs; ++lat) {
    for (int lon = 0; lon < lonSegs; ++lon) {
      const glm::vec3 a = vertex(lat, lon);
      const glm::vec3 b = vertex(lat + 1, lon);
      const glm::vec3 c = vertex(lat + 1, lon + 1);
      const glm::vec3 d = vertex(lat, lon + 1);
      // Two triangles per quad; degenerate at the poles (a==b or c==d) but
      // a zero-area triangle doesn't break ray-parity counting.
      for (glm::vec3 v : {a, b, c, a, c, d}) {
        mesh.positions.push_back(v);
        mesh.normals.push_back(glm::normalize(v));
        mesh.uvs.push_back(glm::vec2(0));
      }
    }
  }
  mesh.boundsMin = glm::vec3(-r);
  mesh.boundsMax = glm::vec3(r);
  mesh.materials.push_back(italy::MaterialAsset{}); // see voxelize_test.cpp's makeUnitCubeShell for why this is required
  return mesh;
}

} // namespace

int main() {
  const float radius = 1.0f;
  const int resolution = 24;
  const MeshAsset sphere = makeUvSphere(radius, /*lonSegs=*/32, /*latSegs=*/16);
  const SdfGrid grid = bakeSdf(sphere, resolution);

  assert(!grid.distances.empty());

  // sdf_baker.cpp's distance estimate is a minimum-hit-distance-over-random-
  // directions approximation, not exact — allow slack proportional to the
  // cell size rather than expecting bit-exact analytic agreement.
  const float tolerance = 1.5f * grid.voxelSize;

  size_t checked = 0, wrongSign = 0;
  double maxAbsError = 0.0;
  for (int z = 0; z < grid.nz; ++z) {
    for (int y = 0; y < grid.ny; ++y) {
      for (int x = 0; x < grid.nx; ++x) {
        const glm::vec3 center = grid.origin + (glm::vec3(x, y, z) + 0.5f) * grid.voxelSize;
        const float analytic = glm::length(center) - radius;
        const float baked = grid.distances[grid.index(x, y, z)];

        // Sign check only makes sense away from the surface itself — right
        // at the boundary, grid quantization can legitimately put the
        // baked and analytic values on opposite sides of zero.
        if (std::fabs(analytic) > tolerance) {
          ++checked;
          if ((analytic < 0.0f) != (baked < 0.0f))
            ++wrongSign;
          maxAbsError = std::max(maxAbsError, static_cast<double>(std::fabs(baked - analytic)));
        }
      }
    }
  }

  std::printf("sdf_sign_test: %zu/%zu cells checked away from the surface, max |error| = %.4f (tolerance %.4f)\n",
              checked - wrongSign, checked, maxAbsError, static_cast<double>(tolerance));
  assert(checked > 0 && "test grid too coarse — no cells landed outside the tolerance band");
  assert(wrongSign == 0 && "sign disagreement between baked SDF and the analytic sphere SDF");
  assert(maxAbsError < 2.0 * tolerance && "baked distance magnitude too far from the analytic sphere SDF");

  std::printf("sdf_sign_test: OK\n");
  return 0;
}
