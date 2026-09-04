// Sanity check for sdf_baker.cpp's sign/magnitude correctness against a unit sphere's known analytic SDF; exercises the real OptiX bake pipeline so it needs a GPU/OptiX SDK at run time.
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

MeshAsset makeUvSphere(float r, int lonSegs, int latSegs) {
  MeshAsset mesh;
  auto vertex = [&](int lat, int lon) {
    const float theta = static_cast<float>(lat) / latSegs * static_cast<float>(M_PI);
    const float phi = static_cast<float>(lon) / lonSegs * 2.0f * static_cast<float>(M_PI);
    return glm::vec3(r * std::sin(theta) * std::cos(phi), r * std::cos(theta), r * std::sin(theta) * std::sin(phi));
  };
  for (int lat = 0; lat < latSegs; ++lat) {
    for (int lon = 0; lon < lonSegs; ++lon) {
      const glm::vec3 a = vertex(lat, lon);
      const glm::vec3 b = vertex(lat + 1, lon);
      const glm::vec3 c = vertex(lat + 1, lon + 1);
      const glm::vec3 d = vertex(lat, lon + 1);
      for (glm::vec3 v : {a, b, c, a, c, d}) {
        mesh.positions.push_back(v);
        mesh.normals.push_back(glm::normalize(v));
        mesh.uvs.push_back(glm::vec2(0));
      }
    }
  }
  mesh.boundsMin = glm::vec3(-r);
  mesh.boundsMax = glm::vec3(r);
  mesh.materials.push_back(italy::MaterialAsset{});
  return mesh;
}

} // namespace

int main() {
  const float radius = 1.0f;
  const int resolution = 24;
  const MeshAsset sphere = makeUvSphere(radius, 32, 16);
  const SdfGrid grid = bakeSdf(sphere, resolution);

  assert(!grid.distances.empty());

  const float tolerance = 1.5f * grid.voxelSize;

  size_t checked = 0, wrongSign = 0;
  double maxAbsError = 0.0;
  for (int z = 0; z < grid.nz; ++z) {
    for (int y = 0; y < grid.ny; ++y) {
      for (int x = 0; x < grid.nx; ++x) {
        const glm::vec3 center = grid.origin + (glm::vec3(x, y, z) + 0.5f) * grid.voxelSize;
        const float analytic = glm::length(center) - radius;
        const float baked = grid.distances[grid.index(x, y, z)];

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
