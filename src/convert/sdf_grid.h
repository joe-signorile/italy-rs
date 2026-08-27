#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace italy {

// Dense (not narrow-band) regular grid of signed distances. The design doc
// says "narrow-band" but narrow-band storage only helps memory for large
// grids and is awkward to sphere-trace through (empty space far from the
// surface needs *some* usable lower-bound distance so marching can safely
// take large steps there) — dense is simpler and unambiguous to render, and
// at MVP resolutions (~64^3) the memory difference doesn't matter. Revisit
// if resolution needs to grow a lot.
struct SdfGrid {
  glm::vec3 origin{0.0f};
  float voxelSize = 1.0f;
  int nx = 0, ny = 0, nz = 0;
  std::vector<float> distances; // nx*ny*nz, negative inside the surface
  glm::vec3 tintColor{1.0f}; // flat color for the whole field — see MATERIAL_SDF's doc comment for why

  glm::vec3 boundsMax() const { return origin + glm::vec3(nx, ny, nz) * voxelSize; }
  size_t index(int x, int y, int z) const { return (static_cast<size_t>(z) * ny + y) * nx + x; }
};

} // namespace italy
