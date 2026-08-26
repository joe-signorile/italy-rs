#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace italy {

// Sparse occupied-cell list rather than a dense 3D array — GLB meshes are
// surfaces, so occupancy at any reasonable resolution is a small fraction of
// resolution^3.
struct VoxelGrid {
  glm::vec3 origin{0.0f};    // world-space corner of cell (0,0,0)
  float voxelSize = 1.0f;    // world-space edge length of one (cubic) voxel
  std::vector<glm::ivec3> cells;
  std::vector<glm::vec3> colors; // parallel to cells

  glm::vec3 cellMin(const glm::ivec3 &c) const { return origin + glm::vec3(c) * voxelSize; }
  glm::vec3 cellMax(const glm::ivec3 &c) const { return cellMin(c) + glm::vec3(voxelSize); }
};

} // namespace italy
