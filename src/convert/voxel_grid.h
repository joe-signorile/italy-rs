#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace italy {

struct VoxelGrid {
  glm::vec3 origin{0.0f};
  float voxelSize = 1.0f;
  std::vector<glm::ivec3> cells;
  std::vector<glm::vec3> colors;

  glm::vec3 cellMin(const glm::ivec3 &c) const { return origin + glm::vec3(c) * voxelSize; }
  glm::vec3 cellMax(const glm::ivec3 &c) const { return cellMin(c) + glm::vec3(voxelSize); }
};

} // namespace italy
