#pragma once

#include <cstddef>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

struct GsplatAsset {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> scales;
  std::vector<glm::vec4> rotations;
  std::vector<float> opacity;
  std::vector<glm::vec3> colorDC;

  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};

  size_t count() const { return positions.size(); }
  glm::vec3 boundsCenter() const { return (boundsMin + boundsMax) * 0.5f; }
  float boundsRadius() const { return glm::length(boundsMax - boundsMin) * 0.5f; }
};

} // namespace italy
