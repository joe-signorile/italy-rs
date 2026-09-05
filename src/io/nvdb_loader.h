#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

struct NvdbVolume {
  std::vector<uint8_t> gridBlob;
  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};
  glm::vec3 sigmaT{1.0f};
  glm::vec3 scatterAlbedo{0.9f};
  float g = 0.0f;
  float densityScale = 1.0f;
  float maxDensity = 1.0f;
};

NvdbVolume buildProceduralFogSphereVolume(glm::vec3 center, float radius, float voxelSize, glm::vec3 sigmaT,
                                           glm::vec3 scatterAlbedo, float g, float densityScale);

} // namespace italy
