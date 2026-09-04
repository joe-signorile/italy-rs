#pragma once

#include <utility>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace italy {

inline std::pair<glm::vec3, glm::vec3> computeSplatAabb(const glm::vec3 &position,
                                                          const glm::vec3 &scale,
                                                          const glm::vec4 &rotationXYZW) {
  constexpr float kSigmaExtent = 3.0f;
  const glm::vec3 halfExtent = scale * kSigmaExtent;
  const glm::quat q(rotationXYZW.w, rotationXYZW.x, rotationXYZW.y, rotationXYZW.z);
  const glm::mat3 r = glm::mat3_cast(glm::normalize(q));

  glm::vec3 worldHalfExtent(0.0f);
  for (int i = 0; i < 3; ++i) {
    worldHalfExtent[i] = std::abs(r[0][i]) * halfExtent.x + std::abs(r[1][i]) * halfExtent.y +
                          std::abs(r[2][i]) * halfExtent.z;
  }
  return {position - worldHalfExtent, position + worldHalfExtent};
}

} // namespace italy
