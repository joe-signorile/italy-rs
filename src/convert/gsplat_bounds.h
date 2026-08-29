#pragma once

#include <utility>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace italy {

// World-space AABB tightly bounding one splat's 3-sigma ellipsoid. Splats are
// often strongly anisotropic (thin, flat disks), so this rotates the
// per-axis 3-sigma half-extents by the splat's orientation rather than using
// a sphere bound (which would be wasteful — one flat splat's bounding sphere
// radius is set by its longest axis, ballooning the AABB on the other two).
//
// Standard rotated-box-to-AABB trick: world-space half-extent along axis i
// is sum_j |R[i][j]| * localHalfExtent[j], where R is the rotation matrix.
// Pure host-side math (no OptiX), so it's unit-testable without a GPU — see
// tests/gsplat_ply_loader_test.cpp.
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
