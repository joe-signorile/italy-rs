#pragma once

// Shared by voxelize.cpp (per-voxel-cell mesh resampling) and sdf_baker.cpp
// (unified-SDF acceleration Phase 1's per-cell material bake, see
// SdfGrid::baseColor's doc comment) — both need "does this triangle overlap
// this axis-aligned cell" and there is exactly one correct way to answer
// that question, so it lives here once rather than as two copies that could
// drift out of sync with each other.

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

namespace italy {

// Classic Akenine-Möller triangle/AABB overlap test (separating axis theorem
// over the box's 3 face normals, the triangle's normal, and the 9 cross
// products of box-edge x triangle-edge). Reference:
// "Fast 3D Triangle-Box Overlap Testing," Akenine-Möller 2001. Chosen over a
// cheaper bbox-only test because a bbox-only test visibly over-thickens thin
// or diagonal surfaces — the exact test costs little extra code.
inline bool triBoxOverlapPlaneTest(const glm::vec3 &normal, const glm::vec3 &vert, const glm::vec3 &maxbox) {
  glm::vec3 vmin, vmax;
  for (int q = 0; q < 3; ++q) {
    const float v = vert[q];
    if (normal[q] > 0.0f) {
      vmin[q] = -maxbox[q] - v;
      vmax[q] = maxbox[q] - v;
    } else {
      vmin[q] = maxbox[q] - v;
      vmax[q] = -maxbox[q] - v;
    }
  }
  if (glm::dot(normal, vmin) > 0.0f)
    return false;
  return glm::dot(normal, vmax) >= 0.0f;
}

// The six axis tests below transliterate Akenine-Möller's AXISTEST_{X01,X2,
// Y02,Y1,Z12,Z0} macros directly (including their sign conventions, which
// differ between the X/Z and Y families — that's not a typo, it falls out of
// the cross-product expansion). Kept separate rather than one falsely-generic
// helper: collapsing them into a single parameterized function risks
// transcribing the sign wrong for exactly the case that's hardest to notice
// in a quick visual check (a slightly-too-thick or slightly-too-thin result).
inline bool triBoxAxisTestX01(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v2,
                               const glm::vec3 &boxHalf) {
  const float p0 = a * v0.y - b * v0.z, p2 = a * v2.y - b * v2.z;
  const float mn = std::min(p0, p2), mx = std::max(p0, p2);
  const float rad = fa * boxHalf.y + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
inline bool triBoxAxisTestX2(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v1,
                              const glm::vec3 &boxHalf) {
  const float p0 = a * v0.y - b * v0.z, p1 = a * v1.y - b * v1.z;
  const float mn = std::min(p0, p1), mx = std::max(p0, p1);
  const float rad = fa * boxHalf.y + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
inline bool triBoxAxisTestY02(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v2,
                               const glm::vec3 &boxHalf) {
  const float p0 = -a * v0.x + b * v0.z, p2 = -a * v2.x + b * v2.z;
  const float mn = std::min(p0, p2), mx = std::max(p0, p2);
  const float rad = fa * boxHalf.x + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
inline bool triBoxAxisTestY1(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v1,
                              const glm::vec3 &boxHalf) {
  const float p0 = -a * v0.x + b * v0.z, p1 = -a * v1.x + b * v1.z;
  const float mn = std::min(p0, p1), mx = std::max(p0, p1);
  const float rad = fa * boxHalf.x + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
inline bool triBoxAxisTestZ12(float a, float b, float fa, float fb, const glm::vec3 &v1, const glm::vec3 &v2,
                               const glm::vec3 &boxHalf) {
  const float p1 = a * v1.x - b * v1.y, p2 = a * v2.x - b * v2.y;
  const float mn = std::min(p1, p2), mx = std::max(p1, p2);
  const float rad = fa * boxHalf.x + fb * boxHalf.y;
  return !(mn > rad || mx < -rad);
}
inline bool triBoxAxisTestZ0(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v1,
                              const glm::vec3 &boxHalf) {
  const float p0 = a * v0.x - b * v0.y, p1 = a * v1.x - b * v1.y;
  const float mn = std::min(p0, p1), mx = std::max(p0, p1);
  const float rad = fa * boxHalf.x + fb * boxHalf.y;
  return !(mn > rad || mx < -rad);
}

inline bool triBoxOverlap(const glm::vec3 &boxCenter, const glm::vec3 &boxHalf, const glm::vec3 &t0,
                           const glm::vec3 &t1, const glm::vec3 &t2) {
  const glm::vec3 v0 = t0 - boxCenter, v1 = t1 - boxCenter, v2 = t2 - boxCenter;
  const glm::vec3 e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;

  float fex = std::fabs(e0.x), fey = std::fabs(e0.y), fez = std::fabs(e0.z);
  if (!triBoxAxisTestX01(e0.z, e0.y, fez, fey, v0, v2, boxHalf))
    return false;
  if (!triBoxAxisTestY02(e0.z, e0.x, fez, fex, v0, v2, boxHalf))
    return false;
  if (!triBoxAxisTestZ12(e0.y, e0.x, fey, fex, v1, v2, boxHalf))
    return false;

  fex = std::fabs(e1.x);
  fey = std::fabs(e1.y);
  fez = std::fabs(e1.z);
  if (!triBoxAxisTestX01(e1.z, e1.y, fez, fey, v0, v2, boxHalf))
    return false;
  if (!triBoxAxisTestY02(e1.z, e1.x, fez, fex, v0, v2, boxHalf))
    return false;
  if (!triBoxAxisTestZ0(e1.y, e1.x, fey, fex, v0, v1, boxHalf))
    return false;

  fex = std::fabs(e2.x);
  fey = std::fabs(e2.y);
  fez = std::fabs(e2.z);
  if (!triBoxAxisTestX2(e2.z, e2.y, fez, fey, v0, v1, boxHalf))
    return false;
  if (!triBoxAxisTestY1(e2.z, e2.x, fez, fex, v0, v1, boxHalf))
    return false;
  if (!triBoxAxisTestZ12(e2.y, e2.x, fey, fex, v1, v2, boxHalf))
    return false;

  // Box-face-normal axis tests (standard AABB/AABB overlap on each axis).
  for (int axis = 0; axis < 3; ++axis) {
    float minV = std::min({v0[axis], v1[axis], v2[axis]});
    float maxV = std::max({v0[axis], v1[axis], v2[axis]});
    if (minV > boxHalf[axis] || maxV < -boxHalf[axis])
      return false;
  }

  // Triangle-plane axis test.
  const glm::vec3 normal = glm::cross(e0, e1);
  return triBoxOverlapPlaneTest(normal, v0, boxHalf);
}

} // namespace italy
