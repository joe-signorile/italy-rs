#pragma once

// Classic Akenine-Möller triangle/AABB separating-axis overlap test, shared by voxelize.cpp and sdf_baker.cpp.

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

namespace italy {

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

  for (int axis = 0; axis < 3; ++axis) {
    float minV = std::min({v0[axis], v1[axis], v2[axis]});
    float maxV = std::max({v0[axis], v1[axis], v2[axis]});
    if (minV > boxHalf[axis] || maxV < -boxHalf[axis])
      return false;
  }

  const glm::vec3 normal = glm::cross(e0, e1);
  return triBoxOverlapPlaneTest(normal, v0, boxHalf);
}

} // namespace italy
