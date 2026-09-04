#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

namespace italy::sdf {

inline float sdSphere(const glm::vec3 &p, float r) { return glm::length(p) - r; }

inline float sdBox(const glm::vec3 &p, const glm::vec3 &b) {
  const glm::vec3 q = glm::abs(p) - b;
  return glm::length(glm::max(q, 0.0f)) + std::min(std::max({q.x, q.y, q.z}), 0.0f);
}

inline float sdCylinder(const glm::vec3 &p, float r, float h) {
  const glm::vec2 d = glm::abs(glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y)) - glm::vec2(r, h);
  return std::min(std::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, 0.0f));
}

inline float sdTorus(const glm::vec3 &p, float major, float minor) {
  const glm::vec2 q(glm::length(glm::vec2(p.x, p.z)) - major, p.y);
  return glm::length(q) - minor;
}

inline float opUnion(float a, float b) { return std::min(a, b); }

inline float opSubtract(float a, float b) { return std::max(a, -b); }

inline float opIntersect(float a, float b) { return std::max(a, b); }

inline float opSmoothUnion(float a, float b, float k) {
  const float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
  return glm::mix(b, a, h) - k * h * (1.0f - h);
}

inline float opSmoothIntersect(float a, float b, float k) { return -opSmoothUnion(-a, -b, k); }

} // namespace italy::sdf
