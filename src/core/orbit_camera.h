#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

namespace italy {

class OrbitCamera {
public:
  void orbit(float dxPixels, float dyPixels);
  void pan(float dxPixels, float dyPixels);
  void zoom(float scrollDelta);

  void frame(const glm::vec3 &center, float boundingRadius) {
    target_ = center;
    distance_ = std::max(boundingRadius / std::tan(fovYRadians * 0.5f) * 1.15f, kMinDistance);
  }

  glm::mat4 viewMatrix() const;
  glm::vec3 position() const;
  glm::vec3 target() const { return target_; }

  float fovYRadians = glm::radians(45.0f);

private:
  glm::vec3 target_{0.0f, 0.0f, 0.0f};
  float yaw_ = glm::radians(45.0f);
  float pitch_ = glm::radians(25.0f);
  float distance_ = 5.0f;

  static constexpr float kMinPitch = glm::radians(-89.0f);
  static constexpr float kMaxPitch = glm::radians(89.0f);
  static constexpr float kMinDistance = 0.1f;
  static constexpr float kOrbitSpeed = 0.01f;
  static constexpr float kPanSpeed = 0.002f;
  static constexpr float kZoomSpeed = 0.1f;
};

} // namespace italy
