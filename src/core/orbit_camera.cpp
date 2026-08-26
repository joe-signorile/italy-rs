#include "core/orbit_camera.h"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

namespace italy {

void OrbitCamera::orbit(float dxPixels, float dyPixels) {
  yaw_ -= dxPixels * kOrbitSpeed;
  pitch_ = std::clamp(pitch_ - dyPixels * kOrbitSpeed, kMinPitch, kMaxPitch);
}

void OrbitCamera::pan(float dxPixels, float dyPixels) {
  const glm::vec3 eye = position();
  const glm::vec3 forward = glm::normalize(target_ - eye);
  const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
  const glm::vec3 up = glm::cross(right, forward);
  const float scale = distance_ * kPanSpeed;
  target_ += (-dxPixels * right + dyPixels * up) * scale;
}

void OrbitCamera::zoom(float scrollDelta) {
  distance_ = std::max(kMinDistance, distance_ * (1.0f - scrollDelta * kZoomSpeed));
}

glm::vec3 OrbitCamera::position() const {
  const float x = distance_ * std::cos(pitch_) * std::sin(yaw_);
  const float y = distance_ * std::sin(pitch_);
  const float z = distance_ * std::cos(pitch_) * std::cos(yaw_);
  return target_ + glm::vec3(x, y, z);
}

glm::mat4 OrbitCamera::viewMatrix() const {
  return glm::lookAt(position(), target_, glm::vec3(0, 1, 0));
}

} // namespace italy
