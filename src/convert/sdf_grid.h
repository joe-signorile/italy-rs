#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

enum class SdfMaterialKind : uint8_t { Opaque = 0, Dielectric = 1 };

struct SdfPaletteMaterial {
  SdfMaterialKind kind = SdfMaterialKind::Opaque;
  glm::vec3 baseColor{1.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
  float ior = 1.0f;
  glm::vec3 extinction{0.0f};
};

struct SdfGrid {
  glm::vec3 origin{0.0f};
  float voxelSize = 1.0f;
  int nx = 0, ny = 0, nz = 0;
  std::vector<float> distances;

  std::vector<glm::vec3> baseColor;
  std::vector<float> metallic;
  std::vector<float> roughness;

  std::vector<uint8_t> branch;
  std::vector<SdfPaletteMaterial> palette;

  glm::vec3 boundsMax() const { return origin + glm::vec3(nx, ny, nz) * voxelSize; }
  size_t index(int x, int y, int z) const { return (static_cast<size_t>(z) * ny + y) * nx + x; }
};

} // namespace italy
