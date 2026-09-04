#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

struct TextureAsset {
  std::vector<uint8_t> pixelsRGBA;
  int width = 0;
  int height = 0;
  bool srgb = false;
};

struct MaterialAsset {
  glm::vec3 baseColorFactor{1.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
  int baseColorTexture = -1;
  int metallicRoughnessTexture = -1;
  int normalTexture = -1;
  float normalScale = 1.0f;

  float transmission = 0.0f;
  float ior = 1.5f;
  glm::vec3 attenuationColor{1.0f};
  float attenuationDistance = std::numeric_limits<float>::infinity();
};

struct MeshAsset {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec4> tangents;

  std::vector<uint32_t> triangleMaterial;
  std::vector<MaterialAsset> materials;
  std::vector<TextureAsset> textures;

  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};

  size_t triangleCount() const { return positions.size() / 3; }
  glm::vec3 boundsCenter() const { return (boundsMin + boundsMax) * 0.5f; }
  float boundsRadius() const { return glm::length(boundsMax - boundsMin) * 0.5f; }

  const MaterialAsset &materialForTriangle(size_t tri) const {
    const uint32_t m = tri < triangleMaterial.size() ? triangleMaterial[tri] : 0u;
    return materials[m < materials.size() ? m : 0];
  }
};

} // namespace italy
