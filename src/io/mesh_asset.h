#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

struct TextureAsset {
  std::vector<uint8_t> pixelsRGBA; // width*height*4, row-major, top-to-bottom
  int width = 0;
  int height = 0;
};

// Flattened triangle soup (3 entries per triangle, no index buffer) — trades
// some duplicated vertex data for a renderer that never needs an index
// lookup alongside optixGetTriangleVertexData(). Fine at MVP asset sizes.
struct MeshAsset {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs; // empty if the source mesh had no UVs

  glm::vec3 baseColorFactor{1.0f, 1.0f, 1.0f};
  bool hasBaseColorTexture = false;
  TextureAsset baseColorTexture;

  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};

  glm::vec3 boundsCenter() const { return (boundsMin + boundsMax) * 0.5f; }
  float boundsRadius() const { return glm::length(boundsMax - boundsMin) * 0.5f; }
};

} // namespace italy
