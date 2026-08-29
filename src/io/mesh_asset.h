#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

struct TextureAsset {
  std::vector<uint8_t> pixelsRGBA; // width*height*4, row-major, top-to-bottom
  int width = 0;
  int height = 0;
  // glTF base-colour textures are sRGB-encoded; metallic-roughness and normal
  // maps are linear. The GPU upload needs to know which, because CUDA does the
  // decode in texture hardware and decoding a normal map would bend every
  // normal toward the +Z pole.
  bool srgb = false;
};

// One glTF material. Texture members index MeshAsset::textures, -1 for absent.
struct MaterialAsset {
  glm::vec3 baseColorFactor{1.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
  int baseColorTexture = -1;
  int metallicRoughnessTexture = -1; // glTF packs roughness in G, metallic in B
  int normalTexture = -1;
  float normalScale = 1.0f;

  // KHR_materials_transmission/ior/volume — glass-like dielectric surfaces
  // on ordinary triangle geometry (as opposed to MATERIAL_GLASS's dedicated
  // analytic-sphere path). Only meaningful once the mesh has passed the
  // watertightness gate (see mesh_validate.h): refraction through triangle
  // geometry needs a well-defined interior, same reasoning the built-in
  // sphere primitive's hollowness fix needed one for MATERIAL_GLASS.
  // Defaults are glTF's own "fully opaque, no absorption" values.
  float transmission = 0.0f;
  float ior = 1.5f;
  glm::vec3 attenuationColor{1.0f};
  float attenuationDistance = std::numeric_limits<float>::infinity();
};

// Flattened triangle soup (3 entries per triangle, no index buffer) — trades
// some duplicated vertex data for a renderer that never needs an index lookup
// alongside optixGetTriangleVertexData().
//
// A whole glTF scene collapses into this one soup: every primitive of every
// mesh reachable from the scene graph, baked into world space by its node
// chain. Per-primitive material identity survives as triangleMaterial, so
// flattening the geometry doesn't flatten the shading.
struct MeshAsset {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;      // (0,0) where the source primitive had none
  std::vector<glm::vec4> tangents; // xyz = tangent, w = bitangent sign; for normal mapping

  std::vector<uint32_t> triangleMaterial; // one entry per triangle
  std::vector<MaterialAsset> materials;   // always at least one (a default)
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
