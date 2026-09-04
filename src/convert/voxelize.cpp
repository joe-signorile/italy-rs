#include "convert/voxelize.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "convert/tri_box_overlap.h"

namespace italy {
namespace {

float srgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

glm::vec3 sampleColorNearest(const TextureAsset &tex, glm::vec2 uv) {
  auto wrap = [](float x) { return x - std::floor(x); };
  const int px = std::clamp(static_cast<int>(wrap(uv.x) * tex.width), 0, tex.width - 1);
  const int py = std::clamp(static_cast<int>(wrap(uv.y) * tex.height), 0, tex.height - 1);
  const size_t idx = (static_cast<size_t>(py) * tex.width + px) * 4;
  const glm::vec3 raw(tex.pixelsRGBA[idx] / 255.0f, tex.pixelsRGBA[idx + 1] / 255.0f,
                      tex.pixelsRGBA[idx + 2] / 255.0f);
  if (!tex.srgb)
    return raw;
  return glm::vec3(srgbToLinear(raw.r), srgbToLinear(raw.g), srgbToLinear(raw.b));
}

glm::vec3 triangleColor(const MeshAsset &mesh, size_t triangle) {
  const MaterialAsset &mat = mesh.materialForTriangle(triangle);
  const size_t base = triangle * 3;
  if (mat.baseColorTexture >= 0 && mat.baseColorTexture < static_cast<int>(mesh.textures.size()) &&
      base + 2 < mesh.uvs.size()) {
    const TextureAsset &tex = mesh.textures[mat.baseColorTexture];
    if (tex.width > 0 && tex.height > 0) {
      const glm::vec2 uv = (mesh.uvs[base] + mesh.uvs[base + 1] + mesh.uvs[base + 2]) / 3.0f;
      return sampleColorNearest(tex, uv) * mat.baseColorFactor;
    }
  }
  return mat.baseColorFactor;
}

struct CellHash {
  size_t operator()(const glm::ivec3 &c) const {
    return (static_cast<size_t>(c.x) * 73856093u) ^ (static_cast<size_t>(c.y) * 19349663u) ^
           (static_cast<size_t>(c.z) * 83492791u);
  }
};
struct CellEq {
  bool operator()(const glm::ivec3 &a, const glm::ivec3 &b) const { return a == b; }
};

} // namespace

VoxelGrid voxelizeMesh(const MeshAsset &mesh, int resolution) {
  VoxelGrid grid;
  const glm::vec3 extent = mesh.boundsMax - mesh.boundsMin;
  const float longestAxis = std::max({extent.x, extent.y, extent.z, 1e-6f});
  grid.voxelSize = longestAxis / static_cast<float>(std::max(resolution, 1));
  grid.origin = mesh.boundsMin;

  std::unordered_map<glm::ivec3, size_t, CellHash, CellEq> cellIndex;
  std::vector<glm::vec3> colorSum;
  std::vector<int> colorCount;

  const size_t triangleCount = mesh.positions.size() / 3;
  for (size_t t = 0; t < triangleCount; ++t) {
    const size_t base = t * 3;
    const glm::vec3 &p0 = mesh.positions[base];
    const glm::vec3 &p1 = mesh.positions[base + 1];
    const glm::vec3 &p2 = mesh.positions[base + 2];

    const glm::vec3 triMin = glm::min(glm::min(p0, p1), p2);
    const glm::vec3 triMax = glm::max(glm::max(p0, p1), p2);
    const glm::ivec3 cellMin =
        glm::clamp(glm::ivec3(glm::floor((triMin - grid.origin) / grid.voxelSize)), glm::ivec3(0),
                   glm::ivec3(resolution - 1));
    const glm::ivec3 cellMax =
        glm::clamp(glm::ivec3(glm::floor((triMax - grid.origin) / grid.voxelSize)), glm::ivec3(0),
                   glm::ivec3(resolution - 1));
    const glm::vec3 color = triangleColor(mesh, t);

    for (int z = cellMin.z; z <= cellMax.z; ++z) {
      for (int y = cellMin.y; y <= cellMax.y; ++y) {
        for (int x = cellMin.x; x <= cellMax.x; ++x) {
          const glm::ivec3 cell(x, y, z);
          const glm::vec3 center = grid.origin + (glm::vec3(cell) + 0.5f) * grid.voxelSize;
          const glm::vec3 half(grid.voxelSize * 0.5f);
          if (!triBoxOverlap(center, half, p0, p1, p2))
            continue;
          auto [it, inserted] = cellIndex.try_emplace(cell, grid.cells.size());
          if (inserted) {
            grid.cells.push_back(cell);
            colorSum.push_back(color);
            colorCount.push_back(1);
          } else {
            colorSum[it->second] += color;
            colorCount[it->second] += 1;
          }
        }
      }
    }
  }

  grid.colors.resize(grid.cells.size());
  for (size_t i = 0; i < grid.cells.size(); ++i)
    grid.colors[i] = colorSum[i] / static_cast<float>(colorCount[i]);

  return grid;
}

} // namespace italy
