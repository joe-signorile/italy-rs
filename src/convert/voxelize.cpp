#include "convert/voxelize.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "convert/tri_box_overlap.h"

namespace italy {
namespace {

// glTF baseColorTexture is sRGB-encoded (baseColorFactor is linear — only
// the texture needs decoding). The OptiX texture-sampling path gets this via
// CUDA's hardware sRGB conversion (see optix_renderer.cpp); this CPU-side
// bake has no such hardware, so it needs the standard sRGB EOTF explicitly.
// Skipping it would systematically darken/mis-tint every voxel color baked
// from a texture.
float srgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

// No V flip: glTF's texture origin is the upper-left, TextureAsset stores rows
// top-to-bottom (see mesh_asset.h), and the GPU path samples tex2D(u, v)
// unflipped (pathtracer.cu's MATERIAL_TEXTURED_DIFFUSE branch). Flipping here
// made the baked voxel colors a vertical mirror of what the same asset shows
// when rendered as a mesh — the two representations have to agree.
glm::vec3 sampleColorNearest(const TextureAsset &tex, glm::vec2 uv) {
  auto wrap = [](float x) { return x - std::floor(x); };
  const int px = std::clamp(static_cast<int>(wrap(uv.x) * tex.width), 0, tex.width - 1);
  const int py = std::clamp(static_cast<int>(wrap(uv.y) * tex.height), 0, tex.height - 1);
  const size_t idx = (static_cast<size_t>(py) * tex.width + px) * 4;
  const glm::vec3 raw(tex.pixelsRGBA[idx] / 255.0f, tex.pixelsRGBA[idx + 1] / 255.0f,
                      tex.pixelsRGBA[idx + 2] / 255.0f);
  // Only base-colour textures are sRGB-encoded; the flag travels with the
  // texture now, so a linear map that somehow reaches here isn't double-
  // decoded (see TextureAsset::srgb).
  if (!tex.srgb)
    return raw;
  return glm::vec3(srgbToLinear(raw.r), srgbToLinear(raw.g), srgbToLinear(raw.b));
}

glm::vec3 triangleColor(const MeshAsset &mesh, size_t triangle) {
  // Per-triangle material lookup: one soup can now carry a whole scene's
  // worth of materials, so the colour has to be resolved per triangle rather
  // than once for the whole mesh.
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

  std::unordered_map<glm::ivec3, size_t, CellHash, CellEq> cellIndex; // cell -> index into grid.cells/colorSum
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
    // Clamped to [0, resolution-1]: a vertex sitting exactly on the mesh's
    // far bounding-box edge (common for axis-aligned geometry, not rare for
    // real meshes either) computes floor(extent/voxelSize) == resolution
    // exactly — one past the last valid cell index — which without this
    // clamp silently added a full phantom extra layer of cells on whichever
    // axis hit the boundary. Caught by voxelize_test's hollow-cube case,
    // where every vertex sits on a boundary on all three axes.
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
