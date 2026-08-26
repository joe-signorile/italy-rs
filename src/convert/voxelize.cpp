#include "convert/voxelize.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace italy {
namespace {

// Classic Akenine-Möller triangle/AABB overlap test (separating axis theorem
// over the box's 3 face normals, the triangle's normal, and the 9 cross
// products of box-edge x triangle-edge). Reference:
// "Fast 3D Triangle-Box Overlap Testing," Akenine-Möller 2001. Chosen over a
// cheaper bbox-only test because a bbox-only test visibly over-thickens thin
// or diagonal surfaces — the exact test costs little extra code.
bool planeBoxOverlap(const glm::vec3 &normal, const glm::vec3 &vert, const glm::vec3 &maxbox) {
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

// The six axis tests below transliterate Akenine-Möller's AXISTEST_{X01,X2,
// Y02,Y1,Z12,Z0} macros directly (including their sign conventions, which
// differ between the X/Z and Y families — that's not a typo, it falls out of
// the cross-product expansion). Kept separate rather than one falsely-generic
// helper: collapsing them into a single parameterized function risks
// transcribing the sign wrong for exactly the case that's hardest to notice
// in a quick visual check (a slightly-too-thick or slightly-too-thin result).
bool axisTestX01(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v2,
                  const glm::vec3 &boxHalf) {
  const float p0 = a * v0.y - b * v0.z, p2 = a * v2.y - b * v2.z;
  const float mn = std::min(p0, p2), mx = std::max(p0, p2);
  const float rad = fa * boxHalf.y + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
bool axisTestX2(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v1,
                 const glm::vec3 &boxHalf) {
  const float p0 = a * v0.y - b * v0.z, p1 = a * v1.y - b * v1.z;
  const float mn = std::min(p0, p1), mx = std::max(p0, p1);
  const float rad = fa * boxHalf.y + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
bool axisTestY02(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v2,
                  const glm::vec3 &boxHalf) {
  const float p0 = -a * v0.x + b * v0.z, p2 = -a * v2.x + b * v2.z;
  const float mn = std::min(p0, p2), mx = std::max(p0, p2);
  const float rad = fa * boxHalf.x + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
bool axisTestY1(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v1,
                 const glm::vec3 &boxHalf) {
  const float p0 = -a * v0.x + b * v0.z, p1 = -a * v1.x + b * v1.z;
  const float mn = std::min(p0, p1), mx = std::max(p0, p1);
  const float rad = fa * boxHalf.x + fb * boxHalf.z;
  return !(mn > rad || mx < -rad);
}
bool axisTestZ12(float a, float b, float fa, float fb, const glm::vec3 &v1, const glm::vec3 &v2,
                  const glm::vec3 &boxHalf) {
  const float p1 = a * v1.x - b * v1.y, p2 = a * v2.x - b * v2.y;
  const float mn = std::min(p1, p2), mx = std::max(p1, p2);
  const float rad = fa * boxHalf.x + fb * boxHalf.y;
  return !(mn > rad || mx < -rad);
}
bool axisTestZ0(float a, float b, float fa, float fb, const glm::vec3 &v0, const glm::vec3 &v1,
                 const glm::vec3 &boxHalf) {
  const float p0 = a * v0.x - b * v0.y, p1 = a * v1.x - b * v1.y;
  const float mn = std::min(p0, p1), mx = std::max(p0, p1);
  const float rad = fa * boxHalf.x + fb * boxHalf.y;
  return !(mn > rad || mx < -rad);
}

bool triBoxOverlap(const glm::vec3 &boxCenter, const glm::vec3 &boxHalf, const glm::vec3 &t0, const glm::vec3 &t1,
                    const glm::vec3 &t2) {
  const glm::vec3 v0 = t0 - boxCenter, v1 = t1 - boxCenter, v2 = t2 - boxCenter;
  const glm::vec3 e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;

  float fex = std::fabs(e0.x), fey = std::fabs(e0.y), fez = std::fabs(e0.z);
  if (!axisTestX01(e0.z, e0.y, fez, fey, v0, v2, boxHalf))
    return false;
  if (!axisTestY02(e0.z, e0.x, fez, fex, v0, v2, boxHalf))
    return false;
  if (!axisTestZ12(e0.y, e0.x, fey, fex, v1, v2, boxHalf))
    return false;

  fex = std::fabs(e1.x);
  fey = std::fabs(e1.y);
  fez = std::fabs(e1.z);
  if (!axisTestX01(e1.z, e1.y, fez, fey, v0, v2, boxHalf))
    return false;
  if (!axisTestY02(e1.z, e1.x, fez, fex, v0, v2, boxHalf))
    return false;
  if (!axisTestZ0(e1.y, e1.x, fey, fex, v0, v1, boxHalf))
    return false;

  fex = std::fabs(e2.x);
  fey = std::fabs(e2.y);
  fez = std::fabs(e2.z);
  if (!axisTestX2(e2.z, e2.y, fez, fey, v0, v1, boxHalf))
    return false;
  if (!axisTestY1(e2.z, e2.x, fez, fex, v0, v1, boxHalf))
    return false;
  if (!axisTestZ12(e2.y, e2.x, fey, fex, v1, v2, boxHalf))
    return false;

  // Box-face-normal axis tests (standard AABB/AABB overlap on each axis).
  for (int axis = 0; axis < 3; ++axis) {
    float minV = std::min({v0[axis], v1[axis], v2[axis]});
    float maxV = std::max({v0[axis], v1[axis], v2[axis]});
    if (minV > boxHalf[axis] || maxV < -boxHalf[axis])
      return false;
  }

  // Triangle-plane axis test.
  const glm::vec3 normal = glm::cross(e0, e1);
  return planeBoxOverlap(normal, v0, boxHalf);
}

glm::vec3 sampleColorNearest(const TextureAsset &tex, glm::vec2 uv) {
  auto wrap = [](float x) { return x - std::floor(x); };
  const int px = std::clamp(static_cast<int>(wrap(uv.x) * tex.width), 0, tex.width - 1);
  const int py = std::clamp(static_cast<int>((1.0f - wrap(uv.y)) * tex.height), 0, tex.height - 1);
  const size_t idx = (static_cast<size_t>(py) * tex.width + px) * 4;
  return glm::vec3(tex.pixelsRGBA[idx] / 255.0f, tex.pixelsRGBA[idx + 1] / 255.0f, tex.pixelsRGBA[idx + 2] / 255.0f);
}

glm::vec3 triangleColor(const MeshAsset &mesh, size_t triangleBase) {
  if (mesh.hasBaseColorTexture && !mesh.uvs.empty()) {
    const glm::vec2 uv =
        (mesh.uvs[triangleBase] + mesh.uvs[triangleBase + 1] + mesh.uvs[triangleBase + 2]) / 3.0f;
    return sampleColorNearest(mesh.baseColorTexture, uv) * mesh.baseColorFactor;
  }
  return mesh.baseColorFactor;
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
    const glm::ivec3 cellMin = glm::ivec3(glm::floor((triMin - grid.origin) / grid.voxelSize));
    const glm::ivec3 cellMax = glm::ivec3(glm::floor((triMax - grid.origin) / grid.voxelSize));
    const glm::vec3 color = triangleColor(mesh, base);

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
