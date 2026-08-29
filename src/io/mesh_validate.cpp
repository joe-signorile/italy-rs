#include "io/mesh_validate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace italy {
namespace {

struct WeldKey {
  int64_t x, y, z;
  bool operator==(const WeldKey &o) const { return x == o.x && y == o.y && z == o.z; }
};

struct WeldKeyHash {
  size_t operator()(const WeldKey &k) const {
    // boost::hash_combine, applied twice — good enough dispersion for a
    // spatial hash of a few hundred thousand vertices, and this project
    // already leans on hand-rolled combinators elsewhere rather than
    // pulling in a hashing library for one function.
    size_t h = std::hash<int64_t>()(k.x);
    h ^= std::hash<int64_t>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

// Undirected edge between two welded vertex indices, packed into one 64-bit
// key (min index in the high 32 bits) so it can live in a plain
// unordered_map<uint64_t, int> without a second hash functor.
uint64_t edgeKey(uint32_t a, uint32_t b) {
  if (a > b)
    std::swap(a, b);
  return (static_cast<uint64_t>(a) << 32) | b;
}

} // namespace

bool isWatertight(const MeshAsset &mesh, std::string &err) {
  const size_t triCount = mesh.triangleCount();
  if (triCount == 0) {
    err = "mesh has no triangles";
    return false;
  }

  // Weld vertices by position so triangles that share a position (but not a
  // buffer index — this is an unindexed triangle soup) count as sharing a
  // vertex for edge-adjacency purposes. Quantize to a grid sized off the
  // mesh's own bounding diagonal, the same "relative, not absolute"
  // tolerance reasoning voxelize.cpp uses for its own degenerate-extent
  // floor (1e-6f).
  const glm::vec3 extent = mesh.boundsMax - mesh.boundsMin;
  const float diagonal = glm::length(extent);
  const float cell = std::max(diagonal * 1e-5f, 1e-6f);

  std::unordered_map<WeldKey, uint32_t, WeldKeyHash> weldMap;
  weldMap.reserve(mesh.positions.size());
  std::vector<uint32_t> welded(mesh.positions.size());
  for (size_t i = 0; i < mesh.positions.size(); ++i) {
    const glm::vec3 &p = mesh.positions[i];
    const WeldKey key{static_cast<int64_t>(std::lround(p.x / cell)), static_cast<int64_t>(std::lround(p.y / cell)),
                       static_cast<int64_t>(std::lround(p.z / cell))};
    const auto [it, inserted] = weldMap.try_emplace(key, static_cast<uint32_t>(weldMap.size()));
    welded[i] = it->second;
  }

  // Every edge of a closed 2-manifold mesh is shared by exactly two
  // triangles; count occurrences and flag anything else.
  std::unordered_map<uint64_t, int> edgeCount;
  edgeCount.reserve(triCount * 3);
  for (size_t t = 0; t < triCount; ++t) {
    const uint32_t a = welded[t * 3 + 0];
    const uint32_t b = welded[t * 3 + 1];
    const uint32_t c = welded[t * 3 + 2];
    // Degenerate (zero-area after welding) triangles don't contribute real
    // edges — same tolerance the loader already extends to degenerate
    // triangles (see gltf_loader.cpp's face-normal fallback).
    if (a == b || b == c || c == a)
      continue;
    ++edgeCount[edgeKey(a, b)];
    ++edgeCount[edgeKey(b, c)];
    ++edgeCount[edgeKey(c, a)];
  }

  int boundary = 0, nonManifold = 0;
  for (const auto &[key, count] : edgeCount) {
    if (count == 1)
      ++boundary;
    else if (count > 2)
      ++nonManifold;
  }

  if (boundary == 0 && nonManifold == 0)
    return true;

  err = "mesh is not watertight (" + std::to_string(boundary) + " boundary edge" + (boundary == 1 ? "" : "s");
  if (nonManifold > 0)
    err += ", " + std::to_string(nonManifold) + " non-manifold edge" + (nonManifold == 1 ? "" : "s");
  err += ") — fill holes and merge duplicate vertices in your DCC tool and re-export";
  return false;
}

} // namespace italy
