#include "io/mesh_validate.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "io/mesh_weld.h"

namespace italy {
namespace {

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

  const std::vector<uint32_t> welded = weldPositions(mesh.positions, mesh.boundsMin, mesh.boundsMax);

  std::unordered_map<uint64_t, int> edgeCount;
  edgeCount.reserve(triCount * 3);
  for (size_t t = 0; t < triCount; ++t) {
    const uint32_t a = welded[t * 3 + 0];
    const uint32_t b = welded[t * 3 + 1];
    const uint32_t c = welded[t * 3 + 2];
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
