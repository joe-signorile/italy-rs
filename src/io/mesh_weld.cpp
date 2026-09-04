#include "io/mesh_weld.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace italy {
namespace {

struct WeldKey {
  int64_t x, y, z;
  bool operator==(const WeldKey &o) const { return x == o.x && y == o.y && z == o.z; }
};

struct WeldKeyHash {
  size_t operator()(const WeldKey &k) const {
    size_t h = std::hash<int64_t>()(k.x);
    h ^= std::hash<int64_t>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

} // namespace

std::vector<uint32_t> weldPositions(const std::vector<glm::vec3> &positions, const glm::vec3 &boundsMin,
                                    const glm::vec3 &boundsMax, std::vector<glm::vec3> *uniquePositions) {
  const glm::vec3 extent = boundsMax - boundsMin;
  const float diagonal = glm::length(extent);
  const float cell = std::max(diagonal * 1e-5f, 1e-6f);

  std::unordered_map<WeldKey, uint32_t, WeldKeyHash> weldMap;
  weldMap.reserve(positions.size());
  std::vector<uint32_t> welded(positions.size());
  if (uniquePositions)
    uniquePositions->clear();

  for (size_t i = 0; i < positions.size(); ++i) {
    const glm::vec3 &p = positions[i];
    const WeldKey key{static_cast<int64_t>(std::lround(p.x / cell)), static_cast<int64_t>(std::lround(p.y / cell)),
                      static_cast<int64_t>(std::lround(p.z / cell))};
    const auto [it, inserted] = weldMap.try_emplace(key, static_cast<uint32_t>(weldMap.size()));
    welded[i] = it->second;
    if (inserted && uniquePositions)
      uniquePositions->push_back(p);
  }
  return welded;
}

size_t weldedIndexCount(const std::vector<uint32_t> &welded) {
  if (welded.empty())
    return 0;
  return static_cast<size_t>(*std::max_element(welded.begin(), welded.end())) + 1;
}

} // namespace italy
