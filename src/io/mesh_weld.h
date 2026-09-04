#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

std::vector<uint32_t> weldPositions(const std::vector<glm::vec3> &positions, const glm::vec3 &boundsMin,
                                    const glm::vec3 &boundsMax,
                                    std::vector<glm::vec3> *uniquePositions = nullptr);

size_t weldedIndexCount(const std::vector<uint32_t> &welded);

} // namespace italy
