#pragma once

#include <functional>
#include <vector>

#include <glm/glm.hpp>

#include "convert/sdf_grid.h"

namespace italy {

struct SampledSdf {
  float distance = 0.0f;
  int branch = 0;
};

struct SdfSurfaceMaterial {
  glm::vec3 baseColor{1.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
};

using SdfDistanceFn = std::function<SampledSdf(const glm::vec3 &p)>;
using SdfMaterialFn = std::function<SdfSurfaceMaterial(int branch, const glm::vec3 &p)>;
using SdfPaletteFn = std::function<SdfPaletteMaterial(int branch)>;

SdfGrid bakeAnalyticSdf(const glm::vec3 &boundsMin, const glm::vec3 &boundsMax, int resolution,
                        const SdfDistanceFn &distance, const SdfMaterialFn &material);

SdfGrid bakeAnalyticSdfPalette(const glm::vec3 &boundsMin, const glm::vec3 &boundsMax, int resolution,
                                const SdfDistanceFn &distance, const SdfPaletteFn &palette, int maxBranches);

SdfGrid makeCupSdf(int resolution, float originY, glm::vec2 centerXZ = glm::vec2(0.0f));

SdfGrid makeGlassCupSdf(int resolution, float originY, glm::vec2 centerXZ = glm::vec2(0.0f));

std::vector<SdfGrid> makeMaterialProbeSdfs(int resolution);

} // namespace italy
