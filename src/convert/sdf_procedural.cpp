#include "convert/sdf_procedural.h"

#include <algorithm>
#include <cmath>

#include "convert/sdf_primitives.h"

namespace italy {

namespace {

SdfGrid bakeSdfGridShape(const glm::vec3 &boundsMin, const glm::vec3 &boundsMax, int resolution) {
  const glm::vec3 extent = boundsMax - boundsMin;
  const float longestAxis = std::max({extent.x, extent.y, extent.z});
  const float voxelSize = longestAxis / static_cast<float>(std::max(resolution, 4));
  const int pad = 2;

  SdfGrid grid;
  grid.voxelSize = voxelSize;
  grid.origin = boundsMin - glm::vec3(static_cast<float>(pad) * voxelSize);
  grid.nx = static_cast<int>(std::ceil(extent.x / voxelSize)) + pad * 2;
  grid.ny = static_cast<int>(std::ceil(extent.y / voxelSize)) + pad * 2;
  grid.nz = static_cast<int>(std::ceil(extent.z / voxelSize)) + pad * 2;
  grid.distances.resize(static_cast<size_t>(grid.nx) * static_cast<size_t>(grid.ny) * static_cast<size_t>(grid.nz));
  return grid;
}

template <typename PerCellFn>
void sampleSdfGrid(SdfGrid &grid, const SdfDistanceFn &distance, PerCellFn &&perCell) {
  for (int z = 0; z < grid.nz; ++z) {
    for (int y = 0; y < grid.ny; ++y) {
      for (int x = 0; x < grid.nx; ++x) {
        const glm::vec3 p = grid.origin + (glm::vec3(static_cast<float>(x), static_cast<float>(y),
                                                     static_cast<float>(z)) +
                                           glm::vec3(0.5f)) *
                                              grid.voxelSize;
        const SampledSdf s = distance(p);
        const size_t i = grid.index(x, y, z);
        grid.distances[i] = s.distance;
        perCell(i, s, p);
      }
    }
  }
}

} // namespace

SdfGrid bakeAnalyticSdf(const glm::vec3 &boundsMin, const glm::vec3 &boundsMax, int resolution,
                        const SdfDistanceFn &distance, const SdfMaterialFn &material) {
  SdfGrid grid = bakeSdfGridShape(boundsMin, boundsMax, resolution);
  const size_t count = grid.distances.size();
  grid.baseColor.resize(count);
  grid.metallic.resize(count);
  grid.roughness.resize(count);

  sampleSdfGrid(grid, distance, [&](size_t i, const SampledSdf &s, const glm::vec3 &p) {
    const SdfSurfaceMaterial m = material(s.branch, p);
    grid.baseColor[i] = m.baseColor;
    grid.metallic[i] = m.metallic;
    grid.roughness[i] = m.roughness;
  });

  return grid;
}

SdfGrid bakeAnalyticSdfPalette(const glm::vec3 &boundsMin, const glm::vec3 &boundsMax, int resolution,
                                const SdfDistanceFn &distance, const SdfPaletteFn &palette, int maxBranches) {
  SdfGrid grid = bakeSdfGridShape(boundsMin, boundsMax, resolution);
  const size_t count = grid.distances.size();
  grid.branch.resize(count);
  grid.palette.resize(static_cast<size_t>(maxBranches));
  for (int b = 0; b < maxBranches; ++b) grid.palette[b] = palette(b);

  sampleSdfGrid(grid, distance, [&](size_t i, const SampledSdf &s, const glm::vec3 &) {
    grid.branch[i] = static_cast<uint8_t>(s.branch);
  });

  return grid;
}

SdfGrid makeCupSdf(int resolution, float originY, glm::vec2 centerXZ) {
  const float outerRadius = 0.55f;
  const float wallThickness = 0.09f;
  const float innerRadius = outerRadius - wallThickness;
  const float innerLift = 0.16f;
  const float rimFraction = 0.62f;

  const glm::vec3 outerCenter(centerXZ.x, originY + outerRadius, centerXZ.y);
  const glm::vec3 innerCenter = outerCenter + glm::vec3(0.0f, innerLift, 0.0f);
  const float rimY = outerCenter.y + outerRadius * rimFraction;

  const glm::vec3 boundsMin(outerCenter.x - outerRadius, outerCenter.y - outerRadius, outerCenter.z - outerRadius);
  const glm::vec3 boundsMax(outerCenter.x + outerRadius, rimY, outerCenter.z + outerRadius);

  const SdfDistanceFn distance = [=](const glm::vec3 &p) -> SampledSdf {
    const float dShell = sdf::opSubtract(sdf::sdSphere(p - outerCenter, outerRadius),
                                         sdf::sdSphere(p - innerCenter, innerRadius));
    const float dCup = sdf::opIntersect(dShell, p.y - rimY);
    return {dCup, 0};
  };
  const SdfMaterialFn material = [](int, const glm::vec3 &) -> SdfSurfaceMaterial {
    return {glm::vec3(0.85f, 0.83f, 0.78f), 0.0f, 0.35f};
  };

  return bakeAnalyticSdf(boundsMin, boundsMax, resolution, distance, material);
}

SdfGrid makeGlassCupSdf(int resolution, float originY, glm::vec2 centerXZ) {
  const float outerRadius = 0.55f;
  const float wallThickness = 0.09f;
  const float innerRadius = outerRadius - wallThickness;
  const float innerLift = 0.16f;
  const float rimFraction = 0.62f;
  const float blendK = 0.012f;

  const glm::vec3 outerCenter(centerXZ.x, originY + outerRadius, centerXZ.y);
  const glm::vec3 innerCenter = outerCenter + glm::vec3(0.0f, innerLift, 0.0f);
  const float rimY = outerCenter.y + outerRadius * rimFraction;
  const float fillY = innerCenter.y - innerRadius + wallThickness * 0.5f + 0.42f * (2.0f * innerRadius);

  const glm::vec3 boundsMin(outerCenter.x - outerRadius, outerCenter.y - outerRadius, outerCenter.z - outerRadius);
  const glm::vec3 boundsMax(outerCenter.x + outerRadius, rimY, outerCenter.z + outerRadius);

  const SdfDistanceFn distance = [=](const glm::vec3 &p) -> SampledSdf {
    const float dShellRaw = sdf::opSubtract(sdf::sdSphere(p - outerCenter, outerRadius),
                                            sdf::sdSphere(p - innerCenter, innerRadius));
    const float dShell = sdf::opIntersect(dShellRaw, p.y - rimY);
    const float dLiquidSphere = sdf::sdSphere(p - innerCenter, innerRadius);
    const float dLiquid = sdf::opSmoothIntersect(dLiquidSphere, p.y - fillY, blendK);
    return dShell <= dLiquid ? SampledSdf{dShell, 0} : SampledSdf{dLiquid, 1};
  };

  const SdfPaletteFn palette = [](int branch) -> SdfPaletteMaterial {
    if (branch == 0) {
      return {SdfMaterialKind::Dielectric, glm::vec3(1.0f), 0.0f, 0.05f, 1.47f, glm::vec3(0.02f, 0.02f, 0.018f)};
    }
    return {SdfMaterialKind::Dielectric, glm::vec3(1.0f), 0.0f, 0.02f, 1.333f, glm::vec3(0.9f, 0.35f, 0.12f)};
  };

  return bakeAnalyticSdfPalette(boundsMin, boundsMax, resolution, distance, palette, 2);
}

std::vector<SdfGrid> makeMaterialProbeSdfs(int resolution) {
  const float radius = 0.4f;

  const auto makeSphereGrid = [&](const glm::vec3 &center, const SdfPaletteMaterial &mat) {
    const glm::vec3 boundsMin = center - glm::vec3(radius);
    const glm::vec3 boundsMax = center + glm::vec3(radius);
    const SdfDistanceFn distance = [=](const glm::vec3 &p) -> SampledSdf {
      return {sdf::sdSphere(p - center, radius), 0};
    };
    const SdfPaletteFn palette = [=](int) -> SdfPaletteMaterial { return mat; };
    return bakeAnalyticSdfPalette(boundsMin, boundsMax, resolution, distance, palette, 1);
  };

  std::vector<SdfGrid> grids;
  grids.reserve(9);

  const float row1X[5] = {-2.2f, -1.1f, 0.0f, 1.1f, 2.2f};
  const float row1Roughness[5] = {0.05f, 0.3f, 0.55f, 0.8f, 1.0f};
  for (int i = 0; i < 5; ++i) {
    SdfPaletteMaterial mat;
    mat.kind = SdfMaterialKind::Opaque;
    mat.baseColor = glm::vec3(0.7f, 0.7f, 0.72f);
    mat.metallic = 0.0f;
    mat.roughness = row1Roughness[i];
    grids.push_back(makeSphereGrid(glm::vec3(row1X[i], radius, -0.6f), mat));
  }

  const float row2X[4] = {-1.65f, -0.55f, 0.55f, 1.65f};
  const float row2Ior[4] = {1.1f, 1.35f, 1.6f, 2.0f};
  for (int i = 0; i < 4; ++i) {
    SdfPaletteMaterial mat;
    mat.kind = SdfMaterialKind::Dielectric;
    mat.baseColor = glm::vec3(1.0f);
    mat.metallic = 0.0f;
    mat.roughness = 0.02f;
    mat.ior = row2Ior[i];
    mat.extinction = glm::vec3(0.05f, 0.05f, 0.05f);
    grids.push_back(makeSphereGrid(glm::vec3(row2X[i], radius, 0.6f), mat));
  }

  return grids;
}

} // namespace italy
