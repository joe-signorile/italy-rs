#define GLM_ENABLE_EXPERIMENTAL

#include "io/mesh_primitive_classify.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include <glm/gtx/pca.hpp>

#include "io/mesh_weld.h"

namespace italy {
namespace {

constexpr float kSphereRmsTol = 0.05f;
constexpr float kBoxRmsTol = 0.06f;
constexpr float kCylinderRmsTol = 0.06f;

float percentile(std::vector<float> v, float p) {
  if (v.empty())
    return 0.0f;
  std::sort(v.begin(), v.end());
  const float f = p * static_cast<float>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(f));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return glm::mix(v[lo], v[hi], f - static_cast<float>(lo));
}

float rms(const std::vector<float> &v) {
  if (v.empty())
    return 0.0f;
  double s = 0.0;
  for (float x : v)
    s += static_cast<double>(x) * x;
  return static_cast<float>(std::sqrt(s / static_cast<double>(v.size())));
}

std::string num(float v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.4g", v);
  return buf;
}

} // namespace

PrimitiveKind classifyPrimitive(const MeshAsset &mesh, std::string &detail) {
  std::vector<glm::vec3> pts;
  weldPositions(mesh.positions, mesh.boundsMin, mesh.boundsMax, &pts);

  const float diag = glm::length(mesh.boundsMax - mesh.boundsMin);
  const float floor = std::max(diag * 1e-4f, 1e-5f);

  if (pts.size() < 4 || diag < 1e-5f) {
    detail = "degenerate mesh (" + std::to_string(pts.size()) + " unique verts, diagonal " + num(diag) + ")";
    return PrimitiveKind::None;
  }

  glm::vec3 c(0.0f);
  for (const glm::vec3 &p : pts)
    c += p;
  c /= static_cast<float>(pts.size());

  glm::vec3 axis[3] = {glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)};
  {
    glm::mat3 cov = glm::computeCovarianceMatrix(pts.data(), pts.size(), c);
    glm::vec3 evals(0.0f);
    glm::mat3 evecs(1.0f);
    if (glm::findEigenvaluesSymReal(cov, evals, evecs) == 3) {
      glm::sortEigenvalues(evals, evecs);
      for (int i = 0; i < 3; ++i) {
        const glm::vec3 e = evecs[i];
        if (glm::length(e) > 1e-8f)
          axis[i] = glm::normalize(e);
      }
    }
  }

  {
    std::vector<glm::vec3> local(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      const glm::vec3 d = pts[i] - c;
      local[i] = glm::vec3(glm::dot(d, axis[0]), glm::dot(d, axis[1]), glm::dot(d, axis[2]));
    }
    glm::vec3 half(0.0f);
    for (int a = 0; a < 3; ++a) {
      std::vector<float> abs(pts.size());
      for (size_t i = 0; i < pts.size(); ++i)
        abs[i] = std::fabs(local[i][a]);
      half[a] = percentile(abs, 0.98f);
    }
    const float scale = std::max({half.x, half.y, half.z});
    if (scale > floor) {
      std::vector<float> rel(pts.size());
      for (size_t i = 0; i < pts.size(); ++i) {
        float best = std::numeric_limits<float>::max();
        for (int a = 0; a < 3; ++a)
          best = std::min(best, std::fabs(half[a] - std::fabs(local[i][a])) / std::max(half[a], floor));
        rel[i] = best;
      }
      const float e = rms(rel);
      if (e < kBoxRmsTol) {
        detail = "box (half-extents " + num(half.x) + "," + num(half.y) + "," + num(half.z) + ", RMS rel. residual " +
                 num(e) + ", p98 " + num(percentile(rel, 0.98f)) + ")";
        return PrimitiveKind::Box;
      }
    }
  }

  {
    std::vector<float> radius(pts.size());
    double meanR = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
      radius[i] = glm::length(pts[i] - c);
      meanR += radius[i];
    }
    meanR /= static_cast<double>(pts.size());
    if (meanR > floor) {
      std::vector<float> rel(pts.size());
      for (size_t i = 0; i < pts.size(); ++i)
        rel[i] = std::fabs(radius[i] - static_cast<float>(meanR)) / static_cast<float>(meanR);
      const float e = rms(rel);
      if (e < kSphereRmsTol) {
        detail = "sphere (radius " + num(static_cast<float>(meanR)) + ", RMS rel. deviation " + num(e) + ", p98 " +
                 num(percentile(rel, 0.98f)) + ")";
        return PrimitiveKind::Sphere;
      }
    }
  }

  {
    float bestErr = std::numeric_limits<float>::max();
    int bestAxis = -1;
    glm::vec2 bestRH(0.0f);
    float bestP98 = 0.0f;
    for (int a = 0; a < 3; ++a) {
      const glm::vec3 ax = axis[a];
      std::vector<float> axial(pts.size()), radial(pts.size());
      for (size_t i = 0; i < pts.size(); ++i) {
        const glm::vec3 d = pts[i] - c;
        const float t = glm::dot(d, ax);
        axial[i] = t;
        radial[i] = glm::length(d - t * ax);
      }
      std::vector<float> absAxial(pts.size());
      for (size_t i = 0; i < pts.size(); ++i)
        absAxial[i] = std::fabs(axial[i]);
      const float h = percentile(absAxial, 0.98f);
      const float R = percentile(radial, 0.98f);
      if (h < floor || R < floor)
        continue;
      const float scale = std::max(h, R);
      std::vector<float> rel(pts.size());
      for (size_t i = 0; i < pts.size(); ++i) {
        const float side = std::fabs(radial[i] - R);
        const float cap = std::fabs(std::fabs(axial[i]) - h);
        rel[i] = std::min(side, cap) / scale;
      }
      const float e = rms(rel);
      if (e < bestErr) {
        bestErr = e;
        bestAxis = a;
        bestRH = glm::vec2(R, h);
        bestP98 = percentile(rel, 0.98f);
      }
    }
    if (bestAxis >= 0 && bestErr < kCylinderRmsTol) {
      detail = "cylinder (radius " + num(bestRH.x) + ", half-height " + num(bestRH.y) + ", PCA axis " +
               std::to_string(bestAxis) + ", RMS rel. residual " + num(bestErr) + ", p98 " + num(bestP98) + ")";
      return PrimitiveKind::Cylinder;
    }
  }

  detail = "no primitive fit within tolerance";
  return PrimitiveKind::None;
}

} // namespace italy
