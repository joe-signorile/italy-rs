// Standalone check for the primitive-fit import gate (io/mesh_primitive_classify.cpp): no framework, just asserts.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <glm/glm.hpp>

#include "io/mesh_primitive_classify.h"

using italy::classifyPrimitive;
using italy::MeshAsset;
using italy::PrimitiveKind;

namespace {

constexpr float kPi = 3.14159265358979323846f;

void addTri(MeshAsset &m, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
  for (glm::vec3 v : {a, b, c}) {
    m.positions.push_back(v);
    m.normals.push_back(glm::vec3(0, 1, 0));
    m.uvs.push_back(glm::vec2(0));
  }
}

void addQuad(MeshAsset &m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
  addTri(m, a, b, c);
  addTri(m, a, c, d);
}

MeshAsset finalize(MeshAsset m) {
  glm::vec3 lo(1e30f), hi(-1e30f);
  for (const glm::vec3 &p : m.positions) {
    lo = glm::min(lo, p);
    hi = glm::max(hi, p);
  }
  m.boundsMin = lo;
  m.boundsMax = hi;
  m.materials.push_back(italy::MaterialAsset{});
  return m;
}

float sp(float x, float e) { return std::copysign(std::pow(std::fabs(x), e), x); }

MeshAsset makeSuperellipsoid(glm::vec3 halfExtent, float n, int res) {
  MeshAsset m;
  const float e = 2.0f / n;
  auto surf = [&](float u, float v) {
    return glm::vec3(halfExtent.x * sp(std::cos(u), e) * sp(std::cos(v), e),
                     halfExtent.y * sp(std::cos(u), e) * sp(std::sin(v), e), halfExtent.z * sp(std::sin(u), e));
  };
  for (int i = 0; i < res; ++i) {
    const float u0 = -kPi / 2.0f + kPi * static_cast<float>(i) / res;
    const float u1 = -kPi / 2.0f + kPi * static_cast<float>(i + 1) / res;
    for (int j = 0; j < 2 * res; ++j) {
      const float v0 = -kPi + 2.0f * kPi * static_cast<float>(j) / (2 * res);
      const float v1 = -kPi + 2.0f * kPi * static_cast<float>(j + 1) / (2 * res);
      addQuad(m, surf(u0, v0), surf(u0, v1), surf(u1, v1), surf(u1, v0));
    }
  }
  return finalize(std::move(m));
}

MeshAsset makeSphere(float radius, int res) {
  return makeSuperellipsoid(glm::vec3(radius), 2.0f, res);
}

MeshAsset makeBox(glm::vec3 half, int sub) {
  MeshAsset m;
  const glm::vec3 axes[3] = {glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)};
  for (int a = 0; a < 3; ++a) {
    const glm::vec3 n = axes[a];
    const glm::vec3 t = axes[(a + 1) % 3];
    const glm::vec3 b = axes[(a + 2) % 3];
    for (int s = -1; s <= 1; s += 2) {
      const glm::vec3 face = n * (static_cast<float>(s) * glm::dot(half, glm::abs(n)));
      for (int i = 0; i < sub; ++i) {
        for (int k = 0; k < sub; ++k) {
          const float ti0 = -1.0f + 2.0f * i / sub, ti1 = -1.0f + 2.0f * (i + 1) / sub;
          const float tk0 = -1.0f + 2.0f * k / sub, tk1 = -1.0f + 2.0f * (k + 1) / sub;
          const float te = glm::dot(half, glm::abs(t));
          const float be = glm::dot(half, glm::abs(b));
          addQuad(m, face + t * (ti0 * te) + b * (tk0 * be), face + t * (ti1 * te) + b * (tk0 * be),
                  face + t * (ti1 * te) + b * (tk1 * be), face + t * (ti0 * te) + b * (tk1 * be));
        }
      }
    }
  }
  return finalize(std::move(m));
}

MeshAsset makeCylinder(float radius, float halfHeight, int seg, int rings) {
  MeshAsset m;
  auto side = [&](float ang, float y) {
    return glm::vec3(radius * std::cos(ang), y, radius * std::sin(ang));
  };
  for (int j = 0; j < seg; ++j) {
    const float a0 = 2.0f * kPi * j / seg, a1 = 2.0f * kPi * (j + 1) / seg;
    for (int i = 0; i < rings; ++i) {
      const float y0 = -halfHeight + 2.0f * halfHeight * i / rings;
      const float y1 = -halfHeight + 2.0f * halfHeight * (i + 1) / rings;
      addQuad(m, side(a0, y0), side(a1, y0), side(a1, y1), side(a0, y1));
    }
    for (int r = 0; r < 4; ++r) {
      const float r0 = radius * r / 4.0f, r1 = radius * (r + 1) / 4.0f;
      for (float yc : {-halfHeight, halfHeight}) {
        addQuad(m, glm::vec3(r0 * std::cos(a0), yc, r0 * std::sin(a0)),
                glm::vec3(r1 * std::cos(a0), yc, r1 * std::sin(a0)),
                glm::vec3(r1 * std::cos(a1), yc, r1 * std::sin(a1)),
                glm::vec3(r0 * std::cos(a1), yc, r0 * std::sin(a1)));
      }
    }
  }
  return finalize(std::move(m));
}

MeshAsset makeOrganic(int res) {
  MeshAsset m;
  auto surf = [&](float u, float v) {
    const float bump = 1.0f + 0.28f * std::sin(3.0f * v) * std::cos(2.0f * u) + 0.15f * std::sin(5.0f * u);
    return glm::vec3(std::cos(u) * std::cos(v), std::cos(u) * std::sin(v), std::sin(u)) * bump;
  };
  for (int i = 0; i < res; ++i) {
    const float u0 = -kPi / 2 + kPi * i / res, u1 = -kPi / 2 + kPi * (i + 1) / res;
    for (int j = 0; j < 2 * res; ++j) {
      const float v0 = -kPi + kPi * j / res, v1 = -kPi + kPi * (j + 1) / res;
      addQuad(m, surf(u0, v0), surf(u0, v1), surf(u1, v1), surf(u1, v0));
    }
  }
  return finalize(std::move(m));
}

PrimitiveKind classify(const MeshAsset &m, const char *label) {
  std::string detail;
  const PrimitiveKind k = classifyPrimitive(m, detail);
  std::printf("  %-24s -> %d  [%s]\n", label, static_cast<int>(k), detail.c_str());
  std::fflush(stdout);
  return k;
}

} // namespace

int main() {
  assert(classify(makeSphere(1.3f, 8), "sphere lo (8)") == PrimitiveKind::Sphere);
  assert(classify(makeSphere(1.3f, 40), "sphere hi (40)") == PrimitiveKind::Sphere);

  assert(classify(makeBox(glm::vec3(1.0f), 6), "cube") == PrimitiveKind::Box);
  assert(classify(makeBox(glm::vec3(1.0f), 1), "cube (8 verts)") == PrimitiveKind::Box);
  assert(classify(makeBox(glm::vec3(1.6f, 0.7f, 1.1f), 1), "box 8 verts") == PrimitiveKind::Box);
  assert(classify(makeBox(glm::vec3(1.6f, 0.7f, 1.1f), 6), "box 1.6x0.7x1.1") == PrimitiveKind::Box);

  assert(classify(makeSuperellipsoid(glm::vec3(1.0f), 16.0f, 24), "beveled cube (n=16)") == PrimitiveKind::Box);

  assert(classify(makeCylinder(0.5f, 2.2f, 48, 12), "cylinder tall") == PrimitiveKind::Cylinder);
  assert(classify(makeCylinder(1.6f, 0.35f, 64, 4), "cylinder squat") == PrimitiveKind::Cylinder);

  assert(classify(makeSuperellipsoid(glm::vec3(1.0f), 2.6f, 24), "ambiguous (n=2.6)") == PrimitiveKind::Sphere);

  assert(classify(makeOrganic(24), "organic blob") == PrimitiveKind::None);

  {
    MeshAsset tiny;
    tiny.positions = {glm::vec3(0), glm::vec3(0, 0, 1e-7f), glm::vec3(0, 1e-7f, 0)};
    tiny.normals.resize(3, glm::vec3(0, 1, 0));
    tiny.uvs.resize(3, glm::vec2(0));
    tiny.materials.push_back(italy::MaterialAsset{});
    assert(classify(tiny, "degenerate") == PrimitiveKind::None);
  }

  std::printf("mesh_primitive_classify_test: OK\n");
  return 0;
}
