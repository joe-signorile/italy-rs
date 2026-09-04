// Standalone sanity check for the watertightness gate (io/mesh_validate.cpp): no framework, just asserts.
#include <cassert>
#include <cstdio>

#include "io/mesh_validate.h"

using italy::MeshAsset;
using italy::isWatertight;

namespace {

void addQuad(MeshAsset &mesh, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
  for (glm::vec3 v : {a, b, c, a, c, d}) {
    mesh.positions.push_back(v);
    mesh.normals.push_back(glm::vec3(0, 1, 0));
    mesh.uvs.push_back(glm::vec2(0));
  }
}

MeshAsset makeUnitCube() {
  MeshAsset mesh;
  const glm::vec3 lo(0, 0, 0), hi(1, 1, 1);
  addQuad(mesh, {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z});
  addQuad(mesh, {lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, lo.y, hi.z});
  addQuad(mesh, {lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z}, {lo.x, lo.y, hi.z});
  addQuad(mesh, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z});
  addQuad(mesh, {lo.x, lo.y, lo.z}, {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, lo.y, lo.z});
  addQuad(mesh, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z});
  mesh.boundsMin = lo;
  mesh.boundsMax = hi;
  mesh.materials.push_back(italy::MaterialAsset{});
  return mesh;
}

} // namespace

int main() {
  {
    const MeshAsset cube = makeUnitCube();
    std::string err;
    assert(isWatertight(cube, err));
    assert(err.empty());
  }

  {
    MeshAsset holey = makeUnitCube();
    holey.positions.resize(holey.positions.size() - 6);
    holey.normals.resize(holey.normals.size() - 6);
    holey.uvs.resize(holey.uvs.size() - 6);
    std::string err;
    assert(!isWatertight(holey, err));
    assert(!err.empty());
  }

  {
    MeshAsset empty;
    empty.materials.push_back(italy::MaterialAsset{});
    std::string err;
    assert(!isWatertight(empty, err));
  }

  std::printf("mesh_validate_test: OK\n");
  return 0;
}
