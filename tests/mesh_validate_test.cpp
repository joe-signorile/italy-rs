// Standalone sanity check for the watertightness gate (io/mesh_validate.cpp),
// matching this repo's other tests' tool-scale: no framework, just asserts,
// run via the `mesh_validate_test` CMake target.
#include <cassert>
#include <cstdio>

#include "io/mesh_validate.h"

using italy::MeshAsset;
using italy::isWatertight;

namespace {

// Same closed-cube-shell fixture voxelize_test.cpp builds, reused here for
// its actual intended purpose: a genuinely closed 2-manifold mesh.
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
  addQuad(mesh, {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z}); // -Z
  addQuad(mesh, {lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, lo.y, hi.z}); // +Z
  addQuad(mesh, {lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z}, {lo.x, lo.y, hi.z}); // -X
  addQuad(mesh, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}); // +X
  addQuad(mesh, {lo.x, lo.y, lo.z}, {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, lo.y, lo.z}); // -Y
  addQuad(mesh, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}); // +Y
  mesh.boundsMin = lo;
  mesh.boundsMax = hi;
  mesh.materials.push_back(italy::MaterialAsset{});
  return mesh;
}

} // namespace

int main() {
  // 1) A genuinely closed cube: watertight.
  {
    const MeshAsset cube = makeUnitCube();
    std::string err;
    assert(isWatertight(cube, err));
    assert(err.empty());
  }

  // 2) The same cube with one face's two triangles deleted: has an open
  //    hole, must be rejected with a diagnostic boundary-edge count.
  {
    MeshAsset holey = makeUnitCube();
    holey.positions.resize(holey.positions.size() - 6);
    holey.normals.resize(holey.normals.size() - 6);
    holey.uvs.resize(holey.uvs.size() - 6);
    std::string err;
    assert(!isWatertight(holey, err));
    assert(!err.empty());
  }

  // 3) An empty mesh is rejected too (not just silently "watertight" by
  //    vacuous truth).
  {
    MeshAsset empty;
    empty.materials.push_back(italy::MaterialAsset{});
    std::string err;
    assert(!isWatertight(empty, err));
  }

  std::printf("mesh_validate_test: OK\n");
  return 0;
}
