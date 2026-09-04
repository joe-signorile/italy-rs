// Standalone sanity check for the triangle-box SAT voxelizer: no framework, just asserts.
#include <cassert>
#include <cstdio>

#include "convert/voxelize.h"

using italy::MeshAsset;
using italy::voxelizeMesh;

namespace {

void addQuad(MeshAsset &mesh, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
  for (glm::vec3 v : {a, b, c, a, c, d}) {
    mesh.positions.push_back(v);
    mesh.normals.push_back(glm::vec3(0, 1, 0));
    mesh.uvs.push_back(glm::vec2(0));
  }
}

MeshAsset makeUnitCubeShell() {
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
  const int resolution = 8;
  const MeshAsset cube = makeUnitCubeShell();
  const auto grid = voxelizeMesh(cube, resolution);

  const size_t maxShellCells = static_cast<size_t>(6 * resolution * resolution);
  assert(!grid.cells.empty());
  assert(grid.cells.size() <= maxShellCells);

  bool foundCorner = false;
  bool foundCenter = false;
  for (const glm::ivec3 &c : grid.cells) {
    if (c == glm::ivec3(0, 0, 0))
      foundCorner = true;
    if (c == glm::ivec3(4, 4, 4))
      foundCenter = true;
  }
  assert(foundCorner && "corner cell should be on the shell");
  assert(!foundCenter && "center cell is interior — must not be marked occupied by a surface voxelizer");

  std::printf("voxelize_test: OK\n");
  return 0;
}
