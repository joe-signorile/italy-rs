// Standalone sanity check for the triangle-box SAT voxelizer — no framework,
// just asserts, matching this repo's tool-scale rather than pulling in a
// test runner for one file. Run via the `voxelize_test` CMake target.
#include <cassert>
#include <cstdio>

#include "convert/voxelize.h"

using italy::MeshAsset;
using italy::voxelizeMesh;

namespace {

void addQuad(MeshAsset &mesh, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
  // Two triangles, CCW; normals/UVs don't matter for this test.
  for (glm::vec3 v : {a, b, c, a, c, d}) {
    mesh.positions.push_back(v);
    mesh.normals.push_back(glm::vec3(0, 1, 0));
    mesh.uvs.push_back(glm::vec2(0));
  }
}

MeshAsset makeUnitCubeShell() {
  MeshAsset mesh;
  const glm::vec3 lo(0, 0, 0), hi(1, 1, 1);
  // 6 faces of a unit cube.
  addQuad(mesh, {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z}); // -Z
  addQuad(mesh, {lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, lo.y, hi.z}); // +Z
  addQuad(mesh, {lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z}, {lo.x, lo.y, hi.z}); // -X
  addQuad(mesh, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}); // +X
  addQuad(mesh, {lo.x, lo.y, lo.z}, {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, lo.y, lo.z}); // -Y
  addQuad(mesh, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}); // +Y
  mesh.boundsMin = lo;
  mesh.boundsMax = hi;
  return mesh;
}

} // namespace

int main() {
  const int resolution = 8;
  const MeshAsset cube = makeUnitCubeShell();
  const auto grid = voxelizeMesh(cube, resolution);

  // 1) Non-trivial occupancy: not empty, and no more than a hollow shell's
  //    worth of cells (this is a shell voxelizer, not a solid-fill one). A
  //    single-voxel-thick shell has at most 6*resolution^2 occupied cells
  //    (6 faces, each at most resolution x resolution — edges/corners are
  //    shared, so real occupancy is always a bit less, never more). Not
  //    "less than half the grid": at resolution 8 a correct thin shell is
  //    already 296 of 512 cells (58%) simply because the grid is coarse
  //    relative to the shell thickness — "half" only looks like a
  //    reasonable ceiling at resolutions high enough that shell volume is
  //    small relative to total volume, and silently stops meaning what it
  //    was meant to at low ones.
  const size_t maxShellCells = static_cast<size_t>(6 * resolution * resolution);
  assert(!grid.cells.empty());
  assert(grid.cells.size() <= maxShellCells);

  // 2) A cell at the cube's corner should be occupied...
  bool foundCorner = false;
  // ...and the cube's exact center cell should NOT be occupied (interior is
  // hollow — this is the property that actually distinguishes a correct SAT
  // surface test from a sloppy "mark every cell near a vertex" heuristic).
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
