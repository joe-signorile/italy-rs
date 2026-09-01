#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace italy {

// Dense (not narrow-band) regular grid of signed distances. The design doc
// says "narrow-band" but narrow-band storage only helps memory for large
// grids and is awkward to sphere-trace through (empty space far from the
// surface needs *some* usable lower-bound distance so marching can safely
// take large steps there) — dense is simpler and unambiguous to render, and
// at MVP resolutions (~64^3) the memory difference doesn't matter. Revisit
// if resolution needs to grow a lot.
struct SdfGrid {
  glm::vec3 origin{0.0f};
  float voxelSize = 1.0f;
  int nx = 0, ny = 0, nz = 0;
  std::vector<float> distances; // nx*ny*nz, negative inside the surface

  // Unified-SDF acceleration, Phase 1 (see /home/joe/.claude/plans/
  // lets-fix-the-bubbly-lecun.md — "Part D"): per-cell material, parallel to
  // `distances` (same index(x,y,z) addressing), replacing the single flat
  // `tintColor` every cell used to share. Populated in sdf_baker.cpp by
  // rasterizing each mesh triangle's material into every cell its bounding
  // box overlaps — the same triBoxOverlap-based algorithm voxelizeMesh()
  // already uses for VoxelGrid::colors (see convert/tri_box_overlap.h),
  // reused rather than reinvented, and run on the CPU as a second pass after
  // the existing GPU distance bake rather than folded into the OptiX bake
  // kernel itself (sdf_bake.cu's ray-parity distance query has no
  // "which triangle is nearest" signal to reuse; adding one would be new,
  // untested device-side machinery for a first phase whose job is proving
  // the *storage and shading* side of per-cell material out, not the bake
  // performance). Cells no triangle overlaps default to (white, 0
  // metallic, 1 roughness) — the pre-Phase-1 flat-tint value would have
  // been indistinguishable from "no data" anyway, since a triangle soup
  // this coarse a grid never reached simply wasn't near the surface.
  std::vector<glm::vec3> baseColor; // nx*ny*nz
  std::vector<float> metallic;      // nx*ny*nz
  std::vector<float> roughness;     // nx*ny*nz

  glm::vec3 boundsMax() const { return origin + glm::vec3(nx, ny, nz) * voxelSize; }
  size_t index(int x, int y, int z) const { return (static_cast<size_t>(z) * ny + y) * nx + x; }
};

} // namespace italy
