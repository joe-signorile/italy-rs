#pragma once

#include <cstddef>
#include <vector>

#include <glm/glm.hpp>

namespace italy {

// One imported Gaussian-splat scene (roadmap phase 3: import pre-generated
// 3DGS `.ply` splats, not fit them from a mesh — see gsplat_ply_loader.h).
// Parallel-array SoA, same shape as MeshAsset, but splats carry their own
// color/opacity instead of indexing a material table.
//
// Values here are already activated (post-exp scale, post-sigmoid opacity,
// SH-DC-converted-to-RGB color) — the loader applies the 3DGS convention's
// activation functions once at import time so nothing downstream (CPU AABB
// math, GPU upload) has to know about the raw fitted parameterization.
//
// realism: only the SH DC term (flat, view-independent color) is kept —
// f_rest_* (higher-order SH bands, i.e. view-dependent color) is dropped on
// import. View-dependent splat color is a real visual cue for shiny/
// reflective training data but at typical splat density in this renderer's
// use case (a single imported object, not a full room-scale capture) flat
// color reads as "close enough" and avoids a per-hit SH evaluation kernel.
struct GsplatAsset {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> scales;     // world-space std-dev per local axis (post-exp)
  std::vector<glm::vec4> rotations;  // quaternion (x,y,z,w), local-to-world
  std::vector<float> opacity;        // post-sigmoid, [0,1]
  std::vector<glm::vec3> colorDC;    // flat RGB, linear

  glm::vec3 boundsMin{0.0f};
  glm::vec3 boundsMax{0.0f};

  size_t count() const { return positions.size(); }
  glm::vec3 boundsCenter() const { return (boundsMin + boundsMax) * 0.5f; }
  float boundsRadius() const { return glm::length(boundsMax - boundsMin) * 0.5f; }
};

} // namespace italy
