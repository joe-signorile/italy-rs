#pragma once

// This is the RHI seam for phase 2: everything above this class (app/,
// core/) talks to italy::OptixRenderer only — nothing outside src/render/
// includes optix.h or cuda_runtime.h directly. A single concrete class
// rather than the Device/Buffer/AccelStructure/CommandList interface split
// sketched in the design doc: there's one backend and one call site today,
// so those interfaces have nothing to abstract yet.
//
// monkey-boy: ceiling chosen — one concrete class as the seam, not a generic
// multi-backend interface set. Upgrade to the fuller interface split if/when
// the Metal backend actually starts.

#include "convert/sdf_grid.h"
#include "convert/voxel_grid.h"
#include "core/orbit_camera.h"
#include "io/mesh_asset.h"
#include "render/environment.h"

namespace italy {

// Exactly one of mesh/voxels/sdf should be set; none set means the fixed
// bring-up scene. A tagged struct rather than overloaded constructors, since
// a bare pointer overload set would be ambiguous for the nullptr default.
// `environment` is orthogonal to the other three (a lighting choice, not a
// geometry one) — when set, it replaces the synthetic quad light those
// other three would otherwise get, for whichever geometry is active.
struct SceneSource {
  const MeshAsset *mesh = nullptr;
  const VoxelGrid *voxels = nullptr;
  const SdfGrid *sdf = nullptr;
  const EnvironmentMap *environment = nullptr;
};

class OptixRenderer {
public:
  // Empty source: renders the fixed diffuse/mirror/glass/light bring-up
  // scene from phase 2. source.mesh set: renders that loaded GLB mesh (as
  // MATERIAL_TEXTURED_DIFFUSE triangles). source.voxels set: renders that
  // voxelized mesh (as MATERIAL_VOXEL custom AABB primitives). source.sdf
  // set: sphere-traces the baked distance field (MATERIAL_SDF). Whichever of
  // those three is picked, it's lit either by source.environment (HDRI,
  // importance sampled) if given, or otherwise by a synthetic quad light
  // sized to the bounding box.
  OptixRenderer(int width, int height, const SceneSource &source = {});
  ~OptixRenderer();

  OptixRenderer(const OptixRenderer &) = delete;
  OptixRenderer &operator=(const OptixRenderer &) = delete;

  // Renders one progressive subframe (accumulates onto the previous one) and
  // updates the GL texture returned by glTextureId(). Call resetAccumulation()
  // first if the camera or scene changed since the last call. samplesPerLaunch
  // trades per-frame cost for faster convergence; exposure only affects the
  // displayed tonemap, not the stored HDR accumulator (see pathtracer.cu),
  // so it's free to change every frame without resetting accumulation.
  // denoise runs the OptiX AI denoiser over the accumulated HDR buffer
  // before tonemapping — same free-to-toggle-any-frame property as
  // exposure, since it never touches the stored accumulator either.
  void render(const OrbitCamera &camera, unsigned int samplesPerLaunch = 1, float exposure = 1.0f,
              bool denoise = false);

  void resetAccumulation();

  unsigned int glTextureId() const { return glTexture_; }
  int width() const { return width_; }
  int height() const { return height_; }
  unsigned int subframeIndex() const { return subframeIndex_; }

  glm::vec3 sceneBoundsCenter() const { return sceneBoundsCenter_; }
  float sceneBoundsRadius() const { return sceneBoundsRadius_; }

private:
  struct Impl;
  Impl *impl_;
  int width_;
  int height_;
  unsigned int glTexture_ = 0;
  unsigned int subframeIndex_ = 0;
  glm::vec3 sceneBoundsCenter_{0.0f};
  float sceneBoundsRadius_ = 3.0f;
};

} // namespace italy
