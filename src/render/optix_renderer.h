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

#include "core/orbit_camera.h"
#include "io/mesh_asset.h"

namespace italy {

class OptixRenderer {
public:
  // mesh == nullptr: renders the fixed diffuse/mirror/glass/light bring-up
  // scene from phase 2. mesh != nullptr: renders that loaded GLB mesh
  // (as MATERIAL_TEXTURED_DIFFUSE triangles) lit by a synthetic quad light
  // sized to its bounding box — there's no scene-graph/multi-object or HDRI
  // lighting yet, so a loaded asset still needs *something* to be lit by.
  OptixRenderer(int width, int height, const MeshAsset *mesh = nullptr);
  ~OptixRenderer();

  OptixRenderer(const OptixRenderer &) = delete;
  OptixRenderer &operator=(const OptixRenderer &) = delete;

  // Renders one progressive subframe (accumulates onto the previous one) and
  // updates the GL texture returned by glTextureId(). Call resetAccumulation()
  // first if the camera or scene changed since the last call.
  void render(const OrbitCamera &camera);

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
