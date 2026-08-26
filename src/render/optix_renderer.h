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

namespace italy {

class OptixRenderer {
public:
  OptixRenderer(int width, int height);
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

private:
  struct Impl;
  Impl *impl_;
  int width_;
  int height_;
  unsigned int glTexture_ = 0;
  unsigned int subframeIndex_ = 0;
};

} // namespace italy
