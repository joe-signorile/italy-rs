#pragma once

// This is the RHI seam for phase 2: everything above this class (app/,
// core/) talks to italy::OptixRenderer only — nothing outside src/render/
// includes optix.h or cuda_runtime.h directly. A single concrete class
// rather than the Device/Buffer/AccelStructure/CommandList interface split
// sketched in the design doc: there's one backend and one call site today,
// so those interfaces have nothing to abstract yet.
//
// claudia: single-backend seam — one concrete class, not a generic
// multi-backend interface set. Upgrade to the fuller Device/Buffer/
// AccelStructure interface split (and a real src/rhi/) if a second backend
// starts. There is nothing to abstract with one backend and one call site;
// see the seam section of CLAUDE.md.

#include "convert/sdf_grid.h"
#include "convert/voxel_grid.h"
#include "core/orbit_camera.h"
#include "io/mesh_asset.h"
#include "render/environment.h"

namespace italy {

// Phase 8: which view transform converts the accumulated (and optionally
// denoised) linear HDR radiance into the displayed 8-bit sRGB image. Plain
// C++ enum, zero CUDA dependency — keeps the RHI seam intact (see the class
// comment below). Numerically mirrored, not shared as a type, by the
// device-side dispatcher in pathtracer.cu (matches how MaterialType etc.
// already stay device-side-only) — see pathtracer_params.h's
// Params::tonemapOperator comment.
enum class TonemapOperator : int {
  AgX = 0,      // default: Blender's modern filmic replacement, graceful highlight rolloff
  Reinhard = 1, // alternate/debugging aid
  Aces = 2,     // alternate/debugging aid (Narkowicz 2015 fit)
  Hable = 3,    // alternate/debugging aid (Uncharted2 filmic curve)
  Clamp = 4,    // alternate/debugging aid: the naive clamp(0,1) every render before this phase used
};

// Everything that can change between one subframe and the next without
// invalidating the accumulated image. A struct rather than a positional
// argument list: these are all independent display/quality knobs, they keep
// arriving one per phase, and at six-plus parameters a call site stops being
// readable. Defaults here are the defaults the UI starts at.
struct RenderSettings {
  unsigned int samplesPerLaunch = 1;
  float exposure = 1.0f; // linear, applied at display time only
  bool denoise = false;
  TonemapOperator tonemap = TonemapOperator::AgX;
  // realism: ceiling on a single sample's radiance, suppressing fireflies at
  // the cost of a little energy in the extreme highlights. <= 0 disables.
  float fireflyClamp = 0.0f;

  // Thin-lens depth of field. aperture is the lens radius in world units, so
  // its useful range scales with the scene — the UI derives its slider bound
  // from the scene radius. 0 is a pinhole. focusDistance <= 0 means "focus on
  // the camera's orbit target", which is what you want almost always.
  float aperture = 0.0f;
  float focusDistance = 0.0f;

  // Environment rotation about +Y, radians.
  float envRotation = 0.0f;
};

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
  // A neutral diffuse plane under the loaded object. Without it a mesh floats
  // in a void: no contact shadow and no bounce light, which is most of what
  // makes a render read as an object somewhere rather than a cutout. Ignored
  // by the fixed test scene, which has its own ground, and by any scene with
  // an environment map — an HDRI already bakes in its own ground/horizon,
  // and a flat grey plane under it reads as a visibly synthetic card rather
  // than as ground (measured: tried resizing and darkening it first, neither
  // fixed the mismatch).
  bool groundPlane = true;
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
  // first if the camera or scene changed since the last call.
  //
  // Every field of RenderSettings except fireflyClamp is display-time only —
  // exposure, tonemap and denoise all read the stored HDR accumulator without
  // modifying it, so they are free to change on any frame with no reset.
  // fireflyClamp, aperture, focusDistance and envRotation are the exceptions:
  // they change what gets *written* into the accumulator, so changing any of
  // them needs a resetAccumulation() to take full effect.
  void render(const OrbitCamera &camera, const RenderSettings &settings = {});

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
