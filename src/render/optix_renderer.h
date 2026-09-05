#pragma once

// The RHI seam: app/ and core/ talk only to italy::OptixRenderer; nothing outside src/render/ includes optix.h or cuda_runtime.h.

#include <vector>

#include "convert/sdf_grid.h"
#include "convert/voxel_grid.h"
#include "core/orbit_camera.h"
#include "io/gsplat_asset.h"
#include "io/mesh_asset.h"
#include "io/nvdb_loader.h"
#include "render/environment.h"

namespace italy {

enum class TonemapOperator : int {
  AgX = 0,
  Reinhard = 1,
  Aces = 2,
  Hable = 3,
  Clamp = 4,
};

struct RenderSettings {
  unsigned int samplesPerLaunch = 1;
  float exposure = 1.0f;
  bool denoise = false;
  bool denoiseTemporal = true;
  TonemapOperator tonemap = TonemapOperator::AgX;
  float fireflyClamp = 10.0f;

  bool lightSubpaths = true;

  unsigned int maxConnectionsPerVertex = 8;

  bool reservoirNEE = false;
  bool reservoirTemporal = true;
  unsigned int extraTestLightCount = 0;

  float aperture = 0.0f;
  float focusDistance = 0.0f;

  float envRotation = 0.0f;

  glm::vec3 backgroundColor{0.05f, 0.06f, 0.08f};
  bool sunEnabled = false;
  glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};
  float sunAngularRadiusDeg = 0.27f;
  glm::vec3 sunRadiance{20.0f, 19.0f, 17.0f};
};

struct SceneSource {
  const MeshAsset *mesh = nullptr;
  const VoxelGrid *voxels = nullptr;
  const SdfGrid *sdf = nullptr;
  const GsplatAsset *splats = nullptr;
  std::vector<const SdfGrid *> extraSdf;
  const NvdbVolume *volume = nullptr;
  const EnvironmentMap *environment = nullptr;
  bool groundPlane = true;
  float groundOffset = 0.0f;
  bool emptyBase = false;
  float emptyBaseFloorY = 0.0f;
};

class OptixRenderer {
public:
  OptixRenderer(int width, int height, const SceneSource &source = {});
  ~OptixRenderer();

  OptixRenderer(const OptixRenderer &) = delete;
  OptixRenderer &operator=(const OptixRenderer &) = delete;

  void render(const OrbitCamera &camera, const RenderSettings &settings = {});

  void resetAccumulation();
  void notifyCameraMoved();

  unsigned int glTextureId() const { return glTexture_; }
  int width() const { return width_; }
  int height() const { return height_; }
  unsigned int subframeIndex() const { return subframeIndex_; }

  bool readAccumulationRgb(std::vector<float> &rgb) const;

  glm::vec3 sceneBoundsCenter() const { return sceneBoundsCenter_; }
  float sceneBoundsRadius() const { return sceneBoundsRadius_; }

private:
  struct Impl;
  Impl *impl_;
  int width_;
  int height_;
  unsigned int glTexture_ = 0;
  unsigned int subframeIndex_ = 0;
  bool cameraMovedPending_ = false;
  glm::vec3 sceneBoundsCenter_{0.0f};
  float sceneBoundsRadius_ = 3.0f;
};

} // namespace italy
