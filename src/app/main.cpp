#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "app/app_window.h"
#include "convert/sdf_baker.h"
#include "convert/sdf_procedural.h"
#include "convert/voxelize.h"
#include "core/orbit_camera.h"
#include "io/gltf_loader.h"
#include "io/gsplat_ply_loader.h"
#include "io/mesh_primitive_classify.h"
#include "io/mesh_validate.h"
#include "io/nvdb_loader.h"
#include "render/environment.h"
#include "render/optix_renderer.h"
#include "render/procedural_sky.h"

namespace {

struct MouseState {
  double lastX = 0.0;
  double lastY = 0.0;
};

bool cameraChanged(const italy::OrbitCamera &a, const italy::OrbitCamera &b) {
  const glm::vec3 da = a.position() - b.position();
  const glm::vec3 dt = a.target() - b.target();
  return glm::dot(da, da) > 1e-10f || glm::dot(dt, dt) > 1e-10f;
}

bool hasExtension(const char *path, const char *ext) {
  const size_t n = std::strlen(path), m = std::strlen(ext);
  if (n < m)
    return false;
  for (size_t i = 0; i < m; ++i)
    if (std::tolower(static_cast<unsigned char>(path[n - m + i])) != ext[i])
      return false;
  return true;
}

bool writeFrameHdr(const italy::OptixRenderer &renderer, const char *path) {
  const int w = renderer.width();
  const int h = renderer.height();
  std::vector<float> rgb;
  if (!renderer.readAccumulationRgb(rgb)) {
    std::fprintf(stderr, "italy: FAILED to write %dx%d frame to %s (nothing accumulated yet)\n", w, h, path);
    return false;
  }
  std::vector<float> flipped(rgb.size());
  for (int row = 0; row < h; ++row)
    std::memcpy(&flipped[static_cast<size_t>(row) * w * 3], &rgb[static_cast<size_t>(h - 1 - row) * w * 3],
                static_cast<size_t>(w) * 3 * sizeof(float));
  const bool ok = stbi_write_hdr(path, w, h, 3, flipped.data()) != 0;
  std::fprintf(stderr, "italy: %s %dx%d linear frame to %s\n", ok ? "wrote" : "FAILED to write", w, h, path);
  return ok;
}

bool writeFramePng(const italy::OptixRenderer &renderer, const char *path) {
  const int w = renderer.width();
  const int h = renderer.height();
  std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
  glBindTexture(GL_TEXTURE_2D, renderer.glTextureId());
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  std::vector<unsigned char> flipped(pixels.size());
  for (int row = 0; row < h; ++row)
    std::memcpy(&flipped[static_cast<size_t>(row) * w * 4], &pixels[static_cast<size_t>(h - 1 - row) * w * 4],
                static_cast<size_t>(w) * 4);
  const bool ok = stbi_write_png(path, w, h, 4, flipped.data(), w * 4) != 0;
  std::fprintf(stderr, "italy: %s %dx%d frame to %s\n", ok ? "wrote" : "FAILED to write", w, h, path);
  return ok;
}

bool writeFrame(const italy::OptixRenderer &renderer, const char *path) {
  return hasExtension(path, ".hdr") ? writeFrameHdr(renderer, path) : writeFramePng(renderer, path);
}

void dumpFrameIfRequested(const italy::OptixRenderer &renderer, GLFWwindow *window) {
  const char *path = std::getenv("ITALY_DUMP_FRAME");
  if (!path)
    return;
  writeFrame(renderer, path);
  glfwSetWindowShouldClose(window, GLFW_TRUE);
}

constexpr int kMinResolution = 1;

enum class Representation : int { Mesh, Voxel, Sdf, Gsplat };
enum class EnvChoice : int { None, Overcast, Midnight, Noon, Custom, ProceduralSky };

struct AppState {
  italy::MeshAsset meshAsset;
  bool haveMesh = false;
  italy::VoxelGrid voxelGrid;
  italy::SdfGrid sdfGrid;
  italy::SdfGrid cupGrid;
  bool wantCup = false;
  static constexpr int kCupResolution = 96;
  italy::SdfGrid glassCupGrid;
  bool wantGlassCup = false;
  static constexpr int kGlassCupResolution = 320;
  std::vector<italy::SdfGrid> materialProbeGrids;
  bool wantMaterialProbe = false;
  static constexpr int kMaterialProbeResolution = 48;
  italy::GsplatAsset splatAsset;
  bool haveSplats = false;
  char gsplatPathBuf[512] = "";
  italy::EnvironmentMap environment;
  bool haveEnvironment = false;

  bool wantFogVolume = false;
  italy::NvdbVolume fogVolume;
  float fogRadius = 1.1f;
  float fogVoxelSize = 0.05f;
  float fogSigmaT = 2.0f;
  glm::vec3 fogScatterAlbedo{0.95f, 0.95f, 0.95f};
  float fogAsymmetry = 0.0f;
  float fogDensityScale = 1.0f;

  char glbPathBuf[512] = "";
  Representation representation = Representation::Mesh;
  int resolution = 128;
  EnvChoice envChoice = EnvChoice::None;
  bool envChoiceExplicit = false;
  char hdriPathBuf[512] = "";
  bool groundPlane = true;

  float skyTurbidity = 3.0f;
  float skySunElevationDeg = 45.0f;
  float skySunAzimuthDeg = 0.0f;
  static constexpr int kSkyResolution = 1024;

  bool sunEnabled = false;
  float sunAngularRadiusDeg = 2.0f;
  glm::vec3 sunColor{1.0f, 0.95f, 0.85f};
  float sunIntensity = 10000.0f;
  glm::vec3 backgroundColor{0.05f, 0.06f, 0.08f};

  italy::RenderSettings render;

  int renderWidth = 960;
  int renderHeight = 540;

  char exportPathBuf[512] = "render.png";
  int exportSamples = 512;
  bool exportPending = false;

  bool isSampling = true;

  std::string statusLine = "Showing the built-in test scene.";
};

struct ProjectBakeState {
  enum class Step { Load, Classify, Representation, Environment, ConstructRenderer };
  bool active = false;
  Step step = Step::Load;
  bool rawMeshLoaded = false;
  int selected = 0;
  std::vector<std::string> log;
};

void syncSunRenderSettings(AppState &state) {
  const float elevRad = glm::radians(state.skySunElevationDeg);
  const float azRad = glm::radians(state.skySunAzimuthDeg);
  state.render.sunDirection = glm::normalize(glm::vec3(std::cos(elevRad) * std::cos(azRad), std::sin(elevRad),
                                                         std::cos(elevRad) * std::sin(azRad)));
  state.render.sunAngularRadiusDeg = state.sunAngularRadiusDeg;
  state.render.sunRadiance = state.sunColor * state.sunIntensity;
  state.render.sunEnabled = state.sunEnabled;
  state.render.backgroundColor = state.backgroundColor;
}

void logBakeLine(ProjectBakeState &bake, const std::string &message) {
  const std::time_t now = std::time(nullptr);
  char stamp[16];
  std::strftime(stamp, sizeof(stamp), "%H:%M:%S", std::localtime(&now));
  bake.log.push_back(std::string("[") + stamp + "] " + message);
  std::fprintf(stderr, "italy: %s\n", message.c_str());
}

void bakeStepLoad(AppState &state, ProjectBakeState &bake) {
  state.haveMesh = false;
  state.haveSplats = false;
  bake.rawMeshLoaded = false;
  if (state.representation == Representation::Gsplat) {
    if (state.gsplatPathBuf[0] != '\0') {
      std::string err;
      if (!italy::loadGsplatPly(state.gsplatPathBuf, state.splatAsset, err)) {
        state.statusLine = "Failed to load " + std::string(state.gsplatPathBuf) + ": " + err;
      } else {
        state.haveSplats = true;
        state.statusLine =
            "Loaded " + std::string(state.gsplatPathBuf) + " (" + std::to_string(state.splatAsset.count()) + " splats)";
      }
    } else {
      state.statusLine = "No gsplat .ply path set.";
    }
  } else if (state.glbPathBuf[0] != '\0') {
    std::string err;
    if (!italy::loadGlb(state.glbPathBuf, state.meshAsset, err)) {
      state.statusLine = "Failed to load " + std::string(state.glbPathBuf) + ": " + err;
    } else {
      bake.rawMeshLoaded = true;
      state.statusLine = "Loaded raw " + std::string(state.glbPathBuf) + ", classifying...";
    }
  } else if (!state.wantMaterialProbe && !state.wantGlassCup && !state.wantCup) {
    state.statusLine = "Showing the built-in test scene.";
  }
  logBakeLine(bake, "Load: " + state.statusLine);
}

void bakeStepClassify(AppState &state, ProjectBakeState &bake) {
  if (bake.rawMeshLoaded) {
    std::string primDetail;
    std::string err;
    if (italy::classifyPrimitive(state.meshAsset, primDetail) == italy::PrimitiveKind::None) {
      state.statusLine = "Rejected " + std::string(state.glbPathBuf) + ": not a primitive shape (" + primDetail +
                         "). GLB import is limited to box/sphere/cylinder prims — use procedural SDF / gsplat "
                         "for other content.";
    } else if (!italy::isWatertight(state.meshAsset, err)) {
      state.statusLine = "Rejected " + std::string(state.glbPathBuf) + ": " + err;
    } else {
      state.haveMesh = true;
      state.statusLine = "Loaded " + std::string(state.glbPathBuf) + " (" +
                          std::to_string(state.meshAsset.triangleCount()) + " tris, " +
                          std::to_string(state.meshAsset.materials.size()) + " materials, " +
                          std::to_string(state.meshAsset.textures.size()) + " textures) — " + primDetail;
    }
    logBakeLine(bake, "Classify/validate: " + state.statusLine);
  } else {
    logBakeLine(bake, "Classify/validate: nothing to classify.");
  }
}

void bakeStepRepresentation(AppState &state, ProjectBakeState &bake) {
  if (state.haveMesh) {
    if (state.representation == Representation::Voxel) {
      state.voxelGrid = italy::voxelizeMesh(state.meshAsset, state.resolution);
      state.statusLine += " -> voxelized (" + std::to_string(state.voxelGrid.cells.size()) + " cells)";
    } else if (state.representation == Representation::Sdf) {
      state.sdfGrid = italy::bakeSdf(state.meshAsset, state.resolution);
      state.statusLine += " -> SDF (" + std::to_string(state.sdfGrid.nx) + "x" + std::to_string(state.sdfGrid.ny) +
                           "x" + std::to_string(state.sdfGrid.nz) + ")";
    }
  } else if (state.wantGlassCup) {
    state.glassCupGrid = italy::makeGlassCupSdf(AppState::kGlassCupResolution, -1.0f, glm::vec2(0.0f));
    state.statusLine += " -> baked glass cup SDF (" + std::to_string(state.glassCupGrid.nx) + "x" +
                         std::to_string(state.glassCupGrid.ny) + "x" + std::to_string(state.glassCupGrid.nz) + ")";
  } else if (state.wantMaterialProbe) {
    state.materialProbeGrids = italy::makeMaterialProbeSdfs(AppState::kMaterialProbeResolution);
    state.statusLine += " -> baked " + std::to_string(state.materialProbeGrids.size()) + " material probe grids at " +
                         std::to_string(AppState::kMaterialProbeResolution) + "^3";
  }
  if (state.wantFogVolume) {
    state.fogVolume = italy::buildProceduralFogSphereVolume(
        glm::vec3(0.0f), state.fogRadius, state.fogVoxelSize, glm::vec3(state.fogSigmaT), state.fogScatterAlbedo,
        state.fogAsymmetry, state.fogDensityScale);
    state.statusLine += " + fog volume (NanoVDB, " + std::to_string(state.fogVolume.gridBlob.size()) + " bytes)";
  }
  logBakeLine(bake, "Bake representation: " + state.statusLine);
}

void bakeStepEnvironment(AppState &state, ProjectBakeState &bake) {
  state.haveEnvironment = false;
  std::string hdriPath;
  switch (state.envChoice) {
  case EnvChoice::Overcast:
    hdriPath = "assets/hdri/overcast_day.hdr";
    break;
  case EnvChoice::Midnight:
    hdriPath = "assets/hdri/midnight.hdr";
    break;
  case EnvChoice::Noon:
    hdriPath = "assets/hdri/noon.hdr";
    break;
  case EnvChoice::Custom:
    hdriPath = state.hdriPathBuf;
    break;
  case EnvChoice::ProceduralSky:
  case EnvChoice::None:
    break;
  }
  if (!hdriPath.empty()) {
    std::string err;
    if (italy::loadEnvironmentMap(hdriPath, state.environment, err)) {
      state.haveEnvironment = true;
    } else {
      state.statusLine += " | environment failed: " + err;
    }
  } else if (state.envChoice == EnvChoice::ProceduralSky) {
    std::string err;
    if (italy::buildProceduralSky(AppState::kSkyResolution, AppState::kSkyResolution / 2, state.skySunElevationDeg,
                                   state.skySunAzimuthDeg, state.skyTurbidity, state.environment, err,
                                   !state.sunEnabled)) {
      state.haveEnvironment = true;
    } else {
      state.statusLine += " | procedural sky failed: " + err;
    }
  }
  logBakeLine(bake, std::string("Build environment: ") + (state.haveEnvironment ? "ready" : "none") +
                         (state.envChoice == EnvChoice::None ? "" : (" (" + state.statusLine + ")")));
}

void bakeStepConstructRenderer(AppState &state, std::unique_ptr<italy::OptixRenderer> &renderer,
                                italy::OrbitCamera &camera, ProjectBakeState &bake, bool reframeCamera = true) {
  italy::SceneSource source;
  const bool haveSdfRepr = state.haveMesh && state.representation == Representation::Sdf;
  const bool haveVoxelRepr = state.haveMesh && state.representation == Representation::Voxel;
  if (state.haveSplats)
    source.splats = &state.splatAsset;
  else if (haveSdfRepr)
    source.sdf = &state.sdfGrid;
  else if (haveVoxelRepr)
    source.voxels = &state.voxelGrid;
  else if (state.haveMesh)
    source.mesh = &state.meshAsset;
  else if (state.wantGlassCup)
    source.sdf = &state.glassCupGrid;
  if (state.haveEnvironment)
    source.environment = &state.environment;
  if (state.wantFogVolume)
    source.volume = &state.fogVolume;
  source.groundPlane = state.groundPlane;
  if (state.wantCup) {
    state.cupGrid = italy::makeCupSdf(AppState::kCupResolution, -1.0f, glm::vec2(-1.6f, 1.6f));
    source.extraSdf = {&state.cupGrid};
  } else if (state.wantMaterialProbe) {
    source.extraSdf.reserve(state.materialProbeGrids.size());
    for (const italy::SdfGrid &grid : state.materialProbeGrids)
      source.extraSdf.push_back(&grid);
    source.emptyBase = true;
  }
  if (state.wantGlassCup) {
    constexpr float kGlassCupOuterRadius = 0.55f;
    source.groundOffset = 0.2f * kGlassCupOuterRadius;
  }

  syncSunRenderSettings(state);

  renderer = std::make_unique<italy::OptixRenderer>(state.renderWidth, state.renderHeight, source);

  const bool sceneHasSdf = source.sdf != nullptr || !source.extraSdf.empty();
  if (reframeCamera && (state.haveMesh || state.haveSplats || sceneHasSdf))
    camera.frame(renderer->sceneBoundsCenter(), renderer->sceneBoundsRadius());

  logBakeLine(bake, "Constructed renderer at " + std::to_string(state.renderWidth) + "x" +
                         std::to_string(state.renderHeight) + ". Handing off to viewport.");
}

void rebuildEnvironmentLive(AppState &state, std::unique_ptr<italy::OptixRenderer> &renderer,
                             italy::OrbitCamera &camera) {
  ProjectBakeState localBake;
  bakeStepEnvironment(state, localBake);
  bakeStepConstructRenderer(state, renderer, camera, localBake, /*reframeCamera=*/false);
}

void rebuildFogVolumeLive(AppState &state, std::unique_ptr<italy::OptixRenderer> &renderer,
                            italy::OrbitCamera &camera) {
  if (state.wantFogVolume) {
    state.fogVolume = italy::buildProceduralFogSphereVolume(glm::vec3(0.0f), state.fogRadius, state.fogVoxelSize,
                                                              glm::vec3(state.fogSigmaT), state.fogScatterAlbedo,
                                                              state.fogAsymmetry, state.fogDensityScale);
  }
  ProjectBakeState localBake;
  bakeStepConstructRenderer(state, renderer, camera, localBake, /*reframeCamera=*/false);
}

void advanceBake(AppState &state, std::unique_ptr<italy::OptixRenderer> &renderer, italy::OrbitCamera &camera,
                  ProjectBakeState &bake, bool &wantProject, bool &wantViewport, bool &wantSettings) {
  switch (bake.step) {
  case ProjectBakeState::Step::Load:
    bakeStepLoad(state, bake);
    bake.step = ProjectBakeState::Step::Classify;
    break;
  case ProjectBakeState::Step::Classify:
    bakeStepClassify(state, bake);
    bake.step = ProjectBakeState::Step::Representation;
    break;
  case ProjectBakeState::Step::Representation:
    bakeStepRepresentation(state, bake);
    bake.step = ProjectBakeState::Step::Environment;
    break;
  case ProjectBakeState::Step::Environment:
    bakeStepEnvironment(state, bake);
    bake.step = ProjectBakeState::Step::ConstructRenderer;
    break;
  case ProjectBakeState::Step::ConstructRenderer:
    bakeStepConstructRenderer(state, renderer, camera, bake);
    bake.active = false;
    bake.step = ProjectBakeState::Step::Load;
    wantProject = false;
    wantViewport = true;
    wantSettings = true;
    break;
  }
}

struct SceneDescriptor {
  const char *name;
  const char *description;
  std::function<void(AppState &)> configure;
};

std::vector<SceneDescriptor> makeSceneRegistry() {
  return {
      {"Bring-up scene",
       "The existing fixed test geometry with mesh/voxel/SDF/gsplat representation controls — loads instantly, "
       "known-good transport baseline.",
       [](AppState &state) {
         state.glbPathBuf[0] = '\0';
         state.gsplatPathBuf[0] = '\0';
         state.representation = Representation::Mesh;
         state.wantCup = false;
         state.wantGlassCup = false;
         state.wantMaterialProbe = false;
         state.envChoice = EnvChoice::None;
         state.statusLine = "Showing the built-in test scene.";
       }},
      {"Material probe",
       "Row of spheres sweeping roughness and IOR under the sun disk — dials in tonemap/exposure by eye.",
       [](AppState &state) {
         state.glbPathBuf[0] = '\0';
         state.representation = Representation::Mesh;
         state.wantCup = false;
         state.wantGlassCup = false;
         state.wantMaterialProbe = true;
         state.envChoice = EnvChoice::ProceduralSky;
         state.sunEnabled = true;
         state.groundPlane = true;
         state.statusLine = "Material probe scene.";
       }},
      {"Glass cup",
       "Borosilicate glass shell with light-blue liquid, HDRI + sun disk, real caustics onto a receiving floor — "
       "the north star scene.",
       [](AppState &state) {
         state.glbPathBuf[0] = '\0';
         state.representation = Representation::Mesh;
         state.wantCup = false;
         state.wantMaterialProbe = false;
         state.wantGlassCup = true;
         state.envChoice = EnvChoice::ProceduralSky;
         state.sunEnabled = true;
         state.groundPlane = true;
         state.statusLine = "Glass cup scene.";
       }},
      {"Caustics cup",
       "Ceramic bowl layered onto the bring-up scene's glass/mirror spheres — a caustic stress test, not a "
       "beauty shot.",
       [](AppState &state) {
         state.glbPathBuf[0] = '\0';
         state.representation = Representation::Mesh;
         state.wantGlassCup = false;
         state.wantMaterialProbe = false;
         state.wantCup = true;
         state.envChoice = EnvChoice::ProceduralSky;
         state.sunEnabled = true;
         state.groundPlane = true;
         state.statusLine = "Caustics cup scene.";
       }},
  };
}

} // namespace

int main(int argc, char **argv) {
  AppState state;

  if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0)
    std::snprintf(state.glbPathBuf, sizeof(state.glbPathBuf), "%s", argv[1]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    const std::string key = eq != std::string::npos ? arg.substr(0, eq) : arg;
    const std::string value = eq != std::string::npos ? arg.substr(eq + 1) : std::string();
    auto parseResolution = [&](const std::string &v) {
      if (v.empty())
        return state.resolution;
      const int n = std::atoi(v.c_str());
      if (n < kMinResolution) {
        std::fprintf(stderr, "italy: resolution %d below the minimum of %d, clamping\n", n, kMinResolution);
      }
      return std::max(n, kMinResolution);
    };
    auto parseDimension = [&](const std::string &v, int fallback, const char *what) {
      const int n = std::atoi(v.c_str());
      if (n < 1) {
        std::fprintf(stderr, "italy: %s %d must be at least 1, keeping %d\n", what, n, fallback);
        return fallback;
      }
      return n;
    };
    if (key == "--voxel") {
      state.representation = Representation::Voxel;
      state.resolution = parseResolution(value);
    } else if (key == "--sdf") {
      state.representation = Representation::Sdf;
      state.resolution = parseResolution(value);
    } else if (key == "--width") {
      state.renderWidth = parseDimension(value, state.renderWidth, "--width");
    } else if (key == "--height") {
      state.renderHeight = parseDimension(value, state.renderHeight, "--height");
    } else if (key == "--env") {
      state.envChoiceExplicit = true;
      if (value == "overcast")
        state.envChoice = EnvChoice::Overcast;
      else if (value == "midnight")
        state.envChoice = EnvChoice::Midnight;
      else if (value == "noon")
        state.envChoice = EnvChoice::Noon;
      else if (value == "sky")
        state.envChoice = EnvChoice::ProceduralSky;
      else {
        state.envChoiceExplicit = false;
        std::fprintf(stderr, "italy: unknown --env preset '%s' (try overcast, midnight, noon, sky)\n", value.c_str());
      }
    } else if (key == "--hdri") {
      state.envChoiceExplicit = true;
      state.envChoice = EnvChoice::Custom;
      std::snprintf(state.hdriPathBuf, sizeof(state.hdriPathBuf), "%s", value.c_str());
    } else if (key == "--gsplat") {
      state.representation = Representation::Gsplat;
      std::snprintf(state.gsplatPathBuf, sizeof(state.gsplatPathBuf), "%s", value.c_str());
    } else if (key == "--fog") {
      state.wantFogVolume = true;
    }
  }

  if (const char *v = std::getenv("ITALY_NVDB_SIGMA_T"))
    state.fogSigmaT = std::max(0.0f, static_cast<float>(std::atof(v)));
  if (const char *v = std::getenv("ITALY_NVDB_DENSITY_SCALE"))
    state.fogDensityScale = std::max(0.0f, static_cast<float>(std::atof(v)));
  if (const char *v = std::getenv("ITALY_NVDB_G"))
    state.fogAsymmetry = std::clamp(static_cast<float>(std::atof(v)), -0.95f, 0.95f);

  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }

  IMGUI_CHECKVERSION();

  italy::OrbitCamera camera;
  italy::OrbitCamera prevCamera = camera;
  MouseState mouse;
  std::unique_ptr<italy::OptixRenderer> renderer;

  if (std::getenv("ITALY_DEBUG_LOOKDOWN")) {
    camera.frame(glm::vec3(-2.2f, -1.0f, -2.2f), 3.0f);
  }
  if (std::getenv("ITALY_FORCE_DENOISE"))
    state.render.denoise = true;
  if (const char *dt = std::getenv("ITALY_DENOISE_TEMPORAL"))
    state.render.denoiseTemporal = std::atoi(dt) != 0;
  if (const char *fc = std::getenv("ITALY_FIREFLY_CLAMP"))
    state.render.fireflyClamp = static_cast<float>(std::atof(fc));
  if (const char *ls = std::getenv("ITALY_LIGHT_SUBPATHS"))
    state.render.lightSubpaths = std::atoi(ls) != 0;
  if (const char *mc = std::getenv("ITALY_MAX_CONNECTIONS"))
    state.render.maxConnectionsPerVertex = static_cast<unsigned int>(std::max(0, std::atoi(mc)));
  if (const char *rn = std::getenv("ITALY_RESERVOIR_NEE"))
    state.render.reservoirNEE = std::atoi(rn) != 0;
  if (const char *rt = std::getenv("ITALY_RESERVOIR_TEMPORAL"))
    state.render.reservoirTemporal = std::atoi(rt) != 0;
  if (const char *el = std::getenv("ITALY_TEST_EXTRA_LIGHTS"))
    state.render.extraTestLightCount = static_cast<unsigned int>(std::max(0, std::atoi(el)));
  if (std::getenv("ITALY_NO_GROUND"))
    state.groundPlane = false;
  if (const char *ap = std::getenv("ITALY_APERTURE"))
    state.render.aperture = static_cast<float>(std::atof(ap));
  if (const char *fd = std::getenv("ITALY_FOCUS_DISTANCE"))
    state.render.focusDistance = static_cast<float>(std::atof(fd));
  if (const char *er = std::getenv("ITALY_ENV_ROTATION"))
    state.render.envRotation = static_cast<float>(std::atof(er));
  if (const char *t = std::getenv("ITALY_SKY_TURBIDITY"))
    state.skyTurbidity = static_cast<float>(std::atof(t));
  if (const char *e = std::getenv("ITALY_SKY_ELEVATION"))
    state.skySunElevationDeg = static_cast<float>(std::atof(e));
  if (const char *a = std::getenv("ITALY_SKY_AZIMUTH"))
    state.skySunAzimuthDeg = static_cast<float>(std::atof(a));
  if (const char *gp = std::getenv("ITALY_GSPLAT_PATH")) {
    state.representation = Representation::Gsplat;
    std::snprintf(state.gsplatPathBuf, sizeof(state.gsplatPathBuf), "%s", gp);
  }
  if (const char *tm = std::getenv("ITALY_TONEMAP")) {
    const std::string v = tm;
    if (v == "agx") state.render.tonemap = italy::TonemapOperator::AgX;
    else if (v == "reinhard") state.render.tonemap = italy::TonemapOperator::Reinhard;
    else if (v == "aces") state.render.tonemap = italy::TonemapOperator::Aces;
    else if (v == "hable") state.render.tonemap = italy::TonemapOperator::Hable;
    else if (v == "clamp") state.render.tonemap = italy::TonemapOperator::Clamp;
    else std::fprintf(stderr, "italy: unknown ITALY_TONEMAP '%s'\n", v.c_str());
  }

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow *shareAnchor = glfwCreateWindow(1, 1, "italy-rs GL anchor", nullptr, nullptr);
  if (!shareAnchor) {
    std::fprintf(stderr, "glfwCreateWindow failed for the hidden GL-share anchor\n");
    return 1;
  }
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

  const std::vector<SceneDescriptor> sceneRegistry = makeSceneRegistry();
  ProjectBakeState bake;

  bool wantProject = true;
  bool wantViewport = false;
  bool wantSettings = false;

  if (std::getenv("ITALY_DUMP_FRAME")) {
    if (const char *sceneEnv = std::getenv("ITALY_SCENE"))
      bake.selected = std::clamp(std::atoi(sceneEnv), 0, static_cast<int>(sceneRegistry.size()) - 1);
    const EnvChoice requestedEnv = state.envChoice;
    sceneRegistry[bake.selected].configure(state);
    if (state.envChoiceExplicit)
      state.envChoice = requestedEnv;
    bake.step = ProjectBakeState::Step::Load;
    bake.active = true;
  }

  std::unique_ptr<italy::AppWindow> projectWindow;
  std::unique_ptr<italy::AppWindow> viewportWindow;
  std::unique_ptr<italy::AppWindow> settingsWindow;

  italy::AppWindow::DrawFn projectDraw = [&](italy::AppWindow &) {
    ImGui::TextWrapped("Pick a scene, then Bake. Reopen this panel later via a checkbox in Viewport or Settings to "
                        "bake something else — it won't tear down the current render until you do.");
    ImGui::Separator();

    for (int i = 0; i < static_cast<int>(sceneRegistry.size()); ++i) {
      ImGui::RadioButton(sceneRegistry[i].name, &bake.selected, i);
      ImGui::SameLine();
      ImGui::TextDisabled("%s", sceneRegistry[i].description);
    }

    ImGui::Separator();
    const bool bakeDisabled = bake.active;
    if (bakeDisabled)
      ImGui::BeginDisabled();
    if (ImGui::Button("Bake")) {
      const SceneDescriptor &scene = sceneRegistry[bake.selected];
      scene.configure(state);
      bake.log.clear();
      bake.rawMeshLoaded = false;
      bake.step = ProjectBakeState::Step::Load;
      bake.active = true;
      logBakeLine(bake, std::string("Baking '") + scene.name + "'.");
      logBakeLine(bake, state.statusLine);
    }
    if (bakeDisabled)
      ImGui::EndDisabled();

    if (bake.active)
      advanceBake(state, renderer, camera, bake, wantProject, wantViewport, wantSettings);

    ImGui::Separator();
    ImGui::Text("Bake log");
    ImGui::BeginChild("bake_log", ImVec2(0, 220), true);
    for (const std::string &line : bake.log)
      ImGui::TextUnformatted(line.c_str());
    if (bake.active)
      ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
  };

  italy::AppWindow::DrawFn viewportDraw = [&](italy::AppWindow &self) {
    ImGui::Checkbox("Project", &wantProject);
    ImGui::SameLine();
    ImGui::Checkbox("Settings", &wantSettings);
    ImGui::Separator();

    double x, y;
    glfwGetCursorPos(self.window(), &x, &y);
    const float dx = static_cast<float>(x - mouse.lastX);
    const float dy = static_cast<float>(y - mouse.lastY);
    mouse.lastX = x;
    mouse.lastY = y;

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float srcAspect = static_cast<float>(renderer->width()) / static_cast<float>(renderer->height());
    ImVec2 imageSize = avail;
    if (avail.x / avail.y > srcAspect)
      imageSize.x = avail.y * srcAspect;
    else
      imageSize.y = avail.x / srcAspect;
    const ImVec2 origin = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(origin.x + (avail.x - imageSize.x) * 0.5f, origin.y + (avail.y - imageSize.y) * 0.5f));
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();

    ImGui::InvisibleButton("##viewport_image", imageSize,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle |
                               ImGuiButtonFlags_MouseButtonRight);
    if (ImGui::IsItemActive()) {
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) camera.orbit(dx, dy);
      if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) || ImGui::IsMouseDown(ImGuiMouseButton_Right))
        camera.pan(dx, dy);
    }
    if (ImGui::IsItemHovered())
      camera.zoom(ImGui::GetIO().MouseWheel);

    if (const char *orbitAfter = std::getenv("ITALY_SCRIPT_ORBIT_AFTER")) {
      static bool scriptedOrbitDone = false;
      if (!scriptedOrbitDone && renderer->subframeIndex() >= static_cast<unsigned int>(std::atoi(orbitAfter))) {
        camera.orbit(15.0f, 0.0f);
        scriptedOrbitDone = true;
      }
    }

    if (state.isSampling) {
      if (cameraChanged(camera, prevCamera)) {
        renderer->notifyCameraMoved();
        prevCamera = camera;
      }
      renderer->render(camera, state.render);
    } else if (cameraChanged(camera, prevCamera)) {
      state.isSampling = true;
      renderer->resetAccumulation();
      prevCamera = camera;
      renderer->render(camera, state.render);
    }

    if (state.exportPending) {
      if (renderer->subframeIndex() >= static_cast<unsigned int>(state.exportSamples)) {
        state.statusLine = writeFrame(*renderer, state.exportPathBuf)
                               ? "Wrote " + std::string(state.exportPathBuf)
                               : "Failed to write " + std::string(state.exportPathBuf);
        state.exportPending = false;
      }
    }

    const char *dumpAfter = std::getenv("ITALY_DUMP_AFTER_SUBFRAME");
    const unsigned int dumpThreshold = dumpAfter ? static_cast<unsigned int>(std::atoi(dumpAfter)) : 128u;
    if (std::getenv("ITALY_DUMP_FRAME") && renderer->subframeIndex() >= dumpThreshold) {
      dumpFrameIfRequested(*renderer, self.window());
      wantProject = false;
      wantViewport = false;
      wantSettings = false;
    }

    ImGui::SetCursorScreenPos(imagePos);
    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(renderer->glTextureId())), imageSize, ImVec2(0, 1),
                 ImVec2(1, 0));
  };

  italy::AppWindow::DrawFn settingsDraw = [&](italy::AppWindow &) {
    ImGui::Checkbox("Project", &wantProject);
    ImGui::SameLine();
    ImGui::Checkbox("Viewport", &wantViewport);
    ImGui::Separator();

    ImGui::TextWrapped("%s", state.statusLine.c_str());
    ImGui::Text("Lighting: %s", state.haveEnvironment ? "environment (HDRI)" : "synthetic quad light");
    ImGui::Separator();

    ImGui::Text("Representation");
    ImGui::RadioButton("Mesh", reinterpret_cast<int *>(&state.representation), 0);
    ImGui::SameLine();
    ImGui::RadioButton("Voxel", reinterpret_cast<int *>(&state.representation), 1);
    ImGui::SameLine();
    ImGui::RadioButton("SDF", reinterpret_cast<int *>(&state.representation), 2);
    ImGui::SameLine();
    ImGui::RadioButton("Gsplat", reinterpret_cast<int *>(&state.representation), 3);
    if (state.representation == Representation::Voxel || state.representation == Representation::Sdf) {
      if (ImGui::InputInt("Resolution", &state.resolution))
        state.resolution = std::max(state.resolution, kMinResolution);
    }

    if (state.representation == Representation::Gsplat)
      ImGui::InputText("Gsplat .ply path", state.gsplatPathBuf, sizeof(state.gsplatPathBuf));
    else
      ImGui::InputText("GLB path", state.glbPathBuf, sizeof(state.glbPathBuf));

    ImGui::TextDisabled("Changing representation/paths here takes effect next time you Bake in the Project "
                         "window.");

    ImGui::Separator();
    ImGui::Text("HDRI");
    bool envChanged = false;
    envChanged |= ImGui::RadioButton("None", reinterpret_cast<int *>(&state.envChoice), 0);
    ImGui::SameLine();
    envChanged |= ImGui::RadioButton("Overcast", reinterpret_cast<int *>(&state.envChoice), 1);
    ImGui::SameLine();
    envChanged |= ImGui::RadioButton("Midnight", reinterpret_cast<int *>(&state.envChoice), 2);
    ImGui::SameLine();
    envChanged |= ImGui::RadioButton("Noon", reinterpret_cast<int *>(&state.envChoice), 3);
    ImGui::SameLine();
    envChanged |= ImGui::RadioButton("Custom", reinterpret_cast<int *>(&state.envChoice), 4);
    ImGui::SameLine();
    envChanged |= ImGui::RadioButton("Procedural Sky", reinterpret_cast<int *>(&state.envChoice), 5);
    if (state.envChoice == EnvChoice::Custom)
      envChanged |= ImGui::InputText("HDRI path", state.hdriPathBuf, sizeof(state.hdriPathBuf),
                                     ImGuiInputTextFlags_EnterReturnsTrue);
    if (envChanged && !(state.envChoice == EnvChoice::Custom && state.hdriPathBuf[0] == '\0'))
      rebuildEnvironmentLive(state, renderer, camera);
    if (state.envChoice == EnvChoice::ProceduralSky) {
      if (ImGui::InputFloat("Turbidity", &state.skyTurbidity)) {
        state.skyTurbidity = std::max(state.skyTurbidity, 1.9f);
        rebuildEnvironmentLive(state, renderer, camera);
      }
      if (ImGui::InputFloat("Sun elevation", &state.skySunElevationDeg)) {
        rebuildEnvironmentLive(state, renderer, camera);
      }
      if (ImGui::InputFloat("Sun azimuth", &state.skySunAzimuthDeg)) {
        rebuildEnvironmentLive(state, renderer, camera);
      }
    }

    ImGui::Separator();
    ImGui::Text("Sun (analytic disk, live)");
    if (ImGui::Checkbox("Sun enabled", &state.sunEnabled)) {
      if (state.envChoice == EnvChoice::ProceduralSky)
        rebuildEnvironmentLive(state, renderer, camera);
      else {
        syncSunRenderSettings(state);
        renderer->resetAccumulation();
      }
    }
    if (ImGui::InputFloat("Sun angular radius (deg)", &state.sunAngularRadiusDeg)) {
      state.sunAngularRadiusDeg = std::clamp(state.sunAngularRadiusDeg, 1e-3f, 89.999f);
      syncSunRenderSettings(state);
      renderer->resetAccumulation();
    }
    if (ImGui::ColorEdit3("Sun color", &state.sunColor.x)) {
      syncSunRenderSettings(state);
      renderer->resetAccumulation();
    }
    if (ImGui::InputFloat("Sun intensity", &state.sunIntensity)) {
      state.sunIntensity = std::max(state.sunIntensity, 0.0f);
      syncSunRenderSettings(state);
      renderer->resetAccumulation();
    }
    if (ImGui::ColorEdit3("Background color", &state.backgroundColor.x)) {
      state.render.backgroundColor = state.backgroundColor;
      renderer->resetAccumulation();
    }
    ImGui::TextDisabled("Sun elevation/azimuth are shared with the procedural sky above; angular radius/color/"
                         "intensity/background are live RenderSettings, no re-bake needed. Turbidity/elevation/"
                         "azimuth/enabled re-bake the sky texture automatically when the environment is the "
                         "procedural sky, so the horizon glow always matches.");

    ImGui::Checkbox("Ground plane", &state.groundPlane);

    ImGui::Separator();
    if (ImGui::Checkbox("Fog volume (NanoVDB)", &state.wantFogVolume))
      rebuildFogVolumeLive(state, renderer, camera);
    if (state.wantFogVolume) {
      bool fogChanged = false;
      fogChanged |= ImGui::SliderFloat("Fog radius", &state.fogRadius, 0.1f, 5.0f);
      fogChanged |= ImGui::SliderFloat("Fog voxel size", &state.fogVoxelSize, 0.01f, 0.2f);
      fogChanged |= ImGui::SliderFloat("Fog sigma_t (extinction)", &state.fogSigmaT, 0.0f, 20.0f);
      fogChanged |= ImGui::ColorEdit3("Fog scatter albedo", &state.fogScatterAlbedo.x);
      fogChanged |= ImGui::SliderFloat("Fog phase g", &state.fogAsymmetry, -0.95f, 0.95f);
      fogChanged |= ImGui::SliderFloat("Fog density scale", &state.fogDensityScale, 0.0f, 5.0f);
      if (fogChanged)
        rebuildFogVolumeLive(state, renderer, camera);
    }

    ImGui::Separator();
    if (ImGui::InputFloat("Exposure", &state.render.exposure))
      state.render.exposure = std::max(state.render.exposure, 0.0f);
    int spl = static_cast<int>(state.render.samplesPerLaunch);
    if (ImGui::InputInt("Samples/launch", &spl))
      state.render.samplesPerLaunch = static_cast<unsigned int>(std::max(spl, 1));
    ImGui::Checkbox("Denoiser", &state.render.denoise);
    if (state.render.denoise) {
      ImGui::SameLine();
      if (ImGui::Checkbox("Temporal", &state.render.denoiseTemporal))
        renderer->resetAccumulation();
    }

    if (ImGui::InputFloat("Firefly clamp (0 = off)", &state.render.fireflyClamp)) {
      state.render.fireflyClamp = std::max(state.render.fireflyClamp, 0.0f);
      renderer->resetAccumulation();
    }

    if (ImGui::Checkbox("Light subpaths (VCM connections/merging)", &state.render.lightSubpaths))
      renderer->resetAccumulation();

    int maxConn = static_cast<int>(state.render.maxConnectionsPerVertex);
    if (ImGui::InputInt("Max BDPT connections/vertex (0 = uncapped)", &maxConn)) {
      state.render.maxConnectionsPerVertex = static_cast<unsigned int>(std::max(maxConn, 0));
      renderer->resetAccumulation();
    }

    if (ImGui::Checkbox("Reservoir NEE (spatial RIS)", &state.render.reservoirNEE))
      renderer->resetAccumulation();
    if (state.render.reservoirNEE) {
      ImGui::SameLine();
      if (ImGui::Checkbox("Temporal", &state.render.reservoirTemporal))
        renderer->resetAccumulation();
    }
    int extraLights = static_cast<int>(state.render.extraTestLightCount);
    if (ImGui::InputInt("Extra test lights (RIS verification)", &extraLights)) {
      state.render.extraTestLightCount = static_cast<unsigned int>(std::clamp(extraLights, 0, 3));
      renderer->resetAccumulation();
    }

    ImGui::Separator();
    ImGui::Text("Lens");
    if (ImGui::InputFloat("Aperture (0 = pinhole)", &state.render.aperture, 0.0f, 0.0f, "%.4f")) {
      state.render.aperture = std::max(state.render.aperture, 0.0f);
      renderer->resetAccumulation();
    }
    if (ImGui::InputFloat("Focus distance (0 = orbit target)", &state.render.focusDistance, 0.0f, 0.0f, "%.3f")) {
      state.render.focusDistance = std::max(state.render.focusDistance, 0.0f);
      renderer->resetAccumulation();
    }

    if (state.haveEnvironment) {
      ImGui::Separator();
      float envRotationDeg = glm::degrees(state.render.envRotation);
      if (ImGui::InputFloat("Environment rotation (deg)", &envRotationDeg)) {
        state.render.envRotation = glm::radians(envRotationDeg);
        renderer->resetAccumulation();
      }
    }

    ImGui::Text("Tonemap");
    ImGui::RadioButton("AgX", reinterpret_cast<int *>(&state.render.tonemap), 0);
    ImGui::SameLine();
    ImGui::RadioButton("Reinhard", reinterpret_cast<int *>(&state.render.tonemap), 1);
    ImGui::SameLine();
    ImGui::RadioButton("ACES", reinterpret_cast<int *>(&state.render.tonemap), 2);
    ImGui::SameLine();
    ImGui::RadioButton("Hable", reinterpret_cast<int *>(&state.render.tonemap), 3);
    ImGui::SameLine();
    ImGui::RadioButton("Clamp", reinterpret_cast<int *>(&state.render.tonemap), 4);

    ImGui::Separator();
    ImGui::Text("Output");
    int dims[2] = {state.renderWidth, state.renderHeight};
    if (ImGui::InputInt2("Render size", dims)) {
      state.renderWidth = std::max(dims[0], 1);
      state.renderHeight = std::max(dims[1], 1);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(re-bake in Project to apply)");

    ImGui::InputText("PNG path", state.exportPathBuf, sizeof(state.exportPathBuf));
    if (ImGui::InputInt("Export samples", &state.exportSamples))
      state.exportSamples = std::max(state.exportSamples, 1);
    if (state.exportPending) {
      ImGui::Text("Rendering... subframe %u / %d", renderer->subframeIndex(), state.exportSamples);
      if (ImGui::Button("Cancel export"))
        state.exportPending = false;
    } else if (ImGui::Button("Render to PNG")) {
      renderer->resetAccumulation();
      state.exportPending = true;
    }

    ImGui::Separator();
    if (ImGui::Button(state.isSampling ? "Pause" : "Resume"))
      state.isSampling = !state.isSampling;
    ImGui::SameLine();
    ImGui::Text(state.isSampling ? "Sampling" : "Paused");
    const glm::vec3 pos = camera.position();
    ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
    ImGui::Text("Subframe: %u", renderer->subframeIndex());
    ImGui::Text("Drag left-click to orbit, right- or middle-click to pan, scroll to zoom.");
  };

  while (wantProject || wantViewport || wantSettings || projectWindow || viewportWindow || settingsWindow) {
    glfwPollEvents();

    if (wantProject && !projectWindow)
      projectWindow = std::make_unique<italy::AppWindow>("Project", 620, 640, shareAnchor, projectDraw);
    if (wantViewport && !viewportWindow)
      viewportWindow = std::make_unique<italy::AppWindow>("Viewport", 980, 580, shareAnchor, viewportDraw);
    if (wantSettings && !settingsWindow)
      settingsWindow = std::make_unique<italy::AppWindow>("Italy R/S", 420, 800, shareAnchor, settingsDraw);

    if (projectWindow && !projectWindow->shouldClose())
      projectWindow->frame();
    if (viewportWindow && !viewportWindow->shouldClose())
      viewportWindow->frame();
    if (settingsWindow && !settingsWindow->shouldClose())
      settingsWindow->frame();

    if (projectWindow && projectWindow->shouldClose())
      wantProject = false;
    if (viewportWindow && viewportWindow->shouldClose())
      wantViewport = false;
    if (settingsWindow && settingsWindow->shouldClose())
      wantSettings = false;

    if (projectWindow && !wantProject)
      projectWindow.reset();
    if (viewportWindow && !wantViewport)
      viewportWindow.reset();
    if (settingsWindow && !wantSettings)
      settingsWindow.reset();
  }

  glfwDestroyWindow(shareAnchor);
  glfwTerminate();
  return 0;
}
