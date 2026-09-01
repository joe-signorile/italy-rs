// Phase 9: live ImGui controls — load a GLB, switch mesh/voxel/SDF
// representation, pick an HDRI preset, tune exposure/samples-per-launch —
// on top of the phase 2 window/camera/viewport shell. CLI args (still
// supported, mainly for scripted verification/ITALY_DUMP_FRAME use) seed
// the same state the UI edits, then both go through one shared rebuild
// path so they can't drift out of sync with each other.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "convert/voxelize.h"
#include "core/orbit_camera.h"
#include "io/gltf_loader.h"
#include "io/gsplat_ply_loader.h"
#include "io/mesh_validate.h"
#include "render/environment.h"
#include "render/optix_renderer.h"
#include "render/procedural_sky.h"

namespace {

// Polled once per frame rather than via GLFW callbacks — simpler than wiring
// a user-pointer + callback for what's just "delta since last frame".
struct MouseState {
  double lastX = 0.0;
  double lastY = 0.0;
};

bool cameraChanged(const italy::OrbitCamera &a, const italy::OrbitCamera &b) {
  const glm::vec3 da = a.position() - b.position();
  const glm::vec3 dt = a.target() - b.target();
  return glm::dot(da, da) > 1e-10f || glm::dot(dt, dt) > 1e-10f;
}

// Reads the renderer's GL texture back and writes it as a PNG. Shared by the
// scripted ITALY_DUMP_FRAME hook and the UI's Render-to-PNG button, so the
// image you export by hand is byte-identical to the one a script captures.
bool writeFramePng(const italy::OptixRenderer &renderer, const char *path) {
  const int w = renderer.width();
  const int h = renderer.height();
  std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
  glBindTexture(GL_TEXTURE_2D, renderer.glTextureId());
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  // Texture was uploaded with V flipped for ImGui (uv0=(0,1),uv1=(1,0)); flip
  // rows back here so the PNG reads right-side-up.
  std::vector<unsigned char> flipped(pixels.size());
  for (int row = 0; row < h; ++row)
    std::memcpy(&flipped[static_cast<size_t>(row) * w * 4], &pixels[static_cast<size_t>(h - 1 - row) * w * 4],
                static_cast<size_t>(w) * 4);
  const bool ok = stbi_write_png(path, w, h, 4, flipped.data(), w * 4) != 0;
  std::fprintf(stderr, "italy: %s %dx%d frame to %s\n", ok ? "wrote" : "FAILED to write", w, h, path);
  return ok;
}

// Debug/verification aid: set ITALY_DUMP_FRAME=path.png to write out the
// accumulated render and exit — lets a render-correctness check happen
// without eyeballing a live window.
void dumpFrameIfRequested(const italy::OptixRenderer &renderer, GLFWwindow *window) {
  const char *path = std::getenv("ITALY_DUMP_FRAME");
  if (!path)
    return;
  writeFramePng(renderer, path);
  glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// Explicit `int` underlying type: ImGui::RadioButton takes an `int*`, and
// the UI loop below reinterpret_casts &state.representation/&state.envChoice
// to one — relying on the (already-guaranteed-by-the-standard-for-scoped-
// enums-with-no-explicit-type) size/alignment match to work is the kind of
// thing worth spelling out rather than leaving implicit.
// Shared by the UI slider and the CLI parser so the two can't disagree about
// what a legal resampling resolution is.
constexpr int kMinResolution = 8;
constexpr int kMaxResolution = 512;

// Gsplat is a distinct load source, not a resampling of the loaded GLB mesh
// like Voxel/Sdf are — see rebuildScene()'s dispatch.
enum class Representation : int { Mesh, Voxel, Sdf, Gsplat };
enum class EnvChoice : int { None, Overcast, Midnight, Noon, Custom, ProceduralSky };

// Everything the UI edits, plus the loaded/resampled assets those edits
// produce. Assets are kept here (not as OptixRenderer members) because
// SceneSource only borrows pointers into them for the duration of the
// OptixRenderer constructor call — this struct is what actually owns them,
// and must outlive whatever SceneSource is built from it.
struct AppState {
  italy::MeshAsset meshAsset;
  bool haveMesh = false;
  italy::VoxelGrid voxelGrid;
  italy::SdfGrid sdfGrid;
  // Roadmap phase 3: an imported Gaussian-splat scene — a distinct load
  // source, not a resampling of meshAsset, so it gets its own path/asset/
  // flag rather than reusing glbPathBuf/meshAsset/haveMesh.
  italy::GsplatAsset splatAsset;
  bool haveSplats = false;
  char gsplatPathBuf[512] = "";
  italy::EnvironmentMap environment;
  bool haveEnvironment = false;

  char glbPathBuf[512] = "";
  Representation representation = Representation::Mesh;
  // 128, not 64. The SDF grid is sized from the mesh's actual extents rather
  // than resolution^3, so raising this is close to free (measured on a 1.9M
  // triangle asset: 6.5s at 64 vs 6.8s at 160) while 64 leaves any structure
  // thinner than a cell — spokes, cables, thin brackets — to break up into
  // floating fragments. Voxel occupancy is sparse, so it scales gently too.
  int resolution = 128;
  EnvChoice envChoice = EnvChoice::None;
  char hdriPathBuf[512] = "";
  bool groundPlane = true;

  // Procedural sky (Preetham/Perez) params — only shown/used when envChoice
  // == ProceduralSky. Defaults are an ordinary clear midday sky.
  float skyTurbidity = 3.0f;
  float skySunElevationDeg = 45.0f;
  float skySunAzimuthDeg = 0.0f;
  static constexpr int kSkyResolution = 1024; // matches the bundled HDRI presets' rough scale

  italy::RenderSettings render;

  // Render resolution. Changing it goes through the same all-or-nothing
  // rebuildScene() as everything else, because OptixRenderer owns the
  // accumulator, PBO, GL texture and denoiser and sizes all four at
  // construction — one rebuild path beats a second, parallel resize path.
  int renderWidth = 960;
  int renderHeight = 540;

  char exportPathBuf[512] = "render.png";
  int exportSamples = 512;
  bool exportPending = false; // set by the UI, serviced by the viewport callback

  // Start/stop toggle for progressive sampling. Continuous accumulation
  // otherwise burns GPU (and shares the one GPU with s-rank's services, per
  // CLAUDE.md) even while just looking at a converged frame. Orbiting while
  // paused auto-resumes (see Viewport's draw callback) rather than leaving a
  // stale, wrong-camera image on screen looking current.
  bool isSampling = true;

  std::string statusLine = "Showing the built-in test scene.";
};

// Re-loads/re-resamples everything from the current UI state and rebuilds
// the renderer from scratch. Deliberately one all-or-nothing "apply" step
// rather than incrementally patching the live scene — OptixRenderer has no
// partial-rebuild API (see its header), and voxelizing/SDF-baking large
// meshes isn't cheap enough to redo on every slider tick anyway, so a single
// explicit rebuild point keeps the cost visible and predictable. Shared by
// startup (seeded from CLI args) and the UI's "Apply" button so the two
// entry points can't drift out of sync with each other.
void rebuildScene(AppState &state, std::unique_ptr<italy::OptixRenderer> &renderer, italy::OrbitCamera &camera) {
  state.haveMesh = false;
  state.haveSplats = false;
  if (state.representation == Representation::Gsplat) {
    // A distinct load source, not a resampling of a loaded GLB — see
    // AppState::splatAsset's doc comment. No watertightness gate: that check
    // is mesh-specific (2-manifold edges) and doesn't apply to a point set.
    if (state.gsplatPathBuf[0] != '\0') {
      std::string err;
      if (!italy::loadGsplatPly(state.gsplatPathBuf, state.splatAsset, err)) {
        state.statusLine = "Failed to load " + std::string(state.gsplatPathBuf) + ": " + err;
        std::fprintf(stderr, "italy: %s\n", state.statusLine.c_str());
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
      std::fprintf(stderr, "italy: %s\n", state.statusLine.c_str());
    } else if (!italy::isWatertight(state.meshAsset, err)) {
      // Rejected, not just warned: SDF baking's sign vote and (once loaded
      // meshes can carry a transmissive material) refraction both depend on
      // a well-defined inside/outside, which only a closed mesh guarantees.
      state.statusLine = "Rejected " + std::string(state.glbPathBuf) + ": " + err;
      std::fprintf(stderr, "italy: %s\n", state.statusLine.c_str());
    } else {
      state.haveMesh = true;
      state.statusLine = "Loaded " + std::string(state.glbPathBuf) + " (" +
                          std::to_string(state.meshAsset.triangleCount()) + " tris, " +
                          std::to_string(state.meshAsset.materials.size()) + " materials, " +
                          std::to_string(state.meshAsset.textures.size()) + " textures)";
    }
  } else {
    state.statusLine = "Showing the built-in test scene.";
  }

  bool haveVoxels = false, haveSdf = false;
  if (state.haveMesh) {
    if (state.representation == Representation::Voxel) {
      state.voxelGrid = italy::voxelizeMesh(state.meshAsset, state.resolution);
      haveVoxels = true;
      state.statusLine += " -> voxelized (" + std::to_string(state.voxelGrid.cells.size()) + " cells)";
    } else if (state.representation == Representation::Sdf) {
      state.sdfGrid = italy::bakeSdf(state.meshAsset, state.resolution);
      haveSdf = true;
      state.statusLine += " -> SDF (" + std::to_string(state.sdfGrid.nx) + "x" + std::to_string(state.sdfGrid.ny) +
                           "x" + std::to_string(state.sdfGrid.nz) + ")";
    }
  }

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
      std::fprintf(stderr, "italy: failed to load environment %s: %s\n", hdriPath.c_str(), err.c_str());
    }
  } else if (state.envChoice == EnvChoice::ProceduralSky) {
    // Populates state.environment directly (no file path involved) — feeds
    // the exact same SceneSource::environment field a loaded HDRI would.
    std::string err;
    if (italy::buildProceduralSky(AppState::kSkyResolution, AppState::kSkyResolution / 2, state.skySunElevationDeg,
                                   state.skySunAzimuthDeg, state.skyTurbidity, state.environment, err)) {
      state.haveEnvironment = true;
    } else {
      state.statusLine += " | procedural sky failed: " + err;
      std::fprintf(stderr, "italy: procedural sky failed: %s\n", err.c_str());
    }
  }

  italy::SceneSource source;
  if (state.haveSplats)
    source.splats = &state.splatAsset;
  else if (haveSdf)
    source.sdf = &state.sdfGrid;
  else if (haveVoxels)
    source.voxels = &state.voxelGrid;
  else if (state.haveMesh)
    source.mesh = &state.meshAsset;
  if (state.haveEnvironment)
    source.environment = &state.environment;
  source.groundPlane = state.groundPlane;

  renderer = std::make_unique<italy::OptixRenderer>(state.renderWidth, state.renderHeight, source);
  if (state.haveMesh || state.haveSplats)
    camera.frame(renderer->sceneBoundsCenter(), renderer->sceneBoundsRadius());
}

} // namespace

int main(int argc, char **argv) {
  AppState state;

  // CLI args seed the same UI state rebuildScene() reads — kept mainly for
  // scripted verification (ITALY_DUMP_FRAME runs, README examples) rather
  // than as the primary way to drive the app now that the UI can do this.
  if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0)
    std::snprintf(state.glbPathBuf, sizeof(state.glbPathBuf), "%s", argv[1]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    const std::string key = eq != std::string::npos ? arg.substr(0, eq) : arg;
    const std::string value = eq != std::string::npos ? arg.substr(eq + 1) : std::string();
    // Resampling resolution is clamped, not trusted: --voxel=0 reached
    // glm::clamp(v, 0, resolution - 1) with a negative upper bound, and
    // absurdly large values allocate a dense SDF grid before anything can
    // report a problem. Same bounds the UI slider enforces.
    auto parseResolution = [&](const std::string &v) {
      if (v.empty())
        return state.resolution;
      const int n = std::atoi(v.c_str());
      if (n < kMinResolution || n > kMaxResolution) {
        std::fprintf(stderr, "italy: resolution %d out of range [%d, %d], clamping\n", n, kMinResolution,
                     kMaxResolution);
      }
      return std::clamp(n, kMinResolution, kMaxResolution);
    };
    auto parseDimension = [&](const std::string &v, int fallback, const char *what) {
      const int n = std::atoi(v.c_str());
      if (n < 64 || n > 8192) {
        std::fprintf(stderr, "italy: %s %d out of range [64, 8192], keeping %d\n", what, n, fallback);
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
      if (value == "overcast")
        state.envChoice = EnvChoice::Overcast;
      else if (value == "midnight")
        state.envChoice = EnvChoice::Midnight;
      else if (value == "noon")
        state.envChoice = EnvChoice::Noon;
      else if (value == "sky")
        state.envChoice = EnvChoice::ProceduralSky;
      else
        std::fprintf(stderr, "italy: unknown --env preset '%s' (try overcast, midnight, noon, sky)\n", value.c_str());
    } else if (key == "--hdri") {
      state.envChoice = EnvChoice::Custom;
      std::snprintf(state.hdriPathBuf, sizeof(state.hdriPathBuf), "%s", value.c_str());
    } else if (key == "--gsplat") {
      state.representation = Representation::Gsplat;
      std::snprintf(state.gsplatPathBuf, sizeof(state.gsplatPathBuf), "%s", value.c_str());
    }
  }

  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }

  IMGUI_CHECKVERSION();

  italy::OrbitCamera camera;
  italy::OrbitCamera prevCamera = camera;
  MouseState mouse;
  std::unique_ptr<italy::OptixRenderer> renderer;
  // Starts true so Viewport's draw callback below performs the initial
  // CLI-seeded rebuildScene() itself, on its own GL context, the first time
  // it runs — same mechanism the Apply button uses, just pre-armed.
  bool applyRequested = true;

  // Testing hooks, same spirit as ITALY_DUMP_FRAME: these toggles are UI-only
  // otherwise, so scripted before/after verification needs a way in that
  // doesn't require actually clicking the checkbox/radio button.
  // Temporary verification aid (VCM energy-check, see humans.md) — points
  // the camera straight down at the fixed test scene's ground point directly
  // under the quad light, so the image *center* pixel is exactly the point a
  // closed-form irradiance prediction is computed for, with no camera-ray
  // trig required to find it. Not a permanent feature; safe to delete once
  // the energy check is re-run after any future change to that scene.
  if (std::getenv("ITALY_DEBUG_LOOKDOWN")) {
    // Retargets at a ground point in the far corner from every sphere (and
    // its shadow/caustic footprint) — default yaw/pitch already gives a
    // clean, unoccluded, low-GI-contamination view of it (confirmed by
    // render: smooth monotonic irradiance gradient, no silhouette/penumbra
    // edges nearby), unlike the point directly under the light, which sits
    // close enough to the mirror/glass spheres for this scene's default
    // orbit angles to clip one of them or land in a soft-shadow penumbra.
    camera.frame(glm::vec3(-2.2f, -1.0f, -2.2f), 3.0f);
  }
  if (std::getenv("ITALY_FORCE_DENOISE"))
    state.render.denoise = true;
  if (const char *fc = std::getenv("ITALY_FIREFLY_CLAMP"))
    state.render.fireflyClamp = static_cast<float>(std::atof(fc));
  // VCM Step 1 A/B verification switch — see RenderSettings::lightSubpaths'
  // doc comment: 0 zero-weights BDPT connections back to today's NEE-only
  // behaviour, for comparing against the closed-form energy-check scene.
  if (const char *ls = std::getenv("ITALY_LIGHT_SUBPATHS"))
    state.render.lightSubpaths = std::atoi(ls) != 0;
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

  std::vector<std::unique_ptr<italy::AppWindow>> windows;

  // Viewport is built first, with shareContext=nullptr, making it the root
  // of the shared GL object namespace: it's the window that owns/samples
  // OptixRenderer's GL texture, so "context of record for the renderer" and
  // "shared-group root" are deliberately the same window.
  windows.push_back(std::make_unique<italy::AppWindow>(
      "Viewport", 980, 580, nullptr, [&](italy::AppWindow &self) {
        // Apply-button subtlety: rebuildScene() constructs a fresh
        // OptixRenderer (owns the GL texture + CUDA-GL PBO interop), which
        // must happen with Viewport's context current. The Apply button
        // lives in Controls' draw callback (a different context), so it only
        // raises applyRequested; this callback — already running under its
        // own context, per AppWindow::frame() — does the actual rebuild.
        if (applyRequested) {
          rebuildScene(state, renderer, camera);
          prevCamera = camera;
          applyRequested = false;
        }

        double x, y;
        glfwGetCursorPos(self.window(), &x, &y);
        const float dx = static_cast<float>(x - mouse.lastX);
        const float dy = static_cast<float>(y - mouse.lastY);
        mouse.lastX = x;
        mouse.lastY = y;

        // Orbit/pan/zoom used to be gated on !io.WantCaptureMouse, but the
        // viewport image fills the whole ImGui window it's drawn in, so
        // ImGui treats hovering it as mouse capture and that gate never
        // opened — orbiting silently did nothing. Fixed below: the image is
        // wrapped in an InvisibleButton and drag/scroll is driven off that
        // item's own active/hovered state instead, which ImGui's own mouse
        // capture doesn't block. Placed before render()/resetAccumulation()
        // so a drag this frame is reflected in this frame's render, not
        // delayed a frame.

        // Fit the render to the available viewport area rather than drawing
        // it at native pixel size — otherwise it's clipped when the OS
        // window is smaller than the render resolution and leaves dead
        // space when larger. Display-only scale: renderer->width()/height()
        // (the accumulator's actual resolution) are untouched, so this never
        // triggers a rebuild or changes render cost.
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

        // Reserve the layout space and read input state via an
        // InvisibleButton *before* rendering the frame below, so a drag this
        // frame is reflected in this frame's render rather than delayed a
        // frame. The actual image is drawn afterward, on top of this same
        // rect, once render() below has refreshed the texture — see "Drive
        // orbit/pan/zoom" comment above.
        ImGui::InvisibleButton("##viewport_image", imageSize);
        if (ImGui::IsItemActive()) {
          if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) camera.orbit(dx, dy);
          if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) camera.pan(dx, dy);
        }
        if (ImGui::IsItemHovered())
          camera.zoom(ImGui::GetIO().MouseWheel);

        if (state.isSampling) {
          if (cameraChanged(camera, prevCamera)) {
            renderer->resetAccumulation();
            prevCamera = camera;
          }
          renderer->render(camera, state.render);
        } else if (cameraChanged(camera, prevCamera)) {
          // Orbiting while paused resumes sampling from the new pose rather
          // than leaving a stale, wrong-camera accumulation on screen looking
          // current.
          state.isSampling = true;
          renderer->resetAccumulation();
          prevCamera = camera;
          renderer->render(camera, state.render);
        }

        // Render-to-PNG: hold the UI while accumulation catches up to the
        // requested sample count, then write through the same path the
        // scripted hook uses. Deliberately not a separate offline render —
        // the accumulator already converges progressively, so "keep going
        // until subframe N" is the whole feature.
        if (state.exportPending) {
          if (renderer->subframeIndex() >= static_cast<unsigned int>(state.exportSamples)) {
            state.statusLine = writeFramePng(*renderer, state.exportPathBuf)
                                   ? "Wrote " + std::string(state.exportPathBuf)
                                   : "Failed to write " + std::string(state.exportPathBuf);
            state.exportPending = false;
          }
        }

        const char *dumpAfter = std::getenv("ITALY_DUMP_AFTER_SUBFRAME");
        const unsigned int dumpThreshold = dumpAfter ? static_cast<unsigned int>(std::atoi(dumpAfter)) : 128u;
        if (std::getenv("ITALY_DUMP_FRAME") && renderer->subframeIndex() >= dumpThreshold) {
          dumpFrameIfRequested(*renderer, self.window());
          // dumpFrameIfRequested only closes Viewport; close every window so
          // the scripted ITALY_DUMP_FRAME flow still exits the process, same
          // external behavior as when there was only one OS window.
          for (auto &w : windows)
            glfwSetWindowShouldClose(w->window(), GLFW_TRUE);
        }

        // Draw on top of the InvisibleButton's rect, now that render() above
        // has refreshed the texture for this frame.
        ImGui::SetCursorScreenPos(imagePos);
        ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(renderer->glTextureId())), imageSize, ImVec2(0, 1),
                     ImVec2(1, 0));
      }));

  windows.push_back(std::make_unique<italy::AppWindow>(
      "Italy R/S", 420, 800, windows.front()->window(), [&](italy::AppWindow &) {
        ImGui::TextWrapped("%s", state.statusLine.c_str());
        ImGui::Text("Lighting: %s", state.haveEnvironment ? "environment (HDRI)" : "synthetic quad light");
        ImGui::Separator();

        ImGui::Text("Representation");
        // Shown unconditionally (not gated on state.haveMesh like the old
        // Mesh/Voxel/SDF-only version): Gsplat is a distinct load source, so
        // the user needs to be able to pick it before anything is loaded, to
        // know which path field below applies.
        bool apply = ImGui::RadioButton("Mesh", reinterpret_cast<int *>(&state.representation), 0);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Voxel", reinterpret_cast<int *>(&state.representation), 1);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("SDF", reinterpret_cast<int *>(&state.representation), 2);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Gsplat", reinterpret_cast<int *>(&state.representation), 3);
        if (state.representation == Representation::Voxel || state.representation == Representation::Sdf) {
          if (ImGui::InputInt("Resolution", &state.resolution))
            state.resolution = std::clamp(state.resolution, kMinResolution, kMaxResolution);
        }

        if (state.representation == Representation::Gsplat)
          ImGui::InputText("Gsplat .ply path", state.gsplatPathBuf, sizeof(state.gsplatPathBuf));
        else
          ImGui::InputText("GLB path", state.glbPathBuf, sizeof(state.glbPathBuf));

        apply |= ImGui::Button("Load / Apply");

        ImGui::Separator();
        ImGui::Text("HDRI");
        apply |= ImGui::RadioButton("None", reinterpret_cast<int *>(&state.envChoice), 0);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Overcast", reinterpret_cast<int *>(&state.envChoice), 1);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Midnight", reinterpret_cast<int *>(&state.envChoice), 2);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Noon", reinterpret_cast<int *>(&state.envChoice), 3);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Custom", reinterpret_cast<int *>(&state.envChoice), 4);
        ImGui::SameLine();
        apply |= ImGui::RadioButton("Procedural Sky", reinterpret_cast<int *>(&state.envChoice), 5);
        if (state.envChoice == EnvChoice::Custom)
          ImGui::InputText("HDRI path", state.hdriPathBuf, sizeof(state.hdriPathBuf));
        if (state.envChoice == EnvChoice::ProceduralSky) {
          if (ImGui::InputFloat("Turbidity", &state.skyTurbidity)) {
            state.skyTurbidity = std::clamp(state.skyTurbidity, 1.9f, 10.0f);
            apply = true;
          }
          if (ImGui::InputFloat("Sun elevation", &state.skySunElevationDeg)) {
            state.skySunElevationDeg = std::clamp(state.skySunElevationDeg, -10.0f, 90.0f);
            apply = true;
          }
          if (ImGui::InputFloat("Sun azimuth", &state.skySunAzimuthDeg)) {
            state.skySunAzimuthDeg = std::clamp(state.skySunAzimuthDeg, 0.0f, 360.0f);
            apply = true;
          }
        }

        apply |= ImGui::Checkbox("Ground plane", &state.groundPlane);

        // Note: RadioButton edits above already trigger `apply` on click,
        // same as the button — representation/HDRI changes need a full scene
        // rebuild either way, so there's no cheaper "preview" path to offer
        // here. This callback runs under Controls' own ImGui/GL context, so
        // it can't call rebuildScene() itself (see the Apply-button comment
        // in Viewport's draw callback above) — it only raises the flag.
        if (apply)
          applyRequested = true;

        ImGui::Separator();
        if (ImGui::InputFloat("Exposure", &state.render.exposure))
          state.render.exposure = std::clamp(state.render.exposure, 0.1f, 8.0f);
        int spl = static_cast<int>(state.render.samplesPerLaunch);
        if (ImGui::InputInt("Samples/launch", &spl))
          state.render.samplesPerLaunch = static_cast<unsigned int>(std::clamp(spl, 1, 16));
        ImGui::Checkbox("Denoiser", &state.render.denoise);

        // Unlike exposure/tonemap/denoise above, everything from here down
        // changes what is written into the accumulator rather than how it is
        // displayed, so each one has to restart accumulation.
        if (ImGui::InputFloat("Firefly clamp (0 = off)", &state.render.fireflyClamp)) {
          state.render.fireflyClamp = std::clamp(state.render.fireflyClamp, 0.0f, 50.0f);
          renderer->resetAccumulation();
        }

        ImGui::Separator();
        ImGui::Text("Lens");
        // Aperture is a world-space radius, so a fixed field range would be
        // meaningless across scenes that differ in scale by orders of
        // magnitude. Derive it from the scene the camera is actually framing.
        const float apertureMax = renderer->sceneBoundsRadius() * 0.25f;
        if (ImGui::InputFloat("Aperture (0 = pinhole)", &state.render.aperture, 0.0f, 0.0f, "%.4f")) {
          state.render.aperture = std::clamp(state.render.aperture, 0.0f, apertureMax);
          renderer->resetAccumulation();
        }
        if (ImGui::InputFloat("Focus distance (0 = orbit target)", &state.render.focusDistance, 0.0f, 0.0f, "%.3f")) {
          state.render.focusDistance = std::clamp(state.render.focusDistance, 0.0f, renderer->sceneBoundsRadius() * 8.0f);
          renderer->resetAccumulation();
        }

        if (state.haveEnvironment) {
          ImGui::Separator();
          float envRotationDeg = glm::degrees(state.render.envRotation);
          if (ImGui::InputFloat("Environment rotation (deg)", &envRotationDeg)) {
            envRotationDeg = std::clamp(envRotationDeg, 0.0f, 360.0f);
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
          state.renderWidth = std::clamp(dims[0], 64, 8192);
          state.renderHeight = std::clamp(dims[1], 64, 8192);
        }
        ImGui::SameLine();
        if (ImGui::Button("Resize"))
          applyRequested = true; // same all-or-nothing rebuild as everything else

        ImGui::InputText("PNG path", state.exportPathBuf, sizeof(state.exportPathBuf));
        if (ImGui::InputInt("Export samples", &state.exportSamples))
          state.exportSamples = std::clamp(state.exportSamples, 16, 4096);
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
        ImGui::Text("Drag left-click to orbit, middle-click to pan, scroll to zoom.");
      }));

  // App quits only once all windows are closed. Destroying the shared-context
  // root window (Viewport) doesn't invalidate the GL object namespace for
  // windows that shared with it (GL spec guarantee), so Controls stays fully
  // functional — just inert, with nothing driving the renderer — if Viewport
  // is closed first. Accepted v1 behavior: no "reopen a closed panel"
  // mechanism exists yet.
  while (!windows.empty()) {
    glfwPollEvents();
    for (auto &w : windows)
      if (!w->shouldClose())
        w->frame();
    std::erase_if(windows, [](const std::unique_ptr<italy::AppWindow> &w) { return w->shouldClose(); });
  }

  glfwTerminate();
  return 0;
}
