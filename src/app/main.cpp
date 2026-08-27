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
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "convert/sdf_baker.h"
#include "convert/voxelize.h"
#include "core/orbit_camera.h"
#include "io/gltf_loader.h"
#include "render/environment.h"
#include "render/optix_renderer.h"

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

// Debug/verification aid: set ITALY_DUMP_FRAME=path.png to write out the
// accumulated render and exit — lets a render-correctness check happen
// without eyeballing a live window.
void dumpFrameIfRequested(const italy::OptixRenderer &renderer, GLFWwindow *window) {
  const char *path = std::getenv("ITALY_DUMP_FRAME");
  if (!path)
    return;
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
  stbi_write_png(path, w, h, 4, flipped.data(), w * 4);
  std::fprintf(stderr, "italy: wrote debug frame to %s\n", path);
  glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// Explicit `int` underlying type: ImGui::RadioButton takes an `int*`, and
// the UI loop below reinterpret_casts &state.representation/&state.envChoice
// to one — relying on the (already-guaranteed-by-the-standard-for-scoped-
// enums-with-no-explicit-type) size/alignment match to work is the kind of
// thing worth spelling out rather than leaving implicit.
enum class Representation : int { Mesh, Voxel, Sdf };
enum class EnvChoice : int { None, Overcast, Midnight, Noon, Custom };

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
  italy::EnvironmentMap environment;
  bool haveEnvironment = false;

  char glbPathBuf[512] = "";
  Representation representation = Representation::Mesh;
  int resolution = 64;
  EnvChoice envChoice = EnvChoice::None;
  char hdriPathBuf[512] = "";

  float exposure = 1.0f;
  int samplesPerLaunch = 1;

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
  if (state.glbPathBuf[0] != '\0') {
    std::string err;
    if (italy::loadGlb(state.glbPathBuf, state.meshAsset, err)) {
      state.haveMesh = true;
      state.statusLine = "Loaded " + std::string(state.glbPathBuf) + " (" +
                          std::to_string(state.meshAsset.positions.size() / 3) + " tris" +
                          (state.meshAsset.hasBaseColorTexture ? ", textured)" : ")");
    } else {
      state.statusLine = "Failed to load " + std::string(state.glbPathBuf) + ": " + err;
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
  }

  italy::SceneSource source;
  if (haveSdf)
    source.sdf = &state.sdfGrid;
  else if (haveVoxels)
    source.voxels = &state.voxelGrid;
  else if (state.haveMesh)
    source.mesh = &state.meshAsset;
  if (state.haveEnvironment)
    source.environment = &state.environment;

  renderer = std::make_unique<italy::OptixRenderer>(960, 540, source);
  if (state.haveMesh)
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
    if (key == "--voxel") {
      state.representation = Representation::Voxel;
      if (!value.empty())
        state.resolution = std::atoi(value.c_str());
    } else if (key == "--sdf") {
      state.representation = Representation::Sdf;
      if (!value.empty())
        state.resolution = std::atoi(value.c_str());
    } else if (key == "--env") {
      if (value == "overcast")
        state.envChoice = EnvChoice::Overcast;
      else if (value == "midnight")
        state.envChoice = EnvChoice::Midnight;
      else if (value == "noon")
        state.envChoice = EnvChoice::Noon;
      else
        std::fprintf(stderr, "italy: unknown --env preset '%s' (try overcast, midnight, noon)\n", value.c_str());
    } else if (key == "--hdri") {
      state.envChoice = EnvChoice::Custom;
      std::snprintf(state.hdriPathBuf, sizeof(state.hdriPathBuf), "%s", value.c_str());
    }
  }

  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(1280, 800, "italy", nullptr, nullptr);
  if (!window) {
    std::fprintf(stderr, "glfwCreateWindow failed\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 410");

  italy::OrbitCamera camera;
  MouseState mouse;

  std::unique_ptr<italy::OptixRenderer> renderer;
  rebuildScene(state, renderer, camera);
  italy::OrbitCamera prevCamera = camera;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    double x, y;
    glfwGetCursorPos(window, &x, &y);
    const float dx = static_cast<float>(x - mouse.lastX);
    const float dy = static_cast<float>(y - mouse.lastY);
    mouse.lastX = x;
    mouse.lastY = y;

    const ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
      const bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      const bool middleDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
      if (leftDown) camera.orbit(dx, dy);
      if (middleDown) camera.pan(dx, dy);
      camera.zoom(io.MouseWheel);
    }

    if (cameraChanged(camera, prevCamera)) {
      renderer->resetAccumulation();
      prevCamera = camera;
    }
    renderer->render(camera, static_cast<unsigned int>(state.samplesPerLaunch), state.exposure);

    if (std::getenv("ITALY_DUMP_FRAME") && renderer->subframeIndex() >= 128)
      dumpFrameIfRequested(*renderer, window);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("italy");
    ImGui::TextWrapped("%s", state.statusLine.c_str());
    ImGui::Text("Lighting: %s", state.haveEnvironment ? "environment (HDRI)" : "synthetic quad light");
    ImGui::Separator();

    ImGui::InputText("GLB path", state.glbPathBuf, sizeof(state.glbPathBuf));

    bool apply = ImGui::Button("Load / Apply");

    if (state.haveMesh) {
      ImGui::Text("Representation");
      apply |= ImGui::RadioButton("Mesh", reinterpret_cast<int *>(&state.representation), 0);
      ImGui::SameLine();
      apply |= ImGui::RadioButton("Voxel", reinterpret_cast<int *>(&state.representation), 1);
      ImGui::SameLine();
      apply |= ImGui::RadioButton("SDF", reinterpret_cast<int *>(&state.representation), 2);
      if (state.representation != Representation::Mesh)
        ImGui::SliderInt("Resolution", &state.resolution, 8, 256);
    }

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
    if (state.envChoice == EnvChoice::Custom)
      ImGui::InputText("HDRI path", state.hdriPathBuf, sizeof(state.hdriPathBuf));

    // Note: RadioButton edits above already trigger `apply` on click, same
    // as the button — representation/HDRI changes need a full scene rebuild
    // either way, so there's no cheaper "preview" path to offer here.
    if (apply)
      rebuildScene(state, renderer, camera);

    ImGui::Separator();
    ImGui::SliderFloat("Exposure", &state.exposure, 0.1f, 8.0f);
    ImGui::SliderInt("Samples/launch", &state.samplesPerLaunch, 1, 16);

    ImGui::Separator();
    const glm::vec3 pos = camera.position();
    ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
    ImGui::Text("Subframe: %u", renderer->subframeIndex());
    ImGui::Text("Drag left-click to orbit, middle-click to pan, scroll to zoom.");
    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(980, 580), ImGuiCond_FirstUseEver);
    ImGui::Begin("Viewport");
    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(renderer->glTextureId())),
                 ImVec2(static_cast<float>(renderer->width()), static_cast<float>(renderer->height())), ImVec2(0, 1),
                 ImVec2(1, 0));
    ImGui::End();

    ImGui::Render();
    int displayW, displayH;
    glfwGetFramebufferSize(window, &displayW, &displayH);
    glViewport(0, 0, displayW, displayH);
    glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
