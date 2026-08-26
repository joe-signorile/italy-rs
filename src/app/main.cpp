// Phase 2: window + Dear ImGui shell + orbit camera + OptiX path-traced
// viewport. The renderer accumulates progressively while the camera is
// still and resets whenever it moves.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "convert/voxelize.h"
#include "core/orbit_camera.h"
#include "io/gltf_loader.h"
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

// Debug/verification aid: set ITALY_DUMP_FRAME=path.png and (optionally)
// ITALY_DUMP_AFTER_SUBFRAME=N to write out the accumulated render and exit —
// lets a render-correctness check happen without eyeballing a live window.
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

} // namespace

int main(int argc, char **argv) {
  // Phase 3: optional GLB path on the command line loads and renders that
  // mesh (as textured triangles) instead of the phase-2 fixed test scene.
  // A file-picker UI comes with phase 9; a CLI arg is the smallest thing
  // that lets ingestion be exercised/verified now.
  italy::MeshAsset meshAsset;
  bool haveMesh = false;
  if (argc > 1) {
    std::string err;
    if (italy::loadGlb(argv[1], meshAsset, err)) {
      haveMesh = true;
      std::fprintf(stderr, "italy: loaded %s (%zu triangles%s)\n", argv[1], meshAsset.positions.size() / 3,
                   meshAsset.hasBaseColorTexture ? ", textured" : "");
    } else {
      std::fprintf(stderr, "italy: failed to load %s: %s\n", argv[1], err.c_str());
    }
  }

  // Phase 4: `--voxel[=N]` resamples the loaded mesh into a voxel grid
  // (N cells along its longest bounding-box axis, default 64) and renders
  // that instead. Same "CLI flag, not UI yet" reasoning as GLB loading.
  italy::VoxelGrid voxelGrid;
  bool haveVoxels = false;
  if (haveMesh) {
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("--voxel", 0) == 0) {
        int resolution = 64;
        const size_t eq = arg.find('=');
        if (eq != std::string::npos)
          resolution = std::atoi(arg.c_str() + eq + 1);
        voxelGrid = italy::voxelizeMesh(meshAsset, resolution);
        haveVoxels = true;
        std::fprintf(stderr, "italy: voxelized at resolution %d -> %zu occupied cells\n", resolution,
                     voxelGrid.cells.size());
      }
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

  // Fixed render resolution for phase 2 bring-up — dynamic viewport resize
  // (reallocating the accum buffer/PBO/texture) lands with the UI controls
  // phase.
  italy::SceneSource source;
  if (haveVoxels)
    source.voxels = &voxelGrid;
  else if (haveMesh)
    source.mesh = &meshAsset;
  italy::OptixRenderer renderer(960, 540, source);
  if (haveMesh)
    camera.frame(renderer.sceneBoundsCenter(), renderer.sceneBoundsRadius());
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
      renderer.resetAccumulation();
      prevCamera = camera;
    }
    renderer.render(camera);

    if (std::getenv("ITALY_DUMP_FRAME") && renderer.subframeIndex() >= 128)
      dumpFrameIfRequested(renderer, window);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("italy");
    if (haveVoxels)
      ImGui::Text("Loaded: %s -> voxelized (%zu occupied cells)", argv[1], voxelGrid.cells.size());
    else if (haveMesh)
      ImGui::Text("Loaded: %s (%zu tris%s)", argv[1], meshAsset.positions.size() / 3,
                   meshAsset.hasBaseColorTexture ? ", textured" : "");
    else
      ImGui::Text("No GLB given on the command line — showing the built-in test scene.");
    ImGui::Separator();
    const glm::vec3 pos = camera.position();
    ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
    ImGui::Text("Subframe: %u", renderer.subframeIndex());
    ImGui::Text("Drag left-click to orbit, middle-click to pan, scroll to zoom.");
    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(980, 580), ImGuiCond_FirstUseEver);
    ImGui::Begin("Viewport");
    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(renderer.glTextureId())),
                 ImVec2(static_cast<float>(renderer.width()), static_cast<float>(renderer.height())), ImVec2(0, 1),
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
