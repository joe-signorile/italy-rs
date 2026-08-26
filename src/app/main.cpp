// Phase 1: window + Dear ImGui shell + orbit camera input. No rendering yet
// (that starts in phase 2 once the OptiX SDK is available) — this just
// proves the windowing/UI/camera plumbing works.

#include <cstdio>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "core/orbit_camera.h"

namespace {

// Polled once per frame rather than via GLFW callbacks — simpler than wiring
// a user-pointer + callback for what's just "delta since last frame".
struct MouseState {
  double lastX = 0.0;
  double lastY = 0.0;
};

} // namespace

int main() {
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

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("italy");
    ImGui::Text("Phase 1 shell: window + ImGui + orbit camera.");
    ImGui::Separator();
    const glm::vec3 pos = camera.position();
    ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
    ImGui::Text("Drag left-click to orbit, middle-click to pan, scroll to zoom.");
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
