#include "app/app_window.h"

#include <cstdio>
#include <cstdlib>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace italy {

namespace {

// Dear ImGui's GLFW backend resolves *which* ImGuiContext a callback should
// touch via the process-global "current context", not via the GLFWwindow*
// the callback receives — fine with one context, wrong the instant a second
// one exists, since glfwPollEvents() is one global call that can dispatch
// events for every open window before we regain control. The fix used
// throughout this file: install_callbacks=false in the constructor below,
// then these 7 trampolines (one per callback ImGui_ImplGlfw would otherwise
// install itself) that pull `this` back out via the window user pointer,
// make that window's ImGuiContext current, and forward to the real
// ImGui_ImplGlfw_*Callback free function (public entry points exposed by
// imgui_impl_glfw.h exactly for this case).
AppWindow *ownerOf(GLFWwindow *window) { return static_cast<AppWindow *>(glfwGetWindowUserPointer(window)); }

void cursorPosTrampoline(GLFWwindow *window, double x, double y) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_CursorPosCallback(window, x, y);
}

void mouseButtonTrampoline(GLFWwindow *window, int button, int action, int mods) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void scrollTrampoline(GLFWwindow *window, double xoffset, double yoffset) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

void keyTrampoline(GLFWwindow *window, int key, int scancode, int action, int mods) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void charTrampoline(GLFWwindow *window, unsigned int c) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_CharCallback(window, c);
}

void windowFocusTrampoline(GLFWwindow *window, int focused) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_WindowFocusCallback(window, focused);
}

void cursorEnterTrampoline(GLFWwindow *window, int entered) {
  ImGui::SetCurrentContext(ownerOf(window)->context());
  ImGui_ImplGlfw_CursorEnterCallback(window, entered);
}

} // namespace

AppWindow::AppWindow(const char *title, int width, int height, GLFWwindow *shareContext, DrawFn draw)
    : title_(title), draw_(std::move(draw)) {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(width, height, title, nullptr, shareContext);
  if (!window_) {
    std::fprintf(stderr, "glfwCreateWindow failed for '%s'\n", title);
    std::exit(1);
  }

  glfwSetWindowUserPointer(window_, this);

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  // CreateContext() internally restores whatever ImGuiContext was current
  // before the call, so it does NOT leave imguiContext_ current — the
  // explicit SetCurrentContext below is required before touching anything
  // ImGui-side (StyleColorsDark, GetIO(), the backend Init calls).
  imguiContext_ = ImGui::CreateContext();
  ImGui::SetCurrentContext(imguiContext_);

  ImGui::StyleColorsDark();
  // monkey-boy: window position/size are fixed defaults, not persisted
  // across restarts — an imgui.ini per OS window has no meaningful
  // intra-window layout left to save now that the OS window IS the panel.
  // Revisit if users find re-arranging windows every launch annoying.
  ImGui::GetIO().IniFilename = nullptr;

  ImGui_ImplGlfw_InitForOpenGL(window_, /*install_callbacks=*/false);
  glfwSetCursorPosCallback(window_, cursorPosTrampoline);
  glfwSetMouseButtonCallback(window_, mouseButtonTrampoline);
  glfwSetScrollCallback(window_, scrollTrampoline);
  glfwSetKeyCallback(window_, keyTrampoline);
  glfwSetCharCallback(window_, charTrampoline);
  glfwSetWindowFocusCallback(window_, windowFocusTrampoline);
  glfwSetCursorEnterCallback(window_, cursorEnterTrampoline);

  ImGui_ImplOpenGL3_Init("#version 410");
}

AppWindow::~AppWindow() {
  ImGui::SetCurrentContext(imguiContext_);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext(imguiContext_);
  glfwDestroyWindow(window_);
}

bool AppWindow::shouldClose() const { return glfwWindowShouldClose(window_); }

void AppWindow::frame() {
  ImGui::SetCurrentContext(imguiContext_);
  glfwMakeContextCurrent(window_);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  const ImGuiIO &io = ImGui::GetIO();
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(io.DisplaySize);
  ImGui::Begin(title_.c_str(), nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
  draw_(*this);
  ImGui::End();

  ImGui::Render();
  int displayW, displayH;
  glfwGetFramebufferSize(window_, &displayW, &displayH);
  glViewport(0, 0, displayW, displayH);
  glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  glfwSwapBuffers(window_);
}

} // namespace italy
