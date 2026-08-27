#pragma once

// One independent OS window: its own GLFWwindow + its own ImGuiContext,
// sharing a GL context/object-namespace with whichever window it was told
// to share with (see AppWindow's constructor comment in app_window.cpp).
// Introduced when the app moved from one OS window hosting two floating
// ImGui panels to one OS window per panel (see the multi-window plan) —
// this class is the unit that gets instantiated once per panel.
//
// Dear ImGui's GLFW backend was never designed for multiple ImGuiContexts
// in one process: its installed callbacks resolve backend state via the
// process-global *current* context, not via the GLFWwindow the callback
// fires for. AppWindow works around that by installing manual trampoline
// callbacks (see app_window.cpp) instead of using
// ImGui_ImplGlfw_InitForOpenGL's own install_callbacks=true path.

#include <functional>
#include <string>

struct GLFWwindow;
struct ImGuiContext;

namespace italy {

class AppWindow {
public:
  // Called once per frame, between ImGui::NewFrame()/Begin() and End()/
  // Render() — i.e. exactly where a panel's widget body runs today.
  using DrawFn = std::function<void(AppWindow &)>;

  // shareContext: nullptr for the first/root window (it defines the shared
  // GL object namespace); pass that window's window() for every subsequent
  // one so textures/buffers created against one are visible from the other.
  AppWindow(const char *title, int width, int height, GLFWwindow *shareContext, DrawFn draw);
  ~AppWindow();

  AppWindow(const AppWindow &) = delete;
  AppWindow &operator=(const AppWindow &) = delete;

  bool shouldClose() const;

  // MakeContextCurrent -> ImGui NewFrame -> draw_(*this) -> ImGui::Render ->
  // GL clear/present -> SwapBuffers. Call once per live window per app frame,
  // after a single global glfwPollEvents().
  void frame();

  GLFWwindow *window() const { return window_; }
  ImGuiContext *context() const { return imguiContext_; }

private:
  std::string title_;
  GLFWwindow *window_ = nullptr;
  ImGuiContext *imguiContext_ = nullptr;
  DrawFn draw_;
};

} // namespace italy
