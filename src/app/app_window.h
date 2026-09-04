#pragma once

#include <functional>
#include <string>

struct GLFWwindow;
struct ImGuiContext;

namespace italy {

class AppWindow {
public:
  using DrawFn = std::function<void(AppWindow &)>;

  AppWindow(const char *title, int width, int height, GLFWwindow *shareContext, DrawFn draw);
  ~AppWindow();

  AppWindow(const AppWindow &) = delete;
  AppWindow &operator=(const AppWindow &) = delete;

  bool shouldClose() const;

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
