// Manually-resolved OpenGL buffer-object entry points (GL_ARB_vertex_buffer_
// object, core since GL 1.5) needed for the CUDA-GL interop PBO. GLFW's
// default include only gets us the legacy GL 1.1 ABI (glViewport, glClear,
// glTex(Sub)Image2D — those we call directly, no loader needed); PBO
// functions aren't part of that ABI on any platform.
//
// monkey-boy: hand-resolving 4 function pointers instead of pulling in a full
// loader (GLAD/GLEW) — upgrade to one if later phases need much more of
// modern GL than this.
#pragma once

#include <GLFW/glfw3.h>

#ifndef GL_PIXEL_UNPACK_BUFFER
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif

namespace italy {

using PFNGLGENBUFFERS = void (*)(GLsizei, GLuint *);
using PFNGLBINDBUFFER = void (*)(GLenum, GLuint);
using PFNGLBUFFERDATA = void (*)(GLenum, ptrdiff_t, const void *, GLenum);
using PFNGLDELETEBUFFERS = void (*)(GLsizei, const GLuint *);

struct GLBufferFns {
  PFNGLGENBUFFERS glGenBuffers = nullptr;
  PFNGLBINDBUFFER glBindBuffer = nullptr;
  PFNGLBUFFERDATA glBufferData = nullptr;
  PFNGLDELETEBUFFERS glDeleteBuffers = nullptr;

  static const GLBufferFns &get() {
    static const GLBufferFns fns = [] {
      GLBufferFns f;
      f.glGenBuffers = reinterpret_cast<PFNGLGENBUFFERS>(glfwGetProcAddress("glGenBuffers"));
      f.glBindBuffer = reinterpret_cast<PFNGLBINDBUFFER>(glfwGetProcAddress("glBindBuffer"));
      f.glBufferData = reinterpret_cast<PFNGLBUFFERDATA>(glfwGetProcAddress("glBufferData"));
      f.glDeleteBuffers = reinterpret_cast<PFNGLDELETEBUFFERS>(glfwGetProcAddress("glDeleteBuffers"));
      return f;
    }();
    return fns;
  }
};

} // namespace italy
