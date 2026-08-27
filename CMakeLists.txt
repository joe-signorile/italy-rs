cmake_minimum_required(VERSION 3.24)
project(italy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)

include(FetchContent)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.6
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG master)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

FetchContent_Declare(tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG v2.9.3
  GIT_SHALLOW TRUE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)
# tinygltf's own CMakeLists (TINYGLTF_HEADER_ONLY) already defines the
# `tinygltf` INTERFACE target, bundling its own stb_image/json.hpp copies —
# nothing more to declare here.

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

find_package(CUDAToolkit REQUIRED)

set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "OptiX SDK install (must contain include/optix.h and SDK/sutil)")
find_path(OPTIX_INCLUDE_DIR optix.h PATHS "${OPTIX_ROOT}/include")
if(NOT OPTIX_INCLUDE_DIR)
  message(FATAL_ERROR "OptiX headers not found. Set OPTIX_ROOT (env var or -DOPTIX_ROOT=...) to the SDK install — see README.md.")
endif()
if(NOT EXISTS "${OPTIX_ROOT}/SDK/sutil/vec_math.h")
  message(FATAL_ERROR "OPTIX_ROOT is set but ${OPTIX_ROOT}/SDK/sutil is missing — need the SDK's sample-support headers (vec_math.h, random.h), not just include/.")
endif()

set(ITALY_CUDA_ARCH "89" CACHE STRING "Target GPU compute capability for the OptiX device kernel (89 = Ada/RTX 40-series)")

# OptiX device programs are consumed at runtime by optixModuleCreate as
# OptiX-IR text, not linked like normal objects — a direct nvcc invocation is
# simpler than fighting CMake's native CUDA_ARCHITECTURES object pipeline for
# these files. The result is read from disk at startup (see
# render_config.h.in) rather than embedded via bin2c: this is a dev tool with
# a fixed, known build-tree layout, so that's a fine simplification for now.
function(italy_compile_optixir cu_file out_var)
  get_filename_component(name ${cu_file} NAME_WE)
  set(output "${CMAKE_BINARY_DIR}/optix_programs/${name}.optixir")
  add_custom_command(
    OUTPUT ${output}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/optix_programs"
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            --optix-ir
            -arch=sm_${ITALY_CUDA_ARCH}
            -std=c++17
            -I "${OPTIX_INCLUDE_DIR}"
            -I "${OPTIX_ROOT}/SDK"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -o ${output}
            ${cu_file}
    DEPENDS ${cu_file} "${CMAKE_CURRENT_SOURCE_DIR}/src/render/kernels/pathtracer_params.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/render/kernels/sdf_bake_params.h"
    COMMENT "Compiling OptiX device program ${name}.cu -> OptiX-IR"
    VERBATIM
  )
  add_custom_target(italy_optixir_${name} DEPENDS ${output})
  set(${out_var} ${output} PARENT_SCOPE)
endfunction()

italy_compile_optixir("${CMAKE_CURRENT_SOURCE_DIR}/src/render/kernels/pathtracer.cu" PATHTRACER_OPTIXIR)
italy_compile_optixir("${CMAKE_CURRENT_SOURCE_DIR}/src/render/kernels/sdf_bake.cu" SDF_BAKE_OPTIXIR)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/render/render_config.h.in
  ${CMAKE_BINARY_DIR}/generated/render_config.h
  @ONLY
)

add_executable(italy
  src/app/main.cpp
  src/core/orbit_camera.cpp
  src/render/optix_renderer.cpp
  src/io/gltf_loader.cpp
  src/convert/voxelize.cpp
  src/convert/sdf_baker.cpp
)
add_dependencies(italy italy_optixir_pathtracer italy_optixir_sdf_bake)
target_include_directories(italy PRIVATE src ${CMAKE_BINARY_DIR}/generated ${OPTIX_INCLUDE_DIR} ${OPTIX_ROOT}/SDK)
target_link_libraries(italy PRIVATE glfw imgui glm::glm OpenGL::GL CUDA::cudart stb tinygltf)

enable_testing()
add_executable(voxelize_test tests/voxelize_test.cpp src/convert/voxelize.cpp)
target_include_directories(voxelize_test PRIVATE src)
target_link_libraries(voxelize_test PRIVATE glm::glm)
add_test(NAME voxelize_test COMMAND voxelize_test)
