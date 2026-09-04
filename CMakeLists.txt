cmake_minimum_required(VERSION 3.24)
project(italy-rs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(agent_filemap ALL
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_filemap.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Regenerating docs/agent/filemap.toon"
  VERBATIM
)

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
            "${CMAKE_CURRENT_SOURCE_DIR}/src/render/kernels/device_onb.h"
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

add_executable(italy-rs
  src/app/main.cpp
  src/app/app_window.cpp
  src/core/orbit_camera.cpp
  src/render/optix_renderer.cpp
  src/io/gltf_loader.cpp
  src/io/mesh_weld.cpp
  src/io/mesh_validate.cpp
  src/io/mesh_primitive_classify.cpp
  src/io/gsplat_ply_loader.cpp
  src/convert/voxelize.cpp
  src/convert/sdf_baker.cpp
  src/convert/sdf_procedural.cpp
  src/render/environment.cpp
  src/render/procedural_sky.cpp
)
add_dependencies(italy-rs italy_optixir_pathtracer italy_optixir_sdf_bake)
target_include_directories(italy-rs PRIVATE src ${CMAKE_BINARY_DIR}/generated ${OPTIX_INCLUDE_DIR} ${OPTIX_ROOT}/SDK)
target_link_libraries(italy-rs PRIVATE glfw imgui glm::glm OpenGL::GL CUDA::cudart stb tinygltf)

enable_testing()
add_executable(voxelize_test tests/voxelize_test.cpp src/convert/voxelize.cpp)
target_include_directories(voxelize_test PRIVATE src)
target_link_libraries(voxelize_test PRIVATE glm::glm)
add_test(NAME voxelize_test COMMAND voxelize_test)

add_executable(environment_cdf_test tests/environment_cdf_test.cpp tests/stb_image_impl.cpp src/render/environment.cpp)
target_include_directories(environment_cdf_test PRIVATE src)
target_link_libraries(environment_cdf_test PRIVATE stb)
add_test(NAME environment_cdf_test COMMAND environment_cdf_test)

add_executable(sdf_sign_test tests/sdf_sign_test.cpp src/convert/sdf_baker.cpp)
add_dependencies(sdf_sign_test italy_optixir_sdf_bake)
target_include_directories(sdf_sign_test PRIVATE src ${CMAKE_BINARY_DIR}/generated ${OPTIX_INCLUDE_DIR} ${OPTIX_ROOT}/SDK)
target_link_libraries(sdf_sign_test PRIVATE glm::glm CUDA::cudart)
add_test(NAME sdf_sign_test COMMAND sdf_sign_test)

add_executable(mesh_validate_test tests/mesh_validate_test.cpp src/io/mesh_validate.cpp src/io/mesh_weld.cpp)
target_include_directories(mesh_validate_test PRIVATE src)
target_link_libraries(mesh_validate_test PRIVATE glm::glm)
add_test(NAME mesh_validate_test COMMAND mesh_validate_test)

add_executable(mesh_primitive_classify_test tests/mesh_primitive_classify_test.cpp
                                             src/io/mesh_primitive_classify.cpp src/io/mesh_weld.cpp)
target_include_directories(mesh_primitive_classify_test PRIVATE src)
target_link_libraries(mesh_primitive_classify_test PRIVATE glm::glm)
add_test(NAME mesh_primitive_classify_test COMMAND mesh_primitive_classify_test)

add_executable(procedural_sky_test tests/procedural_sky_test.cpp tests/stb_image_impl.cpp
                                    src/render/procedural_sky.cpp src/render/environment.cpp)
target_include_directories(procedural_sky_test PRIVATE src)
target_link_libraries(procedural_sky_test PRIVATE glm::glm stb)
add_test(NAME procedural_sky_test COMMAND procedural_sky_test)

add_executable(gsplat_ply_loader_test tests/gsplat_ply_loader_test.cpp src/io/gsplat_ply_loader.cpp)
target_include_directories(gsplat_ply_loader_test PRIVATE src)
target_link_libraries(gsplat_ply_loader_test PRIVATE glm::glm)
add_test(NAME gsplat_ply_loader_test COMMAND gsplat_ply_loader_test)
