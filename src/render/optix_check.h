// Tiny throw-on-error wrappers for OptiX/CUDA calls. NVIDIA's own SDK ships
// an equivalent (SDK/sutil/Exception.h) but it pulls in glad for its GL_CHECK
// macro; we don't use glad (see gl_ext.h), and we only need the OptiX/CUDA
// half, so this is a smaller self-contained copy of just that half.
#pragma once

#include <cuda_runtime_api.h>
#include <optix.h>

#include <sstream>
#include <stdexcept>

namespace italy {

inline void optixCheck(OptixResult res, const char *call, const char *file, unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::ostringstream ss;
    ss << optixGetErrorName(res) << ": call '" << call << "' failed (" << file << ":" << line << ")";
    throw std::runtime_error(ss.str());
  }
}

inline void optixCheckLog(OptixResult res, const char *log, size_t logCapacity, size_t logSize, const char *call,
                           const char *file, unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::ostringstream ss;
    ss << optixGetErrorName(res) << ": call '" << call << "' failed (" << file << ":" << line << ")\nLog:\n"
       << log << (logSize > logCapacity ? "<TRUNCATED>" : "");
    throw std::runtime_error(ss.str());
  }
}

inline void cudaCheck(cudaError_t err, const char *call, const char *file, unsigned int line) {
  if (err != cudaSuccess) {
    std::ostringstream ss;
    ss << "CUDA call '" << call << "' failed: " << cudaGetErrorString(err) << " (" << file << ":" << line << ")";
    throw std::runtime_error(ss.str());
  }
}

} // namespace italy

#define OPTIX_CHECK(call) ::italy::optixCheck((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) ::italy::cudaCheck((call), #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)                                                                                        \
  do {                                                                                                               \
    char LOG[2048];                                                                                                  \
    size_t LOG_SIZE = sizeof(LOG);                                                                                   \
    ::italy::optixCheckLog((call), LOG, sizeof(LOG), LOG_SIZE, #call, __FILE__, __LINE__);                           \
  } while (false)
