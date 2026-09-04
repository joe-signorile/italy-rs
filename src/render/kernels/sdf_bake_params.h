// Own tiny module/pipeline separate from pathtracer_params.h: baking is a one-shot 3D grid launch with a different program shape and gains nothing from sharing the renderer's pipeline/SBT.
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

struct BakeParams {
  OptixTraversableHandle meshHandle;
  float3 gridOrigin;
  float voxelSize;
  unsigned int nx, ny, nz;
  float *output;
};

struct BakeHitGroupData {};
struct BakeMissData {};
struct BakeRayGenData {};
