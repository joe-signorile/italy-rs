// Own tiny module/pipeline, separate from pathtracer_params.h/pathtracer.cu
// — baking is a one-shot preprocess with a totally different program shape
// (3D launch over grid cells, no path tracing) and gains nothing from
// sharing the interactive renderer's pipeline/SBT layout.
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

struct BakeParams {
  OptixTraversableHandle meshHandle; // the mesh's own GAS — NOT an IAS with a light quad in it
  float3 gridOrigin;
  float voxelSize;
  unsigned int nx, ny, nz;
  float *output; // nx*ny*nz signed distances, device buffer
};

struct BakeHitGroupData {};
struct BakeMissData {};
struct BakeRayGenData {};
