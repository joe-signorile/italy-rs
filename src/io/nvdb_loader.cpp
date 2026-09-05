#include "io/nvdb_loader.h"

#include <cstring>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreatePrimitives.h>

namespace italy {

NvdbVolume buildProceduralFogSphereVolume(glm::vec3 center, float radius, float voxelSize, glm::vec3 sigmaT,
                                           glm::vec3 scatterAlbedo, float g, float densityScale) {
  auto handle = nanovdb::tools::createFogVolumeSphere<float>(
      static_cast<double>(radius), nanovdb::Vec3d(center.x, center.y, center.z), static_cast<double>(voxelSize));

  const nanovdb::FloatGrid *grid = handle.grid<float>();

  NvdbVolume volume;
  volume.gridBlob.resize(handle.bufferSize());
  std::memcpy(volume.gridBlob.data(), handle.data(), handle.bufferSize());

  const auto &bbox = grid->worldBBox();
  volume.boundsMin = glm::vec3(static_cast<float>(bbox.min()[0]), static_cast<float>(bbox.min()[1]),
                                static_cast<float>(bbox.min()[2]));
  volume.boundsMax = glm::vec3(static_cast<float>(bbox.max()[0]), static_cast<float>(bbox.max()[1]),
                                static_cast<float>(bbox.max()[2]));
  volume.maxDensity = grid->tree().root().maximum();
  volume.sigmaT = sigmaT;
  volume.scatterAlbedo = scatterAlbedo;
  volume.g = g;
  volume.densityScale = densityScale;
  return volume;
}

} // namespace italy
