#pragma once

#include "convert/voxel_grid.h"
#include "io/mesh_asset.h"

namespace italy {

VoxelGrid voxelizeMesh(const MeshAsset &mesh, int resolution);

} // namespace italy
