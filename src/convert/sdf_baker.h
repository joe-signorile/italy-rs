#pragma once

#include "convert/sdf_grid.h"
#include "io/mesh_asset.h"

namespace italy {

SdfGrid bakeSdf(const MeshAsset &mesh, int resolution);

} // namespace italy
