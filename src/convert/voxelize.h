#pragma once

#include "convert/voxel_grid.h"
#include "io/mesh_asset.h"

namespace italy {

// Surface voxelization: a cell is occupied iff some triangle actually
// overlaps its box (exact SAT test, not just bounding-box overlap — a
// bbox-only test would over-thicken thin/diagonal geometry). `resolution` is
// the cell count along the mesh's longest bounding-box axis; other axes get
// however many whole cells of that same size fit.
VoxelGrid voxelizeMesh(const MeshAsset &mesh, int resolution);

} // namespace italy
