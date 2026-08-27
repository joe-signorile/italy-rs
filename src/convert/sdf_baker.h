#pragma once

#include "convert/sdf_grid.h"
#include "io/mesh_asset.h"

namespace italy {

// Bakes a dense signed-distance grid from a mesh using its own OptiX BVH:
// unsigned distance is the minimum hit distance over a fixed set of sample
// directions per cell (no native "closest point" query exists in OptiX —
// this is the standard ray-tracing-hardware substitute), sign is ray-parity
// counting along a fixed direction. The mesh must be closed/watertight for
// the sign to come out correct; an open mesh will produce plausible-looking
// but not reliably signed results.
//
// Builds and tears down its own temporary OptiX context/pipeline — see
// sdf_baker.cpp for why that's an acceptable simplification over threading
// a shared context through from the caller.
SdfGrid bakeSdf(const MeshAsset &mesh, int resolution);

} // namespace italy
