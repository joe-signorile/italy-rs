#pragma once

#include <string>

#include "io/gsplat_asset.h"

namespace italy {

// Loads a standard 3D Gaussian Splatting `.ply` (the convention used by the
// original INRIA reference implementation and shared by most exporters,
// including s-rank's triposplat) into a GsplatAsset. Supports both `ascii`
// and `binary_little_endian` PLY formats.
//
// Only the vertex properties this renderer needs are read: x,y,z, scale_0-2,
// rot_0-3, opacity, f_dc_0-2. Everything else (normals, f_rest_* / higher-
// order SH bands) is skipped using the header's own property list, so files
// with extra properties in any order still load correctly. See
// gsplat_asset.h for why f_rest_* (view-dependent color) is dropped.
//
// Returns false with a diagnostic in outError on parse failure or if the
// vertex element is missing any of the required properties.
bool loadGsplatPly(const std::string &path, GsplatAsset &outSplats, std::string &outError);

} // namespace italy
