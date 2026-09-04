#pragma once

#include <string>

#include "io/mesh_asset.h"

namespace italy {

bool loadGlb(const std::string &path, MeshAsset &outMesh, std::string &outError);

} // namespace italy
