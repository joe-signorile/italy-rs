#pragma once

#include <string>

#include "io/mesh_asset.h"

namespace italy {

bool isWatertight(const MeshAsset &mesh, std::string &err);

} // namespace italy
