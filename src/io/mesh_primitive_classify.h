#pragma once

#include <string>

#include "io/mesh_asset.h"

namespace italy {

enum class PrimitiveKind { None, Sphere, Box, Cylinder };

PrimitiveKind classifyPrimitive(const MeshAsset &mesh, std::string &detail);

} // namespace italy
