#pragma once

#include <string>

#include "io/mesh_asset.h"

namespace italy {

// Loads the first triangle primitive of the first mesh in a .glb file.
// MVP simplification: single mesh/primitive, no scene graph, no node
// transforms — good enough for "one textured object" ingestion; multi-object
// scenes are future work once there's an actual scene/editor concept to put
// them in.
bool loadGlb(const std::string &path, MeshAsset &outMesh, std::string &outError);

} // namespace italy
