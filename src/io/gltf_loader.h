#pragma once

#include <string>

#include "io/mesh_asset.h"

namespace italy {

// Loads a whole .glb scene into one flat triangle soup: every triangle
// primitive of every mesh reachable from the default scene, baked into world
// space by its node chain, with per-triangle material identity preserved
// (see MeshAsset). Non-triangle primitives and primitives without POSITION
// are skipped rather than failing the load. Files with no scenes fall back to
// every mesh at identity.
//
// Fails only when nothing at all was loadable; outMesh is reset on entry, so
// a failed load leaves an empty asset rather than the previous one.
bool loadGlb(const std::string &path, MeshAsset &outMesh, std::string &outError);

} // namespace italy
