#pragma once

#include <string>

#include "io/mesh_asset.h"

namespace italy {

// Closed/watertight check: every edge of the mesh must be shared by exactly
// two triangles (2-manifold, no boundary or non-manifold edges). Required
// for anything that depends on a well-defined inside/outside — SDF baking's
// ray-parity sign vote (see sdf_baker.h) and, as of the transmissive
// material work, refraction through triangle geometry (see MATERIAL_
// TEXTURED_DIFFUSE's transmission branch in pathtracer.cu): a closed,
// consistently-wound mesh is what makes "the ray is now inside the object"
// a fact rather than a guess.
//
// `MeshAsset` is a flattened triangle soup with no index buffer (see its own
// doc comment), so this welds vertices by position first — there's no
// existing shared-vertex identity to build edge adjacency from directly.
//
// Returns true if watertight. On failure, `err` describes what's wrong
// (boundary edge / non-manifold edge counts) rather than just "no", so a
// rejected load gives the user something actionable.
bool isWatertight(const MeshAsset &mesh, std::string &err);

} // namespace italy
