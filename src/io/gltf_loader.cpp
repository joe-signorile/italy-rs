// tinygltf's header declares stb_image/stb_image_write usage but doesn't
// define their IMPLEMENTATION macros itself — that's on the includer.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STB_IMAGE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "io/gltf_loader.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace italy {
namespace {

// claudia: only FLOAT-componentType accessors are read (the overwhelming
// common case for glTF exporters) — normalized-integer attribute encodings
// and sparse accessors are unsupported. Both are now *detected* rather than
// silently misread: a non-FLOAT accessor, one of the wrong element type, or
// one without a bufferView (the sparse-only form) reports as absent, so the
// caller falls back — geometric normals, (0,0) UVs, derived tangents — or
// skips the primitive entirely, instead of reinterpreting whatever bytes
// happen to sit at offset zero. Upgrade if a real asset needs them.
const uint8_t *accessorFloatData(const tinygltf::Model &model, int accessorIndex, int expectedType,
                                 size_t &outCount, size_t &outStride) {
  outCount = 0;
  outStride = 0;
  if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
    return nullptr;
  const tinygltf::Accessor &acc = model.accessors[accessorIndex];
  if (acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || acc.type != expectedType)
    return nullptr;
  if (acc.bufferView < 0 || acc.bufferView >= static_cast<int>(model.bufferViews.size()))
    return nullptr;
  const tinygltf::BufferView &view = model.bufferViews[acc.bufferView];
  if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size()))
    return nullptr;
  const int stride = acc.ByteStride(view);
  if (stride <= 0)
    return nullptr;
  outCount = acc.count;
  outStride = static_cast<size_t>(stride);
  return model.buffers[view.buffer].data.data() + view.byteOffset + acc.byteOffset;
}

std::vector<uint32_t> readIndices(const tinygltf::Model &model, int accessorIndex) {
  if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
    return {};
  const tinygltf::Accessor &acc = model.accessors[accessorIndex];
  if (acc.bufferView < 0 || acc.bufferView >= static_cast<int>(model.bufferViews.size()))
    return {};
  const tinygltf::BufferView &view = model.bufferViews[acc.bufferView];
  if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size()))
    return {};
  const tinygltf::Buffer &buf = model.buffers[view.buffer];
  const int stride = acc.ByteStride(view);
  if (stride <= 0)
    return {};
  const uint8_t *base = buf.data.data() + view.byteOffset + acc.byteOffset;

  std::vector<uint32_t> out(acc.count);
  for (size_t i = 0; i < acc.count; ++i) {
    const uint8_t *p = base + i * static_cast<size_t>(stride);
    switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      out[i] = *p;
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      out[i] = *reinterpret_cast<const uint16_t *>(p);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      out[i] = *reinterpret_cast<const uint32_t *>(p);
      break;
    default:
      out[i] = 0;
      break;
    }
  }
  return out;
}

// glTF nodes carry their local transform as either a full 4x4 (column-major,
// same order glm stores) or a TRS triple; the two forms are mutually
// exclusive per the spec, so `matrix` wins when present.
glm::mat4 nodeLocalTransform(const tinygltf::Node &node) {
  if (node.matrix.size() == 16) {
    glm::mat4 m(1.0f);
    for (int col = 0; col < 4; ++col)
      for (int row = 0; row < 4; ++row)
        m[col][row] = static_cast<float>(node.matrix[col * 4 + row]);
    return m;
  }

  glm::mat4 m(1.0f);
  if (node.translation.size() == 3)
    m = glm::translate(m, glm::vec3(static_cast<float>(node.translation[0]),
                                    static_cast<float>(node.translation[1]),
                                    static_cast<float>(node.translation[2])));
  if (node.rotation.size() == 4) {
    // glTF stores the quaternion xyzw; glm::quat's ctor takes wxyz.
    const glm::quat q(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                      static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
    m *= glm::mat4_cast(q);
  }
  if (node.scale.size() == 3)
    m = glm::scale(m, glm::vec3(static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]),
                                static_cast<float>(node.scale[2])));
  return m;
}

struct MeshInstance {
  int mesh = -1;
  glm::mat4 transform{1.0f};
};

void collectInstances(const tinygltf::Model &model, int nodeIndex, const glm::mat4 &parent,
                      std::vector<MeshInstance> &out, std::vector<uint8_t> &onStack) {
  if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size()))
    return;
  // The node graph is supposed to be a forest. A malformed file that makes it
  // cyclic would otherwise recurse until the stack dies, and this runs on
  // untrusted input (a .glb the user dropped in). The flag is cleared on the
  // way out so a node legitimately reached twice via two parents still
  // instances twice.
  if (onStack[nodeIndex])
    return;
  onStack[nodeIndex] = 1;

  const tinygltf::Node &node = model.nodes[nodeIndex];
  const glm::mat4 world = parent * nodeLocalTransform(node);
  if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size()))
    out.push_back({node.mesh, world});
  for (int child : node.children)
    collectInstances(model, child, world, out, onStack);

  onStack[nodeIndex] = 0;
}

// Returns an index into outMesh.textures, or -1 if the image is missing or in
// a channel layout this loader doesn't expand.
//
// Keyed on the *image*, not the glTF texture index: several materials
// commonly point at one multi-megabyte JPEG, and a sampler (the only other
// thing a glTF texture carries) is ignored here anyway, so image identity is
// the strictly better dedupe. The colour space is part of the key because one
// image can legitimately be referenced as both sRGB base colour and linear
// data, and a single TextureAsset can only carry one srgb flag — that case
// gets two entries rather than a coin flip.
int addTexture(const tinygltf::Model &model, int gltfTexIndex, bool srgb,
               std::unordered_map<int, int> &cache, MeshAsset &outMesh) {
  if (gltfTexIndex < 0 || gltfTexIndex >= static_cast<int>(model.textures.size()))
    return -1;
  const int imgIndex = model.textures[gltfTexIndex].source;
  if (imgIndex < 0 || imgIndex >= static_cast<int>(model.images.size()))
    return -1;

  const int key = imgIndex * 2 + (srgb ? 1 : 0);
  const auto cached = cache.find(key);
  if (cached != cache.end())
    return cached->second;

  const tinygltf::Image &img = model.images[imgIndex];
  const size_t pixelCount = static_cast<size_t>(img.width) * static_cast<size_t>(img.height);
  // bits != 8 means a 16-bit-per-channel source; img.image would then be two
  // bytes per component and the byte-wise expansion below would read noise.
  if (img.width <= 0 || img.height <= 0 || img.bits != 8) {
    cache.emplace(key, -1);
    return -1;
  }

  TextureAsset tex;
  tex.width = img.width;
  tex.height = img.height;
  tex.srgb = srgb;
  if (img.component == 4 && img.image.size() >= pixelCount * 4) {
    tex.pixelsRGBA = img.image;
  } else if (img.component == 3 && img.image.size() >= pixelCount * 3) {
    tex.pixelsRGBA.resize(pixelCount * 4);
    for (size_t p = 0; p < pixelCount; ++p) {
      tex.pixelsRGBA[p * 4 + 0] = img.image[p * 3 + 0];
      tex.pixelsRGBA[p * 4 + 1] = img.image[p * 3 + 1];
      tex.pixelsRGBA[p * 4 + 2] = img.image[p * 3 + 2];
      tex.pixelsRGBA[p * 4 + 3] = 255;
    }
  } else {
    // Unsupported channel count (1 or 2, i.e. grey/grey+alpha) — fall back to
    // the material's factors rather than guessing a channel mapping.
    cache.emplace(key, -1);
    return -1;
  }

  const int index = static_cast<int>(outMesh.textures.size());
  outMesh.textures.push_back(std::move(tex));
  cache.emplace(key, index);
  return index;
}

// Gram-Schmidt t against n, with every degenerate input funnelled to *some*
// finite perpendicular. A NaN tangent is not a subtle shading error: it
// propagates through the normal-map basis into the throughput and leaves a
// permanently dead pixel that the frame accumulator can never wash out.
// NaN inputs land here too — `NaN > eps` is false.
glm::vec3 orthonormalTangent(const glm::vec3 &n, const glm::vec3 &t) {
  const glm::vec3 v = t - n * glm::dot(n, t);
  const float len = glm::length(v);
  if (len > 1e-12f)
    return v / len;
  const glm::vec3 axis = std::fabs(n.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::vec3 fallback = glm::cross(axis, n);
  const float flen = glm::length(fallback);
  return flen > 1e-12f ? fallback / flen : glm::vec3(1.0f, 0.0f, 0.0f);
}

} // namespace

bool loadGlb(const std::string &path, MeshAsset &outMesh, std::string &outError) {
  // Reset by assignment, not by clearing the vectors individually: the caller
  // (main.cpp's rebuildScene) reuses one long-lived AppState::meshAsset across
  // every Load/Apply, and this function only ever push_back()s into it.
  // Without this, the second Apply rendered two copies of the geometry, the
  // third rendered three, while boundsMin/Max were recomputed from the newest
  // load alone — so camera framing silently desynced from the geometry too.
  // The material and texture tables are equally sticky: loading an untextured
  // asset after a textured one kept the old textures alive and reachable.
  // A fresh value can't be escaped by the next field someone adds here.
  outMesh = MeshAsset{};

  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string warn;
  if (!loader.LoadBinaryFromFile(&model, &outError, &warn, path)) {
    if (outError.empty())
      outError = "tinygltf: failed to load " + path;
    return false;
  }
  if (model.meshes.empty()) {
    outError = path + ": no meshes found";
    return false;
  }

  // --- materials and textures -------------------------------------------
  // Built first so every primitive can just index them. Textures are pulled
  // in lazily by addTexture(), so images the scene never references (glTF
  // files routinely ship spares) never get expanded to RGBA.
  std::unordered_map<int, int> textureCache;
  outMesh.materials.reserve(model.materials.size() + 1);
  for (const tinygltf::Material &src : model.materials) {
    const tinygltf::PbrMetallicRoughness &pbr = src.pbrMetallicRoughness;
    MaterialAsset mat;
    // tinygltf already applies the glTF defaults (baseColorFactor 1,1,1,1;
    // metallic 1; roughness 1) when the fields are absent, so these reads are
    // unconditional — the size check only guards a malformed short array.
    if (pbr.baseColorFactor.size() >= 3)
      mat.baseColorFactor = glm::vec3(static_cast<float>(pbr.baseColorFactor[0]),
                                      static_cast<float>(pbr.baseColorFactor[1]),
                                      static_cast<float>(pbr.baseColorFactor[2]));
    mat.metallic = static_cast<float>(pbr.metallicFactor);
    mat.roughness = static_cast<float>(pbr.roughnessFactor);
    // Only base colour is sRGB-encoded; metallic-roughness and normal maps
    // are linear data, and decoding them would bend roughness dark and every
    // normal toward +Z.
    mat.baseColorTexture = addTexture(model, pbr.baseColorTexture.index, /*srgb=*/true, textureCache, outMesh);
    mat.metallicRoughnessTexture =
        addTexture(model, pbr.metallicRoughnessTexture.index, /*srgb=*/false, textureCache, outMesh);
    mat.normalTexture = addTexture(model, src.normalTexture.index, /*srgb=*/false, textureCache, outMesh);
    mat.normalScale = static_cast<float>(src.normalTexture.scale);
    outMesh.materials.push_back(mat);
  }
  // materialForTriangle() indexes materials unconditionally, so the table can
  // never be empty. This trailing entry doubles as the slot for primitives
  // with material < 0: MaterialAsset's own defaults (dielectric, fully rough)
  // rather than glTF's material defaults, because an asset that declared no
  // material wants plausible clay, not chrome.
  const uint32_t defaultMaterial = static_cast<uint32_t>(outMesh.materials.size());
  outMesh.materials.push_back(MaterialAsset{});

  // --- scene graph -------------------------------------------------------
  std::vector<MeshInstance> instances;
  if (!model.scenes.empty()) {
    const int sceneIndex = model.defaultScene >= 0 && model.defaultScene < static_cast<int>(model.scenes.size())
                               ? model.defaultScene
                               : 0;
    std::vector<uint8_t> onStack(model.nodes.size(), 0);
    for (int root : model.scenes[sceneIndex].nodes)
      collectInstances(model, root, glm::mat4(1.0f), instances, onStack);
  }
  if (instances.empty()) {
    // No scenes, or a scene that reaches no meshes. Fall back to every mesh at
    // identity — a mesh nobody instanced is still geometry the user expects to
    // see, and the alternative is an empty render with no explanation.
    instances.reserve(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i)
      instances.push_back({static_cast<int>(i), glm::mat4(1.0f)});
  }

  // --- reserve -----------------------------------------------------------
  // A single pre-pass over the accessor headers (no buffer reads) so the four
  // parallel vertex arrays each grow exactly once. bike.glb is 1.92M
  // triangles — 5.7M vertices across four arrays is ~275MB, and letting that
  // reallocate geometrically is the difference between a load you wait on and
  // one you don't notice.
  size_t totalIndices = 0;
  for (const MeshInstance &inst : instances) {
    for (const tinygltf::Primitive &prim : model.meshes[inst.mesh].primitives) {
      if (prim.mode != TINYGLTF_MODE_TRIANGLES)
        continue;
      const auto posIt = prim.attributes.find("POSITION");
      if (posIt == prim.attributes.end() || posIt->second < 0 ||
          posIt->second >= static_cast<int>(model.accessors.size()))
        continue;
      if (prim.indices >= 0 && prim.indices < static_cast<int>(model.accessors.size()))
        totalIndices += model.accessors[prim.indices].count;
      else
        totalIndices += model.accessors[posIt->second].count;
    }
  }
  outMesh.positions.reserve(totalIndices);
  outMesh.normals.reserve(totalIndices);
  outMesh.uvs.reserve(totalIndices);
  outMesh.tangents.reserve(totalIndices);
  outMesh.triangleMaterial.reserve(totalIndices / 3);

  glm::vec3 boundsMin(std::numeric_limits<float>::max());
  glm::vec3 boundsMax(std::numeric_limits<float>::lowest());

  // --- geometry ----------------------------------------------------------
  for (const MeshInstance &inst : instances) {
    const glm::mat4 &world = inst.transform;
    const glm::mat3 linear(world);
    const float det = glm::determinant(linear);
    // Normals transform by the inverse transpose, not the matrix itself —
    // under non-uniform scale the two disagree and a plain transform tilts
    // every normal off the surface. A singular chain (a zero scale somewhere)
    // has no inverse; positions still bake fine, so keep the instance and let
    // normals fall back to the flattened face normals.
    const bool singular = !(std::fabs(det) > 1e-12f);
    const glm::mat3 normalMatrix = singular ? glm::mat3(1.0f) : glm::transpose(glm::inverse(linear));
    // A mirroring transform reverses the geometric winding. OptiX front-faces
    // by winding, so swap two corners to put it back — otherwise mirrored
    // parts render inside-out against everything else in the same soup.
    const bool flipWinding = det < 0.0f;

    for (const tinygltf::Primitive &prim : model.meshes[inst.mesh].primitives) {
      // Skip rather than fail: one line/point primitive in a 169-mesh scene
      // shouldn't cost the user the other 168.
      if (prim.mode != TINYGLTF_MODE_TRIANGLES)
        continue;
      const auto posIt = prim.attributes.find("POSITION");
      if (posIt == prim.attributes.end())
        continue;

      size_t posCount = 0, posStride = 0;
      const uint8_t *posData = accessorFloatData(model, posIt->second, TINYGLTF_TYPE_VEC3, posCount, posStride);
      if (!posData || posCount == 0)
        continue;

      auto attribute = [&](const char *name, int type, size_t &count, size_t &stride) -> const uint8_t * {
        const auto it = prim.attributes.find(name);
        if (it == prim.attributes.end())
          return nullptr;
        const uint8_t *data = accessorFloatData(model, it->second, type, count, stride);
        // An attribute array shorter than POSITION's is malformed; treating it
        // as absent is cheaper than bounds-checking every corner read below.
        return (data && count >= posCount) ? data : nullptr;
      };

      size_t normalCount = 0, normalStride = 0;
      const uint8_t *normalData = attribute("NORMAL", TINYGLTF_TYPE_VEC3, normalCount, normalStride);
      size_t uvCount = 0, uvStride = 0;
      const uint8_t *uvData = attribute("TEXCOORD_0", TINYGLTF_TYPE_VEC2, uvCount, uvStride);
      size_t tanCount = 0, tanStride = 0;
      const uint8_t *tanData = attribute("TANGENT", TINYGLTF_TYPE_VEC4, tanCount, tanStride);

      auto position = [&](uint32_t v) {
        const float *f = reinterpret_cast<const float *>(posData + v * posStride);
        return glm::vec3(f[0], f[1], f[2]);
      };
      auto normal = [&](uint32_t v) {
        const float *f = reinterpret_cast<const float *>(normalData + v * normalStride);
        return glm::vec3(f[0], f[1], f[2]);
      };
      auto uv = [&](uint32_t v) -> glm::vec2 {
        if (!uvData)
          return glm::vec2(0.0f);
        const float *f = reinterpret_cast<const float *>(uvData + v * uvStride);
        return glm::vec2(f[0], f[1]);
      };
      auto tangent = [&](uint32_t v) {
        const float *f = reinterpret_cast<const float *>(tanData + v * tanStride);
        return glm::vec4(f[0], f[1], f[2], f[3]);
      };

      std::vector<uint32_t> indices;
      if (prim.indices >= 0) {
        indices = readIndices(model, prim.indices);
        if (indices.empty())
          continue; // unreadable index accessor (sparse-only, or malformed)
      } else {
        indices.resize(posCount);
        for (size_t i = 0; i < posCount; ++i)
          indices[i] = static_cast<uint32_t>(i);
      }

      const uint32_t materialIndex =
          (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size()))
              ? static_cast<uint32_t>(prim.material)
              : defaultMaterial;

      // Trailing indices that don't complete a triangle are dropped rather
      // than failing the load, same reasoning as the mode skip above.
      for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t c[3] = {indices[i], indices[i + 1], indices[i + 2]};
        if (flipWinding)
          std::swap(c[1], c[2]);
        if (c[0] >= posCount || c[1] >= posCount || c[2] >= posCount)
          continue; // out-of-range index in an untrusted file

        glm::vec3 p[3];
        for (int k = 0; k < 3; ++k)
          p[k] = glm::vec3(world * glm::vec4(position(c[k]), 1.0f));

        // NORMAL is optional in glTF, and the bundled assets/test.glb omits
        // it. Substituting a constant up-vector makes every surface shade as
        // though it faces the sky no matter which way it actually points:
        // flat faces come out uniformly lit and rounded edges lose their
        // shading gradient entirely. Derive the triangle's own geometric
        // normal instead — the spec's fallback. Computed from the already-
        // transformed, already-winding-corrected corners, which agrees with
        // the inverse-transpose path used for supplied normals (the corner
        // swap cancels the determinant sign flip). Flat-shaded rather than
        // smooth, which is the honest result for a mesh that shipped no
        // vertex normals.
        const glm::vec3 faceCross = glm::cross(p[1] - p[0], p[2] - p[0]);
        const float faceLen = glm::length(faceCross);
        // Degenerate (zero-area) triangle: no meaningful normal exists, and
        // normalizing would hand the path tracer a NaN that the accumulator
        // can never wash back out.
        const glm::vec3 faceNormal = faceLen > 0.0f ? faceCross / faceLen : glm::vec3(0.0f, 1.0f, 0.0f);

        glm::vec3 n[3];
        glm::vec2 t[3];
        for (int k = 0; k < 3; ++k) {
          t[k] = uv(c[k]);
          if (normalData) {
            const glm::vec3 nn = normalMatrix * normal(c[k]);
            const float len = glm::length(nn);
            n[k] = len > 0.0f ? nn / len : faceNormal;
          } else {
            n[k] = faceNormal;
          }
        }

        // Per-triangle tangent frame from the UV derivatives, used when the
        // primitive shipped no TANGENT. Solves [dp1 dp2] = [T B] * [du1 du2]
        // for T and B; r is the determinant of the UV edge matrix, so a
        // zero-area UV triangle (which includes every primitive with no
        // TEXCOORD_0 at all, where all three UVs are (0,0)) makes it
        // singular and there is simply no information to recover a tangent
        // from — orthonormalTangent's fallback picks an arbitrary one.
        glm::vec3 triT(0.0f), triB(0.0f);
        if (!tanData) {
          const glm::vec3 dp1 = p[1] - p[0], dp2 = p[2] - p[0];
          const glm::vec2 du1 = t[1] - t[0], du2 = t[2] - t[0];
          const float r = du1.x * du2.y - du2.x * du1.y;
          if (std::fabs(r) > 1e-12f) {
            const float inv = 1.0f / r;
            triT = (dp1 * du2.y - dp2 * du1.y) * inv;
            triB = (dp2 * du1.x - dp1 * du2.x) * inv;
          }
        }

        for (int k = 0; k < 3; ++k) {
          outMesh.positions.push_back(p[k]);
          outMesh.normals.push_back(n[k]);
          outMesh.uvs.push_back(t[k]);
          if (tanData) {
            // Supplied tangents transform by the matrix itself (they're
            // directions in the surface, not normals). A mirroring transform
            // inverts the handedness of the frame, so w flips with it —
            // cross(N', T') points the other way once N' has gone through the
            // inverse transpose.
            const glm::vec4 src = tangent(c[k]);
            const glm::vec3 T = orthonormalTangent(n[k], linear * glm::vec3(src));
            const float w = (src.w < 0.0f) != flipWinding ? -1.0f : 1.0f;
            outMesh.tangents.emplace_back(T, w);
          } else {
            const glm::vec3 T = orthonormalTangent(n[k], triT);
            const float w = glm::dot(glm::cross(n[k], T), triB) < 0.0f ? -1.0f : 1.0f;
            outMesh.tangents.emplace_back(T, w);
          }
          boundsMin = glm::min(boundsMin, p[k]);
          boundsMax = glm::max(boundsMax, p[k]);
        }
        outMesh.triangleMaterial.push_back(materialIndex);
      }
    }
  }

  if (outMesh.positions.empty()) {
    outError = path + ": no triangle primitives found";
    return false;
  }
  outMesh.boundsMin = boundsMin;
  outMesh.boundsMax = boundsMax;
  return true;
}

} // namespace italy
