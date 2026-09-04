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

float extensionNumber(const tinygltf::ExtensionMap &ext, const char *extName, const char *key, float def) {
  const auto it = ext.find(extName);
  if (it == ext.end() || !it->second.Has(key))
    return def;
  return static_cast<float>(it->second.Get(key).GetNumberAsDouble());
}

glm::vec3 extensionColor3(const tinygltf::ExtensionMap &ext, const char *extName, const char *key, glm::vec3 def) {
  const auto it = ext.find(extName);
  if (it == ext.end() || !it->second.Has(key))
    return def;
  const tinygltf::Value &arr = it->second.Get(key);
  if (!arr.IsArray() || arr.ArrayLen() < 3)
    return def;
  return glm::vec3(static_cast<float>(arr.Get(0).GetNumberAsDouble()), static_cast<float>(arr.Get(1).GetNumberAsDouble()),
                    static_cast<float>(arr.Get(2).GetNumberAsDouble()));
}

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
    cache.emplace(key, -1);
    return -1;
  }

  const int index = static_cast<int>(outMesh.textures.size());
  outMesh.textures.push_back(std::move(tex));
  cache.emplace(key, index);
  return index;
}

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

  std::unordered_map<int, int> textureCache;
  outMesh.materials.reserve(model.materials.size() + 1);
  for (const tinygltf::Material &src : model.materials) {
    const tinygltf::PbrMetallicRoughness &pbr = src.pbrMetallicRoughness;
    MaterialAsset mat;
    if (pbr.baseColorFactor.size() >= 3)
      mat.baseColorFactor = glm::vec3(static_cast<float>(pbr.baseColorFactor[0]),
                                      static_cast<float>(pbr.baseColorFactor[1]),
                                      static_cast<float>(pbr.baseColorFactor[2]));
    mat.metallic = static_cast<float>(pbr.metallicFactor);
    mat.roughness = static_cast<float>(pbr.roughnessFactor);
    mat.baseColorTexture = addTexture(model, pbr.baseColorTexture.index, /*srgb=*/true, textureCache, outMesh);
    mat.metallicRoughnessTexture =
        addTexture(model, pbr.metallicRoughnessTexture.index, /*srgb=*/false, textureCache, outMesh);
    mat.normalTexture = addTexture(model, src.normalTexture.index, /*srgb=*/false, textureCache, outMesh);
    mat.normalScale = static_cast<float>(src.normalTexture.scale);

    mat.transmission = extensionNumber(src.extensions, "KHR_materials_transmission", "transmissionFactor", 0.0f);
    mat.ior = extensionNumber(src.extensions, "KHR_materials_ior", "ior", 1.5f);
    mat.attenuationColor =
        extensionColor3(src.extensions, "KHR_materials_volume", "attenuationColor", glm::vec3(1.0f));
    mat.attenuationDistance =
        extensionNumber(src.extensions, "KHR_materials_volume", "attenuationDistance", std::numeric_limits<float>::infinity());
    outMesh.materials.push_back(mat);
  }
  const uint32_t defaultMaterial = static_cast<uint32_t>(outMesh.materials.size());
  outMesh.materials.push_back(MaterialAsset{});

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
    instances.reserve(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i)
      instances.push_back({static_cast<int>(i), glm::mat4(1.0f)});
  }

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

  for (const MeshInstance &inst : instances) {
    const glm::mat4 &world = inst.transform;
    const glm::mat3 linear(world);
    const float det = glm::determinant(linear);
    const bool singular = !(std::fabs(det) > 1e-12f);
    const glm::mat3 normalMatrix = singular ? glm::mat3(1.0f) : glm::transpose(glm::inverse(linear));
    const bool flipWinding = det < 0.0f;

    for (const tinygltf::Primitive &prim : model.meshes[inst.mesh].primitives) {
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
          continue;
      } else {
        indices.resize(posCount);
        for (size_t i = 0; i < posCount; ++i)
          indices[i] = static_cast<uint32_t>(i);
      }

      const uint32_t materialIndex =
          (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size()))
              ? static_cast<uint32_t>(prim.material)
              : defaultMaterial;

      for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t c[3] = {indices[i], indices[i + 1], indices[i + 2]};
        if (flipWinding)
          std::swap(c[1], c[2]);
        if (c[0] >= posCount || c[1] >= posCount || c[2] >= posCount)
          continue;

        glm::vec3 p[3];
        for (int k = 0; k < 3; ++k)
          p[k] = glm::vec3(world * glm::vec4(position(c[k]), 1.0f));

        const glm::vec3 faceCross = glm::cross(p[1] - p[0], p[2] - p[0]);
        const float faceLen = glm::length(faceCross);
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
