// tinygltf's header declares stb_image/stb_image_write usage but doesn't
// define their IMPLEMENTATION macros itself — that's on the includer.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STB_IMAGE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "io/gltf_loader.h"

#include <limits>

namespace italy {
namespace {

// monkey-boy: only FLOAT-componentType position/normal/UV accessors are
// handled (the overwhelming common case for glTF exporters) — normalized
// integer attribute encodings are unsupported for now; upgrade if a real
// asset needs them.
const uint8_t *accessorFloatData(const tinygltf::Model &model, int accessorIndex, size_t &outStride) {
  const tinygltf::Accessor &acc = model.accessors[accessorIndex];
  const tinygltf::BufferView &view = model.bufferViews[acc.bufferView];
  const tinygltf::Buffer &buf = model.buffers[view.buffer];
  outStride = static_cast<size_t>(acc.ByteStride(view));
  return buf.data.data() + view.byteOffset + acc.byteOffset;
}

std::vector<uint32_t> readIndices(const tinygltf::Model &model, int accessorIndex) {
  const tinygltf::Accessor &acc = model.accessors[accessorIndex];
  const tinygltf::BufferView &view = model.bufferViews[acc.bufferView];
  const tinygltf::Buffer &buf = model.buffers[view.buffer];
  const uint8_t *base = buf.data.data() + view.byteOffset + acc.byteOffset;
  const size_t stride = acc.ByteStride(view);

  std::vector<uint32_t> out(acc.count);
  for (size_t i = 0; i < acc.count; ++i) {
    const uint8_t *p = base + i * stride;
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

} // namespace

bool loadGlb(const std::string &path, MeshAsset &outMesh, std::string &outError) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string warn;
  if (!loader.LoadBinaryFromFile(&model, &outError, &warn, path)) {
    if (outError.empty())
      outError = "tinygltf: failed to load " + path;
    return false;
  }
  if (model.meshes.empty() || model.meshes[0].primitives.empty()) {
    outError = path + ": no mesh primitives found";
    return false;
  }

  const tinygltf::Primitive &prim = model.meshes[0].primitives[0];
  if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
    outError = path + ": first primitive is not a triangle list (unsupported mode)";
    return false;
  }
  const auto posIt = prim.attributes.find("POSITION");
  if (posIt == prim.attributes.end()) {
    outError = path + ": primitive has no POSITION attribute";
    return false;
  }

  const tinygltf::Accessor &posAcc = model.accessors[posIt->second];
  size_t posStride = 0;
  const uint8_t *posData = accessorFloatData(model, posIt->second, posStride);

  const uint8_t *normalData = nullptr;
  size_t normalStride = 0;
  const auto normIt = prim.attributes.find("NORMAL");
  if (normIt != prim.attributes.end())
    normalData = accessorFloatData(model, normIt->second, normalStride);

  const uint8_t *uvData = nullptr;
  size_t uvStride = 0;
  const auto uvIt = prim.attributes.find("TEXCOORD_0");
  if (uvIt != prim.attributes.end())
    uvData = accessorFloatData(model, uvIt->second, uvStride);

  auto position = [&](size_t vertexIndex) {
    const float *f = reinterpret_cast<const float *>(posData + vertexIndex * posStride);
    return glm::vec3(f[0], f[1], f[2]);
  };
  auto normal = [&](size_t vertexIndex) -> glm::vec3 {
    if (!normalData)
      return glm::vec3(0.0f, 1.0f, 0.0f);
    const float *f = reinterpret_cast<const float *>(normalData + vertexIndex * normalStride);
    return glm::vec3(f[0], f[1], f[2]);
  };
  auto uv = [&](size_t vertexIndex) -> glm::vec2 {
    if (!uvData)
      return glm::vec2(0.0f);
    const float *f = reinterpret_cast<const float *>(uvData + vertexIndex * uvStride);
    return glm::vec2(f[0], f[1]);
  };

  std::vector<uint32_t> indices;
  if (prim.indices >= 0) {
    indices = readIndices(model, prim.indices);
  } else {
    indices.resize(posAcc.count);
    for (size_t i = 0; i < posAcc.count; ++i)
      indices[i] = static_cast<uint32_t>(i);
  }
  if (indices.size() % 3 != 0) {
    outError = path + ": index count is not a multiple of 3";
    return false;
  }

  outMesh.positions.reserve(indices.size());
  outMesh.normals.reserve(indices.size());
  outMesh.uvs.reserve(indices.size());
  glm::vec3 boundsMin(std::numeric_limits<float>::max());
  glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
  for (uint32_t idx : indices) {
    const glm::vec3 p = position(idx);
    outMesh.positions.push_back(p);
    outMesh.normals.push_back(glm::normalize(normal(idx)));
    outMesh.uvs.push_back(uv(idx));
    boundsMin = glm::min(boundsMin, p);
    boundsMax = glm::max(boundsMax, p);
  }
  outMesh.boundsMin = boundsMin;
  outMesh.boundsMax = boundsMax;

  if (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size())) {
    const tinygltf::Material &mat = model.materials[prim.material];
    const auto &bcf = mat.pbrMetallicRoughness.baseColorFactor; // 4 doubles, RGBA
    if (bcf.size() == 4)
      outMesh.baseColorFactor = glm::vec3(bcf[0], bcf[1], bcf[2]);

    const int texIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
    if (texIndex >= 0 && texIndex < static_cast<int>(model.textures.size())) {
      const int imgIndex = model.textures[texIndex].source;
      if (imgIndex >= 0 && imgIndex < static_cast<int>(model.images.size())) {
        const tinygltf::Image &img = model.images[imgIndex];
        outMesh.baseColorTexture.width = img.width;
        outMesh.baseColorTexture.height = img.height;
        outMesh.baseColorTexture.pixelsRGBA.resize(static_cast<size_t>(img.width) * img.height * 4);
        if (img.component == 4) {
          outMesh.baseColorTexture.pixelsRGBA = img.image;
        } else if (img.component == 3) {
          for (size_t p = 0; p < static_cast<size_t>(img.width) * img.height; ++p) {
            outMesh.baseColorTexture.pixelsRGBA[p * 4 + 0] = img.image[p * 3 + 0];
            outMesh.baseColorTexture.pixelsRGBA[p * 4 + 1] = img.image[p * 3 + 1];
            outMesh.baseColorTexture.pixelsRGBA[p * 4 + 2] = img.image[p * 3 + 2];
            outMesh.baseColorTexture.pixelsRGBA[p * 4 + 3] = 255;
          }
        } else {
          outMesh.baseColorTexture.width = 0; // unsupported channel count — fall back to baseColorFactor
          outMesh.baseColorTexture.height = 0;
        }
        outMesh.hasBaseColorTexture = outMesh.baseColorTexture.width > 0;
      }
    }
  }

  return true;
}

} // namespace italy
