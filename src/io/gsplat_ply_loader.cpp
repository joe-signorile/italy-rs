#include "io/gsplat_ply_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace italy {

namespace {

// SH-DC-to-RGB: constant term of a real spherical harmonic basis, the same
// value the 3DGS reference implementation and every compatible exporter use.
constexpr float kShC0 = 0.28209479177387814f;

enum class PlyType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

size_t plyTypeSize(PlyType t) {
  switch (t) {
    case PlyType::Int8:
    case PlyType::UInt8:
      return 1;
    case PlyType::Int16:
    case PlyType::UInt16:
      return 2;
    case PlyType::Int32:
    case PlyType::UInt32:
    case PlyType::Float32:
      return 4;
    case PlyType::Float64:
      return 8;
  }
  return 0;
}

bool parsePlyType(const std::string &tok, PlyType &out) {
  // PLY spec allows both long names and C-style aliases; exporters use both.
  static const std::unordered_map<std::string, PlyType> kNames = {
      {"char", PlyType::Int8},     {"int8", PlyType::Int8},
      {"uchar", PlyType::UInt8},   {"uint8", PlyType::UInt8},
      {"short", PlyType::Int16},   {"int16", PlyType::Int16},
      {"ushort", PlyType::UInt16}, {"uint16", PlyType::UInt16},
      {"int", PlyType::Int32},     {"int32", PlyType::Int32},
      {"uint", PlyType::UInt32},   {"uint32", PlyType::UInt32},
      {"float", PlyType::Float32}, {"float32", PlyType::Float32},
      {"double", PlyType::Float64}, {"float64", PlyType::Float64},
  };
  auto it = kNames.find(tok);
  if (it == kNames.end()) return false;
  out = it->second;
  return true;
}

double readPlyValueAsDouble(const uint8_t *p, PlyType t) {
  switch (t) {
    case PlyType::Int8:
      return static_cast<double>(*reinterpret_cast<const int8_t *>(p));
    case PlyType::UInt8:
      return static_cast<double>(*p);
    case PlyType::Int16: {
      int16_t v;
      std::memcpy(&v, p, sizeof(v));
      return static_cast<double>(v);
    }
    case PlyType::UInt16: {
      uint16_t v;
      std::memcpy(&v, p, sizeof(v));
      return static_cast<double>(v);
    }
    case PlyType::Int32: {
      int32_t v;
      std::memcpy(&v, p, sizeof(v));
      return static_cast<double>(v);
    }
    case PlyType::UInt32: {
      uint32_t v;
      std::memcpy(&v, p, sizeof(v));
      return static_cast<double>(v);
    }
    case PlyType::Float32: {
      float v;
      std::memcpy(&v, p, sizeof(v));
      return static_cast<double>(v);
    }
    case PlyType::Float64: {
      double v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
  }
  return 0.0;
}

struct PlyProperty {
  std::string name;
  PlyType type;
  size_t byteOffset; // offset within one binary vertex record
};

// The subset of per-vertex fields this renderer actually consumes, and where
// each one landed among the header's declared properties (-1 = absent).
struct FieldIndex {
  int x = -1, y = -1, z = -1;
  int scale0 = -1, scale1 = -1, scale2 = -1;
  int rot0 = -1, rot1 = -1, rot2 = -1, rot3 = -1;
  int opacity = -1;
  int fdc0 = -1, fdc1 = -1, fdc2 = -1;

  bool complete() const {
    return x >= 0 && y >= 0 && z >= 0 && scale0 >= 0 && scale1 >= 0 && scale2 >= 0 &&
           rot0 >= 0 && rot1 >= 0 && rot2 >= 0 && rot3 >= 0 && opacity >= 0 && fdc0 >= 0 &&
           fdc1 >= 0 && fdc2 >= 0;
  }
};

FieldIndex indexFields(const std::vector<PlyProperty> &props) {
  FieldIndex fi;
  for (size_t i = 0; i < props.size(); ++i) {
    const std::string &n = props[i].name;
    if (n == "x") fi.x = static_cast<int>(i);
    else if (n == "y") fi.y = static_cast<int>(i);
    else if (n == "z") fi.z = static_cast<int>(i);
    else if (n == "scale_0") fi.scale0 = static_cast<int>(i);
    else if (n == "scale_1") fi.scale1 = static_cast<int>(i);
    else if (n == "scale_2") fi.scale2 = static_cast<int>(i);
    else if (n == "rot_0") fi.rot0 = static_cast<int>(i);
    else if (n == "rot_1") fi.rot1 = static_cast<int>(i);
    else if (n == "rot_2") fi.rot2 = static_cast<int>(i);
    else if (n == "rot_3") fi.rot3 = static_cast<int>(i);
    else if (n == "opacity") fi.opacity = static_cast<int>(i);
    else if (n == "f_dc_0") fi.fdc0 = static_cast<int>(i);
    else if (n == "f_dc_1") fi.fdc1 = static_cast<int>(i);
    else if (n == "f_dc_2") fi.fdc2 = static_cast<int>(i);
  }
  return fi;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

bool loadGsplatPly(const std::string &path, GsplatAsset &outSplats, std::string &outError) {
  outSplats = GsplatAsset{}; // reset any stale state from a previous load

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    outError = "gsplat: could not open " + path;
    return false;
  }

  std::string line;
  if (!std::getline(file, line) || (line != "ply" && line != "ply\r")) {
    outError = "gsplat: not a PLY file (missing 'ply' magic): " + path;
    return false;
  }

  bool binary = false;
  bool haveFormat = false;
  size_t vertexCount = 0;
  bool inVertexElement = false;
  std::vector<PlyProperty> vertexProps;
  size_t runningOffset = 0;

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream iss(line);
    std::string kw;
    iss >> kw;

    if (kw == "comment") {
      continue;
    } else if (kw == "format") {
      std::string fmt;
      iss >> fmt;
      if (fmt == "ascii") {
        binary = false;
      } else if (fmt == "binary_little_endian") {
        binary = true;
      } else {
        outError = "gsplat: unsupported PLY format '" + fmt + "' (only ascii and "
                   "binary_little_endian are supported): " + path;
        return false;
      }
      haveFormat = true;
    } else if (kw == "element") {
      std::string elemName;
      size_t count = 0;
      iss >> elemName >> count;
      inVertexElement = (elemName == "vertex");
      if (inVertexElement) vertexCount = count;
      // Non-vertex elements (e.g. "face") are not expected in a gsplat PLY
      // and aren't handled — real 3DGS exports don't emit them.
    } else if (kw == "property") {
      if (!inVertexElement) continue;
      std::string typeTok;
      iss >> typeTok;
      if (typeTok == "list") {
        outError = "gsplat: list properties on the vertex element are not supported: " + path;
        return false;
      }
      PlyType type;
      if (!parsePlyType(typeTok, type)) {
        outError = "gsplat: unknown PLY property type '" + typeTok + "': " + path;
        return false;
      }
      std::string name;
      iss >> name;
      vertexProps.push_back({name, type, runningOffset});
      runningOffset += plyTypeSize(type);
    } else if (kw == "end_header") {
      break;
    }
    // Unrecognized keywords are ignored, matching a lenient PLY reader.
  }

  if (!haveFormat) {
    outError = "gsplat: PLY file missing 'format' line: " + path;
    return false;
  }
  if (vertexCount == 0 || vertexProps.empty()) {
    outError = "gsplat: PLY file has no vertex element or no properties: " + path;
    return false;
  }

  const FieldIndex fi = indexFields(vertexProps);
  if (!fi.complete()) {
    outError = "gsplat: PLY vertex element is missing one or more required properties "
               "(x,y,z, scale_0-2, rot_0-3, opacity, f_dc_0-2) — not a 3DGS-format splat "
               "file: " + path;
    return false;
  }

  outSplats.positions.reserve(vertexCount);
  outSplats.scales.reserve(vertexCount);
  outSplats.rotations.reserve(vertexCount);
  outSplats.opacity.reserve(vertexCount);
  outSplats.colorDC.reserve(vertexCount);

  const size_t recordSize = runningOffset;

  auto pushVertex = [&](const std::vector<double> &v) {
    const glm::vec3 pos(static_cast<float>(v[fi.x]), static_cast<float>(v[fi.y]),
                         static_cast<float>(v[fi.z]));
    // 3DGS stores log-scale and pre-sigmoid opacity; activate once here so
    // everything downstream works with plain linear values.
    const glm::vec3 scale(std::exp(static_cast<float>(v[fi.scale0])),
                           std::exp(static_cast<float>(v[fi.scale1])),
                           std::exp(static_cast<float>(v[fi.scale2])));
    const glm::vec4 rot(static_cast<float>(v[fi.rot0]), static_cast<float>(v[fi.rot1]),
                         static_cast<float>(v[fi.rot2]), static_cast<float>(v[fi.rot3]));
    const float opacity = sigmoid(static_cast<float>(v[fi.opacity]));
    const glm::vec3 color(0.5f + kShC0 * static_cast<float>(v[fi.fdc0]),
                           0.5f + kShC0 * static_cast<float>(v[fi.fdc1]),
                           0.5f + kShC0 * static_cast<float>(v[fi.fdc2]));

    outSplats.positions.push_back(pos);
    outSplats.scales.push_back(scale);
    outSplats.rotations.push_back(rot);
    outSplats.opacity.push_back(opacity);
    outSplats.colorDC.push_back(glm::clamp(color, 0.0f, 1.0f));
  };

  if (binary) {
    std::vector<uint8_t> record(recordSize);
    std::vector<double> values(vertexProps.size());
    for (size_t i = 0; i < vertexCount; ++i) {
      file.read(reinterpret_cast<char *>(record.data()), static_cast<std::streamsize>(recordSize));
      if (!file) {
        outError = "gsplat: PLY file truncated (binary vertex data ended early): " + path;
        return false;
      }
      for (size_t p = 0; p < vertexProps.size(); ++p) {
        values[p] = readPlyValueAsDouble(record.data() + vertexProps[p].byteOffset, vertexProps[p].type);
      }
      pushVertex(values);
    }
  } else {
    std::vector<double> values(vertexProps.size());
    for (size_t i = 0; i < vertexCount; ++i) {
      if (!std::getline(file, line)) {
        outError = "gsplat: PLY file truncated (ascii vertex data ended early): " + path;
        return false;
      }
      std::istringstream vs(line);
      for (size_t p = 0; p < vertexProps.size(); ++p) {
        if (!(vs >> values[p])) {
          outError = "gsplat: malformed ascii vertex line " + std::to_string(i) + ": " + path;
          return false;
        }
      }
      pushVertex(values);
    }
  }

  outSplats.boundsMin = glm::vec3(std::numeric_limits<float>::max());
  outSplats.boundsMax = glm::vec3(std::numeric_limits<float>::lowest());
  for (const auto &p : outSplats.positions) {
    outSplats.boundsMin = glm::min(outSplats.boundsMin, p);
    outSplats.boundsMax = glm::max(outSplats.boundsMax, p);
  }

  return true;
}

} // namespace italy
