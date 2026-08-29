#include "render/environment.h"

// Deliberately no STB_IMAGE_IMPLEMENTATION here — gltf_loader.cpp already
// provides it (via tinygltf's bundled copy), and stb_image's C API has been
// stable for well over a decade, so declaring against this repo's own `stb`
// FetchContent copy and linking against that already-compiled
// implementation is safe. Defining it a second time here would be the same
// "duplicate global" mistake as optix_function_table_definition.h earlier.
#include <stb_image.h>

namespace italy {

bool loadEnvironmentMap(const std::string &path, EnvironmentMap &out, std::string &err) {
  int w, h, channels;
  float *data = stbi_loadf(path.c_str(), &w, &h, &channels, 3);
  if (!data) {
    err = "failed to load HDR file: " + path;
    return false;
  }
  std::vector<float> pixels(data, data + static_cast<size_t>(w) * h * 3);
  stbi_image_free(data);

  if (!buildEnvironmentCdf(w, h, pixels, out, err)) {
    err = path + ": " + err;
    return false;
  }
  return true;
}

bool buildEnvironmentCdf(int w, int h, const std::vector<float> &pixels, EnvironmentMap &out, std::string &err) {
  out.pixels = pixels;
  out.width = w;
  out.height = h;

  // Build the piecewise-constant 2D distribution, luminance-weighted (Rec.
  // 709 coefficients — the exact weighting only affects *where* NEE biases
  // its samples, not the radiance value returned, so any reasonable
  // luminance formula works equally well here).
  out.conditionalCdf.assign(static_cast<size_t>(h) * (w + 1), 0.0f);
  std::vector<float> rowIntegral(h, 0.0f);
  for (int y = 0; y < h; ++y) {
    float *rowCdf = &out.conditionalCdf[static_cast<size_t>(y) * (w + 1)];
    rowCdf[0] = 0.0f;
    for (int x = 0; x < w; ++x) {
      const float *p = &out.pixels[(static_cast<size_t>(y) * w + x) * 3];
      const float lum = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
      rowCdf[x + 1] = rowCdf[x] + lum;
    }
    rowIntegral[y] = rowCdf[w];
    if (rowIntegral[y] > 0.0f) {
      for (int x = 0; x <= w; ++x)
        rowCdf[x] /= rowIntegral[y];
    } else {
      // Degenerate all-black row: fall back to uniform so sampling never
      // divides by zero. Harmless — the marginal CDF weights this row by
      // rowIntegral (zero), so it's never actually selected.
      for (int x = 0; x <= w; ++x)
        rowCdf[x] = static_cast<float>(x) / w;
    }
  }

  out.marginalCdf.assign(h + 1, 0.0f);
  for (int y = 0; y < h; ++y)
    out.marginalCdf[y + 1] = out.marginalCdf[y] + rowIntegral[y];
  const float total = out.marginalCdf[h];
  if (total <= 0.0f) {
    err = "environment map is entirely black";
    return false;
  }
  for (int y = 0; y <= h; ++y)
    out.marginalCdf[y] /= total;

  return true;
}

} // namespace italy
