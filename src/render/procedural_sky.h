#pragma once

#include <string>

#include "render/environment.h"

namespace italy {

// Synthesizes an equirectangular sky (Preetham/Perez analytic daylight
// model — see procedural_sky.cpp for why this was picked over Hosek-Wilkie)
// into the same EnvironmentMap struct a loaded .hdr file populates, so it
// flows through the exact same GPU upload, importance sampling, NEE, and
// photon-emission code every other environment already uses — nothing
// downstream of this function needs to know the sky wasn't loaded from disk.
//
// `sunElevationDeg` is degrees above the horizon (90 = straight up, 0 = on
// the horizon, negative = below — produces a dim/night-adjacent sky rather
// than being rejected, since "sunset" is a legitimate artist-facing case).
// `sunAzimuthDeg` is compass angle around the horizon, 0 = +X axis.
// `turbidity` is atmospheric haze, roughly 2 (clear) to 10 (hazy); clamped
// to the range the Perez coefficients were fit over.
bool buildProceduralSky(int width, int height, float sunElevationDeg, float sunAzimuthDeg, float turbidity,
                         EnvironmentMap &out, std::string &err);

} // namespace italy
