#pragma once

// Synthesizes an equirectangular sky (Preetham/Perez analytic daylight model) into the same EnvironmentMap struct a loaded .hdr populates, so it flows through the identical GPU upload and sampling path.

#include <string>

#include "render/environment.h"

namespace italy {

bool buildProceduralSky(int width, int height, float sunElevationDeg, float sunAzimuthDeg, float turbidity,
                         EnvironmentMap &out, std::string &err, bool bakeSunDisk = true);

} // namespace italy
