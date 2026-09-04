#pragma once

#include <string>

#include "io/gsplat_asset.h"

namespace italy {

bool loadGsplatPly(const std::string &path, GsplatAsset &outSplats, std::string &outError);

} // namespace italy
