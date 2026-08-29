// environment_cdf_test links environment.cpp, which declares (but per its
// own comment deliberately doesn't define) stb_image's implementation —
// that's normally provided elsewhere in the italy-rs binary by
// io/gltf_loader.cpp. This test binary doesn't link gltf_loader.cpp (no
// glTF loading involved), so it needs its own copy of the definition,
// scoped to this binary only.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
