// Shared by pathtracer.cu (cosine-hemisphere sampling around a shading
// normal) and sdf_bake.cu (jittering sample directions around a base
// direction) — same orthonormal-basis-around-a-vector construction either
// way, no reason to duplicate it in both device files.
#pragma once

#include <sutil/vec_math.h>

struct Onb {
  __forceinline__ __device__ Onb(const float3 &normal) {
    m_normal = normal;
    if (fabsf(m_normal.x) > fabsf(m_normal.z)) {
      m_binormal = make_float3(-m_normal.y, m_normal.x, 0.0f);
    } else {
      m_binormal = make_float3(0.0f, -m_normal.z, m_normal.y);
    }
    m_binormal = normalize(m_binormal);
    m_tangent = cross(m_binormal, m_normal);
  }

  __forceinline__ __device__ float3 toWorld(const float3 &p) const {
    return p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
  }

  float3 m_tangent, m_binormal, m_normal;
};
