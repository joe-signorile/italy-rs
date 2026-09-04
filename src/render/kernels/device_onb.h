// Orthonormal-basis-around-a-vector construction shared by pathtracer.cu (cosine-hemisphere sampling) and sdf_bake.cu (sample-direction jittering).
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

  __forceinline__ __device__ float3 toLocal(const float3 &v) const {
    return make_float3(dot(v, m_tangent), dot(v, m_binormal), dot(v, m_normal));
  }

  float3 m_tangent, m_binormal, m_normal;
};
