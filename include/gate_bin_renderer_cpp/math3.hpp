// Minimal 3-vector / quaternion math, callable on both host and CUDA device.
//
// Quaternions are SCALAR-LAST [x, y, z, w] (Hamilton), matching the JAX
// gate_traversal env (dva-quad-jax) and rl_infer's gate/task.py xyzw helpers.
#ifndef GATE_BIN_RENDERER_CPP__MATH3_HPP_
#define GATE_BIN_RENDERER_CPP__MATH3_HPP_

#include <cmath>

#if defined(__CUDACC__)
#define GBR_HD __host__ __device__
#else
#define GBR_HD
#endif

namespace gbr {

struct Vec3 {
  float x, y, z;
};

struct Quat {  // [x, y, z, w], scalar last
  float x, y, z, w;
};

GBR_HD inline Vec3 v_add(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
GBR_HD inline Vec3 v_sub(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
GBR_HD inline Vec3 v_scale(const Vec3 &a, float s) { return {a.x * s, a.y * s, a.z * s}; }
GBR_HD inline Vec3 v_cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// v_world = R(q) * v_body. Matches gate/task.quat_apply_xyzw / ju.quat_apply.
GBR_HD inline Vec3 quat_apply(const Quat &q, const Vec3 &v) {
  const Vec3 u{q.x, q.y, q.z};
  const Vec3 t = v_scale(v_cross(u, v), 2.0f);
  return v_add(v_add(v, v_scale(t, q.w)), v_cross(u, t));
}

// v_body = R(q)^T * v_world. Matches gate/task.quat_rotate_inverse_xyzw.
GBR_HD inline Vec3 quat_apply_inv(const Quat &q, const Vec3 &v) {
  const Vec3 u{q.x, q.y, q.z};
  const Vec3 t = v_scale(v_cross(u, v), 2.0f);
  return v_add(v_sub(v, v_scale(t, q.w)), v_cross(u, t));
}

}  // namespace gbr

#endif  // GATE_BIN_RENDERER_CPP__MATH3_HPP_
