// PX4 (NED/FRD) <-> policy (ENU/FLU) frame transforms, host-only. Faithful to
// rl_infer/rl_infer/util/math.py (ned_to_enu_vec, ned_frd_quat_to_enu_flu) and
// gate/task.py (euler_to_quat_xyzw, quat_mul_xyzw), so the rendered camera pose
// matches what gate_jax_node feeds the policy.
#ifndef GATE_BIN_RENDERER_CPP__FRAME_TRANSFORMS_HPP_
#define GATE_BIN_RENDERER_CPP__FRAME_TRANSFORMS_HPP_

#include <array>
#include <cmath>

#include "gate_bin_renderer_cpp/math3.hpp"

namespace gbr {

// World vector NED -> ENU: (N,E,D) -> (E,N,U) = (y, x, -z).
inline Vec3 ned_to_enu(const Vec3 &v) { return {v.y, v.x, -v.z}; }

// ZYX Tait-Bryan -> quaternion [x,y,z,w]. Matches gate/task.euler_to_quat_xyzw
// (and Gazebo <pose> RPY).
inline Quat euler_zyx_to_quat_xyzw(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  Quat q{static_cast<float>(sr * cp * cy - cr * sp * sy),
         static_cast<float>(cr * sp * cy + sr * cp * sy),
         static_cast<float>(cr * cp * sy - sr * sp * cy),
         static_cast<float>(cr * cp * cy + sr * sp * sy)};
  const double n = std::sqrt(static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y +
                             static_cast<double>(q.z) * q.z + static_cast<double>(q.w) * q.w);
  const double inv = n > 1e-9 ? 1.0 / n : 1.0;
  return {static_cast<float>(q.x * inv), static_cast<float>(q.y * inv),
          static_cast<float>(q.z * inv), static_cast<float>(q.w * inv)};
}

// Quaternion product q (x) r, scalar-last; matches gate/task.quat_mul_xyzw.
inline Quat quat_mul_xyzw(const Quat &q, const Quat &r) {
  return {q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y,
          q.w * r.y - q.x * r.z + q.y * r.w + q.z * r.x,
          q.w * r.z + q.x * r.y - q.y * r.x + q.z * r.w,
          q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z};
}

using Mat3 = std::array<double, 9>;  // row-major

// Quaternion (w,x,y,z) -> 3x3 rotation matrix (matches util/math.quat_to_rotmat).
inline Mat3 quat_wxyz_to_rotmat(double w, double x, double y, double z) {
  return Mat3{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
              2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
              2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
}

inline Mat3 mat_mul(const Mat3 &a, const Mat3 &b) {
  Mat3 c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      c[i * 3 + j] = a[i * 3 + 0] * b[0 * 3 + j] + a[i * 3 + 1] * b[1 * 3 + j] +
                     a[i * 3 + 2] * b[2 * 3 + j];
  return c;
}

// 3x3 rotation matrix -> quaternion [x,y,z,w] (matches util/math.rotmat_to_quat,
// reordered to scalar-last).
inline Quat rotmat_to_quat_xyzw(const Mat3 &m) {
  const double tr = m[0] + m[4] + m[8];
  double w, x, y, z;
  if (tr > 0.0) {
    double s = std::sqrt(tr + 1.0) * 2.0;
    w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
    w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
    w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s;
  } else {
    double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
    w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s;
  }
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  const double inv = n > 1e-12 ? 1.0 / n : 1.0;
  return {static_cast<float>(x * inv), static_cast<float>(y * inv),
          static_cast<float>(z * inv), static_cast<float>(w * inv)};
}

// Attitude quaternion body->world from PX4 NED/FRD to policy ENU/FLU.
// Input q is VehicleAttitude.q layout [w,x,y,z] (Hamilton, scalar first).
// Output [x,y,z,w]. R_enuflu = NED_TO_ENU * R_nedfrd * FRD_TO_FLU^T, with
//   NED_TO_ENU = [[0,1,0],[1,0,0],[0,0,-1]], FRD_TO_FLU = diag(1,-1,-1).
inline Quat ned_frd_quat_to_enu_flu(double qw, double qx, double qy, double qz) {
  const Mat3 R = quat_wxyz_to_rotmat(qw, qx, qy, qz);
  const Mat3 NED_TO_ENU{0, 1, 0, 1, 0, 0, 0, 0, -1};
  const Mat3 FRD_TO_FLU_T{1, 0, 0, 0, -1, 0, 0, 0, -1};  // diagonal, == its own T
  return rotmat_to_quat_xyzw(mat_mul(mat_mul(NED_TO_ENU, R), FRD_TO_FLU_T));
}

}  // namespace gbr

#endif  // GATE_BIN_RENDERER_CPP__FRAME_TRANSFORMS_HPP_
