#include "gate_bin_renderer_cpp/camera_presets.hpp"

#include <cmath>

namespace gbr {

namespace {

// Fill a CameraSpec's R_bc (row-major) from a 3x3 given as 9 row-major values.
void set_rbc(CameraSpec &c, const float r[9]) {
  for (int i = 0; i < 9; ++i) c.R_bc[i] = r[i];
}

// ── 1-cam: CameraConfig() with optical axis up-tilted by CAM_PITCH_DEG ───────
// R_bc = R_y(-p): [[cp,0,-sp],[0,1,0],[sp,0,cp]]  (binary_mask.py).
CameraSpec one_cam() {
  CameraSpec c;
  c.width = 84; c.height = 84;
  c.fx = 40.0f; c.fy = 40.0f; c.cx = 42.0f; c.cy = 42.0f;
  c.near = 0.001f; c.line_thickness = 1.5f;
  c.offset_b[0] = 0.10f; c.offset_b[1] = 0.0f; c.offset_b[2] = 0.04f;  // CAM_OFFSET_B
  const float p = static_cast<float>(45.0 * M_PI / 180.0);  // CAM_PITCH_DEG
  const float cp = std::cos(p), sp = std::sin(p);
  const float r[9] = {cp, 0.0f, -sp, 0.0f, 1.0f, 0.0f, sp, 0.0f, cp};
  set_rbc(c, r);
  return c;
}

// ── OAK-FFC-4P calibration (2026-04-06), body FLU; R_bc rows from the env ────
CameraSpec full_cam(const float offset_b[3], const float r[9]) {
  CameraSpec c;
  c.width = 150; c.height = 84;
  c.fx = 10.1f; c.fy = 10.1f; c.cx = 75.0f; c.cy = 42.0f;
  c.near = 0.001f; c.line_thickness = 1.5f;
  c.offset_b[0] = offset_b[0]; c.offset_b[1] = offset_b[1]; c.offset_b[2] = offset_b[2];
  set_rbc(c, r);
  return c;
}

// cam0 (CAM_A): front-right, yaw ~ -45 deg
const float CAM0_POS[3] = {0.100007f, -0.041956f, 0.058455f};
const float CAM0_RBC[9] = {0.707506f,  0.706200f,  0.026788f,
                           -0.706624f, 0.707497f,  0.011422f,
                           -0.010886f, -0.027010f, 0.999576f};
// cam1 (CAM_B): rear-right, yaw ~ -135 deg
const float CAM1_POS[3] = {-0.114741f, -0.048489f, 0.058851f};
const float CAM1_RBC[9] = {-0.701632f, 0.712364f,  0.015848f,
                           -0.712530f, -0.701563f, -0.010478f,
                           0.003654f,  -0.018643f, 0.999820f};
// cam2 (CAM_C): rear-left, yaw ~ +135 deg
const float CAM2_POS[3] = {-0.114514f, 0.038092f, 0.060604f};
const float CAM2_RBC[9] = {-0.705534f, -0.708287f, 0.023485f,
                           0.708234f,  -0.705877f, -0.011913f,
                           0.025015f,  0.008228f,  0.999653f};
// cam3 (CAM_D): front-left, yaw ~ +45 deg
const float CAM3_POS[3] = {0.096321f, 0.040626f, 0.062413f};
const float CAM3_RBC[9] = {0.705098f,  -0.708626f, 0.026189f,
                           0.709017f,  0.705123f,  -0.009853f,
                           -0.011485f, 0.025516f,  0.999608f};

}  // namespace

std::vector<CameraSpec> make_cameras(int ncam) {
  switch (ncam) {
    case 1:
      return {one_cam()};
    case 2:
      // [left = cam3 (FL) | right = cam0 (FR)] — default_front_two_cam_configs.
      return {full_cam(CAM3_POS, CAM3_RBC), full_cam(CAM0_POS, CAM0_RBC)};
    case 4:
      // [cam0 (FR) | cam1 (BR) | cam2 (BL) | cam3 (FL)] — default_four_cam_configs.
      return {full_cam(CAM0_POS, CAM0_RBC), full_cam(CAM1_POS, CAM1_RBC),
              full_cam(CAM2_POS, CAM2_RBC), full_cam(CAM3_POS, CAM3_RBC)};
    default:
      throw std::invalid_argument("ncam must be 1, 2 or 4");
  }
}

}  // namespace gbr
