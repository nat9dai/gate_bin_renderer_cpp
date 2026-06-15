// Core gate binary-mask rendering — a faithful C++ port of the dva-quad-jax
// reference rasterisers:
//   envs/tasks/gate_traversal_binary_mask.py        (1-cam)
//   envs/tasks/gate_traversal_binary_mask_2_cam.py  (2-cam, FullCameraConfig)
//   envs/tasks/gate_traversal_binary_mask_4_cam.py  (4-cam)
//
// Two stages, deliberately split so the CPU loop and the CUDA kernel run the
// IDENTICAL per-pixel arithmetic (the host pre-solve happens once per camera):
//
//   1. host  : solve_wire() / solve_fill() project the gate geometry into each
//              camera (corner projection + near-plane clip + off-screen
//              extension, or the ray/plane setup) -> a tiny per-camera struct.
//   2. device: wire_value() / fill_value() evaluate one pixel from that struct.
//
// Frames (verbatim from the reference):
//   * quaternion [x,y,z,w]; body FLU (x-fwd, y-left, z-up); world ENU.
//   * OpenCV camera frame: x_cam=-y_mount, y_cam=-z_mount, z_cam=+x_mount.
//   * body->mount: v_mount_row = v_body_row @ R_bc  (R_bc rows = R_bc[r*3+c]).
#ifndef GATE_BIN_RENDERER_CPP__GATE_RENDER_HPP_
#define GATE_BIN_RENDERER_CPP__GATE_RENDER_HPP_

#include "gate_bin_renderer_cpp/math3.hpp"

namespace gbr {

// Gate geometry (gate_traversal.py: GATE_WIDTH=0.60, GATE_HEIGHT=0.25).
constexpr float GATE_WIDTH = 0.60f;
constexpr float GATE_HEIGHT = 0.25f;
constexpr float GATE_HW = GATE_WIDTH * 0.5f;   // inner-opening half-width  (0.30)
constexpr float GATE_HH = GATE_HEIGHT * 0.5f;  // inner-opening half-height (0.125)
// Outer-frame half-extents (binary_mask.py: derived from gate.usd mesh).
constexpr float GATE_FRAME_HW_OUTER = GATE_WIDTH * 0.937f / 1.524f;
constexpr float GATE_FRAME_HH_OUTER = GATE_HEIGHT * 1.067f / 1.524f;

enum class Mode { kWireframe = 0, kFilled = 1, kSoft = 2 };

// Pinhole camera + extrinsics for one camera. (1-cam: R_bc = Ry(-pitch).)
struct CameraSpec {
  int width = 84;
  int height = 84;
  float fx = 40.0f, fy = 40.0f, cx = 42.0f, cy = 42.0f;
  float near = 0.001f;
  float line_thickness = 1.5f;
  float offset_b[3] = {0.10f, 0.0f, 0.04f};  // body FLU optical-centre offset
  float R_bc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // body->mount, row-major
};

// Per-camera solution for the wireframe / soft rasteriser: the 4 opening edges
// already projected to pixel space (near-clipped + off-screen-extended).
struct WireSolved {
  float u1[4], v1[4], u2[4], v2[4];
  int valid[4];
};

// Per-camera solution for the filled (ray-cast) rasteriser.
struct FillSolved {
  float M[9];          // d_gate = M * d_mount (column conv): M[i*3+j]
  float cam_pos_g[3];  // camera optical centre in gate frame
};

// ── device-callable per-pixel evaluators ────────────────────────────────────

// Squared distance (px^2) from sample (px,py) to the nearest valid opening edge.
GBR_HD inline float wire_min_dist2(float px, float py, const WireSolved &s) {
  float best = 1e30f;
  for (int e = 0; e < 4; ++e) {
    if (!s.valid[e]) continue;
    const float dx = s.u2[e] - s.u1[e];
    const float dy = s.v2[e] - s.v1[e];
    const float len2 = dx * dx + dy * dy + 1e-9f;
    float t = ((px - s.u1[e]) * dx + (py - s.v1[e]) * dy) / len2;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const float qx = s.u1[e] + t * dx;
    const float qy = s.v1[e] + t * dy;
    const float d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy);
    if (d2 < best) best = d2;
  }
  return best;
}

// Wireframe/soft pixel value in [0,1]. (px,py) sampled at integer pixel coords,
// matching the JAX `jnp.arange(W)` grid (no +0.5 centre offset).
GBR_HD inline float wire_value(int u, int v, const CameraSpec &c, const WireSolved &s,
                               Mode mode, float tau) {
  const float d2 = wire_min_dist2(static_cast<float>(u), static_cast<float>(v), s);
  if (mode == Mode::kSoft) return expf(-d2 / (2.0f * tau * tau));
  const float thr = c.line_thickness * 0.5f;
  return (d2 < thr * thr) ? 1.0f : 0.0f;
}

// Filled (ray-cast) pixel value in {0,1}: 1 where the back-projected ray hits
// the gate frame band (inside outer rect, outside inner opening).
GBR_HD inline float fill_value(int u, int v, const CameraSpec &c, const FillSolved &s) {
  const float dmx = 1.0f;
  const float dmy = -(static_cast<float>(u) - c.cx) / c.fx;
  const float dmz = -(static_cast<float>(v) - c.cy) / c.fy;
  const float gx = s.M[0] * dmx + s.M[1] * dmy + s.M[2] * dmz;
  const float gy = s.M[3] * dmx + s.M[4] * dmy + s.M[5] * dmz;
  const float gz = s.M[6] * dmx + s.M[7] * dmy + s.M[8] * dmz;
  if (fabsf(gx) <= 1e-9f) return 0.0f;
  const float t = -s.cam_pos_g[0] / gx;
  if (!(t > 0.0f)) return 0.0f;
  const float yg = s.cam_pos_g[1] + t * gy;
  const float zg = s.cam_pos_g[2] + t * gz;
  const bool in_outer = (fabsf(yg) <= GATE_FRAME_HW_OUTER) && (fabsf(zg) <= GATE_FRAME_HH_OUTER);
  const bool in_inner = (fabsf(yg) <= GATE_HW) && (fabsf(zg) <= GATE_HH);
  return (in_outer && !in_inner) ? 1.0f : 0.0f;
}

// ── host pre-solve (runs once per camera per frame) ─────────────────────────

// Camera optical centre in world: CG + R(drone_quat) * offset_b.
inline Vec3 camera_pos_world(const Vec3 &drone_pos, const Quat &drone_quat,
                             const CameraSpec &c) {
  const Vec3 off{c.offset_b[0], c.offset_b[1], c.offset_b[2]};
  return v_add(drone_pos, quat_apply(drone_quat, off));
}

// Project the 4 inner-opening corners into one camera and prepare the edges.
// Mirrors _compute_vis_obs_jax_wireframe / _per_cam_wireframe_jax exactly.
inline WireSolved solve_wire(const CameraSpec &c, const Vec3 &drone_pos,
                             const Quat &drone_quat, const Vec3 &gate_pos,
                             const Quat &gate_quat) {
  const Vec3 corners_local[4] = {
      {0.0f, GATE_HW, GATE_HH},    // top-right
      {0.0f, -GATE_HW, GATE_HH},   // top-left
      {0.0f, -GATE_HW, -GATE_HH},  // bottom-left
      {0.0f, GATE_HW, -GATE_HH},   // bottom-right
  };
  const Vec3 cam_pos_w = camera_pos_world(drone_pos, drone_quat, c);

  // corners -> world -> body(FLU) -> mount -> OpenCV camera frame.
  Vec3 ccam[4];
  for (int k = 0; k < 4; ++k) {
    const Vec3 cw = v_add(gate_pos, quat_apply(gate_quat, corners_local[k]));
    const Vec3 cb = quat_apply_inv(drone_quat, v_sub(cw, cam_pos_w));
    // mount: cm[j] = sum_i cb[i] * R_bc[i*3+j]   (row-vector @ R_bc)
    const float mx = cb.x * c.R_bc[0] + cb.y * c.R_bc[3] + cb.z * c.R_bc[6];
    const float my = cb.x * c.R_bc[1] + cb.y * c.R_bc[4] + cb.z * c.R_bc[7];
    const float mz = cb.x * c.R_bc[2] + cb.y * c.R_bc[5] + cb.z * c.R_bc[8];
    ccam[k] = {-my, -mz, mx};  // x_cam=-y_mount, y_cam=-z_mount, z_cam=+x_mount
  }

  const float near = c.near;
  const float ext = 5.0f * static_cast<float>(c.width > c.height ? c.width : c.height);

  WireSolved s;
  for (int k = 0; k < 4; ++k) {
    const Vec3 p1 = ccam[k];
    const Vec3 p2 = ccam[(k + 1) & 3];
    const float z1 = p1.z, z2 = p2.z;

    // Near-plane clip per endpoint (no branching on the value, as in JAX).
    const float dz12 = z2 - z1;
    const float safe12 = (fabsf(dz12) > 1e-9f) ? dz12 : 1.0f;
    const float t1 = (near - z1) / safe12;
    const Vec3 p1c = (z1 < near) ? v_add(p1, v_scale(v_sub(p2, p1), t1)) : p1;
    const float dz21 = z1 - z2;
    const float safe21 = (fabsf(dz21) > 1e-9f) ? dz21 : 1.0f;
    const float t2 = (near - z2) / safe21;
    const Vec3 p2c = (z2 < near) ? v_add(p2, v_scale(v_sub(p1, p2), t2)) : p2;

    const int valid = (z1 > near) || (z2 > near);

    const float z1c = p1c.z > near ? p1c.z : near;
    const float z2c = p2c.z > near ? p2c.z : near;
    float u1 = c.fx * p1c.x / z1c + c.cx;
    float v1 = c.fy * p1c.y / z1c + c.cy;
    float u2 = c.fx * p2c.x / z2c + c.cx;
    float v2 = c.fy * p2c.y / z2c + c.cy;

    // Off-screen extension of near-clipped endpoints (sequential, as in JAX:
    // the endpoint-2 direction uses the already-updated endpoint 1).
    if (z1 < near) {
      const float d1x = u1 - u2, d1y = v1 - v2;
      const float n1 = sqrtf(d1x * d1x + d1y * d1y) + 1e-9f;
      u1 = u2 + d1x / n1 * ext;
      v1 = v2 + d1y / n1 * ext;
    }
    if (z2 < near) {
      const float d2x = u2 - u1, d2y = v2 - v1;
      const float n2 = sqrtf(d2x * d2x + d2y * d2y) + 1e-9f;
      u2 = u1 + d2x / n2 * ext;
      v2 = v1 + d2y / n2 * ext;
    }

    s.u1[k] = u1; s.v1[k] = v1; s.u2[k] = u2; s.v2[k] = v2; s.valid[k] = valid;
  }
  return s;
}

// Prepare the ray-cast (filled) per-camera transform. Mirrors
// _compute_vis_obs_jax_filled: d_gate = R_gate^T R_drone R_bc d_mount.
inline FillSolved solve_fill(const CameraSpec &c, const Vec3 &drone_pos,
                             const Quat &drone_quat, const Vec3 &gate_pos,
                             const Quat &gate_quat) {
  FillSolved s;
  for (int j = 0; j < 3; ++j) {
    const Vec3 rbc_col{c.R_bc[0 * 3 + j], c.R_bc[1 * 3 + j], c.R_bc[2 * 3 + j]};
    const Vec3 w = quat_apply(drone_quat, rbc_col);   // mount basis -> world
    const Vec3 g = quat_apply_inv(gate_quat, w);      // world -> gate
    s.M[0 * 3 + j] = g.x; s.M[1 * 3 + j] = g.y; s.M[2 * 3 + j] = g.z;
  }
  const Vec3 cam_pos_w = camera_pos_world(drone_pos, drone_quat, c);
  const Vec3 cg = quat_apply_inv(gate_quat, v_sub(cam_pos_w, gate_pos));
  s.cam_pos_g[0] = cg.x; s.cam_pos_g[1] = cg.y; s.cam_pos_g[2] = cg.z;
  return s;
}

}  // namespace gbr

#endif  // GATE_BIN_RENDERER_CPP__GATE_RENDER_HPP_
