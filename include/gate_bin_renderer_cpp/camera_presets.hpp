// Camera presets matching the dva-quad-jax reference configs, per camera count:
//   1-cam : CameraConfig()            84x84,  fx=fy=40,  cx=cy=42, pitch 45 deg
//   2-cam : default_front_two_cam     150x84, fx=fy=10.1,cx=75,cy=42 (FL, FR)
//   4-cam : default_four_cam          150x84, fx=fy=10.1,cx=75,cy=42 (FR,BR,BL,FL)
//
// The mask is the horizontal concat of the per-camera masks in this order, so
// the published image has width = ncam * W (e.g. 2-cam => 84x300).
#ifndef GATE_BIN_RENDERER_CPP__CAMERA_PRESETS_HPP_
#define GATE_BIN_RENDERER_CPP__CAMERA_PRESETS_HPP_

#include <stdexcept>
#include <string>
#include <vector>

#include "gate_bin_renderer_cpp/gate_render.hpp"

namespace gbr {

// Build the ordered camera list for `ncam` in {1,2,4}. `tilt_deg` overrides the
// 1-cam optical-axis up-tilt (default 45 deg = CAM_PITCH_DEG); ignored for 2/4.
std::vector<CameraSpec> make_cameras(int ncam);

// Total mask width in pixels for a camera list (sum of per-cam widths; all
// cameras in a preset share intrinsics, so this is ncam * W).
inline int total_width(const std::vector<CameraSpec> &cams) {
  int w = 0;
  for (const auto &c : cams) w += c.width;
  return w;
}

inline Mode parse_mode(const std::string &m) {
  if (m == "wireframe") return Mode::kWireframe;
  if (m == "filled") return Mode::kFilled;
  if (m == "soft") return Mode::kSoft;
  throw std::invalid_argument("mode must be wireframe|filled|soft, got: " + m);
}

}  // namespace gbr

#endif  // GATE_BIN_RENDERER_CPP__CAMERA_PRESETS_HPP_
