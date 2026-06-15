// Backend-agnostic gate-mask renderer interface + factory.
#ifndef GATE_BIN_RENDERER_CPP__RENDERER_HPP_
#define GATE_BIN_RENDERER_CPP__RENDERER_HPP_

#include <memory>
#include <vector>

#include "gate_bin_renderer_cpp/camera_presets.hpp"
#include "gate_bin_renderer_cpp/gate_render.hpp"

namespace gbr {

class IGateRenderer {
 public:
  virtual ~IGateRenderer() = default;

  // Render the full (height x total_width) mask, row-major, into `out` (caller
  // owns; must hold height*total_width floats). Values in [0,1].
  virtual void render(const Vec3 &drone_pos, const Quat &drone_quat,
                      const Vec3 &gate_pos, const Quat &gate_quat,
                      float *out) = 0;

  virtual int height() const = 0;
  virtual int total_width() const = 0;
  virtual const char *backend_name() const = 0;
};

std::unique_ptr<IGateRenderer> make_cpu_renderer(const std::vector<CameraSpec> &cams,
                                                 Mode mode, float tau);

// Returns a CUDA renderer when `want_gpu` and a usable CUDA device is present
// (and the package was built with CUDA); otherwise falls back to CPU. If
// `using_gpu` is non-null it is set to the chosen backend.
std::unique_ptr<IGateRenderer> make_renderer(const std::vector<CameraSpec> &cams,
                                             Mode mode, float tau, bool want_gpu,
                                             bool *using_gpu);

}  // namespace gbr

#endif  // GATE_BIN_RENDERER_CPP__RENDERER_HPP_
