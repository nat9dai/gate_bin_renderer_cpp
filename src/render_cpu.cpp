#include <cstring>
#include <vector>

#include "gate_bin_renderer_cpp/renderer.hpp"

namespace gbr {

namespace {

class CpuRenderer : public IGateRenderer {
 public:
  CpuRenderer(const std::vector<CameraSpec> &cams, Mode mode, float tau)
      : cams_(cams), mode_(mode), tau_(tau) {
    height_ = cams_.empty() ? 0 : cams_[0].height;
    total_w_ = gbr::total_width(cams_);
  }

  void render(const Vec3 &drone_pos, const Quat &drone_quat, const Vec3 &gate_pos,
              const Quat &gate_quat, float *out) override {
    int xoff = 0;
    for (const auto &c : cams_) {
      const int W = c.width, H = c.height;
      if (mode_ == Mode::kFilled) {
        const FillSolved s = solve_fill(c, drone_pos, drone_quat, gate_pos, gate_quat);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int v = 0; v < H; ++v) {
          float *row = out + static_cast<long>(v) * total_w_ + xoff;
          for (int u = 0; u < W; ++u) row[u] = fill_value(u, v, c, s);
        }
      } else {
        const WireSolved s = solve_wire(c, drone_pos, drone_quat, gate_pos, gate_quat);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int v = 0; v < H; ++v) {
          float *row = out + static_cast<long>(v) * total_w_ + xoff;
          for (int u = 0; u < W; ++u) row[u] = wire_value(u, v, c, s, mode_, tau_);
        }
      }
      xoff += W;
    }
  }

  int height() const override { return height_; }
  int total_width() const override { return total_w_; }
  const char *backend_name() const override { return "cpu"; }

 private:
  std::vector<CameraSpec> cams_;
  Mode mode_;
  float tau_;
  int height_ = 0;
  int total_w_ = 0;
};

}  // namespace

std::unique_ptr<IGateRenderer> make_cpu_renderer(const std::vector<CameraSpec> &cams,
                                                 Mode mode, float tau) {
  return std::unique_ptr<IGateRenderer>(new CpuRenderer(cams, mode, tau));
}

#ifdef GBR_HAVE_CUDA
std::unique_ptr<IGateRenderer> make_cuda_renderer(const std::vector<CameraSpec> &cams,
                                                  Mode mode, float tau);
bool cuda_device_available();
#endif

std::unique_ptr<IGateRenderer> make_renderer(const std::vector<CameraSpec> &cams,
                                             Mode mode, float tau, bool want_gpu,
                                             bool *using_gpu) {
#ifdef GBR_HAVE_CUDA
  if (want_gpu && cuda_device_available()) {
    if (using_gpu) *using_gpu = true;
    return make_cuda_renderer(cams, mode, tau);
  }
#else
  (void)want_gpu;
#endif
  if (using_gpu) *using_gpu = false;
  return make_cpu_renderer(cams, mode, tau);
}

}  // namespace gbr
