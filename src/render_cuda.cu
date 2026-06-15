// CUDA backend. Compiled only when the package is configured with a CUDA
// toolkit (see CMakeLists). The per-pixel arithmetic comes from the SAME
// __host__ __device__ evaluators in gate_render.hpp that the CPU path uses, so
// the two backends are bit-for-bit consistent up to float ordering.
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "gate_bin_renderer_cpp/renderer.hpp"

namespace gbr {

namespace {

#define GBR_CUDA_CHECK(call)                                                     \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "[gate_bin_renderer cuda] %s:%d %s\n", __FILE__,     \
                   __LINE__, cudaGetErrorString(_e));                           \
    }                                                                          \
  } while (0)

__global__ void wire_kernel(CameraSpec c, WireSolved s, Mode mode, float tau,
                            int xoff, int total_w, float *out) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= c.width || v >= c.height) return;
  out[static_cast<long>(v) * total_w + xoff + u] = wire_value(u, v, c, s, mode, tau);
}

__global__ void fill_kernel(CameraSpec c, FillSolved s, int xoff, int total_w,
                            float *out) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= c.width || v >= c.height) return;
  out[static_cast<long>(v) * total_w + xoff + u] = fill_value(u, v, c, s);
}

class CudaRenderer : public IGateRenderer {
 public:
  CudaRenderer(const std::vector<CameraSpec> &cams, Mode mode, float tau)
      : cams_(cams), mode_(mode), tau_(tau) {
    height_ = cams_.empty() ? 0 : cams_[0].height;
    total_w_ = gbr::total_width(cams_);
    n_ = static_cast<size_t>(height_) * total_w_;
    GBR_CUDA_CHECK(cudaMalloc(&d_out_, n_ * sizeof(float)));
  }

  ~CudaRenderer() override {
    if (d_out_) cudaFree(d_out_);
  }

  void render(const Vec3 &drone_pos, const Quat &drone_quat, const Vec3 &gate_pos,
              const Quat &gate_quat, float *out) override {
    const dim3 block(16, 16);
    int xoff = 0;
    for (const auto &c : cams_) {
      const dim3 grid((c.width + block.x - 1) / block.x,
                      (c.height + block.y - 1) / block.y);
      if (mode_ == Mode::kFilled) {
        const FillSolved s = solve_fill(c, drone_pos, drone_quat, gate_pos, gate_quat);
        fill_kernel<<<grid, block>>>(c, s, xoff, total_w_, d_out_);
      } else {
        const WireSolved s = solve_wire(c, drone_pos, drone_quat, gate_pos, gate_quat);
        wire_kernel<<<grid, block>>>(c, s, mode_, tau_, xoff, total_w_, d_out_);
      }
      xoff += c.width;
    }
    GBR_CUDA_CHECK(cudaGetLastError());
    GBR_CUDA_CHECK(cudaMemcpy(out, d_out_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
  }

  int height() const override { return height_; }
  int total_width() const override { return total_w_; }
  const char *backend_name() const override { return "cuda"; }

 private:
  std::vector<CameraSpec> cams_;
  Mode mode_;
  float tau_;
  int height_ = 0;
  int total_w_ = 0;
  size_t n_ = 0;
  float *d_out_ = nullptr;
};

}  // namespace

bool cuda_device_available() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    // Clear the sticky error so later CUDA calls aren't poisoned.
    cudaGetLastError();
    return false;
  }
  return n > 0;
}

std::unique_ptr<IGateRenderer> make_cuda_renderer(const std::vector<CameraSpec> &cams,
                                                  Mode mode, float tau) {
  return std::unique_ptr<IGateRenderer>(new CudaRenderer(cams, mode, tau));
}

}  // namespace gbr
