// Offline gate-mask renderer for parity testing / debugging (no ROS deps).
//
// Reads pose cases from stdin (one per line, 14 floats):
//   dpx dpy dpz dqx dqy dqz dqw  gpx gpy gpz gqx gqy gqz gqw
// (drone position + quaternion[xyzw], gate position + quaternion[xyzw], in the
//  sim world ENU / body FLU frame — i.e. the same convention as the JAX env).
//
// Writes the masks to stdout. Default format is raw little-endian float32:
//   n_cases * height * total_width values, row-major per case. Use --format txt
//   for a human-readable 0/1 grid (one blank line between cases).
//
// Usage:
//   gate_render_cli --ncam 2 --mode wireframe --backend auto [--tau 1.5]
//                   [--format f32|txt] < cases.txt > out.f32
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gate_bin_renderer_cpp/renderer.hpp"

int main(int argc, char **argv) {
  int ncam = 2;
  std::string mode = "wireframe";
  std::string backend = "auto";
  float tau = 1.5f;
  std::string format = "f32";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (a == "--ncam") ncam = std::stoi(next());
    else if (a == "--mode") mode = next();
    else if (a == "--backend") backend = next();
    else if (a == "--tau") tau = std::stof(next());
    else if (a == "--format") format = next();
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  std::vector<gbr::CameraSpec> cams;
  gbr::Mode m;
  try {
    cams = gbr::make_cameras(ncam);
    m = gbr::parse_mode(mode);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "config error: %s\n", e.what());
    return 2;
  }

  bool using_gpu = false;
  std::unique_ptr<gbr::IGateRenderer> r;
  if (backend == "cpu") {
    r = gbr::make_cpu_renderer(cams, m, tau);
  } else {  // auto or cuda -> prefer GPU when available
    r = gbr::make_renderer(cams, m, tau, /*want_gpu=*/true, &using_gpu);
  }
  const int H = r->height(), W = r->total_width();
  std::fprintf(stderr, "[gate_render_cli] ncam=%d mode=%s backend=%s dims=%dx%d\n",
               ncam, mode.c_str(), r->backend_name(), H, W);

  std::vector<float> mask(static_cast<size_t>(H) * W);
  std::string line;
  long n_cases = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    float f[14];
    bool ok = true;
    for (int k = 0; k < 14; ++k) {
      if (!(ss >> f[k])) { ok = false; break; }
    }
    if (!ok) { std::fprintf(stderr, "skipping malformed line\n"); continue; }

    const gbr::Vec3 dp{f[0], f[1], f[2]};
    const gbr::Quat dq{f[3], f[4], f[5], f[6]};
    const gbr::Vec3 gp{f[7], f[8], f[9]};
    const gbr::Quat gq{f[10], f[11], f[12], f[13]};
    r->render(dp, dq, gp, gq, mask.data());

    if (format == "f32") {
      std::fwrite(mask.data(), sizeof(float), mask.size(), stdout);
    } else {
      for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
          const float val = mask[static_cast<size_t>(v) * W + u];
          std::putchar(val > 0.5f ? '#' : '.');
        }
        std::putchar('\n');
      }
      std::putchar('\n');
    }
    ++n_cases;
  }
  std::fprintf(stderr, "[gate_render_cli] rendered %ld case(s)\n", n_cases);
  return 0;
}
