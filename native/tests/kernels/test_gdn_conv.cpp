// Decode-step conv1d + SiLU and the gated RMSNorm of the Qwen3.x linear attention (src/kernels/gdn/conv.*)
// against FP64 host references. The CPU halves run everywhere; the CUDA comparison runs only on a 12.0
// device because the pack builds for sm_120a.
#include "common/test_util.h"

#include "kernels/gdn/conv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ingine::test;

namespace {

namespace gdn = ingine::kernels::gdn;
constexpr int K = gdn::CONV_K;
constexpr int VD = 128;

double silu64(double v) { return v / (1.0 + std::exp(-v)); }

// FP64 reference of the same conv, stepping the history exactly as the kernel contract says.
void conv_reference(const std::vector<double>& x, const std::vector<double>& weight, std::vector<double>& history,
                    std::vector<double>& y, int conv_dim) {
  for (int c = 0; c < conv_dim; ++c) {
    const double x0 = history[c];
    const double x1 = history[conv_dim + c];
    const double x2 = history[2 * conv_dim + c];
    const double xn = x[c];
    const double* w = weight.data() + std::size_t(c) * K;
    const double acc = w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * xn;
    y[c] = silu64(acc);
    history[c] = x1;
    history[conv_dim + c] = x2;
    history[2 * conv_dim + c] = xn;
  }
}

bool close(const char* tag, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
  double worst = 0.0;
  double rms = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(double(got[i]) - ref[i]);
    worst = std::max(worst, d);
    rms += ref[i] * ref[i];
  }
  rms = std::sqrt(rms / double(ref.size()));
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= tol || worst == 0.0;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", tag, worst, rms,
              ratio, tol);
  return ok;
}

int zeros_conv_identity() {
  const int d = 8;
  std::vector<float> x(d), weight(std::size_t(d) * K, 0.0f), history(3 * d, 0.0f), y(d, 0.0f);
  for (int c = 0; c < d; ++c) {
    x[c] = float(c) - 3.5f;
    weight[std::size_t(c) * K + 3] = 1.0f;  // only the current sample
  }
  gdn::ConvStepArgs a;
  a.x = x.data();
  a.weight = weight.data();
  a.history = history.data();
  a.y = y.data();
  a.conv_dim = d;
  gdn::conv_silu_step_cpu(a);

  std::vector<double> ref(d);
  for (int c = 0; c < d; ++c) ref[c] = silu64(double(x[c]));
  bool ok = close("conv identity taps", y, ref, 1e-6);
  for (int c = 0; c < d; ++c) {
    ok &= history[c] == 0.0f && history[d + c] == 0.0f && history[2 * d + c] == x[c];
  }
  return ok ? 0 : 1;
}

// Six random steps over a small and a model-sized channel count, against the FP64 reference.
int random_steps(int d, unsigned seed) {
  Rng rng(seed);
  std::vector<float> weight(std::size_t(d) * K), x(d), history(3 * d), y(d);
  for (auto& w : weight) w = float(rng.normal());
  for (auto& h : history) h = float(rng.normal());

  std::vector<double> ref_history(history.begin(), history.end());
  std::vector<double> ref_weight(weight.begin(), weight.end());
  std::vector<double> ref_x(d), ref_y(d);
  std::vector<double> ref_all;
  std::vector<float> got_all;

  gdn::ConvStepArgs a;
  a.weight = weight.data();
  a.history = history.data();
  a.y = y.data();
  a.conv_dim = d;
  for (int step = 0; step < 6; ++step) {
    for (int c = 0; c < d; ++c) {
      const float v = float(rng.normal());
      x[c] = v;
      ref_x[c] = double(v);
    }
    a.x = x.data();
    gdn::conv_silu_step_cpu(a);
    conv_reference(ref_x, ref_weight, ref_history, ref_y, d);
    got_all.insert(got_all.end(), y.begin(), y.end());
    ref_all.insert(ref_all.end(), ref_y.begin(), ref_y.end());
  }
  bool ok = close(("conv random d=" + std::to_string(d) + " y").c_str(), got_all, ref_all, 1e-5);
  ok &= close(("conv random d=" + std::to_string(d) + " history").c_str(), history, ref_history, 1e-5);
  return ok ? 0 : 1;
}

int gated_norm_case() {
  const int heads = 3;
  const int n = heads * VD;
  Rng rng(7);
  std::vector<float> x(n), z(n), weight(VD);
  for (auto& v : x) v = float(rng.normal());
  for (auto& v : z) v = float(rng.normal());
  for (auto& v : weight) v = float(0.5 + rng.normal() * 0.1);

  std::vector<double> ref(x.begin(), x.end());
  for (int h = 0; h < heads; ++h) {
    const int base = h * VD;
    double sum = 0.0;
    for (int d = 0; d < VD; ++d) sum += ref[base + d] * ref[base + d];
    const double inv = 1.0 / std::sqrt(sum / VD + 1e-6);
    for (int d = 0; d < VD; ++d) {
      ref[base + d] = ref[base + d] * inv * double(weight[d]) * silu64(double(z[base + d]));
    }
  }

  gdn::GatedNormArgs a;
  a.x = x.data();
  a.z = z.data();
  a.weight = weight.data();
  a.v_heads = heads;
  a.v_dim = VD;
  a.eps = 1e-6f;
  gdn::gated_norm_cpu(a);

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    const double d = std::fabs(double(x[i]) - ref[i]);
    const double scale = std::max(1e-6, std::fabs(ref[i]));
    if (d / scale > 1e-6) ok = false;
  }
  std::printf("%s gated norm vs FP64\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("skip CUDA comparison: no CUDA device\n");
    return false;
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
    std::printf("skip CUDA comparison: cannot query device 0\n");
    return false;
  }
  if (p.major != 12 || p.minor != 0) {
    std::printf("skip CUDA comparison: %s is sm_%d%d, the kernel builds for sm_120a\n", p.name, p.major, p.minor);
    return false;
  }
  return true;
}

int cuda_case() {
  const int d = 8192;  // the 35B-A3B conv width: 2 * 16 * 128 + 32 * 128
  const int heads = 32;
  Rng rng(11);
  std::vector<float> weight(std::size_t(d) * K), x(d), history(3 * d), y(d);
  for (auto& w : weight) w = float(rng.normal());
  for (auto& h : history) h = float(rng.normal());
  for (auto& v : x) v = float(rng.normal());

  std::vector<float> cpu_history = history, cpu_y(d);
  gdn::ConvStepArgs cpu;
  cpu.x = x.data();
  cpu.weight = weight.data();
  cpu.history = cpu_history.data();
  cpu.y = cpu_y.data();
  cpu.conv_dim = d;
  gdn::conv_silu_step_cpu(cpu);

  DeviceArray<float> dw(weight), dx(x), dh(history), dy(d);
  gdn::ConvStepArgs dev;
  dev.x = dx.get();
  dev.weight = dw.get();
  dev.history = dh.get();
  dev.y = dy.get();
  dev.conv_dim = d;
  if (gdn::conv_silu_step(dev, nullptr) != cudaSuccess) {
    std::printf("FAIL conv launch\n");
    return 1;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  bool ok = true;
  {
    const std::vector<float> gy = dy.download(), gh = dh.download();
    std::vector<double> ry(cpu_y.begin(), cpu_y.end());
    std::vector<double> rh(cpu_history.begin(), cpu_history.end());
    ok &= close("cuda conv y", gy, ry, 1e-6);
    ok &= close("cuda conv history", gh, rh, 1e-6);
  }

  const int n = heads * VD;
  std::vector<float> gx(n), gz(n), gw(VD);
  for (auto& v : gx) v = float(rng.normal());
  for (auto& v : gz) v = float(rng.normal());
  for (auto& v : gw) v = float(0.5 + rng.normal() * 0.1);
  std::vector<float> cpu_gx = gx;
  gdn::GatedNormArgs ca;
  ca.x = cpu_gx.data();
  ca.z = gz.data();
  ca.weight = gw.data();
  ca.v_heads = heads;
  ca.v_dim = VD;
  gdn::gated_norm_cpu(ca);

  DeviceArray<float> dgx(gx), dgz(gz), dgw(gw);
  gdn::GatedNormArgs da;
  da.x = dgx.get();
  da.z = dgz.get();
  da.weight = dgw.get();
  da.v_heads = heads;
  da.v_dim = VD;
  if (gdn::gated_norm(da, nullptr) != cudaSuccess) {
    std::printf("FAIL gated norm launch\n");
    return 1;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  {
    const std::vector<float> g = dgx.download();
    std::vector<double> r(cpu_gx.begin(), cpu_gx.end());
    ok &= close("cuda gated norm", g, r, 1e-5);
  }
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int failures = zeros_conv_identity() + random_steps(17, 3) + random_steps(8192, 4) + gated_norm_case();
  if (have_matching_gpu()) failures += cuda_case();
  if (failures == 0) std::printf("ALL PASS\n");
  return failures == 0 ? 0 : 1;
}
