// Routed-expert math of the Qwen3.6-35B-A3B MoE layers: h_k = silu(gate_k x) * (up_k x), then
// y = sum_k w_k * (down_k h_k). Weights are synthetic NVFP4 bytes; the reference decodes the same bytes with
// the CPU GEMV and does the gating and weighting in FP64.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/moe/moe_apply.h"
#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace moe = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;
constexpr int K = moe::MOE_TOP_K;
constexpr int INTER = moe::MOE_INTER;
constexpr int HID = moe::MOE_HIDDEN;
constexpr int E = 16;  // only the selected ids need real data

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  if (p.major != 12 || p.minor != 0) {
    std::printf("skip CUDA comparison: %s is sm_%d%d, the kernel builds for sm_120a\n", p.name, p.major, p.minor);
    return false;
  }
  return true;
}

struct ExpertData {
  std::vector<std::uint8_t> gate_w, gate_s, up_w, up_s, down_w, down_s;
  float gate_scale = 0.0f, up_scale = 0.0f, down_scale = 0.0f;
};

ExpertData make_expert(Rng& rng) {
  ExpertData e;
  e.gate_w.resize(std::size_t(INTER) * HID / 2);
  e.up_w.resize(std::size_t(INTER) * HID / 2);
  e.down_w.resize(std::size_t(HID) * INTER / 2);
  e.gate_s.resize(std::size_t(INTER) * HID / 16);
  e.up_s.resize(std::size_t(INTER) * HID / 16);
  e.down_s.resize(std::size_t(HID) * INTER / 16);
  for (auto& b : e.gate_w) b = std::uint8_t(rng.uniform() * 255.0);
  for (auto& b : e.up_w) b = std::uint8_t(rng.uniform() * 255.0);
  for (auto& b : e.down_w) b = std::uint8_t(rng.uniform() * 255.0);
  for (auto& b : e.gate_s) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  for (auto& b : e.up_s) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  for (auto& b : e.down_s) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  e.gate_scale = 0.001f + float(rng.uniform()) * 0.001f;
  e.up_scale = 0.001f + float(rng.uniform()) * 0.001f;
  e.down_scale = 0.001f + float(rng.uniform()) * 0.001f;
  return e;
}

// FP64 reference over the same bytes.
std::vector<double> reference(const std::vector<ExpertData>& data, const std::vector<float>& x,
                              const std::vector<int>& ids, const std::vector<float>& w) {
  std::vector<double> y(HID, 0.0);
  for (int k = 0; k < K; ++k) {
    const ExpertData& e = data[std::size_t(ids[k])];
    std::vector<float> g(INTER), u(INTER), h(INTER), d(HID);
    quant::Nvfp4GemvArgs a;
    a.w = e.gate_w.data();
    a.scales = e.gate_s.data();
    a.scale = e.gate_scale;
    a.x = const_cast<float*>(x.data());
    a.y = g.data();
    a.rows = INTER;
    a.cols = HID;
    quant::nvfp4_gemv_cpu(a);
    a.w = e.up_w.data();
    a.scales = e.up_s.data();
    a.scale = e.up_scale;
    a.y = u.data();
    quant::nvfp4_gemv_cpu(a);
    for (int i = 0; i < INTER; ++i) {
      const double v = g[i];
      h[i] = float(v / (1.0 + std::exp(-v)) * double(u[i]));
    }
    a.w = e.down_w.data();
    a.scales = e.down_s.data();
    a.scale = e.down_scale;
    a.x = h.data();
    a.y = d.data();
    a.rows = HID;
    a.cols = INTER;
    quant::nvfp4_gemv_cpu(a);
    for (int i = 0; i < HID; ++i) y[i] += double(w[k]) * double(d[i]);
  }
  return y;
}

bool run_case() {
  Rng rng(31);
  std::vector<ExpertData> data(E);
  for (auto& e : data) e = make_expert(rng);
  std::vector<float> x(HID);
  for (auto& v : x) v = float(rng.normal());
  std::vector<int> ids = {3, 0, 9, 15, 7, 1, 12, 5};
  std::vector<float> w(K);
  float sum = 0.0f;
  for (auto& v : w) v = float(rng.uniform() + 0.01f);
  for (float v : w) sum += v;
  for (auto& v : w) v /= sum;

  const std::vector<double> ref = reference(data, x, ids, w);

  DeviceArray<float> dx(x), dy(HID), dh(INTER), dup(INTER), dd(HID);
  std::vector<quant::Nvfp4GemvArgs> dev(0);
  std::vector<DeviceArray<std::uint8_t>> gw, gs, uw, us, dw, ds;
  gw.reserve(E);
  for (int e = 0; e < E; ++e) {
    gw.emplace_back(data[e].gate_w);
    gs.emplace_back(data[e].gate_s);
    uw.emplace_back(data[e].up_w);
    us.emplace_back(data[e].up_s);
    dw.emplace_back(data[e].down_w);
    ds.emplace_back(data[e].down_s);
  }
  std::vector<moe::MoeExpert> experts(E);
  for (int e = 0; e < E; ++e) {
    experts[e].gate = {gw[e].get(), gs[e].get(), data[e].gate_scale, dx.get(), dh.get(), INTER, HID};
    experts[e].up = {uw[e].get(), us[e].get(), data[e].up_scale, dx.get(), dup.get(), INTER, HID};
    experts[e].down = {dw[e].get(), ds[e].get(), data[e].down_scale, dh.get(), dd.get(), HID, INTER};
  }

  moe::MoeApplyArgs a;
  a.x = dx.get();
  a.ids = ids.data();
  a.weights = w.data();
  a.experts = experts.data();
  a.y = dy.get();
  a.scratch_h = dh.get();
  a.scratch_up = dup.get();
  a.scratch_down = dd.get();
  if (moe::moe_apply(a, nullptr) != cudaSuccess) {
    std::printf("FAIL moe launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> got = dy.download();

  double worst = 0.0, rms = 0.0;
  for (int i = 0; i < HID; ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - ref[i]));
    rms += ref[i] * ref[i];
  }
  rms = std::sqrt(rms / HID);
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= 1e-3;
  std::printf("%s cuda moe y: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol 1.0e-03)\n", ok ? "PASS" : "FAIL", worst,
              rms, ratio);
  return ok;
}

bool silu_mul_cpu_case() {
  Rng rng(32);
  std::vector<float> g(64), u(64), out(64);
  for (auto& v : g) v = float(rng.normal());
  for (auto& v : u) v = float(rng.normal());
  moe::silu_mul_cpu(out.data(), g.data(), u.data(), 64);
  bool ok = true;
  for (int i = 0; i < 64; ++i) {
    const double want = double(g[i]) / (1.0 + std::exp(-double(g[i]))) * double(u[i]);
    if (std::fabs(double(out[i]) - want) > 1e-6) ok = false;
  }
  std::printf("%s silu_mul reference\n", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = silu_mul_cpu_case();
  if (have_matching_gpu()) {
    ok &= run_case();
  } else {
    std::printf("skip CUDA comparison: needs sm_120\n");
  }
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
