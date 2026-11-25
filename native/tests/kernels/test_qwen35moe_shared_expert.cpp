// Shared expert of the Qwen3.6-35B-A3B MoE layers against an FP64 reference over the same synthetic bytes.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/moe/shared_expert.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace moe = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;
constexpr int INTER = moe::MOE_INTER;
constexpr int HID = moe::MOE_HIDDEN;

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

bool cpu_case() {
  Rng rng(41);
  std::vector<float> y(HID), x(HID);
  std::vector<__nv_bfloat16> w(HID);
  for (auto& v : y) v = float(rng.normal());
  for (auto& v : x) v = float(rng.normal());
  for (auto& v : w) v = __float2bfloat16(float(rng.normal()));

  std::vector<float> got = y;
  moe::scale_sigmoid_dot_cpu(got.data(), w.data(), x.data(), HID);

  double dot = 0.0;
  for (int i = 0; i < HID; ++i) dot += double(__bfloat162float(w[i])) * double(x[i]);
  const double gate = 1.0 / (1.0 + std::exp(-dot));
  bool ok = true;
  for (int i = 0; i < HID; ++i) {
    if (std::fabs(double(got[i]) - double(y[i]) * gate) > 1e-6) ok = false;
  }
  std::printf("%s sigmoid dot reference\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool cuda_case() {
  Rng rng(42);
  std::vector<std::uint8_t> gw(std::size_t(INTER) * HID / 2), gs(std::size_t(INTER) * HID / 16);
  std::vector<std::uint8_t> uw(gw.size()), us(gs.size());
  std::vector<std::uint8_t> dw(std::size_t(HID) * INTER / 2), ds(std::size_t(HID) * INTER / 16);
  for (auto& b : gw) b = std::uint8_t(rng.uniform() * 255.0);
  for (auto& b : uw) b = std::uint8_t(rng.uniform() * 255.0);
  for (auto& b : dw) b = std::uint8_t(rng.uniform() * 255.0);
  for (auto& b : gs) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  for (auto& b : us) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  for (auto& b : ds) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  const float sx = 0.0012f, su = 0.0014f, sd = 0.0011f;

  std::vector<float> x(HID), gwf(HID);
  // Small activations keep the gate sigmoid away from saturation; unit-variance inputs make the 2048-wide
  // dot saturate and the whole output collapse to zero on both sides.
  for (auto& v : x) v = float(rng.normal()) * 0.02f;
  std::vector<__nv_bfloat16> wbf(HID);
  for (auto& v : wbf) v = __float2bfloat16(float(rng.normal()));
  for (int i = 0; i < HID; ++i) gwf[i] = __bfloat162float(wbf[i]);

  // Reference: three CPU GEMVs, FP64 silu, FP64 sigmoid gate.
  std::vector<float> g(INTER), u(INTER), h(INTER), ref(HID);
  quant::Nvfp4GemvArgs a;
  a.w = gw.data();
  a.scales = gs.data();
  a.scale = sx;
  a.x = x.data();
  a.y = g.data();
  a.rows = INTER;
  a.cols = HID;
  quant::nvfp4_gemv_cpu(a);
  a.w = uw.data();
  a.scales = us.data();
  a.scale = su;
  a.y = u.data();
  quant::nvfp4_gemv_cpu(a);
  for (int i = 0; i < INTER; ++i) {
    const double v = g[i];
    h[i] = float(v / (1.0 + std::exp(-v)) * double(u[i]));
  }
  a.w = dw.data();
  a.scales = ds.data();
  a.scale = sd;
  a.x = h.data();
  a.y = ref.data();
  a.rows = HID;
  a.cols = INTER;
  quant::nvfp4_gemv_cpu(a);
  double dot = 0.0;
  for (int i = 0; i < HID; ++i) dot += double(gwf[i]) * double(x[i]);
  const double gate = 1.0 / (1.0 + std::exp(-dot));
  for (int i = 0; i < HID; ++i) ref[i] = float(double(ref[i]) * gate);

  DeviceArray<std::uint8_t> dgw(gw), dgs(gs), duw(uw), dus(us), ddw(dw), dds(ds);
  DeviceArray<float> dx(x), dy(HID), dh(INTER), dup(INTER);
  DeviceArray<__nv_bfloat16> dwbf(wbf);
  moe::SharedExpertArgs s;
  s.gate = {dgw.get(), dgs.get(), sx, dx.get(), dh.get(), INTER, HID};
  s.up = {duw.get(), dus.get(), su, dx.get(), dup.get(), INTER, HID};
  s.down = {ddw.get(), dds.get(), sd, dh.get(), dy.get(), HID, INTER};
  s.gate_weight = dwbf.get();
  s.x = dx.get();
  s.y = dy.get();
  s.scratch_h = dh.get();
  s.scratch_up = dup.get();
  if (moe::shared_expert(s, nullptr) != cudaSuccess) {
    std::printf("FAIL shared expert launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> got = dy.download();

  double worst = 0.0, rms = 0.0;
  for (int i = 0; i < HID; ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])));
    rms += double(ref[i]) * double(ref[i]);
  }
  rms = std::sqrt(rms / HID);
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= 1e-3;
  std::printf("%s cuda shared expert: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol 1.0e-03)\n",
              ok ? "PASS" : "FAIL", worst, rms, ratio);
  return ok;
}

}  // namespace

int main() {
  bool ok = cpu_case();
  if (have_matching_gpu()) {
    ok &= cuda_case();
  } else {
    std::printf("skip CUDA comparison: needs sm_120\n");
  }
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
