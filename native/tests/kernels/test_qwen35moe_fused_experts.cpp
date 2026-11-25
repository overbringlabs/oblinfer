// Token-major fused MoE experts: the three-launch path of fused_experts.cu against an independent FP64
// oracle over the same synthetic NVFP4 bytes. Weights live in two packed arenas (one stride per expert, the
// shape model/placement.h produces) and the layer table is indexed by expert id on the device, so the ids
// and router weights never reach the host between the router and the experts.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/moe/fused_experts.h"
#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace moe = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;
constexpr int K = moe::MOE_TOP_K;
constexpr int INTER = moe::MOE_INTER;
constexpr int HID = moe::MOE_HIDDEN;
constexpr int E = 16;  // experts in the packed block

// One expert stride: gate, up, down, each weight tensor followed by its block scales.
constexpr int GATE_W = INTER * HID / 2;
constexpr int UP_W = INTER * HID / 2;
constexpr int DOWN_W = HID * INTER / 2;
constexpr int W_STRIDE = GATE_W + UP_W + DOWN_W;
constexpr int GATE_S = INTER * HID / 16;
constexpr int UP_S = INTER * HID / 16;
constexpr int DOWN_S = HID * INTER / 16;
constexpr int S_STRIDE = GATE_S + UP_S + DOWN_S;
constexpr int GATE_W_OFF = 0, UP_W_OFF = GATE_W, DOWN_W_OFF = GATE_W + UP_W;
constexpr int GATE_S_OFF = 0, UP_S_OFF = GATE_S, DOWN_S_OFF = GATE_S + UP_S;

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

struct Scale2 {
  float gate = 0.0f, up = 0.0f, down = 0.0f;
};

struct Pack {
  std::vector<std::uint8_t> w;
  std::vector<std::uint8_t> s;
  std::vector<Scale2> scale2;
};

Pack make_pack(Rng& rng) {
  Pack p;
  p.w.resize(std::size_t(E) * W_STRIDE);
  p.s.resize(std::size_t(E) * S_STRIDE);
  p.scale2.resize(std::size_t(E));
  for (auto& b : p.w) b = std::uint8_t(rng.uniform() * 255.0);
  // e4m3 block scales kept in the finite positive range the checkpoint uses.
  for (auto& b : p.s) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);
  for (auto& v : p.scale2) {
    v.gate = 0.001f + float(rng.uniform()) * 0.001f;
    v.up = 0.001f + float(rng.uniform()) * 0.001f;
    v.down = 0.001f + float(rng.uniform()) * 0.001f;
  }
  return p;
}

// The layer table differs between the host and device passes only by its base pointers.
std::vector<moe::FusedExpertLayer> make_layers(const std::uint8_t* w, const std::uint8_t* s,
                                               const std::vector<Scale2>& scale2) {
  std::vector<moe::FusedExpertLayer> layers(E);
  for (int e = 0; e < E; ++e) {
    moe::FusedExpertLayer& L = layers[std::size_t(e)];
    L.w_base = w;
    L.s_base = s;
    L.stride_bytes = W_STRIDE;
    L.scale_stride_bytes = S_STRIDE;
    L.gate_weight_off = GATE_W_OFF;
    L.up_weight_off = UP_W_OFF;
    L.down_weight_off = DOWN_W_OFF;
    L.gate_scale_off = GATE_S_OFF;
    L.up_scale_off = UP_S_OFF;
    L.down_scale_off = DOWN_S_OFF;
    L.gate_scale2 = scale2[std::size_t(e)].gate;
    L.up_scale2 = scale2[std::size_t(e)].up;
    L.down_scale2 = scale2[std::size_t(e)].down;
  }
  return layers;
}

// FP64 oracle that derives every expert pointer from the arena by hand, so a wrong stride or offset in the
// layer table cannot agree with both sides of the comparison.
std::vector<double> reference(const Pack& p, const std::vector<float>& x, const std::vector<int>& ids,
                              const std::vector<float>& w) {
  std::vector<double> y(std::size_t(HID), 0.0);
  for (int k = 0; k < K; ++k) {
    const int e = ids[std::size_t(k)];
    const std::size_t wb = std::size_t(e) * W_STRIDE;
    const std::size_t sb = std::size_t(e) * S_STRIDE;
    std::vector<float> g(INTER), u(INTER), h(INTER), d(HID);
    quant::Nvfp4GemvArgs a;
    a.x = x.data();
    a.rows = INTER;
    a.cols = HID;
    a.w = p.w.data() + wb + GATE_W_OFF;
    a.scales = p.s.data() + sb + GATE_S_OFF;
    a.scale = p.scale2[std::size_t(e)].gate;
    a.y = g.data();
    quant::nvfp4_gemv_cpu(a);
    a.w = p.w.data() + wb + UP_W_OFF;
    a.scales = p.s.data() + sb + UP_S_OFF;
    a.scale = p.scale2[std::size_t(e)].up;
    a.y = u.data();
    quant::nvfp4_gemv_cpu(a);
    for (int i = 0; i < INTER; ++i) {
      const double v = g[std::size_t(i)];
      h[std::size_t(i)] = float(v / (1.0 + std::exp(-v)) * double(u[std::size_t(i)]));
    }
    a.w = p.w.data() + wb + DOWN_W_OFF;
    a.scales = p.s.data() + sb + DOWN_S_OFF;
    a.scale = p.scale2[std::size_t(e)].down;
    a.x = h.data();
    a.y = d.data();
    a.rows = HID;
    a.cols = INTER;
    quant::nvfp4_gemv_cpu(a);
    for (int i = 0; i < HID; ++i) y[std::size_t(i)] += double(w[std::size_t(k)]) * double(d[std::size_t(i)]);
  }
  return y;
}

bool check(const char* what, const std::vector<float>& got, const std::vector<double>& ref) {
  return report(what, compare(got.data(), ref.data(), ref.size()), 1e-3);
}

std::vector<float> normalised_weights(Rng& rng) {
  std::vector<float> w(K);
  float sum = 0.0f;
  for (auto& v : w) v = float(rng.uniform() + 0.01f);
  for (float v : w) sum += v;
  for (auto& v : w) v /= sum;
  return w;
}

// The FP64 reference path, runnable on every machine.
bool cpu_case(const Pack& p, const std::vector<float>& x, const std::vector<int>& ids, const std::vector<float>& w,
              const std::vector<double>& ref) {
  const std::vector<moe::FusedExpertLayer> layers = make_layers(p.w.data(), p.s.data(), p.scale2);
  std::vector<float> y(std::size_t(HID), -1.0f), sg(std::size_t(K) * INTER), su(std::size_t(K) * INTER),
      sd(std::size_t(K) * HID, 0.0f);
  moe::FusedMoeArgs a;
  a.layers = layers.data();
  a.x = x.data();
  a.ids = const_cast<int*>(ids.data());
  a.weights = w.data();
  a.y = y.data();
  a.scratch_gate = sg.data();
  a.scratch_up = su.data();
  a.scratch_down = sd.data();
  moe::fused_moe_apply_cpu(a);

  bool ok = check("cpu fused_moe y", y, ref);

  // scratch_up carries the up projection and scratch_gate has been overwritten in place by
  // h_k = silu(gate_k) * up_k, which is what the down step consumes.
  std::vector<float> want_sg(std::size_t(K) * INTER), want_su(std::size_t(K) * INTER);
  for (int k = 0; k < K; ++k) {
    const int e = ids[std::size_t(k)];
    quant::Nvfp4GemvArgs g;
    g.w = p.w.data() + std::size_t(e) * W_STRIDE + GATE_W_OFF;
    g.scales = p.s.data() + std::size_t(e) * S_STRIDE + GATE_S_OFF;
    g.scale = p.scale2[std::size_t(e)].gate;
    g.x = x.data();
    g.y = want_sg.data() + std::size_t(k) * INTER;
    g.rows = INTER;
    g.cols = HID;
    quant::nvfp4_gemv_cpu(g);
    g.w = p.w.data() + std::size_t(e) * W_STRIDE + UP_W_OFF;
    g.scales = p.s.data() + std::size_t(e) * S_STRIDE + UP_S_OFF;
    g.scale = p.scale2[std::size_t(e)].up;
    g.y = want_su.data() + std::size_t(k) * INTER;
    quant::nvfp4_gemv_cpu(g);
  }
  double scale = 0.0, worst_h = 0.0, worst_u = 0.0;
  for (std::size_t k = 0; k < std::size_t(K); ++k) {
    for (int i = 0; i < INTER; ++i) {
      const std::size_t idx = k * INTER + std::size_t(i);
      const double gv = want_sg[idx], uv = want_su[idx];
      const double hv = gv / (1.0 + std::exp(-gv)) * uv;
      scale = std::max({scale, std::fabs(hv), std::fabs(uv)});
      worst_h = std::max(worst_h, std::fabs(double(sg[idx]) - hv));
      worst_u = std::max(worst_u, std::fabs(double(su[idx]) - uv));
    }
  }
  const bool scratch_ok = worst_h <= 1e-3 * scale && worst_u <= 1e-3 * scale;
  std::printf("%s cpu fused scratch gate->h/up: max_abs=%.3e max_abs_up=%.3e (scale %.3e)\n",
              scratch_ok ? "PASS" : "FAIL", worst_h, worst_u, scale);

  // scratch_down holds the weighted per-slot down outputs the CUDA path folds into registers.
  double worst_d = 0.0, dscale = 0.0;
  for (int k = 0; k < K; ++k) {
    const int e = ids[std::size_t(k)];
    std::vector<float> h(INTER), d(HID);
    for (int i = 0; i < INTER; ++i) h[std::size_t(i)] = sg[std::size_t(k) * INTER + i];
    quant::Nvfp4GemvArgs g;
    g.w = p.w.data() + std::size_t(e) * W_STRIDE + DOWN_W_OFF;
    g.scales = p.s.data() + std::size_t(e) * S_STRIDE + DOWN_S_OFF;
    g.scale = p.scale2[std::size_t(e)].down;
    g.x = h.data();
    g.y = d.data();
    g.rows = HID;
    g.cols = INTER;
    quant::nvfp4_gemv_cpu(g);
    for (int i = 0; i < HID; ++i) {
      const double want = double(w[std::size_t(k)]) * double(d[std::size_t(i)]);
      dscale = std::max(dscale, std::fabs(want));
      worst_d = std::max(worst_d, std::fabs(double(sd[std::size_t(k) * HID + i]) - want));
    }
  }
  const bool down_ok = worst_d <= 1e-3 * dscale;
  std::printf("%s cpu fused scratch_down: max_abs=%.3e (scale %.3e)\n", down_ok ? "PASS" : "FAIL", worst_d, dscale);
  return ok && scratch_ok && down_ok;
}

bool cuda_case(const Pack& p, const std::vector<float>& x, const std::vector<int>& ids, const std::vector<float>& w,
               const std::vector<int>& ids_b, const std::vector<float>& w_b, const std::vector<double>& ref,
               const std::vector<double>& ref_b) {
  DeviceArray<std::uint8_t> dw(p.w), ds(p.s);
  DeviceArray<float> dx(x), dy(HID), dsg(std::size_t(K) * INTER), dsu(std::size_t(K) * INTER),
      dsd(std::size_t(K) * HID);
  DeviceArray<int> dids(ids);
  DeviceArray<float> dwt(w);
  const std::vector<moe::FusedExpertLayer> dev_layers = make_layers(dw.get(), ds.get(), p.scale2);
  DeviceArray<moe::FusedExpertLayer> dlayers(dev_layers);

  moe::FusedMoeArgs a;
  a.layers = dlayers.get();
  a.x = dx.get();
  a.ids = dids.get();
  a.weights = dwt.get();
  a.y = dy.get();
  a.scratch_gate = dsg.get();
  a.scratch_up = dsu.get();
  a.scratch_down = dsd.get();
  if (moe::fused_moe_apply(a, nullptr) != cudaSuccess) {
    std::printf("FAIL fused launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> got = dy.download();
  bool ok = check("cuda fused_moe y", got, ref);

  // Re-route the same buffers to a different expert set and different weights: if the kernel had baked the
  // first ids or weights in, this run would reproduce the first output.
  dids.upload(ids_b);
  dwt.upload(w_b);
  if (moe::fused_moe_apply(a, nullptr) != cudaSuccess) {
    std::printf("FAIL fused reroute launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> rerouted = dy.download();
  ok &= check("cuda fused_moe y rerouted", rerouted, ref_b);

  double diff = 0.0, rms = 0.0;
  for (int i = 0; i < HID; ++i) {
    diff = std::max(diff, std::fabs(double(got[std::size_t(i)]) - double(rerouted[std::size_t(i)])));
    rms += ref_b[std::size_t(i)] * ref_b[std::size_t(i)];
  }
  rms = std::sqrt(rms / HID);
  const bool moved = diff > 1e-3 * rms;
  std::printf("%s cuda fused_moe reads ids and weights on device: max_abs_delta=%.3e rms(ref)=%.3e\n",
              moved ? "PASS" : "FAIL", diff, rms);
  return ok && moved;
}

}  // namespace

int main() {
  INGINE_CHECK(W_STRIDE < (1 << 30) && S_STRIDE < (1 << 30));  // the stride fields are ints
  Rng rng(41);
  const Pack p = make_pack(rng);
  std::vector<float> x(HID);
  for (auto& v : x) v = float(rng.normal());
  // Non-trivial, unsorted, non-contiguous ids: they must come from the device array, not be derived from k.
  const std::vector<int> ids = {3, 0, 9, 15, 7, 1, 12, 5};
  const std::vector<float> w = normalised_weights(rng);
  const std::vector<int> ids_b = {11, 4, 2, 13, 6, 14, 8, 10};
  const std::vector<float> w_b = normalised_weights(rng);

  const std::vector<double> ref = reference(p, x, ids, w);
  const std::vector<double> ref_b = reference(p, x, ids_b, w_b);

  bool ok = cpu_case(p, x, ids, w, ref);
  if (have_matching_gpu()) {
    ok &= cuda_case(p, x, ids, w, ids_b, w_b, ref, ref_b);
  } else {
    std::printf("skip CUDA comparison: needs sm_120\n");
  }
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
