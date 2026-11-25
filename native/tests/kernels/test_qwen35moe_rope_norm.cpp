// q/k RMSNorm + partial rotary and the output gate of the Qwen3.6-35B-A3B attention layers, against FP64
// references. The rotary window is 64 of 256 dims at theta 1e7; dims past the window must pass through
// untouched.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/rope_norm.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace attn = ingine::arch::qwen35moe::kernels;
constexpr int H = attn::NORM_HEADS;
constexpr int KV = attn::NORM_KV_HEADS;
constexpr int D = attn::NORM_HEAD_DIM;
constexpr int ROT = attn::ROTARY_DIM;

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("skip CUDA comparison: no CUDA device\n");
    return false;
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  if (p.major != 12 || p.minor != 0) {
    std::printf("skip CUDA comparison: %s is sm_%d%d, the kernel builds for sm_120a\n", p.name, p.major, p.minor);
    return false;
  }
  return true;
}

struct Case {
  std::vector<float> q, k, qw, kw, gate, x;
};

Case make_case(unsigned seed) {
  Case c;
  Rng rng(seed);
  c.q.resize(H * D);
  c.k.resize(KV * D);
  c.qw.resize(D);
  c.kw.resize(D);
  c.gate.resize(H * D);
  c.x.resize(H * D);
  for (auto& v : c.q) v = float(rng.normal());
  for (auto& v : c.k) v = float(rng.normal());
  for (auto& v : c.qw) v = float(0.5 + rng.normal() * 0.1);
  for (auto& v : c.kw) v = float(0.5 + rng.normal() * 0.1);
  for (auto& v : c.gate) v = float(rng.normal());
  for (auto& v : c.x) v = float(rng.normal());
  return c;
}

// Independent FP64 reference of the norm and the rotary window.
std::vector<double> reference_norm_rope(const std::vector<float>& src, int heads, const std::vector<float>& weight,
                                        int pos) {
  std::vector<double> x(src.begin(), src.end());
  for (int h = 0; h < heads; ++h) {
    const int base = h * D;
    double sum = 0.0;
    for (int d = 0; d < D; ++d) sum += x[base + d] * x[base + d];
    const double inv = 1.0 / std::sqrt(sum / D + 1e-6);
    for (int d = 0; d < D; ++d) x[base + d] *= inv * double(weight[d]);
    for (int i = 0; i < ROT / 2; ++i) {
      const double angle = double(pos) * std::pow(1e7, -2.0 * double(i) / double(ROT));
      const double c = std::cos(angle), s = std::sin(angle);
      const double x0 = x[base + i], x1 = x[base + i + ROT / 2];
      x[base + i] = x0 * c - x1 * s;
      x[base + i + ROT / 2] = x0 * s + x1 * c;
    }
  }
  return x;
}

bool compare(const char* tag, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
  double worst = 0.0, rms = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - ref[i]));
    rms += ref[i] * ref[i];
  }
  rms = std::sqrt(rms / double(ref.size()));
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= tol;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", tag, worst, rms,
              ratio, tol);
  return ok;
}

bool cpu_case() {
  Case c = make_case(3);
  const std::vector<double> want_q = reference_norm_rope(c.q, H, c.qw, 37);
  const std::vector<double> want_k = reference_norm_rope(c.k, KV, c.kw, 37);

  attn::NormRopeArgs a;
  a.q = c.q.data();
  a.k = c.k.data();
  a.q_weight = c.qw.data();
  a.k_weight = c.kw.data();
  a.pos = 37;
  attn::qk_norm_rope_cpu(a);
  bool ok = compare("cpu q", c.q, want_q, 1e-6);
  ok &= compare("cpu k", c.k, want_k, 1e-6);

  std::vector<float> x = c.x;
  std::vector<double> ref_gate(c.x.size());
  for (size_t i = 0; i < c.x.size(); ++i) {
    ref_gate[i] = double(c.x[i]) / (1.0 + std::exp(-double(c.gate[i])));
  }
  attn::OutputGateArgs g;
  g.x = x.data();
  g.gate = c.gate.data();
  g.count = int(x.size());
  attn::output_gate_cpu(g);
  ok &= compare("cpu output gate", x, ref_gate, 1e-6);

  // Dims past the rotary window hold the normalised value: no rotation touched them.
  bool untouched = true;
  for (int h = 0; h < H; ++h) {
    for (int d = ROT; d < D; ++d) {
      if (std::fabs(double(c.q[h * D + d]) - want_q[h * D + d]) > 1e-6) untouched = false;
    }
  }
  std::printf("%s dims past the rotary window untouched\n", untouched ? "PASS" : "FAIL");
  return ok && untouched;
}

bool cuda_case() {
  Case c = make_case(4);
  const std::vector<double> want_q = reference_norm_rope(c.q, H, c.qw, 120);
  const std::vector<double> want_k = reference_norm_rope(c.k, KV, c.kw, 120);

  DeviceArray<float> dq(c.q), dk(c.k), dqw(c.qw), dkw(c.kw);
  attn::NormRopeArgs a;
  a.q = dq.get();
  a.k = dk.get();
  a.q_weight = dqw.get();
  a.k_weight = dkw.get();
  a.pos = 120;
  if (attn::qk_norm_rope(a, nullptr) != cudaSuccess) {
    std::printf("FAIL norm/rope launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  bool ok = compare("cuda q", dq.download(), want_q, 1e-5);
  ok &= compare("cuda k", dk.download(), want_k, 1e-5);

  std::vector<double> ref_gate(c.x.size());
  for (size_t i = 0; i < c.x.size(); ++i) {
    ref_gate[i] = double(c.x[i]) / (1.0 + std::exp(-double(c.gate[i])));
  }
  DeviceArray<float> dx(c.x), dgate(c.gate);
  attn::OutputGateArgs g;
  g.x = dx.get();
  g.gate = dgate.get();
  g.count = int(c.x.size());
  if (attn::output_gate(g, nullptr) != cudaSuccess) {
    std::printf("FAIL gate launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  ok &= compare("cuda output gate", dx.download(), ref_gate, 1e-5);
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
