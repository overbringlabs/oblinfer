// Decode step of the Qwen3.x gated delta-rule linear attention (src/kernels/gdn) against the FP64
// host reference of docs/arch/gdn-recurrent.md. Closed-form cases pin the parts of the loop a matching reference
// and kernel cannot catch on their own (l2 norm + 1/sqrt(K_DIM) scale, repeat_interleave of the grouped k heads,
// the decay of the state); the FP64 reference is checked against those, and the CUDA kernel against the reference.
//
// The pack compiles for sm_120a only, so the CUDA comparison skips on any other device (this checkout has sm_89
// cards). The CPU reference half runs everywhere, and the test is not labelled `gpu`.
#include "common/test_util.h"

#include "kernels/gdn/gdn.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace gdn = ingine::kernels::gdn;
constexpr int KD = gdn::GDN_K_DIM;
constexpr int VD = gdn::GDN_V_DIM;
constexpr int KH = gdn::GDN_K_HEADS;

struct Case {
  int v_heads = 0;
  std::vector<float> q, k, v, beta, g;
  std::vector<double> state, out;

  gdn::GdnStepArgs args() const {
    gdn::GdnStepArgs a;
    a.q = q.data();
    a.k = k.data();
    a.v = v.data();
    a.beta = beta.data();
    a.g = g.data();
    a.v_heads = v_heads;
    return a;
  }
};

Case blank_case(int v_heads) {
  Case c;
  c.v_heads = v_heads;
  c.q.assign(size_t(KH) * KD, 0.0f);
  c.k.assign(size_t(KH) * KD, 0.0f);
  c.v.assign(size_t(v_heads) * VD, 0.0f);
  c.beta.assign(v_heads, 1.0f);
  c.g.assign(v_heads, 0.0f);
  c.state.assign(size_t(v_heads) * KD * VD, 0.0);
  c.out.assign(size_t(v_heads) * VD, 0.0);
  return c;
}

Case random_case(int v_heads, Rng& rng) {
  Case c = blank_case(v_heads);
  for (auto& x : c.q) x = float(rng.normal());
  for (auto& x : c.k) x = float(rng.normal());
  for (auto& x : c.v) x = float(rng.normal());
  for (auto& x : c.state) x = rng.normal() * 0.5;
  for (int h = 0; h < v_heads; ++h) {
    c.beta[h] = float(rng.uniform());
    c.g[h] = float(rng.uniform(-3.0, 0.0));  // decay exp(g) in [0.05, 1]
  }
  return c;
}

struct Ref {
  std::vector<double> state, out;
};

Ref run_reference(const Case& c) {
  Ref r;
  r.state = c.state;
  r.out.assign(size_t(c.v_heads) * VD, 0.0);
  gdn::gdn_recurrent_step_cpu(c.args(), r.state.data(), r.out.data());
  return r;
}

bool close(const char* what, const double* got, const double* ref, size_t n, double tol_rel) {
  double sq = 0.0, max_abs = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < n; ++i) sq += ref[i] * ref[i];
  for (size_t i = 0; i < n; ++i) {
    const double d = std::fabs(got[i] - ref[i]);
    if (!(d <= max_abs)) max_abs = d, worst = i;
  }
  const double rms = std::sqrt(sq / double(n ? n : 1));
  const double ratio = rms > 0.0 ? max_abs / rms : max_abs;
  const bool ok = ratio <= tol_rel;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e) worst=%zu\n",
              ok ? "PASS" : "FAIL", what, max_abs, rms, ratio, tol_rel, worst);
  return ok;
}

// q_hat and k_hat of one k head as the reference defines them (independent of the kernel).
void norms(const Case& c, int kh, double q_hat[KD], double k_hat[KD]) {
  double qn = 0.0, kn = 0.0;
  for (int i = 0; i < KD; ++i) {
    qn += double(c.q[kh * KD + i]) * double(c.q[kh * KD + i]);
    kn += double(c.k[kh * KD + i]) * double(c.k[kh * KD + i]);
  }
  const double q_scale = 1.0 / std::sqrt(qn + 1e-6) / std::sqrt(double(KD));
  const double k_scale = 1.0 / std::sqrt(kn + 1e-6);
  for (int i = 0; i < KD; ++i) {
    q_hat[i] = double(c.q[kh * KD + i]) * q_scale;
    k_hat[i] = double(c.k[kh * KD + i]) * k_scale;
  }
}

// Zero state, beta = 1, g = 0, q and k a unit vector: S = k_hat v^T and out = (q_hat . k_hat) v.
bool unit_case() {
  Case c = blank_case(48);
  for (int kh = 0; kh < KH; ++kh) c.q[kh * KD] = 1.0f, c.k[kh * KD] = 1.0f;
  for (int h = 0; h < c.v_heads; ++h)
    for (int d = 0; d < VD; ++d) c.v[h * VD + d] = 0.25f + 0.001f * float(h) + 0.001f * float(d);

  const Ref r = run_reference(c);
  const double inv = 1.0 / std::sqrt(1.0 + 1e-6);
  std::vector<double> out(size_t(c.v_heads) * VD), state(size_t(c.v_heads) * KD * VD, 0.0);
  for (int h = 0; h < c.v_heads; ++h) {
    for (int d = 0; d < VD; ++d) out[size_t(h) * VD + d] = double(c.v[h * VD + d]) * inv * inv / std::sqrt(double(KD));
    for (int d = 0; d < VD; ++d) state[(size_t(h) * KD) * VD + d] = double(c.v[h * VD + d]) * inv;
  }
  bool ok = close("unit out", r.out.data(), out.data(), out.size(), 1e-9);
  ok &= close("unit state", r.state.data(), state.data(), state.size(), 1e-9);
  return ok;
}

// Zero state, beta = 1, g = 0, arbitrary q/k: S = k_hat v^T, out = (q_hat . k_hat) v for every v head.
bool rank_one_case() {
  Rng rng(11);
  Case c = blank_case(48);
  for (auto& x : c.q) x = float(rng.normal());
  for (auto& x : c.k) x = float(rng.normal());
  for (auto& x : c.v) x = float(rng.normal());
  const Ref r = run_reference(c);
  const int group = c.v_heads / KH;
  std::vector<double> out(size_t(c.v_heads) * VD), state(size_t(c.v_heads) * KD * VD, 0.0);
  for (int h = 0; h < c.v_heads; ++h) {
    double q_hat[KD], k_hat[KD];
    norms(c, h / group, q_hat, k_hat);
    double dot = 0.0;
    for (int i = 0; i < KD; ++i) dot += q_hat[i] * k_hat[i];
    for (int d = 0; d < VD; ++d) {
      out[size_t(h) * VD + d] = dot * double(c.v[h * VD + d]);
      for (int i = 0; i < KD; ++i) state[(size_t(h) * KD + i) * VD + d] = k_hat[i] * double(c.v[h * VD + d]);
    }
  }
  bool ok = close("rank-1 out", r.out.data(), out.data(), out.size(), 1e-9);
  ok &= close("rank-1 state", r.state.data(), state.data(), state.size(), 1e-9);
  return ok;
}

// Only k head 0 has data, so only the first v_heads / KH value heads may produce output.
bool grouping_case(int v_heads) {
  Case c = blank_case(v_heads);
  for (int d = 0; d < KD; ++d) c.q[d] = 1.0f, c.k[d] = 1.0f;
  for (auto& x : c.v) x = 1.0f;
  const Ref r = run_reference(c);
  const int group = v_heads / KH;
  // q_hat . k_hat of two all-ones vectors: sqrt(KD) / (KD + 1e-6).
  const double dot = std::sqrt(double(KD)) / (double(KD) + 1e-6);
  bool ok = true, mismatch = false;
  for (int h = 0; h < v_heads && !mismatch; ++h) {
    const double expected = h < group ? dot : 0.0;
    for (int d = 0; d < VD; ++d) {
      const double got = r.out[size_t(h) * VD + d];
      if (std::fabs(got - expected) > 1e-12) {
        std::printf("FAIL grouping v_heads=%d head %d dim %d: got %.6e expected %.6e\n", v_heads, h, d, got,
                    expected);
        ok = mismatch = true;
        break;
      }
    }
  }
  if (ok) std::printf("PASS grouping v_heads=%d: only heads 0..%d read k head 0\n", v_heads, group - 1);
  return ok;
}

// beta = 0 leaves the rank-1 update at zero, so the state is only decayed by exp(g) and out = S^T q_hat.
bool decay_case() {
  Rng rng(22);
  Case c = random_case(48, rng);
  for (auto& b : c.beta) b = 0.0f;
  for (auto& g : c.g) g = std::log(0.5f);
  const Ref r = run_reference(c);
  const int group = c.v_heads / KH;
  std::vector<double> out(size_t(c.v_heads) * VD), state = c.state;
  for (int h = 0; h < c.v_heads; ++h) {
    double q_hat[KD], k_hat[KD];
    norms(c, h / group, q_hat, k_hat);
    const double decay = std::exp(double(c.g[h]));
    for (int d = 0; d < VD; ++d) {
      double o = 0.0;
      for (int i = 0; i < KD; ++i) {
        double& s = state[(size_t(h) * KD + i) * VD + d];
        s *= decay;
        o += s * q_hat[i];
      }
      out[size_t(h) * VD + d] = o;
    }
  }
  bool ok = close("decay out", r.out.data(), out.data(), out.size(), 1e-9);
  ok &= close("decay state", r.state.data(), state.data(), state.size(), 1e-9);
  return ok;
}

// The kernel is compiled for sm_120a, so it can only run on a 12.0 device.
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

bool cuda_case(int v_heads, Rng& rng) {
  Case c = random_case(v_heads, rng);
  const Ref r = run_reference(c);

  DeviceArray<float> dq(c.q), dk(c.k), dv(c.v), dbeta(c.beta), dg(c.g);
  const std::vector<float> state0(c.state.begin(), c.state.end());
  DeviceArray<float> dstate(state0), dout(c.out.size());

  gdn::GdnStepArgs a = c.args();
  // The kernel runs on the device: point every input at its device copy (c.args() returns host pointers).
  a.q = dq.get();
  a.k = dk.get();
  a.v = dv.get();
  a.beta = dbeta.get();
  a.g = dg.get();
  a.state = dstate.get();
  a.out = dout.get();
  const cudaError_t err = gdn::gdn_recurrent_step(a, nullptr);
  if (err != cudaSuccess) {
    std::printf("FAIL cuda v_heads=%d: launch: %s\n", v_heads, cudaGetErrorString(err));
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));

  const std::vector<float> state = dstate.download(), out = dout.download();
  const std::string tag = "cuda v_heads=" + std::to_string(v_heads);
  bool ok = report((tag + " state").c_str(), compare(state.data(), r.state.data(), state.size()), 1e-4);
  ok &= report((tag + " out").c_str(), compare(out.data(), r.out.data(), out.size()), 1e-4);
  return ok;
}

}  // namespace

int main() {
  Rng rng(20261125);
  bool ok = true;
  ok &= unit_case();
  ok &= rank_one_case();
  ok &= grouping_case(48);
  ok &= grouping_case(32);
  ok &= decay_case();

  // CPU reference and CUDA kernel at both value-head counts of the qwen38 packs.
  for (int v_heads : {48, 32}) {
    const Case c = random_case(v_heads, rng);
    const Ref r = run_reference(c);
    bool finite = true;
    for (double x : r.out) finite = finite && std::isfinite(x);
    for (double x : r.state) finite = finite && std::isfinite(x);
    INGINE_CHECK(finite);
    std::printf("PASS reference v_heads=%d: %zu state / %zu out values finite\n", v_heads, r.state.size(), r.out.size());
  }

  if (have_matching_gpu())
    for (int v_heads : {48, 32}) ok &= cuda_case(v_heads, rng);

  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
