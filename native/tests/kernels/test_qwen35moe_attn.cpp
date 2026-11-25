// Qwen3.6-35B-A3B decode attention (src/arch/qwen35moe/kernels/attn) against the FP64 reference over the
// same BF16 cache bytes. Covers the single-position case, a chunk that fits one pass, a position count
// that spans two chunks (the running-softmax merge), and the query-head-to-KV-head grouping.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ingine::test;

namespace {

namespace attn = ingine::arch::qwen35moe::kernels;
constexpr int H = attn::ATTN_HEADS;
constexpr int KV = attn::ATTN_KV_HEADS;
constexpr int D = attn::ATTN_HEAD_DIM;

bool close(const char* tag, const std::vector<float>& got, const std::vector<float>& ref) {
  double worst = 0.0;
  double rms = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])));
    rms += double(ref[i]) * double(ref[i]);
  }
  rms = std::sqrt(rms / double(ref.size()));
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= 1e-5;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol 1.0e-05)\n", ok ? "PASS" : "FAIL", tag, worst, rms,
              ratio);
  return ok;
}

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
  int max_pos = 1024;
  int pos = 0;
  std::vector<float> q, k_new, v_new;
  std::vector<__nv_bfloat16> k_cache, v_cache;  // pre-filled past positions
};

Case make_case(int pos, unsigned seed) {
  Case c;
  c.pos = pos;
  Rng rng(seed);
  c.q.resize(H * D);
  c.k_new.resize(KV * D);
  c.v_new.resize(KV * D);
  for (auto& v : c.q) v = float(rng.normal());
  for (auto& v : c.k_new) v = float(rng.normal());
  for (auto& v : c.v_new) v = float(rng.normal());

  // Positions before `pos` are already in the cache; fill them with values the kernel and the reference
  // must both read.
  c.k_cache.resize(std::size_t(c.max_pos) * KV * D);
  c.v_cache.resize(std::size_t(c.max_pos) * KV * D);
  for (int i = 0; i < pos; ++i) {
    for (int h = 0; h < KV; ++h) {
      for (int d = 0; d < D; ++d) {
        const std::size_t at = (std::size_t(i) * KV + h) * D + d;
        c.k_cache[at] = __float2bfloat16(float(rng.normal()));
        c.v_cache[at] = __float2bfloat16(float(rng.normal()));
      }
    }
  }
  return c;
}

attn::DecodeAttnArgs host_args(Case& c, std::vector<float>& out) {
  attn::DecodeAttnArgs a;
  a.q = c.q.data();
  a.k_new = c.k_new.data();
  a.v_new = c.v_new.data();
  a.k_cache = c.k_cache.data();
  a.v_cache = c.v_cache.data();
  a.out = out.data();
  a.pos = c.pos;
  a.max_pos = c.max_pos;
  return a;
}

bool run_case(const char* tag, int pos, unsigned seed) {
  Case c = make_case(pos, seed);

  std::vector<float> ref_out(H * D, 0.0f);
  attn::DecodeAttnArgs ref = host_args(c, ref_out);
  attn::decode_attention_cpu(ref);

  Case dev = c;
  std::vector<float> dev_out(H * D, 0.0f);
  DeviceArray<float> dq(dev.q), dk(dev.k_new), dv(dev.v_new), dout(dev_out);
  DeviceArray<__nv_bfloat16> dkc(dev.k_cache), dvc(dev.v_cache);
  attn::DecodeAttnArgs a;
  a.q = dq.get();
  a.k_new = dk.get();
  a.v_new = dv.get();
  a.k_cache = dkc.get();
  a.v_cache = dvc.get();
  a.out = dout.get();
  a.pos = pos;
  a.max_pos = c.max_pos;
  if (attn::decode_attention(a, nullptr) != cudaSuccess) {
    std::printf("FAIL %s: launch\n", tag);
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  dev_out = dout.download();

  bool ok = close(tag, dev_out, ref_out);

  // The store must have written the new row as BF16 in the cache.
  const std::vector<__nv_bfloat16> kc = dkc.download();
  for (int h = 0; h < KV; ++h) {
    for (int d = 0; d < D; ++d) {
      const std::size_t at = (std::size_t(pos) * KV + h) * D + d;
      if (kc[at] != __float2bfloat16(dev.k_new[h * D + d])) ok = false;
    }
  }
  return ok;
}

// With one position the softmax is exactly 1, so the reference must return v (through the BF16 cache).
// Runs on every machine, so the oracle itself is checked without a GPU.
bool closed_form_cpu() {
  Case c = make_case(0, 9);
  std::vector<float> out(H * D, 0.0f);
  attn::DecodeAttnArgs a = host_args(c, out);
  attn::decode_attention_cpu(a);
  bool ok = true;
  for (int h = 0; h < H; ++h) {
    const int kh = h / (H / KV);
    for (int d = 0; d < D; ++d) {
      const float want = __bfloat162float(__float2bfloat16(c.v_new[kh * D + d]));
      if (std::fabs(out[h * D + d] - want) > 1e-6f) ok = false;
    }
  }
  std::printf("%s closed form: pos 0 returns v\n", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = closed_form_cpu();
  if (!have_matching_gpu()) {
    std::printf("skip CUDA comparison: needs sm_120\n");
  } else {
    ok &= run_case("pos 0", 0, 1);
    ok &= run_case("pos 7", 7, 2);
    ok &= run_case("pos 511 (one chunk)", 511, 3);
    ok &= run_case("pos 600 (two chunks)", 600, 4);
    ok &= run_case("pos 1023 (two chunks, full)", 1023, 5);
  }
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
