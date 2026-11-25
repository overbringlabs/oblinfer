// Qwen3.6-35B-A3B decode attention (src/arch/qwen35moe/kernels/attn) against the FP64 reference over the
// same BF16 cache bytes. Covers the single-position case, a chunk that fits one pass, a position count
// that spans two chunks (the running-softmax merge), the query-head-to-KV-head grouping, and the head
// ranges the two-device KV-head split launches: disjoint ranges of one output buffer must rebuild the
// full-range result bit for bit.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
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

// The head ranges of the split must reproduce the full-range result exactly, not to a tolerance: the same
// kernel, the same cache bytes and the same per-head arithmetic either match bit for bit or the mapping of
// query heads to KV heads shifted. kSentinel marks output the launch was not allowed to touch.
constexpr float kSentinel = 1234.5f;

std::vector<float> sentinel_out() { return std::vector<float>(H * D, kSentinel); }

bool bitwise_equal(const char* tag, const std::vector<float>& got, const std::vector<float>& ref) {
  for (size_t i = 0; i < got.size() && i < ref.size(); ++i) {
    if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
      std::printf("FAIL %s: first mismatch at head %zu dim %d got %.9g want %.9g\n", tag, i / D, int(i % D),
                  double(got[i]), double(ref[i]));
      return false;
    }
  }
  if (got.size() != ref.size()) {
    std::printf("FAIL %s: size %zu vs %zu\n", tag, got.size(), ref.size());
    return false;
  }
  std::printf("PASS %s: bitwise equal\n", tag);
  return true;
}

bool split_matches_full_cpu(int pos, unsigned seed) {
  Case c = make_case(pos, seed);
  std::vector<float> want(H * D, 0.0f);
  attn::DecodeAttnArgs ref = host_args(c, want);
  attn::decode_attention_cpu(ref);

  bool ok = true;
  for (int split : {1, 5, H / 2, H - 1}) {
    Case c2 = c;
    std::vector<float> got(H * D, kSentinel);
    attn::DecodeAttnArgs a = host_args(c2, got);
    a.head_begin = 0;
    a.head_count = split;
    attn::decode_attention_cpu(a);
    const std::vector<float> partial = got;
    a.head_begin = split;
    a.head_count = H - split;
    attn::decode_attention_cpu(a);

    char tag[96];
    std::snprintf(tag, sizeof tag, "cpu heads 0..%d + %d..%d (pos %d)", split - 1, split, H - 1, pos);
    ok &= bitwise_equal(tag, got, want);
    bool above_untouched = true;
    for (int i = split * D; i < H * D; ++i) above_untouched &= (partial[i] == kSentinel);
    if (!above_untouched) {
      std::printf("FAIL cpu heads 0..%d wrote outside its range (pos %d)\n", split - 1, pos);
      ok = false;
    }
  }

  // A range in the middle of the head block: its own heads match, every other head keeps its old bytes.
  {
    constexpr int kBegin = 3, kCount = 2;  // inside one KV group, with heads left on both sides
    Case c2 = c;
    std::vector<float> got(H * D, kSentinel);
    attn::DecodeAttnArgs a = host_args(c2, got);
    a.head_begin = kBegin;
    a.head_count = kCount;
    attn::decode_attention_cpu(a);
    bool exact = true;
    for (int i = 0; i < H * D; ++i) {
      const int head = i / D;
      const bool mine = head >= kBegin && head < kBegin + kCount;
      exact &= mine ? (std::memcmp(&got[i], &want[i], sizeof(float)) == 0) : (got[i] == kSentinel);
    }
    std::printf("%s cpu heads %d..%d writes only those heads\n", exact ? "PASS" : "FAIL", kBegin,
                kBegin + kCount - 1);
    ok &= exact;
  }

  {
    Case c2 = c;
    const std::vector<float> sentinel = sentinel_out();
    std::vector<float> got(H * D, kSentinel);
    attn::DecodeAttnArgs a = host_args(c2, got);
    const int bad[][2] = {{-1, H}, {0, 0}, {H - 1, 2}, {H, 1}};
    bool refused = true;
    for (const auto& r : bad) {
      a.head_begin = r[0];
      a.head_count = r[1];
      attn::decode_attention_cpu(a);
      refused &= std::memcmp(got.data(), sentinel.data(), got.size() * sizeof(float)) == 0;
    }
    std::printf("%s cpu invalid head ranges: no write\n", refused ? "PASS" : "FAIL");
    ok &= refused;
  }
  return ok;
}

attn::DecodeAttnArgs device_args(const DeviceArray<float>& q, const DeviceArray<float>& k_new,
                                 const DeviceArray<float>& v_new, const DeviceArray<__nv_bfloat16>& kc,
                                 const DeviceArray<__nv_bfloat16>& vc, float* out, int pos, int max_pos) {
  attn::DecodeAttnArgs a;
  a.q = q.get();
  a.k_new = k_new.get();
  a.v_new = v_new.get();
  a.k_cache = kc.get();
  a.v_cache = vc.get();
  a.out = out;
  a.pos = pos;
  a.max_pos = max_pos;
  return a;
}

bool launch(const attn::DecodeAttnArgs& a, const char* tag) {
  if (attn::decode_attention(a, nullptr) != cudaSuccess) {
    std::printf("FAIL %s: launch\n", tag);
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  return true;
}

// One launch per head range into a shared output buffer, which is what the two cards do. The ranges must
// rebuild the full-range launch bit for bit, and still match the FP64 oracle.
bool run_split_case(const char* tag, int pos, unsigned seed) {
  Case c = make_case(pos, seed);
  std::vector<float> ref(H * D, 0.0f);
  attn::DecodeAttnArgs ref_args = host_args(c, ref);
  attn::decode_attention_cpu(ref_args);

  const std::vector<float> sentinel = sentinel_out();
  DeviceArray<float> dq(c.q), dk(c.k_new), dv(c.v_new);
  DeviceArray<__nv_bfloat16> dkc(c.k_cache), dvc(c.v_cache);
  DeviceArray<float> dout_full(H * D), dout_split(sentinel), dout_refused(sentinel);

  if (!launch(device_args(dq, dk, dv, dkc, dvc, dout_full.get(), pos, c.max_pos), tag)) return false;
  const std::vector<float> full = dout_full.download();

  bool ok = true;
  for (int split : {1, 5, H / 2, H - 1}) {
    dout_split.upload(sentinel);
    attn::DecodeAttnArgs lo = device_args(dq, dk, dv, dkc, dvc, dout_split.get(), pos, c.max_pos);
    lo.head_begin = 0;
    lo.head_count = split;
    if (!launch(lo, tag)) return false;
    const std::vector<float> partial = dout_split.download();
    attn::DecodeAttnArgs hi = lo;
    hi.head_begin = split;
    hi.head_count = H - split;
    if (!launch(hi, tag)) return false;

    char name[96];
    std::snprintf(name, sizeof name, "%s heads 0..%d + %d..%d", tag, split - 1, split, H - 1);
    const std::vector<float> got = dout_split.download();
    ok &= bitwise_equal(name, got, full);
    ok &= close(name, got, ref);
    bool above_untouched = true;
    for (int i = split * D; i < H * D; ++i) above_untouched &= (partial[i] == kSentinel);
    if (!above_untouched) {
      std::printf("FAIL %s: heads 0..%d wrote outside its range\n", tag, split - 1);
      ok = false;
    }
  }

  const int bad[][2] = {{-1, H}, {0, 0}, {H - 1, 2}, {H, 1}};
  bool refused = true;
  for (const auto& r : bad) {
    attn::DecodeAttnArgs a = device_args(dq, dk, dv, dkc, dvc, dout_refused.get(), pos, c.max_pos);
    a.head_begin = r[0];
    a.head_count = r[1];
    refused &= attn::decode_attention(a, nullptr) == cudaErrorInvalidValue;
  }
  const std::vector<float> untouched = dout_refused.download();
  const bool clean =
      refused && std::memcmp(untouched.data(), sentinel.data(), sentinel.size() * sizeof(float)) == 0;
  std::printf("%s cuda invalid head ranges: cudaErrorInvalidValue, nothing launched\n", clean ? "PASS" : "FAIL");
  ok &= clean;
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
  ok &= split_matches_full_cpu(0, 1);
  ok &= split_matches_full_cpu(511, 3);
  ok &= split_matches_full_cpu(600, 4);
  if (!have_matching_gpu()) {
    std::printf("skip CUDA comparison: needs sm_120\n");
  } else {
    ok &= run_case("pos 0", 0, 1);
    ok &= run_case("pos 7", 7, 2);
    ok &= run_case("pos 511 (one chunk)", 511, 3);
    ok &= run_case("pos 600 (two chunks)", 600, 4);
    ok &= run_case("pos 1023 (two chunks, full)", 1023, 5);
    ok &= run_split_case("split pos 600 (two chunks)", 600, 4);
    ok &= run_split_case("split pos 1023 (two chunks, full)", 1023, 5);
  }
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
