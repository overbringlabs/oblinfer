// Split (flash-decoding) path of the Qwen3.6-35B-A3B decode attention, which runs sixteen dims per lane: a
// position is covered by sixteen lanes, two positions are in flight per warp, and the warp merges its two
// position streams before the eight warps of the block merge. Cases are built against the FP64 reference over
// the *same cache bytes*, history included, so the q8_0 decode and its per-block scales are exercised with
// real data rather than a zeroed cache.
//
// Covered: both KvType selections per side (the kernel is instantiated once per pair), head ranges, splits
// larger than the position count (the empty-split early return), and split lengths that are not a multiple of
// the sixteen-position block step — the cases where one of a warp's two streams, or a whole warp, gets no
// position at all.
//
// The CUDA comparison runs only on sm_120 (the pack builds for sm_120a) unless `--force` is given; the host
// halves run everywhere. `--bench [depth] [splits] [q8_0|bf16]` times one token's attention pass.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace ingine::test;

namespace {

namespace attn = ingine::arch::qwen35moe::kernels;
constexpr int H = attn::ATTN_HEADS;
constexpr int KV = attn::ATTN_KV_HEADS;
constexpr int D = attn::ATTN_HEAD_DIM;
constexpr int DIMS_PER_LANE = 16;
constexpr int LANES_PER_POS = D / DIMS_PER_LANE;  // sixteen lanes cover one position
constexpr int POS_PER_BLOCK_STEP = (D / 32) * (32 / LANES_PER_POS);  // eight warps x two streams = 16

// The CUDA comparison needs the arch the pack builds for; `--force` runs it on any device, which is how the
// kernel gets checked on a local card before the sm_120 run.
bool have_usable_gpu(bool force) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("skip CUDA comparison: no CUDA device\n");
    return false;
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  if (!force && (p.major != 12 || p.minor != 0)) {
    std::printf("skip CUDA comparison: %s is sm_%d%d, the kernel builds for sm_120a\n", p.name, p.major, p.minor);
    return false;
  }
  std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  return true;
}

bool close(const char* tag, const std::vector<float>& got, const std::vector<float>& ref, double tol = 1e-5) {
  double worst = 0.0, rms = 0.0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double e = std::fabs(double(got[i]) - double(ref[i]));
    rms += double(ref[i]) * double(ref[i]);
    if (e > worst) {
      worst = e;
      at = i;
    }
  }
  rms = std::sqrt(rms / double(ref.size()));
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e) worst=head %zu dim %d\n",
              ratio <= tol ? "PASS" : "FAIL", tag, worst, rms, ratio, tol, at / D, int(at % D));
  return ratio <= tol;
}

// The kernel's own quantisation, mirrored on the host: one fp16 scale per 32-value block over the values array.
// Rows past `pos` stay zero, which both sides read as zero.
std::vector<std::uint8_t> fill_cache(const std::vector<float>& src, int max_pos, attn::KvType type, int pos) {
  const std::size_t values = std::size_t(max_pos) * KV * D;
  std::vector<std::uint8_t> bytes(std::size_t(attn::kv_cache_bytes(type, max_pos)));
  if (type == attn::KvType::Bf16) {
    auto* as_bf = reinterpret_cast<__nv_bfloat16*>(bytes.data());
    for (std::size_t i = 0; i < std::size_t(pos) * KV * D; ++i) as_bf[i] = __float2bfloat16(src[i]);
    return bytes;
  }
  auto* vals = attn::kv_q8_values(bytes.data());
  auto* scales = const_cast<__half*>(attn::kv_q8_scales(bytes.data(), values));
  const std::size_t blocks = std::size_t(pos) * KV * D / attn::KV_BLOCK;
  for (std::size_t blk = 0; blk < blocks; ++blk) {
    float vmax = 0.0f;
    for (int i = 0; i < attn::KV_BLOCK; ++i) vmax = std::max(vmax, std::fabs(src[blk * attn::KV_BLOCK + i]));
    const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
    scales[blk] = __float2half(d);
    for (int i = 0; i < attn::KV_BLOCK; ++i)
      vals[blk * attn::KV_BLOCK + i] = std::int8_t(std::lround(src[blk * attn::KV_BLOCK + i] / d));
  }
  return bytes;
}

struct Case {
  int pos = 0;
  int max_pos = 0;
  attn::KvType tk = attn::KvType::Bf16, tv = attn::KvType::Bf16;
  std::vector<float> q, k_new, v_new;
  std::vector<std::uint8_t> kc, vc;
};

Case make_case(int pos, int max_pos, attn::KvType tk, attn::KvType tv, unsigned seed) {
  Case c;
  c.pos = pos;
  c.max_pos = max_pos;
  c.tk = tk;
  c.tv = tv;
  Rng rng(seed);
  c.q.resize(std::size_t(H) * D);
  c.k_new.resize(std::size_t(KV) * D);
  c.v_new.resize(std::size_t(KV) * D);
  for (auto& v : c.q) v = float(rng.normal());
  for (auto& v : c.k_new) v = float(rng.normal());
  for (auto& v : c.v_new) v = float(rng.normal());
  std::vector<float> hist(std::size_t(pos) * KV * D), hist_v(std::size_t(pos) * KV * D);
  for (auto& v : hist) v = float(rng.normal());
  for (auto& v : hist_v) v = float(rng.normal());
  c.kc = fill_cache(hist, max_pos, tk, pos);
  c.vc = fill_cache(hist_v, max_pos, tv, pos);
  return c;
}

attn::DecodeAttnArgs base_args(Case& c, float* out) {
  attn::DecodeAttnArgs a;
  a.q = c.q.data();
  a.k_new = c.k_new.data();
  a.v_new = c.v_new.data();
  a.k_cache = c.kc.data();
  a.v_cache = c.vc.data();
  a.out = out;
  a.pos = c.pos;
  a.max_pos = c.max_pos;
  a.head_begin = 0;
  a.head_count = H;
  a.kv_type_k = c.tk;
  a.kv_type_v = c.tv;
  return a;
}

const char* cache_name(attn::KvType tk, attn::KvType tv) {
  if (tk == tv) return tk == attn::KvType::Q8_0 ? "q8_0" : "bf16";
  return tk == attn::KvType::Q8_0 ? "q8_0 K, bf16 V" : "bf16 K, q8_0 V";
}

// One split launch on the device against the FP64 reference over the same bytes. The reference stores the new
// row first, exactly as the device store kernel does, and its bytes are what the device then reloads.
bool run_split_case(const char* tag, int pos, int max_pos, int splits, attn::KvType tk, attn::KvType tv,
                    unsigned seed) {
  Case c = make_case(pos, max_pos, tk, tv, seed);
  std::vector<float> ref(std::size_t(H) * D, 0.0f);
  attn::DecodeAttnArgs ref_args = base_args(c, ref.data());
  attn::decode_attention_cpu(ref_args);

  DeviceArray<float> dq(c.q), dk(c.k_new), dv(c.v_new), dout(std::size_t(H) * D);
  DeviceArray<std::uint8_t> dkc(c.kc), dvc(c.vc);
  DeviceArray<attn::SplitState> dpart(std::size_t(H) * splits);
  attn::DecodeAttnArgs a = base_args(c, dout.get());
  a.q = dq.get();
  a.k_new = dk.get();
  a.v_new = dv.get();
  a.k_cache = dkc.get();
  a.v_cache = dvc.get();
  a.partial = dpart.get();
  a.splits = splits;
  if (attn::decode_attention(a, nullptr) != cudaSuccess) {
    std::printf("FAIL %s: launch\n", tag);
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> got = dout.download();

  char name[160];
  std::snprintf(name, sizeof name, "%s (%s cache, %d splits)", tag, cache_name(tk, tv), splits);
  bool ok = close(name, got, ref);

  // The split states the combine kernel consumed must be finite: an unmerged or NaN partial shows up here even
  // when the normalised output hides it.
  const std::vector<attn::SplitState> states = dpart.download();
  int empty = 0;
  bool sane = true;
  for (int h = 0; h < H; ++h) {
    for (int s = 0; s < splits; ++s) {
      const attn::SplitState& st = states[std::size_t(h) * splits + s];
      if (st.m == -INFINITY) {
        ++empty;
        if (st.l != 0.0f) sane = false;
        for (int d = 0; d < D; ++d)
          if (st.acc[d] != 0.0f) sane = false;
        continue;
      }
      if (!std::isfinite(st.m) || !std::isfinite(st.l) || st.l <= 0.0f) sane = false;
      for (int d = 0; d < D; ++d)
        if (!std::isfinite(st.acc[d])) sane = false;
    }
  }
  std::printf("%s %s: partial states sane (%d of %d empty)\n", sane ? "PASS" : "FAIL", name, empty, H * splits);
  return ok && sane;
}

// Disjoint head-range launches must rebuild the full-range launch bit for bit: same kernel, same bytes, so any
// difference means the lane-to-dimension mapping shifted with the head index.
constexpr float kSentinel = 1234.5f;

bool bitwise_equal(const char* tag, const std::vector<float>& got, const std::vector<float>& ref) {
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
      std::printf("FAIL %s: first mismatch at head %zu dim %d got %.9g want %.9g\n", tag, i / D, int(i % D),
                  double(got[i]), double(ref[i]));
      return false;
    }
  }
  std::printf("PASS %s: bitwise equal\n", tag);
  return true;
}

bool head_range_case(int pos, int splits, attn::KvType tk, attn::KvType tv) {
  Case c = make_case(pos, 2048, tk, tv, 21);
  DeviceArray<float> dq(c.q), dk(c.k_new), dv(c.v_new);
  DeviceArray<std::uint8_t> dkc(c.kc), dvc(c.vc);
  DeviceArray<attn::SplitState> dpart(std::size_t(H) * splits);
  DeviceArray<float> dfull(std::size_t(H) * D), dgot(std::size_t(H) * D);

  attn::DecodeAttnArgs full = base_args(c, dfull.get());
  full.q = dq.get();
  full.k_new = dk.get();
  full.v_new = dv.get();
  full.k_cache = dkc.get();
  full.v_cache = dvc.get();
  full.partial = dpart.get();
  full.splits = splits;
  INGINE_TEST_CUDA(attn::decode_attention(full, nullptr));
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> want = dfull.download();

  bool ok = true;
  char name[128];
  for (int cut : {1, H / 2, H - 1}) {
    std::vector<float> sentinel(std::size_t(H) * D, kSentinel);
    dgot.upload(sentinel);
    attn::DecodeAttnArgs lo = full;
    lo.out = dgot.get();
    lo.head_begin = 0;
    lo.head_count = cut;
    ok &= attn::decode_attention(lo, nullptr) == cudaSuccess;
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    const std::vector<float> partial = dgot.download();  // snapshot before the second range runs
    attn::DecodeAttnArgs hi = lo;
    hi.head_begin = cut;
    hi.head_count = H - cut;
    ok &= attn::decode_attention(hi, nullptr) == cudaSuccess;
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    std::snprintf(name, sizeof name, "heads 0..%d + %d..%d, pos %d, %d splits", cut - 1, cut, H - 1, pos, splits);
    ok &= bitwise_equal(name, dgot.download(), want);
    bool untouched = true;
    for (int i = cut * D; i < H * D; ++i) untouched &= (partial[i] == kSentinel);
    if (!untouched) std::printf("FAIL heads 0..%d wrote outside its range\n", cut - 1);
    ok &= untouched;
  }

  // A range in the middle of the head block, so the launch's local head index differs from the absolute one.
  std::vector<float> sentinel(std::size_t(H) * D, kSentinel);
  dgot.upload(sentinel);
  attn::DecodeAttnArgs mid = full;
  mid.out = dgot.get();
  mid.head_begin = 3;
  mid.head_count = 2;
  ok &= attn::decode_attention(mid, nullptr) == cudaSuccess;
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> got = dgot.download();
  bool exact = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const int head = int(i / D);
    const bool mine = head >= 3 && head < 5;
    exact &= mine ? (std::memcmp(&got[i], &want[i], sizeof(float)) == 0) : (got[i] == kSentinel);
  }
  std::printf("%s heads 3..4 writes only those heads, bitwise\n", exact ? "PASS" : "FAIL");
  return ok && exact;
}

// An empty split range must write the inert state (m = -inf, l = 0, acc = 0) and must not poison the merge.
bool empty_range_case() {
  const int pos = 33, splits = attn::kMaxSplits;  // 34 positions over 64 ranges: 30 ranges get nothing
  Case c = make_case(pos, 64, attn::KvType::Q8_0, attn::KvType::Q8_0, 31);
  std::vector<float> ref(std::size_t(H) * D, 0.0f);
  attn::DecodeAttnArgs ref_args = base_args(c, ref.data());
  attn::decode_attention_cpu(ref_args);

  DeviceArray<float> dq(c.q), dk(c.k_new), dv(c.v_new), dout(std::size_t(H) * D);
  DeviceArray<std::uint8_t> dkc(c.kc), dvc(c.vc);
  DeviceArray<attn::SplitState> dpart(std::size_t(H) * splits);
  attn::DecodeAttnArgs a = base_args(c, dout.get());
  a.q = dq.get();
  a.k_new = dk.get();
  a.v_new = dv.get();
  a.k_cache = dkc.get();
  a.v_cache = dvc.get();
  a.partial = dpart.get();
  a.splits = splits;
  if (attn::decode_attention(a, nullptr) != cudaSuccess) {
    std::printf("FAIL empty ranges: launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<attn::SplitState> states = dpart.download();
  int inert = 0;
  bool ok = true;
  for (int s = pos + 1; s < splits; ++s) {
    const attn::SplitState& st = states[std::size_t(0) * splits + s];
    if (st.m == -INFINITY && st.l == 0.0f) {
      bool zeros = true;
      for (int d = 0; d < D; ++d) zeros &= (st.acc[d] == 0.0f);
      ok &= zeros;
      ++inert;
    } else {
      ok = false;
    }
  }
  std::printf("%s empty ranges: %d ranges kept the inert state, no NaN in the merge\n",
              ok ? "PASS" : "FAIL", inert);
  return ok && close("empty ranges merged", dout.download(), ref);
}

bool invalid_range_case() {
  Case c = make_case(64, 128, attn::KvType::Q8_0, attn::KvType::Bf16, 41);
  DeviceArray<float> dq(c.q), dk(c.k_new), dv(c.v_new), dout(std::size_t(H) * D);
  DeviceArray<std::uint8_t> dkc(c.kc), dvc(c.vc);
  DeviceArray<attn::SplitState> dpart(std::size_t(H) * 32);
  std::vector<float> sentinel(std::size_t(H) * D, kSentinel);
  dout.upload(sentinel);
  attn::DecodeAttnArgs a = base_args(c, dout.get());
  a.q = dq.get();
  a.k_new = dk.get();
  a.v_new = dv.get();
  a.k_cache = dkc.get();
  a.v_cache = dvc.get();
  a.partial = dpart.get();
  a.splits = 32;
  const int bad[][2] = {{-1, H}, {0, 0}, {H - 1, 2}, {H, 1}};
  bool refused = true;
  for (const auto& r : bad) {
    a.head_begin = r[0];
    a.head_count = r[1];
    refused &= attn::decode_attention(a, nullptr) == cudaErrorInvalidValue;
  }
  const bool clean = refused && dout.download() == sentinel;
  std::printf("%s cuda invalid head ranges: cudaErrorInvalidValue, nothing launched\n", clean ? "PASS" : "FAIL");
  return clean;
}

// Manual on the target card: one token's attention pass at the serving depth.
int bench_mode(int depth, int splits, attn::KvType type) {
  Case c = make_case(depth - 1, depth, type, type, 5);
  DeviceArray<float> dq(c.q), dk(c.k_new), dv(c.v_new), dout(std::size_t(H) * D);
  DeviceArray<std::uint8_t> dkc(c.kc), dvc(c.vc);
  DeviceArray<attn::SplitState> dpart(std::size_t(H) * splits);
  attn::DecodeAttnArgs a = base_args(c, dout.get());
  a.q = dq.get();
  a.k_new = dk.get();
  a.v_new = dv.get();
  a.k_cache = dkc.get();
  a.v_cache = dvc.get();
  a.partial = dpart.get();
  a.splits = splits;
  if (attn::decode_attention(a, nullptr) != cudaSuccess) {
    std::printf("FAIL bench launch\n");
    return 1;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  constexpr int kIters = 20;
  cudaEventRecord(e0);
  for (int i = 0; i < kIters; ++i) INGINE_TEST_CUDA(attn::decode_attention(a, nullptr));
  cudaEventRecord(e1);
  INGINE_TEST_CUDA(cudaEventSynchronize(e1));
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);
  const double us = ms * 1000.0 / kIters;
  // The eight query heads of a KV group each walk the same rows, so the traffic the kernel asks for is the
  // cache read once per query head.
  const double demanded = 2.0 * double(attn::kv_cache_bytes(type, depth)) * (H / KV);
  std::printf("bench %s depth %d splits %d: %.1f us, %.1f GB/s demanded\n",
              type == attn::KvType::Q8_0 ? "q8_0" : "bf16", depth, splits, us, demanded / (us * 1e-6) / 1e9);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool bench = false, force = false;
  std::vector<std::string> rest;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--bench") bench = true;
    else if (a == "--force") force = true;
    else rest.push_back(a);
  }

  static_assert(DIMS_PER_LANE * LANES_PER_POS == D, "sixteen dims over sixteen lanes must cover the head");
  static_assert(POS_PER_BLOCK_STEP == 16, "a block step covers eight warps times two position streams");

  if (!have_usable_gpu(force)) return kSkip;
  if (bench) {
    const int depth = rest.size() > 0 ? std::atoi(rest[0].c_str()) : 262144;
    const int splits = rest.size() > 1 ? std::atoi(rest[1].c_str()) : attn::kMaxSplits;
    const attn::KvType type = rest.size() > 2 && rest[2] == "bf16" ? attn::KvType::Bf16 : attn::KvType::Q8_0;
    return bench_mode(depth, splits, type);
  }

  bool ok = true;
  // Tail cases: per-split lengths around the sixteen-position block step, so some warps and some second
  // streams get no position (33 -> two steps minus one, 32 -> exact, 31, 17, 1).
  const int positions[] = {1, 16, 17, 31, 32, 33, 600, 1023, 2049};
  const int splits[] = {2, 3, 16, 32, attn::kMaxSplits};
  const attn::KvType pairs[][2] = {
      {attn::KvType::Q8_0, attn::KvType::Q8_0},
      {attn::KvType::Bf16, attn::KvType::Bf16},
      {attn::KvType::Q8_0, attn::KvType::Bf16},
      {attn::KvType::Bf16, attn::KvType::Q8_0},
  };
  char tag[96];
  for (const auto& pr : pairs) {
    for (int pos : positions) {
      for (int s : splits) {
        if (pos < s && s != attn::kMaxSplits) continue;  // keep the empty-range mix manageable
        const int max_pos = pos + 8 > 64 ? pos + 8 : 64;
        std::snprintf(tag, sizeof tag, "pos %d", pos);
        ok &= run_split_case(tag, pos, max_pos, s, pr[0], pr[1], unsigned(pos * 7 + s));
      }
    }
    ok &= run_split_case("long tail of one split", 4095, 4104, 2, pr[0], pr[1], 77);
  }
  ok &= empty_range_case();
  ok &= head_range_case(1023, attn::kMaxSplits, attn::KvType::Q8_0, attn::KvType::Q8_0);
  ok &= head_range_case(600, 32, attn::KvType::Bf16, attn::KvType::Q8_0);
  ok &= invalid_range_case();

  if (!ok) {
    std::printf("FAILURES\n");
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
