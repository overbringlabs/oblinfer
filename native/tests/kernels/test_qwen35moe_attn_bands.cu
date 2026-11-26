// Position-band decode attention (src/arch/qwen35moe/kernels/attn/kv_bands.*) against the un-banded cache.
//
// The band plan never cuts a flash-decode column: band 1 mirrors the rows just below the boundary, so each
// card runs whole columns of the same (pos, splits) grid an un-banded launch would use. Every SplitState is
// then bit for bit what the un-banded launch writes, which is what the device cases assert: the banded output
// must match the single-cache device output on every bit, and both must match the FP64 reference over the
// same cache bytes within tolerance. Depths 1k and 32k are the ones the decoder runs at; the rest are the
// shapes that decide whether the arithmetic is right at all - a query below the boundary, one inside the
// mirror, one exactly on it, the last position, a query-head range, the widest and narrowest column grids, and
// both cache formats.
//
// A two-band plan with both bands on one device (an explicit cut, no staging) covers the band arithmetic on
// any sm_120; the cross-card shape adds the pinned exchange and the event chain and needs two, so it is
// skipped otherwise. The band math and the argument checks need no device.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/attn/kv_bands.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ingine::test;
namespace attn = ingine::arch::qwen35moe::kernels;
namespace kva = ingine::arch::qwen35moe;

namespace {

constexpr int H = attn::ATTN_HEADS;
constexpr int KV = attn::ATTN_KV_HEADS;
constexpr int D = attn::ATTN_HEAD_DIM;
constexpr std::size_t kValuesPerPos = std::size_t(KV) * D;  // 512 values per position, per cache side
constexpr float kSentinel = 1234.5f;

bool expect_eq(const char* tag, long long got, long long want) {
  const bool ok = got == want;
  std::printf("%s %s: got %lld want %lld\n", ok ? "PASS" : "FAIL", tag, got, want);
  return ok;
}

// ---------------------------------------------------------------- host halves, no CUDA needed

// Card 0 is about twice the bandwidth of card 1, so the boundary sits near two thirds of the positions, on a
// KV_BLOCK line, so each band's q8_0 blocks coincide with the un-banded cache's.
bool band_math() {
  bool ok = true;
  ok &= expect_eq("cut 262144", kva::kv_band_cut(262144), 174784);
  ok &= expect_eq("cut 6144", kva::kv_band_cut(6144), 4096);
  ok &= expect_eq("cut 32768", kva::kv_band_cut(32768), 21856);
  ok &= expect_eq("cut 1024", kva::kv_band_cut(1024), 704);
  ok &= expect_eq("cut 64 (one block each)", kva::kv_band_cut(64), 32);
  ok &= expect_eq("cut 63 (too small to split)", kva::kv_band_cut(63), 63);
  ok &= expect_eq("cut 1", kva::kv_band_cut(1), 1);
  ok &= expect_eq("cut 6144 ratio 1", kva::kv_band_cut(6144, 1.0f), 3072);
  ok &= expect_eq("cut 6144 ratio 3", kva::kv_band_cut(6144, 3.0f), 4608);
  ok &= expect_eq("cut 6144 ratio 0 (off)", kva::kv_band_cut(6144, 0.0f), 6144);

  // The mirror holds the widest column the decoder's depth rule can produce at this capacity.
  ok &= expect_eq("mirror 262144", kva::kv_band_overlap(262144, 174784), 4096);
  ok &= expect_eq("mirror 32768", kva::kv_band_overlap(32768, 21856), 2048);
  ok &= expect_eq("mirror 6144", kva::kv_band_overlap(6144, 4096), 2048);
  ok &= expect_eq("mirror 1024 (capped at the cut)", kva::kv_band_overlap(1024, 704), 704);
  ok &= expect_eq("mirror unsplit", kva::kv_band_overlap(63, 63), 0);

  for (int max_pos : {64, 65, 96, 128, 511, 512, 1024, 4096, 6143, 6144, 32768, 131072, 262143, 262144}) {
    const int cut = kva::kv_band_cut(max_pos);
    const int overlap = kva::kv_band_overlap(max_pos, cut);
    const bool splittable = max_pos >= 2 * attn::KV_BLOCK;
    const bool aligned = !splittable || cut % attn::KV_BLOCK == 0;
    const bool leaves_room = !splittable || (cut >= attn::KV_BLOCK && max_pos - cut >= attn::KV_BLOCK);
    const bool mirror_in_range = !splittable || (overlap >= 1 && overlap <= cut);
    if (!aligned || !leaves_room || !mirror_in_range || cut <= 0 || cut > max_pos) {
      std::printf("FAIL cut %d: cut %d overlap %d (aligned=%d leaves_room=%d mirror=%d)\n", max_pos, cut, overlap,
                  aligned, leaves_room, mirror_in_range);
      ok = false;
    }
    // Band bytes are slices: the two bands hold exactly what one cache of max_pos holds, plus the mirror.
    const std::size_t whole = kva::kv_band_bytes(max_pos, attn::KvType::Q8_0);
    const std::size_t split = kva::kv_band_bytes(cut, attn::KvType::Q8_0) +
                              kva::kv_band_bytes(max_pos - cut + overlap, attn::KvType::Q8_0);
    if (splittable && split != whole + kva::kv_band_bytes(overlap, attn::KvType::Q8_0)) {
      std::printf("FAIL cut %d: bands %zu vs whole %zu plus mirror\n", max_pos, split, whole);
      ok = false;
    }
    // The decoder's own depth rule must always be placeable on this plan: no query at any position may
    // reach a geometry the launcher would refuse.
    for (int pos = 0; pos < max_pos; pos += std::max(1, max_pos / 9)) {
      const int splits = std::max(1, std::min(attn::kMaxSplits, (pos + kva::KV_BAND_SPLIT_POSITIONS) /
                                                                    kva::KV_BAND_SPLIT_POSITIONS));
      if (splittable && kva::kv_band_columns_at(cut, overlap, pos, splits) < 0) {
        std::printf("FAIL cut %d: pos %d splits %d cannot be placed (cut %d mirror %d)\n", max_pos, pos, splits, cut,
                    overlap);
        ok = false;
      }
    }
  }
  std::printf("%s cut, mirror and byte-slice invariants over a sweep of capacities\n", ok ? "PASS" : "FAIL");

  const double share = double(kva::kv_band_cut(262144)) / 262144.0;
  const bool proportionate = std::fabs(share - 2.0 / 3.0) <= double(attn::KV_BLOCK) / 262144.0;
  std::printf("%s 262144 split gives card 0 share %.6f (want 0.666667 +- one block)\n",
              proportionate ? "PASS" : "FAIL", share);
  return ok && proportionate;
}

// The column split decides whether a banded result can be bit-identical to the un-banded one: every column of
// the grid must land wholly inside one band, and the same rule the launcher uses must say so.
bool column_math() {
  bool ok = true;
  struct Shape {
    int max_pos;
    float ratio;
  };
  for (Shape s : {Shape{1024, 2.0f}, Shape{6144, 2.0f}, Shape{32768, 2.0f}, Shape{262144, 2.0f},
                  Shape{262144, 1.0f}, Shape{262144, 3.0f}}) {
    const int cut = kva::kv_band_cut(s.max_pos, s.ratio);
    const int overlap = kva::kv_band_overlap(s.max_pos, cut);
    const int peer_begin = cut - overlap;
    const int step = std::max(1, s.max_pos / 13);
    int refusals = 0;
    for (int pos = 0; pos < s.max_pos; pos += step) {
      for (int splits : {1, 2, 3, 8, 16, 32, attn::kMaxSplits}) {
        const int per = (pos + 1 + splits - 1) / splits;
        const int c1 = kva::kv_band_columns_at(cut, overlap, pos, splits);
        if (c1 < 0) {
          // The only legal reason to refuse is a column wider than the mirror.
          if (per <= overlap) {
            std::printf("FAIL columns max_pos=%d pos=%d splits=%d: refused with per %d <= mirror %d\n", s.max_pos,
                        pos, splits, per, overlap);
            ok = false;
          }
          ++refusals;
          continue;
        }
        if (c1 > splits) {
          std::printf("FAIL columns max_pos=%d pos=%d splits=%d: c1 %d outside [0,%d]\n", s.max_pos, pos, splits, c1,
                      splits);
          ok = false;
          continue;
        }
        // Band 0's columns, clipped to the query, end at or before the boundary.
        if (std::min(pos + 1, c1 * per) > cut) {
          std::printf("FAIL columns max_pos=%d pos=%d splits=%d: band 0 reaches %d past cut %d\n", s.max_pos, pos,
                      splits, c1 * per, cut);
          ok = false;
        }
        // Band 1's first column starts on a row it holds.
        if (c1 < splits && c1 * per < peer_begin) {
          std::printf("FAIL columns max_pos=%d pos=%d splits=%d: band 1 starts at %d before its rows %d\n",
                      s.max_pos, pos, splits, c1 * per, peer_begin);
          ok = false;
        }
      }
    }
    std::printf("[columns] max_pos=%d ratio %.1f: %d refusals at a column wider than the mirror\n", s.max_pos,
                double(s.ratio), refusals);
  }
  std::printf("%s every flash column lands whole inside one band\n", ok ? "PASS" : "FAIL");
  return ok;
}

// A malformed request must be refused before anything is launched, so these need no device. The band and
// buffer pointers are placeholders: every guard they trip is host-side.
bool rejects_bad_requests() {
  kva::KvBands b{};
  b.count = 2;
  b.max_pos = 6144;
  b.cut = 4096;
  b.overlap = 2048;
  b.bands[0] = {0, 0, 4096, reinterpret_cast<void*>(8), reinterpret_cast<void*>(8)};
  b.bands[1] = {1, 2048, 6144, reinterpret_cast<void*>(8), reinterpret_cast<void*>(8)};
  attn::SplitState* const fake = reinterpret_cast<attn::SplitState*>(std::uintptr_t(8));
  cudaStream_t const stream = reinterpret_cast<cudaStream_t>(std::uintptr_t(8));
  kva::KvBandExchange exch{};
  exch.host_device = 0;
  exch.peer_device = 1;

  attn::DecodeAttnArgs a{};
  a.q = reinterpret_cast<const float*>(8);
  a.k_new = reinterpret_cast<const float*>(8);
  a.v_new = reinterpret_cast<const float*>(8);
  a.out = reinterpret_cast<float*>(8);
  a.pos = 5000;
  a.max_pos = 6144;
  a.head_count = H;

  std::string err;
  auto refused = [&](const attn::DecodeAttnArgs& args, const kva::KvBands& bs, int splits, cudaStream_t peer,
                     kva::KvBandExchange* ex, const char* reason) {
    err.clear();
    const bool got = !kva::decode_attention_bands(args, bs, fake, splits, stream, peer, &err, ex);
    const bool named = got && err.find(reason) != std::string::npos;
    std::printf("%s %s: %s\n", named ? "PASS" : "FAIL", reason, got ? err.c_str() : "accepted");
    return named;
  };

  bool ok = refused(a, b, 0, stream, &exch, "split count out of range");
  ok &= refused(a, b, attn::kMaxSplits + 1, stream, &exch, "split count out of range");
  ok &= refused(a, b, attn::kMaxSplits, stream, nullptr, "needs a KvBandExchange");
  ok &= refused(a, b, attn::kMaxSplits, nullptr, &exch, "peer stream");

  attn::DecodeAttnArgs past = a;
  past.pos = 6144;
  ok &= refused(past, b, attn::kMaxSplits, stream, &exch, "position outside");

  attn::DecodeAttnArgs no_heads = a;
  no_heads.head_count = 0;
  ok &= refused(no_heads, b, attn::kMaxSplits, stream, &exch, "head range");

  attn::DecodeAttnArgs no_out = a;
  no_out.out = nullptr;
  ok &= refused(no_out, b, attn::kMaxSplits, stream, &exch, "are required");

  kva::KvBands short_cover = b;
  short_cover.bands[1].pos_end = 6143;
  ok &= refused(a, short_cover, attn::kMaxSplits, stream, &exch, "cover max_pos");

  kva::KvBands shifted = b;
  shifted.bands[0].pos_begin = 1;
  ok &= refused(a, shifted, attn::kMaxSplits, stream, &exch, "band 0");

  // A grid whose column is wider than the mirror cannot be placed: the call refuses rather than let a band
  // read rows it does not hold.
  kva::KvBands narrow = b;
  narrow.overlap = 16;
  narrow.bands[1].pos_begin = narrow.cut - narrow.overlap;
  ok &= refused(a, narrow, 2, stream, &exch, "no column boundary fits");
  return ok;
}

// ---------------------------------------------------------------- cache packing

// A band's cache bytes are a contiguous slice of the un-banded cache on both sides of the q8_0 layout: the
// values array is position-major and the scale array follows it, so a band shifts both by its own start.
std::vector<std::uint8_t> pack_band(attn::KvType type, const std::vector<std::uint8_t>& whole, int max_pos,
                                    int begin, int end) {
  const std::size_t voff = std::size_t(begin) * kValuesPerPos;
  const std::size_t vlen = std::size_t(end - begin) * kValuesPerPos;
  std::vector<std::uint8_t> out;
  if (type == attn::KvType::Bf16) {
    out.assign(whole.begin() + std::ptrdiff_t(2 * voff), whole.begin() + std::ptrdiff_t(2 * (voff + vlen)));
    return out;
  }
  const std::size_t values_all = std::size_t(max_pos) * kValuesPerPos;
  const std::size_t sc_bytes = vlen / attn::KV_BLOCK * sizeof(__half);
  const std::size_t sc_off = values_all + voff / attn::KV_BLOCK * sizeof(__half);
  out.resize(vlen + sc_bytes);
  std::memcpy(out.data(), whole.data() + voff, vlen);
  std::memcpy(out.data() + vlen, whole.data() + sc_off, sc_bytes);
  return out;
}

// Positions past the query hold valid encodings of their own, so a band that reads past its rows, or a store
// that lands in the wrong place, shows up as a changed output rather than as an accidental zero.
std::vector<std::uint8_t> fill_cache(attn::KvType type, int max_pos, Rng& rng) {
  const std::size_t values = std::size_t(max_pos) * kValuesPerPos;
  std::vector<std::uint8_t> buf(std::size_t(attn::kv_cache_bytes(type, max_pos)));
  if (type == attn::KvType::Bf16) {
    auto* p = reinterpret_cast<__nv_bfloat16*>(buf.data());
    for (std::size_t i = 0; i < values; ++i) p[i] = __float2bfloat16(float(rng.normal()) * 0.3f);
    return buf;
  }
  auto* v = attn::kv_q8_values(buf.data());
  for (std::size_t i = 0; i < values; ++i) v[i] = std::int8_t(rng.next() % 255) - 127;
  auto* s = const_cast<__half*>(attn::kv_q8_scales(buf.data(), values));
  for (std::size_t b = 0; b < values / attn::KV_BLOCK; ++b) s[b] = __float2half(0.002f + 0.001f * float(b % 13));
  return buf;
}

// The device rounds the stored row with __float2int_rn and the FP64 reference with std::lround, which differ
// on an exact half; that row may be one LSB off. Everything else, including every scale, is bit-identical.
bool band_bytes_match(attn::KvType type, const std::vector<std::uint8_t>& got,
                      const std::vector<std::uint8_t>& want, int begin, int end, int written_row) {
  if (got.size() != want.size()) {
    std::printf("FAIL band bytes: size %zu vs %zu\n", got.size(), want.size());
    return false;
  }
  const std::size_t vlen = std::size_t(end - begin) * kValuesPerPos;
  for (std::size_t i = 0; i < vlen; ++i) {
    const int row = begin + int(i / kValuesPerPos);
    if (type == attn::KvType::Q8_0 && row == written_row) {
      const int diff = std::abs(int(std::int8_t(got[i])) - int(std::int8_t(want[i])));
      if (diff > 1) {
        std::printf("FAIL band bytes: value at row %d offset %zu differs by %d\n", row, i, diff);
        return false;
      }
      continue;
    }
    if (got[i] != want[i]) {
      std::printf("FAIL band bytes: byte %zu (row %d) is %02x want %02x\n", i, row, got[i], want[i]);
      return false;
    }
  }
  if (type != attn::KvType::Q8_0) return true;
  for (std::size_t i = vlen; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      std::printf("FAIL band bytes: scale byte %zu differs\n", i);
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------- device halves

// Frees on the device it allocated from, so a case may leave the current device wherever it likes.
struct DeviceBuf {
  void* p = nullptr;
  int device = 0;
  ~DeviceBuf() {
    if (!p) return;
    cudaSetDevice(device);
    cudaFree(p);
    p = nullptr;
  }
  void alloc(int dev, std::size_t bytes) {
    device = dev;
    INGINE_TEST_CUDA(cudaSetDevice(dev));
    INGINE_TEST_CUDA(cudaMalloc(&p, bytes));
    INGINE_TEST_CUDA(cudaMemset(p, 0, bytes));
  }
  template <class T>
  T* get() const {
    return static_cast<T*>(p);
  }
  template <class T>
  void upload(const std::vector<T>& h) const {
    INGINE_TEST_CUDA(cudaSetDevice(device));
    INGINE_TEST_CUDA(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  template <class T>
  std::vector<T> download(std::size_t n) const {
    std::vector<T> h(n);
    INGINE_TEST_CUDA(cudaSetDevice(device));
    INGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// sm_120 devices in order: the kernels are built for sm_120a and will not load anywhere else.
std::vector<int> sm120_devices() {
  std::vector<int> out;
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return out;
  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
    if (p.major == 12 && p.minor == 0) out.push_back(d);
  }
  return out;
}

// One case: an un-banded cache of max_pos positions, a query at pos, and the head range the launches may
// touch. The FP64 reference has already written row `pos` into the host cache bytes.
struct Case {
  int max_pos = 0;
  int pos = 0;
  int head_begin = 0;
  int head_count = H;
  int splits = 0;  // 0 takes the decoder's own depth rule
  attn::KvType type = attn::KvType::Q8_0;
  std::vector<float> q, k_new, v_new;
  std::vector<std::uint8_t> kc, vc;
  std::vector<float> ref;

  int columns() const {
    if (splits) return splits;
    return std::max(1, std::min(attn::kMaxSplits, (pos + kva::KV_BAND_SPLIT_POSITIONS) /
                                                      kva::KV_BAND_SPLIT_POSITIONS));
  }
};

Case make_case(int max_pos, int pos, int head_begin, int head_count, attn::KvType type, unsigned seed,
               int splits = 0) {
  Case c;
  c.max_pos = max_pos;
  c.pos = pos;
  c.head_begin = head_begin;
  c.head_count = head_count;
  c.splits = splits;
  c.type = type;
  Rng rng(seed);
  c.q.assign(std::size_t(H) * D, 0.0f);
  c.k_new.assign(std::size_t(KV) * D, 0.0f);
  c.v_new.assign(std::size_t(KV) * D, 0.0f);
  for (auto& v : c.q) v = float(rng.normal());
  for (auto& v : c.k_new) v = float(rng.normal());
  for (auto& v : c.v_new) v = float(rng.normal());
  c.kc = fill_cache(type, max_pos, rng);
  c.vc = fill_cache(type, max_pos, rng);

  c.ref.assign(std::size_t(H) * D, kSentinel);
  attn::DecodeAttnArgs o{};
  o.q = c.q.data();
  o.k_new = c.k_new.data();
  o.v_new = c.v_new.data();
  o.k_cache = c.kc.data();
  o.v_cache = c.vc.data();
  o.out = c.ref.data();
  o.pos = pos;
  o.max_pos = max_pos;
  o.kv_type_k = type;
  o.kv_type_v = type;
  o.head_begin = head_begin;
  o.head_count = head_count;
  attn::decode_attention_cpu(o);
  return c;
}

std::vector<double> ref_range(const Case& c) {
  std::vector<double> ref(std::size_t(c.head_count) * D);
  for (int hh = 0; hh < c.head_count; ++hh) {
    for (int d = 0; d < D; ++d) {
      ref[std::size_t(hh) * D + d] = double(c.ref[std::size_t(c.head_begin + hh) * D + d]);
    }
  }
  return ref;
}

// Heads outside the range must keep the bytes the case put there.
bool untouched(const char* tag, const std::vector<float>& got, const Case& c) {
  bool ok = true;
  for (int i = 0; i < H * D; ++i) {
    const int head = i / D;
    if (head < c.head_begin || head >= c.head_begin + c.head_count) ok &= (got[std::size_t(i)] == kSentinel);
  }
  std::printf("%s %s: heads outside [%d,%d) keep their bytes\n", ok ? "PASS" : "FAIL", tag, c.head_begin,
              c.head_begin + c.head_count);
  return ok;
}

// Activations and the SplitState row, on the host card, filled from a case.
struct Activations {
  DeviceBuf q, k_new, v_new, out, partial;
  void alloc(const Case& c, int dev) {
    q.alloc(dev, c.q.size() * sizeof(float));
    k_new.alloc(dev, c.k_new.size() * sizeof(float));
    v_new.alloc(dev, c.v_new.size() * sizeof(float));
    out.alloc(dev, std::size_t(H) * D * sizeof(float));
    partial.alloc(dev, std::size_t(H) * attn::kMaxSplits * sizeof(attn::SplitState));
    reset(c);
  }
  void reset(const Case& c) {
    q.upload(c.q);
    k_new.upload(c.k_new);
    v_new.upload(c.v_new);
    out.upload(std::vector<float>(std::size_t(H) * D, kSentinel));
  }
  attn::DecodeAttnArgs args(const Case& c) const {
    attn::DecodeAttnArgs a{};
    a.q = q.get<float>();
    a.k_new = k_new.get<float>();
    a.v_new = v_new.get<float>();
    a.out = out.get<float>();
    a.pos = c.pos;
    a.max_pos = c.max_pos;
    a.kv_type_k = c.type;
    a.kv_type_v = c.type;
    a.head_begin = c.head_begin;
    a.head_count = c.head_count;
    a.partial = partial.get<attn::SplitState>();
    a.splits = c.columns();
    return a;
  }
  std::vector<float> result() const { return out.download<float>(std::size_t(H) * D); }
};

// Cache bytes into a plan's bands, from the case's un-banded whole cache.
void fill_plan(const Case& c, const kva::KvBands& bands) {
  for (const kva::KvBand& band : bands.bands) {
    if (band.device < 0 || !band.k) continue;
    const std::vector<std::uint8_t> pk = pack_band(c.type, c.kc, c.max_pos, band.pos_begin, band.pos_end);
    const std::vector<std::uint8_t> pv = pack_band(c.type, c.vc, c.max_pos, band.pos_begin, band.pos_end);
    INGINE_TEST_CUDA(cudaSetDevice(band.device));
    INGINE_TEST_CUDA(cudaMemcpy(band.k, pk.data(), pk.size(), cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(band.v, pv.data(), pv.size(), cudaMemcpyHostToDevice));
  }
}

// One banded case, checked against two references: the single-cache device run over the same bytes (must be
// bit-identical) and the FP64 host reference (must be within tolerance).
bool run_banded_case(const char* tag, const Case& c, const std::vector<int>& devs) {
  std::string err;
  const int splits = c.columns();
  const int dev0 = devs[0];
  const int dev1 = devs.size() > 1 ? devs[1] : devs[0];
  // With one device the plan is forced by an explicit cut, so the band arithmetic is covered everywhere.
  const int forced_cut = devs.size() > 1 ? 0 : kva::kv_band_cut(c.max_pos, kva::KV_BAND_RATIO);

  kva::KvBands bands{};
  if (!kva::kv_bands_init(&bands, c.max_pos, dev0, dev1, c.type, c.type, &err, kva::KV_BAND_RATIO, forced_cut)) {
    std::printf("FAIL %s: %s\n", tag, err.c_str());
    return false;
  }
  const int c1 = kva::kv_band_columns(bands, c.pos, splits);
  std::printf("[%s] max_pos=%d pos=%d heads=[%d,%d) type=%s bands=%d cut=%d mirror=%d splits=%d -> card 0 runs "
              "columns [0,%d), card 1 [%d,%d)\n",
              tag, c.max_pos, c.pos, c.head_begin, c.head_begin + c.head_count,
              c.type == attn::KvType::Q8_0 ? "q8_0" : "bf16", bands.count, bands.cut, bands.overlap, splits, c1, c1,
              splits);
  if (c1 < 0) {
    std::printf("FAIL %s: no column boundary fits (cut %d mirror %d)\n", tag, bands.cut, bands.overlap);
    kva::kv_bands_free(&bands);
    return false;
  }
  fill_plan(c, bands);

  Activations act;
  act.alloc(c, dev0);
  const attn::DecodeAttnArgs a = act.args(c);

  cudaStream_t s1 = nullptr;
  kva::KvBandExchange exch{};
  const bool cross = bands.count == 2 && bands.bands[1].device != dev0;
  if (cross) {
    INGINE_TEST_CUDA(cudaSetDevice(bands.bands[1].device));
    INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    if (!kva::kv_band_exchange_init(&exch, dev0, bands.bands[1].device, &err)) {
      std::printf("FAIL %s: %s\n", tag, err.c_str());
      INGINE_TEST_CUDA(cudaStreamDestroy(s1));
      kva::kv_bands_free(&bands);
      return false;
    }
  }
  INGINE_TEST_CUDA(cudaSetDevice(dev0));

  // Reference: one un-banded cache over the whole capacity, on the host card, split and combine.
  kva::KvBands whole{};
  if (!kva::kv_bands_init(&whole, c.max_pos, dev0, dev0, c.type, c.type, &err)) {
    std::printf("FAIL %s: %s\n", tag, err.c_str());
    kva::kv_bands_free(&bands);
    return false;
  }
  fill_plan(c, whole);
  attn::DecodeAttnArgs ref = act.args(c);
  ref.k_cache = whole.bands[0].k;
  ref.v_cache = whole.bands[0].v;
  INGINE_TEST_CUDA(cudaSetDevice(dev0));
  if (attn::decode_attention(ref, nullptr) != cudaSuccess) {
    std::printf("FAIL %s: single-cache decode_attention\n", tag);
    std::exit(1);
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> single = act.result();

  // The banded run, from the same activation bytes.
  act.reset(c);
  bool ok = true;
  const bool launched = kva::decode_attention_bands(a, bands, act.partial.get<attn::SplitState>(), splits,
                                                   nullptr, s1, &err, cross ? &exch : nullptr);
  if (!launched) {
    std::printf("FAIL %s: %s\n", tag, err.c_str());
    ok = false;
  } else {
    INGINE_TEST_CUDA(cudaSetDevice(dev0));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    if (s1) {
      INGINE_TEST_CUDA(cudaSetDevice(bands.bands[1].device));
      INGINE_TEST_CUDA(cudaStreamSynchronize(s1));
    }
    const std::vector<float> got = act.result();

    // 1. bit-identical to the single-cache device run over the same bytes.
    int first_diff = -1;
    double worst = 0.0;
    for (int hh = 0; hh < c.head_count; ++hh) {
      for (int d = 0; d < D; ++d) {
        const std::size_t i = std::size_t(c.head_begin + hh) * D + d;
        if (std::memcmp(&got[i], &single[i], sizeof(float)) != 0) {
          if (first_diff < 0) first_diff = int(i);
          worst = std::max(worst, std::fabs(double(got[i]) - double(single[i])));
        }
      }
    }
    std::printf("%s %s: banded output is bit-identical to the single-cache run", first_diff < 0 ? "PASS" : "FAIL",
                tag);
    if (first_diff >= 0) std::printf(" (first diff at element %d, max delta %.3e)", first_diff, worst);
    std::printf("\n");
    ok &= first_diff < 0;

    // 2. and both match the FP64 reference over the same cache bytes.
    ok &= report(tag,
                 compare(got.data() + std::size_t(c.head_begin) * D, ref_range(c).data(),
                         std::size_t(c.head_count) * D),
                 1e-4);
    ok &= untouched(tag, got, c);

    // 3. each band still holds exactly the reference bytes for the positions it owns: that catches a store
    // that landed in the wrong band, a band reading past its rows, and any packing mistake.
    for (const kva::KvBand& band : bands.bands) {
      if (band.device < 0 || !band.k) continue;
      const std::vector<std::uint8_t> want_k = pack_band(c.type, c.kc, c.max_pos, band.pos_begin, band.pos_end);
      const std::vector<std::uint8_t> want_v = pack_band(c.type, c.vc, c.max_pos, band.pos_begin, band.pos_end);
      INGINE_TEST_CUDA(cudaSetDevice(band.device));
      std::vector<std::uint8_t> got_k(want_k.size()), got_v(want_v.size());
      INGINE_TEST_CUDA(cudaMemcpy(got_k.data(), band.k, want_k.size(), cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(got_v.data(), band.v, want_v.size(), cudaMemcpyDeviceToHost));
      const bool stored = c.pos >= band.pos_begin && c.pos < band.pos_end;
      const int row = stored ? c.pos : -1;
      const bool same = band_bytes_match(c.type, got_k, want_k, band.pos_begin, band.pos_end, row) &&
                        band_bytes_match(c.type, got_v, want_v, band.pos_begin, band.pos_end, row);
      std::printf("%s %s: band [%d,%d) on dev %d holds the reference bytes\n", same ? "PASS" : "FAIL", tag,
                  band.pos_begin, band.pos_end, band.device);
      ok &= same;
    }
  }
  if (s1) {
    INGINE_TEST_CUDA(cudaSetDevice(bands.bands[1].device));
    INGINE_TEST_CUDA(cudaStreamDestroy(s1));
  }
  kva::kv_band_exchange_free(&exch);
  kva::kv_bands_free(&whole);
  kva::kv_bands_free(&bands);
  INGINE_TEST_CUDA(cudaSetDevice(dev0));
  return ok;
}

// Informational, not asserted: the split kernel's realised cache bandwidth per card. The boundary is placed
// by a bandwidth ratio, so this is the number behind the choice. Timing lines never fail a case.
void bandwidth_probe(int depth, const std::vector<int>& devs) {
  const Case c = make_case(depth, depth - 1, 0, H, attn::KvType::Q8_0, 91, attn::kMaxSplits);
  const double bytes = double(attn::kv_cache_bytes(attn::KvType::Q8_0, depth)) * 2.0;
  std::vector<double> rates;
  for (int dev : devs) {
    kva::KvBands plan{};
    std::string err;
    if (!kva::kv_bands_init(&plan, depth, dev, dev, c.type, c.type, &err)) {
      std::printf("[bw] dev %d: %s\n", dev, err.c_str());
      continue;
    }
    fill_plan(c, plan);
    Activations act;
    act.alloc(c, dev);
    attn::DecodeAttnArgs a = act.args(c);
    a.k_cache = plan.bands[0].k;
    a.v_cache = plan.bands[0].v;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    INGINE_TEST_CUDA(cudaSetDevice(dev));
    bool good = true;
    for (int i = 0; i < 3 && good; ++i) {
      if (attn::decode_split(a, attn::kMaxSplits, nullptr) != cudaSuccess) good = false;
    }
    const int rounds = 20;
    cudaEventRecord(e0);
    for (int i = 0; i < rounds; ++i) attn::decode_split(a, attn::kMaxSplits, nullptr);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    INGINE_TEST_CUDA(cudaGetLastError());
    if (!good) {
      std::printf("[bw] dev %d: split launch failed\n", dev);
    } else {
      const double per_call = ms / rounds;
      const double gbs = bytes / (per_call / 1000.0) / 1e9;
      rates.push_back(gbs);
      std::printf("[bw] dev %d: %d positions, %.1f MB of K+V per launch, %.3f ms -> %.0f GB/s\n", dev, depth,
                  bytes / 1e6, per_call, gbs);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    kva::kv_bands_free(&plan);
  }
  if (rates.size() == 2 && rates[1] > 0.0) {
    const double ratio = rates[0] / rates[1];
    std::printf("[bw] measured card0:card1 split-kernel rate %.3f -> boundary at %.4f of the positions\n", ratio,
                ratio / (1.0 + ratio));
  }
}

}  // namespace

int main() {
  bool ok = band_math();
  ok &= column_math();
  ok &= rejects_bad_requests();

  const std::vector<int> devs = sm120_devices();
  if (devs.empty()) {
    std::printf("skip device halves: no sm_120 device present\n");
    std::printf("%s\n", ok ? "ALL PASS (host halves)" : "FAIL");
    return ok ? 0 : 1;
  }

  // Both bands on one device: the band arithmetic with no staging, so it is covered on any sm_120.
  const std::vector<int> one = {devs[0]};
  ok &= run_banded_case("local q8_0 1k", make_case(1024, 1000, 0, H, attn::KvType::Q8_0, 41, 8), one);
  ok &= run_banded_case("local q8_0 32k", make_case(32768, 32000, 0, H, attn::KvType::Q8_0, 42), one);
  ok &= run_banded_case("local bf16 32k", make_case(32768, 32000, 0, H, attn::KvType::Bf16, 43), one);
  ok &= run_banded_case("local q8_0 on the boundary", make_case(6144, 4096, 0, H, attn::KvType::Q8_0, 44), one);
  ok &= run_banded_case("local q8_0 in the mirror", make_case(6144, 3000, 0, H, attn::KvType::Q8_0, 45), one);
  ok &= run_banded_case("local q8_0 head range", make_case(6144, 6000, 8, 4, attn::KvType::Q8_0, 46), one);
  ok &= run_banded_case("local q8_0 widest column", make_case(6144, 5000, 0, H, attn::KvType::Q8_0, 47, 3), one);
  ok &= run_banded_case("local q8_0 all columns", make_case(6144, 6000, 0, H, attn::KvType::Q8_0, 48,
                                                            attn::kMaxSplits),
                         one);
  ok &= run_banded_case("local q8_0 pos 0", make_case(6144, 0, 0, H, attn::KvType::Q8_0, 49, 2), one);

  if (devs.size() < 2) {
    std::printf("skip cross-card cases and the bandwidth probe: %zu sm_120 device(s), they need two\n",
                devs.size());
    if (ok) std::printf("ALL PASS\n");
    return ok ? 0 : 1;
  }
  const std::vector<int> pair = {devs[0], devs[1]};
  // Depth 1k: the query never reaches the boundary, so card 1 reads nothing and nothing crosses.
  ok &= run_banded_case("cross q8_0 1k", make_case(1024, 1000, 0, H, attn::KvType::Q8_0, 31, 8), pair);
  // Depth 32k: past the boundary; both cards stream and the mirror carries the straddling column.
  ok &= run_banded_case("cross q8_0 32k", make_case(32768, 32000, 0, H, attn::KvType::Q8_0, 32), pair);
  ok &= run_banded_case("cross bf16 32k", make_case(32768, 32000, 0, H, attn::KvType::Bf16, 33), pair);
  ok &= run_banded_case("cross q8_0 head range", make_case(32768, 30000, 4, 8, attn::KvType::Q8_0, 34), pair);
  ok &= run_banded_case("cross q8_0 on the boundary", make_case(6144, 4096, 0, H, attn::KvType::Q8_0, 35), pair);
  ok &= run_banded_case("cross q8_0 in the mirror", make_case(6144, 3000, 0, H, attn::KvType::Q8_0, 36), pair);
  ok &= run_banded_case("cross q8_0 last position", make_case(6144, 6143, 0, H, attn::KvType::Q8_0, 37), pair);
  ok &= run_banded_case("cross q8_0 widest column", make_case(6144, 5000, 0, H, attn::KvType::Q8_0, 38, 3), pair);
  ok &= run_banded_case("cross q8_0 all columns", make_case(6144, 6000, 0, H, attn::KvType::Q8_0, 39,
                                                            attn::kMaxSplits),
                         pair);

  bandwidth_probe(32768, pair);

  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
