#include "arch/qwen35moe/kernels/attn/decode_attn.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "arch/qwen35moe/model/model_data.h"

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kDim = ATTN_HEAD_DIM;  // 256 threads, one per output dim
constexpr int kWarps = kDim / 32;    // 8 warps
constexpr int kGroup = ATTN_HEADS / ATTN_KV_HEADS;
static_assert(ATTN_HEADS % ATTN_KV_HEADS == 0, "query heads must divide evenly over KV heads");

// Split-kernel lane mapping: a lane owns sixteen consecutive dims, so sixteen lanes cover one position and a
// warp step advances two positions. The score butterfly then stays inside a half-warp (four shuffles instead
// of five) and every load is a full 16-byte or 32-byte access.
constexpr int kLaneDims = 16;
constexpr int kPosLanes = kDim / kLaneDims;               // 16 lanes per position
constexpr int kPosPerWarp = 32 / kPosLanes;               // 2 position streams per warp
constexpr int kStepPos = kWarps * kPosPerWarp;            // 16 positions per block step
constexpr int kQ8Words = kLaneDims / 4;                   // int32 words of q8 values per lane
constexpr int kQ4Words = kLaneDims / 8;                   // uint32 words of q4 nibbles per lane
constexpr int kBfWords = kLaneDims / 2;                   // uint32 words of bf16 values per lane
static_assert(kLaneDims * kPosLanes == kDim, "sixteen dims over sixteen lanes must cover the head");
static_assert(kBfWords <= 8 && kQ8Words <= 8 && kQ4Words <= 8, "the load staging holds the wider bf16 case");

// head_count > 0, so ATTN_HEADS - head_begin bounds the largest legal range without overflowing.
bool head_range_valid(const DecodeAttnArgs& args) {
  return args.head_begin >= 0 && args.head_count > 0 && args.head_count <= ATTN_HEADS - args.head_begin;
}

// A band either is absent (pos_end < 0, whole cache) or is a non-empty window inside the capacity.
bool band_valid(const DecodeAttnArgs& args) {
  if (args.pos_begin < 0 || args.max_pos <= 0) return false;
  if (args.pos_end < 0) return args.pos_begin == 0;
  return args.pos_begin < args.pos_end && args.pos_end <= args.max_pos;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Element decode of one 32-bit load. The shift is always a compile-time constant after unrolling, so the byte
// select folds into the conversion.
__device__ __forceinline__ float q8_byte(std::uint32_t w, int shift) { return float(std::int8_t(w >> shift)); }
// q4_0: one nibble, shifted by the -8 offset the format stores. `shift` is a compile-time constant per
// unrolled step, so the extract is a single bitfield instruction.
__device__ __forceinline__ float q4_nibble(std::uint32_t w, int shift) {
  return float(int((w >> shift) & 0xFu)) - 8.0f;
}
__device__ __forceinline__ float bf16_lo(std::uint32_t w) {
  return __bfloat162float(__ushort_as_bfloat16(std::uint16_t(w & 0xFFFFu)));
}
__device__ __forceinline__ float bf16_hi(std::uint32_t w) {
  return __bfloat162float(__ushort_as_bfloat16(std::uint16_t(w >> 16)));
}

// Partial dot of one lane's value words against its q registers. `USED` is the word count for the format: four
// int32 words of q8_0, two of q4_0's nibbles, eight of bf16, all read out of the same eight-word staging.
// Four independent accumulator chains: a single chain of sixteen dependent multiplies serialises the score at
// one latency per value, and the block runs only eight (HPAIR = 1) or four (HPAIR = 2) warps per scheduler.
template <int USED>
__device__ __forceinline__ float dot_q8(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; ++u) {
    d0 = fmaf(q[u * 4 + 0], q8_byte(w[u], 0), d0);
    d1 = fmaf(q[u * 4 + 1], q8_byte(w[u], 8), d1);
    d2 = fmaf(q[u * 4 + 2], q8_byte(w[u], 16), d2);
    d3 = fmaf(q[u * 4 + 3], q8_byte(w[u], 24), d3);
  }
  return (d0 + d1) + (d2 + d3);
}
template <int USED>
__device__ __forceinline__ float dot_bf16(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; u += 2) {
    d0 = fmaf(q[u * 2], bf16_lo(w[u]), d0);
    d1 = fmaf(q[u * 2 + 1], bf16_hi(w[u]), d1);
    d2 = fmaf(q[u * 2 + 2], bf16_lo(w[u + 1]), d2);
    d3 = fmaf(q[u * 2 + 3], bf16_hi(w[u + 1]), d3);
  }
  return (d0 + d1) + (d2 + d3);
}
// q4_0's sixteen dims are two words of eight nibbles, low nibble first, so the unrolled shift is 4 * b.
template <int USED>
__device__ __forceinline__ float dot_q4(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; ++u) {
#pragma unroll
    for (int b = 0; b < 8; b += 4) {
      d0 = fmaf(q[u * 8 + b + 0], q4_nibble(w[u], 4 * (b + 0)), d0);
      d1 = fmaf(q[u * 8 + b + 1], q4_nibble(w[u], 4 * (b + 1)), d1);
      d2 = fmaf(q[u * 8 + b + 2], q4_nibble(w[u], 4 * (b + 2)), d2);
      d3 = fmaf(q[u * 8 + b + 3], q4_nibble(w[u], 4 * (b + 3)), d3);
    }
  }
  return (d0 + d1) + (d2 + d3);
}
// The running numerator update acc = acc * corr + p * v, in its two halves: the rescale by the max correction and
// the term the new position adds. The row scale of the V side folds into p, so the add costs one multiply per
// dim.
__device__ __forceinline__ void acc_scale(float (&acc)[kLaneDims], float corr) {
#pragma unroll
  for (int j = 0; j < kLaneDims; ++j) acc[j] *= corr;
}
__device__ __forceinline__ void acc_add_q8(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p) {
#pragma unroll
  for (int u = 0; u < kQ8Words; ++u) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int j = u * 4 + b;
      acc[j] = fmaf(p, q8_byte(w[u], 8 * b), acc[j]);
    }
  }
}
__device__ __forceinline__ void acc_add_bf16(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p) {
#pragma unroll
  for (int u = 0; u < kBfWords; ++u) {
    acc[u * 2] = fmaf(p, bf16_lo(w[u]), acc[u * 2]);
    acc[u * 2 + 1] = fmaf(p, bf16_hi(w[u]), acc[u * 2 + 1]);
  }
}
__device__ __forceinline__ void acc_add_q4(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p) {
#pragma unroll
  for (int u = 0; u < kQ4Words; ++u) {
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int j = u * 8 + b;
      acc[j] = fmaf(p, q4_nibble(w[u], 4 * b), acc[j]);
    }
  }
}

// Sum inside the sixteen-lane half that owns one position. The xor distances never cross the half-warp, so the
// full mask is the correct mask and needs no convergence bookkeeping.
__device__ __forceinline__ float lane_sum16(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

__device__ __forceinline__ float kv_at(const void* base, KvType type, std::size_t index, std::uint64_t values) {
  if (type == KvType::Bf16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]);
  }
  if (type == KvType::Q4_0) {
    const std::uint8_t packed = kv_q4_values(base)[index / 2];
    const float q = float(int((packed >> (4 * (index & 1))) & 0xFu)) - 8.0f;
    return q * __half2float(kv_q4_scales(base, values)[index / KV_BLOCK]);
  }
  return float(kv_q8_values(const_cast<void*>(base))[index]) *
         __half2float(kv_q8_scales(base, values)[index / KV_BLOCK]);
}

__global__ void kv_store_bf16_kernel(DecodeAttnArgs args) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  if (h >= ATTN_KV_HEADS || d >= kDim) return;
  const std::size_t at = (std::size_t(args.pos - args.pos_begin) * ATTN_KV_HEADS + h) * kDim + d;
  static_cast<__nv_bfloat16*>(args.k_cache)[at] = __float2bfloat16(args.k_new[h * kDim + d]);
  static_cast<__nv_bfloat16*>(args.v_cache)[at] = __float2bfloat16(args.v_new[h * kDim + d]);
}

// q8_0 store: one warp per 32-value block, so the block scale and its 32 values never leave the warp.
__global__ __launch_bounds__(256) void kv_store_q8_kernel(DecodeAttnArgs args) {
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (h >= ATTN_KV_HEADS || warp >= kDim / KV_BLOCK) return;
  const int base_d = warp * KV_BLOCK;
  const int src = h * kDim + base_d + lane;
  const std::size_t at = (std::size_t(args.pos - args.pos_begin) * ATTN_KV_HEADS + h) * kDim + base_d;

  float k = args.k_new[src];
  float v = args.v_new[src];
  float kmax = fabsf(k), vmax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) {
    kmax = fmaxf(kmax, __shfl_down_sync(0xffffffffu, kmax, off));
    vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
  }
  kmax = __shfl_sync(0xffffffffu, kmax, 0);
  vmax = __shfl_sync(0xffffffffu, vmax, 0);
  const float kd = kmax > 0.0f ? kmax / 127.0f : 1.0f;
  const float vd = vmax > 0.0f ? vmax / 127.0f : 1.0f;

  const std::uint64_t values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;
  const std::size_t block = at / KV_BLOCK;
  if (lane == 0) {
    const_cast<__half*>(kv_q8_scales(args.k_cache, values))[block] = __float2half(kd);
    const_cast<__half*>(kv_q8_scales(args.v_cache, values))[block] = __float2half(vd);
  }
  kv_q8_values(args.k_cache)[at + lane] = std::int8_t(__float2int_rn(k / kd));
  kv_q8_values(args.v_cache)[at + lane] = std::int8_t(__float2int_rn(v / vd));
}

// q4_0 store: one warp per 32-value block, one byte per lane of the low half-warp (values 2j and 2j + 1), so
// the block's sixteen bytes leave in one coalesced store. The high half-warp contributes 0 to the max, which
// cannot change a maximum of absolute values.
__global__ __launch_bounds__(256) void kv_store_q4_kernel(DecodeAttnArgs args) {
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (h >= ATTN_KV_HEADS || warp >= kDim / KV_BLOCK) return;
  const int base_d = warp * KV_BLOCK;
  const bool packs = lane < KV_BLOCK / 2;
  float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
  if (packs) {
    k0 = args.k_new[h * kDim + base_d + 2 * lane];
    k1 = args.k_new[h * kDim + base_d + 2 * lane + 1];
    v0 = args.v_new[h * kDim + base_d + 2 * lane];
    v1 = args.v_new[h * kDim + base_d + 2 * lane + 1];
  }
  float kmax = fmaxf(fabsf(k0), fabsf(k1)), vmax = fmaxf(fabsf(v0), fabsf(v1));
  for (int off = 16; off > 0; off >>= 1) {
    kmax = fmaxf(kmax, __shfl_down_sync(0xffffffffu, kmax, off));
    vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
  }
  kmax = __shfl_sync(0xffffffffu, kmax, 0);
  vmax = __shfl_sync(0xffffffffu, vmax, 0);
  const float kd = kmax > 0.0f ? kmax / 8.0f : 1.0f;
  const float vd = vmax > 0.0f ? vmax / 8.0f : 1.0f;

  const std::uint64_t values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;
  const std::size_t at = (std::size_t(args.pos - args.pos_begin) * ATTN_KV_HEADS + h) * kDim + base_d;
  if (lane == 0) {
    const_cast<__half*>(kv_q4_scales(args.k_cache, values))[at / KV_BLOCK] = __float2half(kd);
    const_cast<__half*>(kv_q4_scales(args.v_cache, values))[at / KV_BLOCK] = __float2half(vd);
  }
  if (packs) {
    kv_q4_values(args.k_cache)[at / 2 + lane] = std::uint8_t(kv_q4_nibble(k0, kd) | kv_q4_nibble(k1, kd) << 4);
    kv_q4_values(args.v_cache)[at / 2 + lane] = std::uint8_t(kv_q4_nibble(v0, vd) | kv_q4_nibble(v1, vd) << 4);
  }
}

__global__ __launch_bounds__(kDim) void decode_kernel(DecodeAttnArgs args) {
  const int h = args.head_begin + blockIdx.x;  // absolute query head; the grid is the head range
  const int d = threadIdx.x;
  const int kh = h / kGroup;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint64_t cache_values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;

  __shared__ float scores[ATTN_CHUNK];
  __shared__ float warp_sums[kWarps];
  __shared__ float run_max, run_sum, chunk_max, chunk_sum;

  // q for this head in registers: this lane covers dims lane, lane + 32, ... lane + 224.
  float q_reg[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) q_reg[i] = args.q[h * kDim + lane + i * 32];

  if (threadIdx.x == 0) {
    run_max = -INFINITY;
    run_sum = 0.0f;
  }
  __syncthreads();

  const float scale = rsqrtf(float(kDim));
  float acc = 0.0f;
  // Un-banded only: decode_attention refuses a band on this path, so global positions are rows.
  for (int base = 0; base <= args.pos; base += ATTN_CHUNK) {
    const int len = min(ATTN_CHUNK, args.pos + 1 - base);

    // One warp per position: score[i] = q . k(base + i).
    for (int i = warp; i < len; i += kWarps) {
      const std::size_t kbase = (std::size_t(base + i) * ATTN_KV_HEADS + kh) * kDim;
      float dot = 0.0f;
#pragma unroll
      for (int r = 0; r < 8; ++r) dot += q_reg[r] * kv_at(args.k_cache, args.kv_type_k, kbase + lane + r * 32, cache_values);
      dot = warp_sum(dot);
      if (lane == 0) scores[i] = dot * scale;
    }
    __syncthreads();

    // Block max over the chunk, merged with the running max.
    float local_max = -INFINITY;
    for (int i = d; i < len; i += kDim) local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));
    if (lane == 0) warp_sums[warp] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
      float m = -INFINITY;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) m = fmaxf(m, warp_sums[w]);
      chunk_max = m;
    }
    __syncthreads();

    const float m_new = fmaxf(run_max, chunk_max);
    const float rescale = expf(run_max - m_new);

    // Block sum of exp(score - m_new).
    float local_sum = 0.0f;
    for (int i = d; i < len; i += kDim) local_sum += expf(scores[i] - m_new);
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_down_sync(0xffffffffu, local_sum, off);
    __syncthreads();
    if (lane == 0) warp_sums[warp] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      float s = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) s += warp_sums[w];
      chunk_sum = s;
    }
    __syncthreads();

    acc *= rescale;
    for (int i = 0; i < len; ++i) {
      const float p = expf(scores[i] - m_new);
      acc += p * kv_at(args.v_cache, args.kv_type_v, (std::size_t(base + i) * ATTN_KV_HEADS + kh) * kDim + d, cache_values);
    }

    if (threadIdx.x == 0) {
      run_sum = run_sum * rescale + chunk_sum;
      run_max = m_new;
    }
    __syncthreads();
  }

  args.out[h * kDim + d] = acc / run_sum;
}

// Mixed K/V types: the first half of the block stores K, the second half V, each in its own format. Only used
// when the two sides differ; the uniform cases take the simpler kernels above.
__global__ __launch_bounds__(512) void kv_store_mixed_kernel(DecodeAttnArgs args) {
  const int half = threadIdx.x / 256;
  const int t = threadIdx.x % 256;
  const int h = blockIdx.x;
  const int warp = t >> 5;
  const int lane = t & 31;
  if (h >= ATTN_KV_HEADS) return;
  // decode_kv_store never launches this for a position outside the band, so the row is always in range.
  const int row = args.pos - args.pos_begin;
  const float* src = half == 0 ? args.k_new : args.v_new;
  void* dst = half == 0 ? args.k_cache : args.v_cache;
  const KvType type = half == 0 ? args.kv_type_k : args.kv_type_v;
  if (type == KvType::Bf16) {
    const int d = t;
    if (d >= kDim) return;
    const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + d;
    static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(src[h * kDim + d]);
    return;
  }
  if (warp >= kDim / KV_BLOCK) return;
  const int base_d = warp * KV_BLOCK;
  const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + base_d;
  const std::size_t block = at / KV_BLOCK;
  const std::uint64_t values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;
  if (type == KvType::Q8_0) {
    float v = src[h * kDim + base_d + lane];
    float vmax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
    vmax = __shfl_sync(0xffffffffu, vmax, 0);
    const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
    if (lane == 0) const_cast<__half*>(kv_q8_scales(dst, values))[block] = __float2half(d);
    kv_q8_values(dst)[at + lane] = std::int8_t(__float2int_rn(v / d));
    return;
  }
  const bool packs = lane < KV_BLOCK / 2;
  float v0 = 0.0f, v1 = 0.0f;
  if (packs) {
    v0 = src[h * kDim + base_d + 2 * lane];
    v1 = src[h * kDim + base_d + 2 * lane + 1];
  }
  float vmax = fmaxf(fabsf(v0), fabsf(v1));
  for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
  vmax = __shfl_sync(0xffffffffu, vmax, 0);
  const float d = vmax > 0.0f ? vmax / 8.0f : 1.0f;
  if (lane == 0) const_cast<__half*>(kv_q4_scales(dst, values))[block] = __float2half(d);
  if (packs) kv_q4_values(dst)[at / 2 + lane] = std::uint8_t(kv_q4_nibble(v0, d) | kv_q4_nibble(v1, d) << 4);
}

// Flash decode: one warp walks a strided subset of the split's positions and keeps its running softmax state in
// registers, so the score and the value update happen in the same pass with no shared score array and no
// per-chunk barriers. Sixteen dims per lane means sixteen lanes cover a position, so a warp step advances two
// positions and the score butterfly is four shuffles inside a half-warp instead of five across the warp.
//
// Per warp step (two positions) a lane does one aligned 16-byte read of K plus one scale, the same for V,
// sixteen K multiplies and sixteen fused value updates. The 16-byte value read is the widest the q8_0 layout
// allows — sixteen dims always sit inside one 32-value block, so one scale covers them — and it halves the
// load-instruction count of the eight-dims-per-lane mapping. This is what the depth cost needs: 2.85 GB of
// cache per token at 262k. q4_0 keeps the same mapping: the lane's sixteen dims are eight bytes, the aligned
// half of the 32-value block it shares with its neighbour lane, so the pair still reads one 16-byte block.
//
// The two positions a warp handles are two independent streams, one per half-warp, each with its own (m, l);
// the halves hold the same sixteen dims, so they merge through a lane-width shuffle before the block merge,
// which stays eight-wide and keeps the 8 KB shared stage.
//
// The cache formats are template arguments, not a runtime test: the measured loop is 1.2-1.4x faster for
// q8_0 and no slower for bf16 when each instantiation carries only its own decode, and it keeps the register
// count at the 64 that holds four blocks per SM.
//
// HPAIR query heads share one block. They sit in the same KV group, so they read the same K and V rows: one
// decode of the sixteen dims feeds HPAIR dots and HPAIR value updates. That decode is the largest single block
// of instructions in the step — 56 of its 170 warp instructions are byte extraction and int8 conversion, and the
// conversions run on the half-rate pipe — so serving a pair is what pays: 694 us per layer at 258048 on sm_120
// against 937 for the committed single-head step. HPAIR = 1 is the shape used for a head range that cannot be
// paired.
template <int HPAIR>
struct SplitHeadState {
  float q[HPAIR][kLaneDims];
  float acc[HPAIR][kLaneDims];
  float m[HPAIR];
  float l[HPAIR];
};

// One position of one lane as loaded: its sixteen K and sixteen V values plus the two block scales. Kept apart
// from the arithmetic so a step reads its operands from here and nothing hides the loads behind the reduction.
struct KvStage {
  std::uint32_t kw[8];
  std::uint32_t vw[8];
  float ks = 1.0f;  // bf16 stores no block scale, and the multiplies by it are folded away
  float vs = 1.0f;
};

// The lane's sixteen dims of one position, for every cache format. `GUARD` is the form used where the position
// may be past the end of the column, which is only ever the final step of a warp: it reads the last real row
// instead, so the staged words stay finite — the value update of that stream multiplies them by a zero weight
// and zero times an infinity would be a NaN.
template <KvType TK, KvType TV, bool GUARD>
__device__ __forceinline__ void load_kv(const std::uint64_t cache_values, int band, int kh, int d0, int p, int end,
                                        void* k_cache, void* v_cache, KvStage& s) {
  const int row = (GUARD && p >= end) ? end - 1 : p;
  const std::size_t at = (std::size_t(row - band) * ATTN_KV_HEADS + kh) * kDim + d0;

  if constexpr (TK == KvType::Q8_0) {
    const int4 r = *reinterpret_cast<const int4*>(kv_q8_values(k_cache) + at);
    s.kw[0] = r.x; s.kw[1] = r.y; s.kw[2] = r.z; s.kw[3] = r.w;
    s.ks = __half2float(kv_q8_scales(k_cache, cache_values)[at / KV_BLOCK]);
  } else if constexpr (TK == KvType::Q4_0) {
    const uint2 r = *reinterpret_cast<const uint2*>(kv_q4_values(k_cache) + at / 2);
    s.kw[0] = r.x; s.kw[1] = r.y;
    s.ks = __half2float(kv_q4_scales(k_cache, cache_values)[at / KV_BLOCK]);
  } else {
    const uint4* p4 = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(k_cache) + at);
    const uint4 a = p4[0], b = p4[1];
    s.kw[0] = a.x; s.kw[1] = a.y; s.kw[2] = a.z; s.kw[3] = a.w;
    s.kw[4] = b.x; s.kw[5] = b.y; s.kw[6] = b.z; s.kw[7] = b.w;
  }
  if constexpr (TV == KvType::Q8_0) {
    const int4 r = *reinterpret_cast<const int4*>(kv_q8_values(v_cache) + at);
    s.vw[0] = r.x; s.vw[1] = r.y; s.vw[2] = r.z; s.vw[3] = r.w;
    s.vs = __half2float(kv_q8_scales(v_cache, cache_values)[at / KV_BLOCK]);
  } else if constexpr (TV == KvType::Q4_0) {
    const uint2 r = *reinterpret_cast<const uint2*>(kv_q4_values(v_cache) + at / 2);
    s.vw[0] = r.x; s.vw[1] = r.y;
    s.vs = __half2float(kv_q4_scales(v_cache, cache_values)[at / KV_BLOCK]);
  } else {
    const uint4* p4 = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(v_cache) + at);
    const uint4 a = p4[0], b = p4[1];
    s.vw[0] = a.x; s.vw[1] = a.y; s.vw[2] = a.z; s.vw[3] = a.w;
    s.vw[4] = b.x; s.vw[5] = b.y; s.vw[6] = b.z; s.vw[7] = b.w;
  }
}

// One position of one lane's two position streams, for every head of the block. The staged K and V are decoded
// once and used by all HPAIR heads.
template <KvType TK, KvType TV, int HPAIR, bool GUARD>
__device__ __forceinline__ void split_step(const KvStage& st_in, int p, int end, SplitHeadState<HPAIR>& st) {
  // The final step's second stream can be past the end of the column. It must not leave the warp — the score
  // butterfly is a full-warp shuffle — so it computes from whatever was staged and its contribution is selected
  // away below: an absent position leaves (m, l, acc) exactly as it found them.
  const bool live = !GUARD || p < end;

  // The block scale is per lane and applied before the reduction: the sixteen dims of a lane sit in one
  // 32-value block, but different lanes sit in different blocks, so the scales are not common to the butterfly.
  const float scale = rsqrtf(float(kDim));
  float score[HPAIR], corr[HPAIR], px[HPAIR];
#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) {
    float dot;
    if constexpr (TK == KvType::Q8_0) dot = dot_q8<kQ8Words>(st_in.kw, st.q[hh]) * st_in.ks;
    else if constexpr (TK == KvType::Q4_0) dot = dot_q4<kQ4Words>(st_in.kw, st.q[hh]) * st_in.ks;
    else dot = dot_bf16<kBfWords>(st_in.kw, st.q[hh]);
    score[hh] = lane_sum16(dot) * scale;
  }
#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) {
    const float m_new = live ? fmaxf(st.m[hh], score[hh]) : st.m[hh];
    corr[hh] = live ? expf(st.m[hh] - m_new) : 1.0f;
    px[hh] = live ? expf(score[hh] - m_new) : 0.0f;
    st.l[hh] = st.l[hh] * corr[hh] + px[hh];
    st.m[hh] = m_new;
  }
  // The block scale of the V side folds into p, which keeps the value update at one multiply per dim; in the
  // bf16 instantiation vs is the constant 1.0 and the multiply is not generated at all.
#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) acc_scale(st.acc[hh], corr[hh]);
#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) {
    const float pv = px[hh] * st_in.vs;
    if constexpr (TV == KvType::Q8_0) acc_add_q8(st.acc[hh], st_in.vw, pv);
    else if constexpr (TV == KvType::Q4_0) acc_add_q4(st.acc[hh], st_in.vw, pv);
    else acc_add_bf16(st.acc[hh], st_in.vw, pv);
  }
}

// HPAIR query heads per block: grid.x is the head count divided by HPAIR, grid.y the column range.
template <KvType KTYPE_K, KvType KTYPE_V, int HPAIR>
__global__ __launch_bounds__(kDim) void decode_split_kernel(DecodeAttnArgs args) {
  const int h0 = args.head_begin + int(blockIdx.x) * HPAIR;
  const int kh = h0 / kGroup;  // HPAIR divides the group, so every head of the block shares one KV head
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint64_t cache_values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;
  const int band = args.pos_begin;  // global position of this cache's row 0

  // The column grid is cut from pos and splits alone — ceil((pos + 1) / splits) positions per column — exactly
  // as a single un-banded launch cuts it, so a launch that owns columns [col_begin, col_begin + gridDim.y)
  // writes the same states that launch would. A column past the query is empty and writes the empty state the
  // combine kernel skips, which is what keeps a stale column out of the merge.
  const int col = args.col_begin + int(blockIdx.y);
  const int per_split = (args.pos + 1 + args.splits - 1) / args.splits;
  const int begin = col * per_split;
  const int end = min(args.pos + 1, begin + per_split);
  const std::size_t row0 = std::size_t(blockIdx.x) * HPAIR * std::size_t(args.splits) + std::size_t(col);
  if (begin >= end) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int hh = 0; hh < HPAIR; ++hh) {
        args.partial[row0 + std::size_t(hh) * std::size_t(args.splits)].m = -INFINITY;
        args.partial[row0 + std::size_t(hh) * std::size_t(args.splits)].l = 0.0f;
      }
    }
#pragma unroll
    for (int hh = 0; hh < HPAIR; ++hh) args.partial[row0 + std::size_t(hh) * std::size_t(args.splits)].acc[threadIdx.x] = 0.0f;
    return;
  }

  __shared__ float s_m[HPAIR][kWarps];
  __shared__ float s_l[HPAIR][kWarps];
  __shared__ float s_max[HPAIR], s_total[HPAIR];  // the block-level state, kept off the per-warp rows
  __shared__ float s_acc[HPAIR][kWarps][kDim];

  const int sub = lane >> 4;                            // which of the warp's two position streams this lane is on
  const int d0 = (lane & (kPosLanes - 1)) * kLaneDims;  // this lane's sixteen dims, the same in both streams

  SplitHeadState<HPAIR> st;
#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) {
#pragma unroll
    for (int i = 0; i < kLaneDims; ++i) st.q[hh][i] = args.q[(h0 + hh) * kDim + d0 + i];
    st.m[hh] = -INFINITY;
    st.l[hh] = 0.0f;
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) st.acc[hh][j] = 0.0f;
  }

  // The steady-state steps carry no position test, so the hot loop has no divergent branch; only the final step of
  // a warp is guarded, because its second stream may already be past the end.
  int i0 = begin + warp * kPosPerWarp;
  KvStage stage;
  for (; i0 + kStepPos <= end; i0 += kStepPos) {
    load_kv<KTYPE_K, KTYPE_V, false>(cache_values, band, kh, d0, i0 + sub, end, args.k_cache, args.v_cache, stage);
    split_step<KTYPE_K, KTYPE_V, HPAIR, false>(stage, i0 + sub, end, st);
  }
  if (i0 < end) {
    load_kv<KTYPE_K, KTYPE_V, true>(cache_values, band, kh, d0, i0 + sub, end, args.k_cache, args.v_cache, stage);
    split_step<KTYPE_K, KTYPE_V, HPAIR, true>(stage, i0 + sub, end, st);
  }

  // Merge the warp's two streams. The partner half holds the same sixteen dims of the other position, so a
  // lane-width shuffle is enough; an empty stream carries m = -INFINITY and contributes nothing.
#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) {
    const float m_p = __shfl_xor_sync(0xffffffffu, st.m[hh], kPosLanes);
    const float l_p = __shfl_xor_sync(0xffffffffu, st.l[hh], kPosLanes);
    const float m_warp = fmaxf(st.m[hh], m_p);
    const float w_own = m_warp == -INFINITY ? 0.0f : expf(st.m[hh] - m_warp);
    const float w_partner = m_warp == -INFINITY ? 0.0f : expf(m_p - m_warp);
    st.m[hh] = m_warp;
    st.l[hh] = st.l[hh] * w_own + l_p * w_partner;
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) {
      st.acc[hh][j] = st.acc[hh][j] * w_own + __shfl_xor_sync(0xffffffffu, st.acc[hh][j], kPosLanes) * w_partner;
    }
    if (lane == 0) {
      s_m[hh][warp] = st.m[hh];
      s_l[hh][warp] = st.l[hh];
    }
    if (sub == 0) {  // one half-warp writes the merged head: 16 lanes x 16 dims = 256
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) s_acc[hh][warp][d0 + j] = st.acc[hh][j];
    }
  }
  __syncthreads();

  // The eight warps of every head of the block merge through one log-sum-exp. The per-warp maxima stay where the
  // warp merge wrote them: the block maximum is its own shared slot, so each warp keeps the weight exp(m_w - M)
  // it earned.
  if (threadIdx.x == 0) {
#pragma unroll
    for (int hh = 0; hh < HPAIR; ++hh) {
      float M = -INFINITY;
      for (int w = 0; w < kWarps; ++w) M = fmaxf(M, s_m[hh][w]);
      float L = 0.0f;
      for (int w = 0; w < kWarps; ++w) L += s_l[hh][w] * expf(s_m[hh][w] - M);
      s_max[hh] = M;
      s_total[hh] = L;
    }
  }
  __syncthreads();

#pragma unroll
  for (int hh = 0; hh < HPAIR; ++hh) {
    float total = 0.0f;
    for (int w = 0; w < kWarps; ++w) total += s_acc[hh][w][threadIdx.x] * expf(s_m[hh][w] - s_max[hh]);
    SplitState& out = args.partial[row0 + std::size_t(hh) * std::size_t(args.splits)];
    out.acc[threadIdx.x] = total;
    if (threadIdx.x == 0) {
      out.m = s_max[hh];
      out.l = s_total[hh];
    }
  }
}

// Merge the splits per head: log-sum-exp over the split maxima, then normalise.
__global__ __launch_bounds__(kDim) void combine_splits_kernel(DecodeAttnArgs args) {
  const int h_local = blockIdx.x;
  const int h = args.head_begin + h_local;
  const int d = threadIdx.x;

  __shared__ float shared_max;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s < args.splits; ++s) m = fmaxf(m, args.partial[std::size_t(h_local) * args.splits + s].m);
    shared_max = m;
  }
  __syncthreads();

  float acc = 0.0f;
  float l = 0.0f;
  for (int s = 0; s < args.splits; ++s) {
    const SplitState& p = args.partial[std::size_t(h_local) * args.splits + s];
    if (p.m == -INFINITY) continue;
    const float w = expf(p.m - shared_max);
    acc += p.acc[d] * w;
    if (d == 0) l += p.l * w;
  }
  if (d == 0) shared_max = l;  // reuse the shared slot for the denominator
  __syncthreads();
  args.out[h * kDim + d] = acc / shared_max;
}

}  // namespace

namespace {

// Query heads one block serves: the pair path needs both heads of the pair inside one KV group, so the range
// must start and end on an even boundary. ATTN_HEADS / ATTN_KV_HEADS is 8, so any even pair qualifies.
int split_pairing(const DecodeAttnArgs& args) {
  return (args.head_begin % 2 == 0 && args.head_count % 2 == 0) ? 2 : 1;
}

template <KvType TK, KvType TV>
cudaError_t launch_split(const DecodeAttnArgs& args, dim3 grid, cudaStream_t stream) {
  if (split_pairing(args) == 2) decode_split_kernel<TK, TV, 2><<<grid, kDim, 0, stream>>>(args);
  else decode_split_kernel<TK, TV, 1><<<grid, kDim, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace

cudaError_t decode_kv_store(const DecodeAttnArgs& args, cudaStream_t stream) {
  if (!args.k_new || !args.v_new || !args.k_cache || !args.v_cache) return cudaErrorInvalidValue;
  if (!band_valid(args)) return cudaErrorInvalidValue;
  // A band that does not hold `pos` has no row to write: the owning band's store covers the new token.
  if (args.pos < args.pos_begin || (args.pos_end >= 0 && args.pos >= args.pos_end)) return cudaSuccess;
  if (args.kv_type_k == KvType::Bf16 && args.kv_type_v == KvType::Bf16) {
    kv_store_bf16_kernel<<<ATTN_KV_HEADS, kDim, 0, stream>>>(args);
  } else if (args.kv_type_k == KvType::Q8_0 && args.kv_type_v == KvType::Q8_0) {
    kv_store_q8_kernel<<<ATTN_KV_HEADS, 256, 0, stream>>>(args);
  } else if (args.kv_type_k == KvType::Q4_0 && args.kv_type_v == KvType::Q4_0) {
    kv_store_q4_kernel<<<ATTN_KV_HEADS, 256, 0, stream>>>(args);
  } else {
    kv_store_mixed_kernel<<<ATTN_KV_HEADS, 2 * 256, 0, stream>>>(args);
  }
  return cudaGetLastError();
}

cudaError_t decode_split(const DecodeAttnArgs& args, int band_splits, cudaStream_t stream) {
  // `out` is deliberately not required: a band launch never writes it, and the peer band must not be handed
  // a pointer into another device's memory.
  if (!args.q || !args.k_cache || !args.v_cache || !args.partial || band_splits < 1) return cudaErrorInvalidValue;
  if (args.col_begin < 0 || band_splits > args.splits - args.col_begin) return cudaErrorInvalidValue;
  if (!head_range_valid(args) || !band_valid(args)) return cudaErrorInvalidValue;
  if (args.pos < 0 || args.pos >= args.max_pos) return cudaErrorInvalidValue;
  // A column must sit wholly inside the cache the launch is handed: the kernel indexes it band-locally.
  const int per_split = (args.pos + 1 + args.splits - 1) / args.splits;
  const int first = args.col_begin * per_split;
  const int last = std::min(args.pos + 1, first + band_splits * per_split);
  if (first < args.pos_begin || last > args.pos_begin + kv_cache_rows(args)) return cudaErrorInvalidValue;
  // One head pair per block shares the K/V decode; a range that cannot be paired keeps the single-head shape.
  const dim3 grid(args.head_count / split_pairing(args), band_splits);
  // One instantiation per (K, V) pair, so the measured loop carries only its own decode and the register count
  // stays at the four blocks per SM the layout was chosen for.
  const KvType tk = args.kv_type_k;
  const KvType tv = args.kv_type_v;
  if (tk == KvType::Bf16) {
    if (tv == KvType::Bf16) return launch_split<KvType::Bf16, KvType::Bf16>(args, grid, stream);
    if (tv == KvType::Q8_0) return launch_split<KvType::Bf16, KvType::Q8_0>(args, grid, stream);
    return launch_split<KvType::Bf16, KvType::Q4_0>(args, grid, stream);
  }
  if (tk == KvType::Q8_0) {
    if (tv == KvType::Bf16) return launch_split<KvType::Q8_0, KvType::Bf16>(args, grid, stream);
    if (tv == KvType::Q8_0) return launch_split<KvType::Q8_0, KvType::Q8_0>(args, grid, stream);
    return launch_split<KvType::Q8_0, KvType::Q4_0>(args, grid, stream);
  }
  if (tv == KvType::Bf16) return launch_split<KvType::Q4_0, KvType::Bf16>(args, grid, stream);
  if (tv == KvType::Q8_0) return launch_split<KvType::Q4_0, KvType::Q8_0>(args, grid, stream);
  return launch_split<KvType::Q4_0, KvType::Q4_0>(args, grid, stream);
}

cudaError_t decode_combine(const DecodeAttnArgs& args, cudaStream_t stream) {
  if (!args.out || !args.partial || args.splits < 1) return cudaErrorInvalidValue;
  if (!head_range_valid(args)) return cudaErrorInvalidValue;
  combine_splits_kernel<<<args.head_count, kDim, 0, stream>>>(args);
  return cudaGetLastError();
}

cudaError_t decode_attention(const DecodeAttnArgs& args, cudaStream_t stream) {
  if (!args.q || !args.k_new || !args.v_new || !args.k_cache || !args.v_cache || !args.out) {
    return cudaErrorInvalidValue;
  }
  if (args.pos < 0 || args.pos >= args.max_pos) return cudaErrorInvalidValue;
  if (!head_range_valid(args) || !band_valid(args)) return cudaErrorInvalidValue;
  // The single-pass kernel reads the cache through kv_at, which needs the rows of the buffer it is given; a
  // band is only ever walked by the split kernel, whose column window is band-local.
  if (args.pos_end >= 0 && !(args.partial && args.splits > 1)) return cudaErrorInvalidValue;
  cudaError_t err = decode_kv_store(args, stream);
  if (err != cudaSuccess) return err;
  if (args.partial && args.splits > 1) {
    err = decode_split(args, args.splits, stream);
    if (err != cudaSuccess) return err;
    return decode_combine(args, stream);
  }
  decode_kernel<<<args.head_count, kDim, 0, stream>>>(args);
  return cudaGetLastError();
}

namespace {

// Host mirror of the store kernels, per side, so the reference sees exactly the bytes the device wrote.
void store_side_cpu(const float* src, void* dst, KvType type, int pos, int h, int max_pos) {
  if (type == KvType::Bf16) {
    for (int d = 0; d < kDim; ++d) {
      const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + d;
      static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(src[h * kDim + d]);
    }
    return;
  }
  if (type == KvType::Q4_0) {
    const std::uint64_t values = std::uint64_t(max_pos) * ATTN_KV_HEADS * kDim;
    for (int block = 0; block < kDim / KV_BLOCK; ++block) {
      const int base = h * kDim + block * KV_BLOCK;
      float amax = 0.0f;
      for (int i = 0; i < KV_BLOCK; ++i) amax = std::max(amax, std::fabs(src[base + i]));
      const float d = amax > 0.0f ? amax / 8.0f : 1.0f;
      const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + block * KV_BLOCK;
      const_cast<__half*>(kv_q4_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
      for (int i = 0; i < KV_BLOCK / 2; ++i) {
        kv_q4_values(dst)[at / 2 + i] = std::uint8_t(kv_q4_nibble(src[base + 2 * i], d) |
                                                     kv_q4_nibble(src[base + 2 * i + 1], d) << 4);
      }
    }
    return;
  }
  for (int block = 0; block < kDim / KV_BLOCK; ++block) {
    float vmax = 0.0f;
    for (int i = 0; i < KV_BLOCK; ++i) {
      vmax = std::max(vmax, std::fabs(src[h * kDim + block * KV_BLOCK + i]));
    }
    const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
    const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + block * KV_BLOCK;
    const std::uint64_t values = std::uint64_t(max_pos) * ATTN_KV_HEADS * kDim;
    const_cast<__half*>(kv_q8_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
    for (int i = 0; i < KV_BLOCK; ++i) {
      kv_q8_values(dst)[at + i] = std::int8_t(std::lround(src[h * kDim + block * KV_BLOCK + i] / d));
    }
  }
}

void store_cpu(const DecodeAttnArgs& args) {
  for (int h = 0; h < ATTN_KV_HEADS; ++h) {
    store_side_cpu(args.k_new, args.k_cache, args.kv_type_k, args.pos, h, args.max_pos);
    store_side_cpu(args.v_new, args.v_cache, args.kv_type_v, args.pos, h, args.max_pos);
  }
}

float read_cpu(const void* base, KvType type, std::size_t index, std::uint64_t values) {
  if (type == KvType::Bf16) return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]);
  if (type == KvType::Q4_0) {
    const std::uint8_t packed = kv_q4_values(base)[index / 2];
    const float q = float(int((packed >> (4 * (index & 1))) & 0xFu)) - 8.0f;
    return q * __half2float(kv_q4_scales(base, values)[index / KV_BLOCK]);
  }
  return float(kv_q8_values(const_cast<void*>(base))[index]) *
         __half2float(kv_q8_scales(base, values)[index / KV_BLOCK]);
}

}  // namespace

void decode_attention_cpu(const DecodeAttnArgs& args) {
  if (!head_range_valid(args)) return;
  // The reference models one un-banded cache: a band would need its own row base and its own scale offset.
  if (args.pos_begin != 0 || args.pos_end >= 0) return;
  store_cpu(args);

  const double scale = 1.0 / std::sqrt(double(kDim));
  std::vector<double> q(ATTN_HEADS * kDim);
  for (int i = 0; i < ATTN_HEADS * kDim; ++i) q[std::size_t(i)] = double(args.q[i]);
  std::vector<double> score(args.pos + 1);
  const std::uint64_t cpu_values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;
  for (int hh = 0; hh < args.head_count; ++hh) {
    const int h = args.head_begin + hh;
    const int kh = h / kGroup;
    double m = -INFINITY;
    for (int i = 0; i <= args.pos; ++i) {
      double dot = 0.0;
      for (int d = 0; d < kDim; ++d) {
        dot += q[h * kDim + d] * double(read_cpu(args.k_cache, args.kv_type_k, (std::size_t(i) * ATTN_KV_HEADS + kh) * kDim + d, cpu_values));
      }
      score[i] = dot * scale;
      m = std::max(m, score[i]);
    }
    double sum = 0.0;
    for (int i = 0; i <= args.pos; ++i) {
      score[i] = std::exp(score[i] - m);
      sum += score[i];
    }
    for (int d = 0; d < kDim; ++d) {
      double acc = 0.0;
      for (int i = 0; i <= args.pos; ++i) {
        const std::size_t at = (std::size_t(i) * ATTN_KV_HEADS + kh) * kDim + d;
        acc += score[i] * double(read_cpu(args.v_cache, args.kv_type_v, at, cpu_values));
      }
      args.out[h * kDim + d] = float(acc / sum);
    }
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
