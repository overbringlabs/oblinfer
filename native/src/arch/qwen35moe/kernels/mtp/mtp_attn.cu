#include "arch/qwen35moe/kernels/mtp/mtp_attn.h"

#include <algorithm>
#include <cmath>

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kDim = ATTN_HEAD_DIM;  // 256 threads, one per output dim
constexpr int kWarps = kDim / 32;    // 8 warps
constexpr int kGroup = ATTN_HEADS / ATTN_KV_HEADS;

// Same lane mapping as the shipped q8_0 split kernel: a lane owns sixteen consecutive dims, sixteen lanes cover
// one position, and a warp step advances two positions. Sixteen dims are eight nibble bytes, so a lane reads
// its K and V with one eight-byte load each and its scale (one per 32-value block) never straddles a block.
constexpr int kLaneDims = 16;
constexpr int kPosLanes = kDim / kLaneDims;  // 16 lanes per position
constexpr int kPosPerWarp = 32 / kPosLanes;  // 2 position streams per warp
constexpr int kStepPos = kWarps * kPosPerWarp;
constexpr int kQ4Words = kLaneDims / 8;  // two 32-bit words of nibbles per lane

static_assert(kLaneDims * kPosLanes == kDim, "sixteen dims over sixteen lanes must cover the head");
static_assert(kQ4Words == 2, "a lane's sixteen dims are two 32-bit words");
static_assert(kLaneDims % MTP_KV_BLOCK == 0 || MTP_KV_BLOCK % kLaneDims == 0,
              "a lane's dims must stay inside one block scale");

bool head_range_valid(const DecodeAttnArgs& args) {
  return args.head_begin >= 0 && args.head_count > 0 && args.head_count <= ATTN_HEADS - args.head_begin;
}

bool band_valid(const DecodeAttnArgs& args) {
  if (args.pos_begin < 0 || args.max_pos <= 0) return false;
  if (args.pos_end < 0) return args.pos_begin == 0;
  return args.pos_begin < args.pos_end && args.pos_end <= args.max_pos;
}

// One nibble of a 32-bit load, offsets zero centred: x = scale * (nibble - 8).
__device__ __forceinline__ float q4_value(std::uint32_t w, int shift) {
  return float(int((w >> shift) & 0x0Fu) - 8);
}

// Partial dot of a lane's two nibble words against its sixteen q registers. The block scale is applied by the
// caller, so the butterfly stays a plain sum.
__device__ __forceinline__ float dot_q4(const std::uint32_t (&w)[kQ4Words], const float (&q)[kLaneDims]) {
  float dot = 0.0f;
#pragma unroll
  for (int u = 0; u < kQ4Words; ++u) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      dot = fmaf(q[u * 8 + 2 * b], q4_value(w[u], 8 * b), dot);
      dot = fmaf(q[u * 8 + 2 * b + 1], q4_value(w[u], 8 * b + 4), dot);
    }
  }
  return dot;
}

__device__ __forceinline__ void accum_q4(float (&acc)[kLaneDims], const std::uint32_t (&w)[kQ4Words], float p,
                                         float corr) {
#pragma unroll
  for (int u = 0; u < kQ4Words; ++u) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int j = u * 8 + 2 * b;
      acc[j] = fmaf(p, q4_value(w[u], 8 * b), acc[j] * corr);
      acc[j + 1] = fmaf(p, q4_value(w[u], 8 * b + 4), acc[j + 1] * corr);
    }
  }
}

// One warp per 32-value block; lanes 0..15 own one byte each, so a block's scale and its thirty-two values never
// leave the warp. The first half of the block stores K, the second half V.
__global__ __launch_bounds__(512) void mtp_kv_store_q4_kernel(DecodeAttnArgs args) {
  const int half = threadIdx.x / kDim;
  const int t = threadIdx.x % kDim;
  const int h = blockIdx.x;
  const int warp = t >> 5;
  const int lane = t & 31;
  if (h >= ATTN_KV_HEADS || warp >= kDim / MTP_KV_BLOCK) return;
  const float* src = half == 0 ? args.k_new : args.v_new;
  void* dst = half == 0 ? args.k_cache : args.v_cache;

  const int base_d = warp * MTP_KV_BLOCK;
  const std::size_t at = (std::size_t(args.pos - args.pos_begin) * ATTN_KV_HEADS + h) * kDim + base_d;

  // All thirty-two lanes take part in the reduction; the upper half carries zeros so the block max is the
  // lower half's.
  const bool owns = lane < MTP_KV_BLOCK / 2;
  const float v0 = owns ? src[h * kDim + base_d + 2 * lane] : 0.0f;
  const float v1 = owns ? src[h * kDim + base_d + 2 * lane + 1] : 0.0f;
  float amax = fmaxf(fabsf(v0), fabsf(v1));
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
  amax = __shfl_sync(0xffffffffu, amax, 0);
  const float d = amax > 0.0f ? amax / 8.0f : 1.0f;

  if (owns) {
    const int n0 = max(0, min(15, __float2int_rn(v0 / d) + 8));
    const int n1 = max(0, min(15, __float2int_rn(v1 / d) + 8));
    mtp_q4_values(dst)[at / 2 + lane] = std::uint8_t(n0 | (n1 << 4));
  }
  if (lane == 0) {
    const std::uint64_t values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;
    const_cast<__half*>(mtp_q4_scales(dst, values))[at / MTP_KV_BLOCK] = __float2half(d);
  }
}

// Flash decode over a Q4_0 cache. Structurally the shipped decode_split_kernel: one warp walks a strided subset
// of its column's positions and keeps (m, l, acc) in registers, the two positions of a warp step are two
// independent streams that merge through a lane-width shuffle, and the eight warps of the block merge through
// shared memory into one SplitState per (head, column).
__global__ __launch_bounds__(kDim) void mtp_decode_split_q4_kernel(DecodeAttnArgs args) {
  const int h = args.head_begin + blockIdx.x;
  const int kh = h / kGroup;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint64_t cache_values = std::uint64_t(kv_cache_rows(args)) * ATTN_KV_HEADS * kDim;
  const int band = args.pos_begin;  // global position of this cache's row 0

  const int col = args.col_begin + int(blockIdx.y);
  const int per_split = (args.pos + 1 + args.splits - 1) / args.splits;
  const int begin = col * per_split;
  const int end = min(args.pos + 1, begin + per_split);
  const std::size_t slot = std::size_t(blockIdx.x) * std::size_t(args.splits) + std::size_t(col);
  if (begin >= end) {
    // Empty column: the state decode_combine skips, so a column past the query never enters the merge.
    if (threadIdx.x == 0) {
      args.partial[slot].m = -INFINITY;
      args.partial[slot].l = 0.0f;
    }
    args.partial[slot].acc[threadIdx.x] = 0.0f;
    return;
  }

  __shared__ float s_m[kWarps];
  __shared__ float s_l[kWarps];
  __shared__ float s_acc[kWarps][kDim];
  __shared__ float s_total, s_max;

  const int sub = lane >> 4;
  const int d0 = (lane & (kPosLanes - 1)) * kLaneDims;
  const unsigned hmask = 0x0000ffffu << (sub * kPosLanes);

  float q_reg[kLaneDims];
#pragma unroll
  for (int i = 0; i < kLaneDims; ++i) q_reg[i] = args.q[h * kDim + d0 + i];

  const float scale = rsqrtf(float(kDim));
  float m = -INFINITY;
  float l = 0.0f;
  float acc[kLaneDims];
#pragma unroll
  for (int j = 0; j < kLaneDims; ++j) acc[j] = 0.0f;

  for (int i0 = begin + warp * kPosPerWarp; i0 < end; i0 += kStepPos) {
    const int p = i0 + sub;
    if (p >= end) continue;  // tail step: the second stream runs out first
    const std::size_t kbase = (std::size_t(p - band) * ATTN_KV_HEADS + kh) * kDim + d0;
    const std::size_t vbase = kbase;  // K and V share the layout, so the same index addresses both caches

    // Both loads are independent of the score, so they are issued before the reduction.
    std::uint32_t kw[kQ4Words], vw[kQ4Words];
    const uint2 kr = *reinterpret_cast<const uint2*>(mtp_q4_values(args.k_cache) + kbase / 2);
    const uint2 vr = *reinterpret_cast<const uint2*>(mtp_q4_values(args.v_cache) + vbase / 2);
    kw[0] = kr.x; kw[1] = kr.y;
    vw[0] = vr.x; vw[1] = vr.y;
    const float ks = __half2float(mtp_q4_scales(args.k_cache, cache_values)[kbase / MTP_KV_BLOCK]);
    const float vs = __half2float(mtp_q4_scales(args.v_cache, cache_values)[vbase / MTP_KV_BLOCK]);

    // The block scale is per lane (different lanes sit in different blocks), so it is applied before the
    // butterfly; the value side's scale folds into p instead, which keeps the update at one multiply per dim.
    float dot = dot_q4(kw, q_reg) * ks;
    for (int off = kPosLanes / 2; off > 0; off >>= 1) dot += __shfl_xor_sync(hmask, dot, off);
    const float score = dot * scale;

    const float m_new = fmaxf(m, score);
    const float corr = expf(m - m_new);
    const float p_ex = expf(score - m_new);
    l = l * corr + p_ex;
    m = m_new;
    accum_q4(acc, vw, p_ex * vs, corr);
  }

  // Merge the warp's two streams: the partner half holds the same sixteen dims of the other position.
  const float m_p = __shfl_xor_sync(0xffffffffu, m, kPosLanes);
  const float l_p = __shfl_xor_sync(0xffffffffu, l, kPosLanes);
  const float m_warp = fmaxf(m, m_p);
  const float w_own = m_warp == -INFINITY ? 0.0f : expf(m - m_warp);
  const float w_partner = m_warp == -INFINITY ? 0.0f : expf(m_p - m_warp);
  const float l_warp = l * w_own + l_p * w_partner;
#pragma unroll
  for (int j = 0; j < kLaneDims; ++j) {
    acc[j] = acc[j] * w_own + __shfl_xor_sync(0xffffffffu, acc[j], kPosLanes) * w_partner;
  }

  if (lane == 0) {
    s_m[warp] = m_warp;
    s_l[warp] = l_warp;
  }
  if (sub == 0) {  // one half-warp writes the merged head: 16 lanes x 16 dims = 256
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) s_acc[warp][d0 + j] = acc[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float M = -INFINITY;
    for (int w = 0; w < kWarps; ++w) M = fmaxf(M, s_m[w]);
    float L = 0.0f;
    for (int w = 0; w < kWarps; ++w) L += s_l[w] * expf(s_m[w] - M);
    s_max = M;
    s_total = L;
  }
  __syncthreads();

  float total = 0.0f;
  for (int w = 0; w < kWarps; ++w) total += s_acc[w][threadIdx.x] * expf(s_m[w] - s_max);
  args.partial[slot].acc[threadIdx.x] = total;
  if (threadIdx.x == 0) {
    args.partial[slot].m = s_max;
    args.partial[slot].l = s_total;
  }
}

}  // namespace

const char* mtp_kv_type_name(MtpKvType type) {
  switch (type) {
    case MtpKvType::Q4_0: return "q4_0";
    case MtpKvType::Q8_0: return "q8_0";
    case MtpKvType::Bf16: return "bf16";
  }
  return "unknown";
}

std::uint64_t mtp_kv_bytes(MtpKvType type, int max_pos) {
  if (max_pos <= 0) return 0;
  if (type == MtpKvType::Bf16) return kv_cache_bytes(KvType::Bf16, max_pos);
  if (type == MtpKvType::Q8_0) return kv_cache_bytes(KvType::Q8_0, max_pos);
  const std::uint64_t values = std::uint64_t(max_pos) * ATTN_KV_HEADS * ATTN_HEAD_DIM;
  return values / 2 + (values / MTP_KV_BLOCK) * sizeof(__half);
}

cudaError_t mtp_kv_store_q4(const DecodeAttnArgs& args, cudaStream_t stream) {
  if (!args.k_new || !args.v_new || !args.k_cache || !args.v_cache) return cudaErrorInvalidValue;
  if (!band_valid(args)) return cudaErrorInvalidValue;
  if (args.pos < args.pos_begin || (args.pos_end >= 0 && args.pos >= args.pos_end)) return cudaSuccess;
  mtp_kv_store_q4_kernel<<<ATTN_KV_HEADS, 2 * kDim, 0, stream>>>(args);
  return cudaGetLastError();
}

cudaError_t mtp_decode_split_q4(const DecodeAttnArgs& args, int band_splits, cudaStream_t stream) {
  if (!args.q || !args.k_cache || !args.v_cache || !args.partial || band_splits < 1) {
    return cudaErrorInvalidValue;
  }
  if (args.col_begin < 0 || band_splits > args.splits - args.col_begin) return cudaErrorInvalidValue;
  if (!head_range_valid(args) || !band_valid(args)) return cudaErrorInvalidValue;
  if (args.pos < 0 || args.pos >= args.max_pos) return cudaErrorInvalidValue;
  // A column must sit wholly inside the cache the launch is handed: the kernel indexes it band-locally.
  const int per_split = (args.pos + 1 + args.splits - 1) / args.splits;
  const int first = args.col_begin * per_split;
  const int last = std::min(args.pos + 1, first + band_splits * per_split);
  if (first < args.pos_begin || last > args.pos_begin + kv_cache_rows(args)) return cudaErrorInvalidValue;
  const dim3 grid(args.head_count, band_splits);
  mtp_decode_split_q4_kernel<<<grid, kDim, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace ingine::arch::qwen35moe::kernels
