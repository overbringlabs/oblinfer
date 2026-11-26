#include "arch/qwen35moe/kernels/batch/batch_attn.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kDim = ATTN_HEAD_DIM;  // 256 threads, one per output dim
constexpr int kWarps = kDim / 32;    // 8 warps
constexpr int kGroup = ATTN_HEADS / ATTN_KV_HEADS;
constexpr int kNewRow = ATTN_KV_HEADS * ATTN_HEAD_DIM;  // one position of k_new / v_new

// Lane mapping: eight consecutive dims per lane, a whole warp per position, eight warps per block. The
// single-token kernel's shape is sixteen dims and two position streams, which needs 64 registers; at verify
// width that shape costs 181 registers at n = 4, one block per SM, and the streaming loop loses three quarters
// of the memory parallelism it needs (measured: a four-query pass over the 258k cache cost 4x a single-query
// pass). Eight dims halves every per-query register set, which is what keeps two blocks per SM resident, and
// the cache read is what this kernel is for. Every load is still one aligned 8- (q8_0), 4- (q4_0) or 16-byte
// (bf16) access per lane.
constexpr int kLaneDims = 8;
constexpr int kPosLanes = kDim / kLaneDims;     // 32 lanes per position: a whole warp
constexpr int kPosPerWarp = 32 / kPosLanes;     // one position stream per warp
constexpr int kStepPos = kWarps * kPosPerWarp;  // 8 positions per block step
constexpr int kQ8Words = kLaneDims / 4;         // int32 words of q8 values per lane
constexpr int kQ4Words = kLaneDims / 8;         // uint32 words of q4 nibbles per lane
constexpr int kBfWords = kLaneDims / 2;         // uint32 words of bf16 values per lane
static_assert(kLaneDims * kPosLanes == kDim, "the lanes of a warp must cover the head exactly");
static_assert(kBfWords <= 8 && kQ8Words <= 8 && kQ4Words <= 8, "the load staging holds the wider bf16 case");

__device__ __forceinline__ float q8_byte(std::uint32_t w, int shift) { return float(std::int8_t(w >> shift)); }
__device__ __forceinline__ float q4_nibble(std::uint32_t w, int shift) {
  return float(int((w >> shift) & 0xFu)) - 8.0f;
}
__device__ __forceinline__ float bf16_lo(std::uint32_t w) {
  return __bfloat162float(__ushort_as_bfloat16(std::uint16_t(w & 0xFFFFu)));
}
__device__ __forceinline__ float bf16_hi(std::uint32_t w) {
  return __bfloat162float(__ushort_as_bfloat16(std::uint16_t(w >> 16)));
}

template <int USED>
__device__ __forceinline__ float dot_q8(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float dot = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; ++u) {
#pragma unroll
    for (int b = 0; b < 4; ++b) dot = fmaf(q[u * 4 + b], q8_byte(w[u], 8 * b), dot);
  }
  return dot;
}
template <int USED>
__device__ __forceinline__ float dot_q4(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float dot = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; ++u) {
#pragma unroll
    for (int b = 0; b < 8; ++b) dot = fmaf(q[u * 8 + b], q4_nibble(w[u], 4 * b), dot);
  }
  return dot;
}
template <int USED>
__device__ __forceinline__ float dot_bf16(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float dot = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; ++u) {
    dot = fmaf(q[u * 2], bf16_lo(w[u]), dot);
    dot = fmaf(q[u * 2 + 1], bf16_hi(w[u]), dot);
  }
  return dot;
}
// acc = acc * corr + p * v, with the cache's block scale folded into p.
__device__ __forceinline__ void accum_q8(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p,
                                         float corr) {
#pragma unroll
  for (int u = 0; u < kQ8Words; ++u) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int j = u * 4 + b;
      acc[j] = fmaf(p, q8_byte(w[u], 8 * b), acc[j] * corr);
    }
  }
}
__device__ __forceinline__ void accum_q4(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p,
                                         float corr) {
#pragma unroll
  for (int u = 0; u < kQ4Words; ++u) {
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int j = u * 8 + b;
      acc[j] = fmaf(p, q4_nibble(w[u], 4 * b), acc[j] * corr);
    }
  }
}
__device__ __forceinline__ void accum_bf16(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p,
                                           float corr) {
#pragma unroll
  for (int u = 0; u < kBfWords; ++u) {
    acc[u * 2] = fmaf(p, bf16_lo(w[u]), acc[u * 2] * corr);
    acc[u * 2 + 1] = fmaf(p, bf16_hi(w[u]), acc[u * 2 + 1] * corr);
  }
}

// This lane's sixteen dims of the cache row that starts at value index `at`, plus the block scale that covers
// them: one lane's sixteen dims always sit inside one 32-value block, so a scale is per lane.
template <KvType TYPE>
__device__ __forceinline__ void load_row(const void* base, std::size_t at, std::uint64_t values,
                                         std::uint32_t (&w)[8], float* scale) {
  if constexpr (TYPE == KvType::Q8_0) {
    const int2 r = *reinterpret_cast<const int2*>(kv_q8_values(const_cast<void*>(base)) + at);
    w[0] = unsigned(r.x);
    w[1] = unsigned(r.y);
    *scale = __half2float(kv_q8_scales(base, values)[at / KV_BLOCK]);
  } else if constexpr (TYPE == KvType::Q4_0) {
    // Eight nibbles are four bytes: two lanes share one 32-value block, as the format expects.
    w[0] = *reinterpret_cast<const std::uint32_t*>(kv_q4_values(base) + at / 2);
    *scale = __half2float(kv_q4_scales(base, values)[at / KV_BLOCK]);
  } else {
    const uint4 a = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(base) + at);
    w[0] = a.x;
    w[1] = a.y;
    w[2] = a.z;
    w[3] = a.w;
    *scale = 1.0f;
  }
}

// Store one new row into the cache's own window: grid (n * ATTN_KV_HEADS), one block per (position, KV head),
// 256 threads over the 256 dims. One warp per 32-value block for the block formats, one thread per dim for
// bf16, so the block scale and the values it covers never leave the warp. A band that does not hold the row
// has nothing to write, and the band that holds it stores it.
template <KvType TYPE>
__global__ __launch_bounds__(kDim) void batch_kv_store_kernel(const BatchAttnArgs args, int side) {
  const int t = blockIdx.x / ATTN_KV_HEADS;
  const int h = blockIdx.x % ATTN_KV_HEADS;
  const float* in = (side == 0 ? args.k_new : args.v_new) + std::size_t(t) * kNewRow + h * kDim;
  void* dst = side == 0 ? args.k_cache : args.v_cache;
  const int pos = args.pos0 + t;
  if (pos < args.pos_begin || pos >= args.pos_begin + args.cache_rows) return;
  const int row = pos - args.pos_begin;
  const std::uint64_t values = std::uint64_t(args.cache_rows) * ATTN_KV_HEADS * kDim;

  if constexpr (TYPE == KvType::Bf16) {
    const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + threadIdx.x;
    static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(in[threadIdx.x]);
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int base_d = warp * KV_BLOCK;
  if constexpr (TYPE == KvType::Q8_0) {
    const float v = in[base_d + lane];
    float vmax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
    vmax = __shfl_sync(0xffffffffu, vmax, 0);
    const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
    const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + base_d;
    if (lane == 0) const_cast<__half*>(kv_q8_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
    kv_q8_values(dst)[at + lane] = std::int8_t(__float2int_rn(v / d));
  } else {
    // q4_0: the low half-warp packs one byte per lane (values 2j and 2j + 1), a coalesced sixteen-byte store.
    const bool packs = lane < KV_BLOCK / 2;
    float v0 = 0.0f, v1 = 0.0f;
    if (packs) {
      v0 = in[base_d + 2 * lane];
      v1 = in[base_d + 2 * lane + 1];
    }
    float vmax = fmaxf(fabsf(v0), fabsf(v1));
    for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
    vmax = __shfl_sync(0xffffffffu, vmax, 0);
    const float d = vmax > 0.0f ? vmax / 8.0f : 1.0f;
    const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + base_d;
    if (lane == 0) const_cast<__half*>(kv_q4_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
    if (packs) {
      kv_q4_values(dst)[at / 2 + lane] =
          std::uint8_t(kv_q4_nibble(v0, d) | (kv_q4_nibble(v1, d) << 4));
    }
  }
}

// One block per (head, split), eight warps, n accumulator sets: the cache row is read once and feeds every
// query of the block, which is the whole point of the batched path.
template <int N, KvType KK, KvType KV>
__global__ __launch_bounds__(kDim) void batch_decode_split_kernel(const BatchAttnArgs args) {
  const int h_local = blockIdx.x;
  const int h = args.head_begin + h_local;
  const int kh = h / kGroup;
  const int split = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int sub = lane / kPosLanes;            // which of the warp's position streams (one, at this width)
  const int d0 = (lane % kPosLanes) * kLaneDims;  // this lane's dims; the whole warp covers the head
  const unsigned hmask = 0xffffffffu >> (32 - kPosLanes * kPosPerWarp) << (sub * kPosLanes);
  const std::uint64_t cache_values = std::uint64_t(args.cache_rows) * ATTN_KV_HEADS * kDim;

  // The column grid is cut from pos and splits alone, as the single-token kernel cuts it, so the splits a
  // caller asks for mean the same thing here as there. A band runs only the part of its column it holds, and
  // its states merge with the other band's through the combine kernel's log-sum-exp, exactly as two splits do.
  const int count = args.pos0 + N;  // cache positions 0..pos0+N-1 are in scope
  const int per_split = (count + args.splits - 1) / args.splits;
  const int begin = split * per_split;
  const int end = min(count, begin + per_split);
  const int lo = max(begin, args.pos_begin);
  const int hi = min(end, args.pos_begin + args.cache_rows);
  const int col_end = args.col_end < 0 ? args.splits : args.col_end;

  BatchSplitState& st = args.partial[std::size_t(h_local) * args.splits + split];
  const int d = threadIdx.x;
  // A block outside this launch's column window, or outside the cache this launch reads, writes the empty
  // state the combine skips: the other band's launch owns that column.
  if (split < args.col_begin || split >= col_end || begin >= end || lo >= hi) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int t = 0; t < N; ++t) {
        st.m[t] = -INFINITY;
        st.l[t] = 0.0f;
      }
    }
#pragma unroll
    for (int t = 0; t < N; ++t) st.acc[t * kDim + d] = 0.0f;
    return;
  }

  float q_reg[N][kLaneDims];
#pragma unroll
  for (int t = 0; t < N; ++t) {
    const float* qp = args.q + (std::size_t(t) * ATTN_HEADS + h) * kDim + d0;
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) q_reg[t][j] = qp[j];
  }

  const float scale = rsqrtf(float(kDim));
  float m[N], l[N], acc[N][kLaneDims];
#pragma unroll
  for (int t = 0; t < N; ++t) {
    m[t] = -INFINITY;
    l[t] = 0.0f;
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) acc[t][j] = 0.0f;
  }

  // Two rows per iteration: the loads of the next row issue while the dots of the current one run, which is
  // what a warp needs when it carries only one position stream. The kernel is latency-bound at verify width,
  // not instruction-bound, so the duplicated staging is worth its eight registers.
#pragma unroll 2
  for (int i0 = lo + warp * kPosPerWarp; i0 < hi; i0 += kStepPos) {
    const int p = i0 + sub;
    if (p >= hi) continue;  // tail step: the second stream runs out first
    const std::size_t row = (std::size_t(p - args.pos_begin) * ATTN_KV_HEADS + kh) * kDim + d0;
    // Both loads are independent of the score, so they are issued before the reduction.
    std::uint32_t kw[8], vw[8];
    float ks = 1.0f, vs = 1.0f;
    load_row<KK>(args.k_cache, row, cache_values, kw, &ks);
    load_row<KV>(args.v_cache, row, cache_values, vw, &vs);

#pragma unroll
    for (int t = 0; t < N; ++t) {
      if (p > args.pos0 + t) continue;  // causal: a query does not see the later rows of its own block
      float dot = 0.0f;
      if constexpr (KK == KvType::Q8_0) dot = dot_q8<kQ8Words>(kw, q_reg[t]) * ks;
      else if constexpr (KK == KvType::Q4_0) dot = dot_q4<kQ4Words>(kw, q_reg[t]) * ks;
      else dot = dot_bf16<kBfWords>(kw, q_reg[t]);
      for (int off = kPosLanes / 2; off > 0; off >>= 1) dot += __shfl_xor_sync(hmask, dot, off);
      const float score = dot * scale;

      const float m_new = fmaxf(m[t], score);
      const float corr = expf(m[t] - m_new);
      const float p_ex = expf(score - m_new);
      l[t] = l[t] * corr + p_ex;
      m[t] = m_new;
      // The block scale of the V side folds into p, which keeps the value update at one multiply per dim.
      const float pv = p_ex * vs;
      if constexpr (KV == KvType::Q8_0) accum_q8(acc[t], vw, pv, corr);
      else if constexpr (KV == KvType::Q4_0) accum_q4(acc[t], vw, pv, corr);
      else accum_bf16(acc[t], vw, pv, corr);
    }
  }

  __shared__ float s_m[kWarps];
  __shared__ float s_l[kWarps];
  __shared__ float s_acc[kWarps][kDim];
  __shared__ float s_max, s_total;

  // Merge the warp's two streams through a lane-width shuffle, then the block's eight warps through shared
  // memory, one query at a time so the stage stays 8 KB. Order and weights are the single-token kernel's.
#pragma unroll
  for (int t = 0; t < N; ++t) {
    // With one position stream per warp there is nothing to merge across the halves.
    float m_warp = m[t];
    float l_warp = l[t];
    if constexpr (kPosPerWarp > 1) {
      const float m_p = __shfl_xor_sync(0xffffffffu, m[t], kPosLanes);
      const float l_p = __shfl_xor_sync(0xffffffffu, l[t], kPosLanes);
      m_warp = fmaxf(m[t], m_p);
      const float w_own = m_warp == -INFINITY ? 0.0f : expf(m[t] - m_warp);
      const float w_partner = m_warp == -INFINITY ? 0.0f : expf(m_p - m_warp);
      l_warp = l[t] * w_own + l_p * w_partner;
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) {
        acc[t][j] = acc[t][j] * w_own + __shfl_xor_sync(0xffffffffu, acc[t][j], kPosLanes) * w_partner;
      }
    }
    if (lane == 0) {
      s_m[warp] = m_warp;
      s_l[warp] = l_warp;
    }
    if (sub == 0) {  // one half-warp writes the merged head: 16 lanes x 16 dims = 256
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) s_acc[warp][d0 + j] = acc[t][j];
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
    for (int w = 0; w < kWarps; ++w) total += s_acc[w][d] * expf(s_m[w] - s_max);
    st.acc[t * kDim + d] = total;
    if (d == 0) {
      st.m[t] = s_max;
      st.l[t] = s_total;
    }
    __syncthreads();  // the stage is reused by the next query
  }
}

// Merge the splits per (head, query): log-sum-exp over the split maxima, then normalise.
template <int N>
__global__ __launch_bounds__(kDim) void batch_combine_kernel(const BatchAttnArgs args) {
  const int h_local = blockIdx.x;
  const int h = args.head_begin + h_local;
  const int t = blockIdx.y;
  const int d = threadIdx.x;
  const BatchSplitState* base = args.partial + std::size_t(h_local) * args.splits;

  const int col_end = args.col_end < 0 ? args.splits : args.col_end;
  __shared__ float s_max, s_total;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = args.col_begin; s < col_end; ++s) m = fmaxf(m, base[s].m[t]);
    float l = 0.0f;
    if (m != -INFINITY) {
      for (int s = args.col_begin; s < col_end; ++s) {
        if (base[s].m[t] == -INFINITY) continue;
        l += base[s].l[t] * expf(base[s].m[t] - m);
      }
    }
    s_max = m;
    s_total = l;
  }
  __syncthreads();

  float acc = 0.0f;
  if (s_max != -INFINITY) {
    for (int s = args.col_begin; s < col_end; ++s) {
      if (base[s].m[t] == -INFINITY) continue;
      acc += base[s].acc[t * kDim + d] * expf(base[s].m[t] - s_max);
    }
  }
  args.out[(std::size_t(t) * ATTN_HEADS + h) * kDim + d] = acc / s_total;
}

// One band's merge of its own columns, written as a state instead of a normalised output, so the host can join
// two bands with one more combine over the two states they left per head.
template <int N>
__global__ __launch_bounds__(kDim) void batch_band_states_kernel(const BatchAttnArgs args) {
  const int h_local = blockIdx.x;
  const int t = blockIdx.y;
  const int d = threadIdx.x;
  const BatchSplitState* base = args.partial + std::size_t(h_local) * args.splits;
  const int col_end = args.col_end < 0 ? args.splits : args.col_end;
  BatchSplitState& dst = args.band_states[std::size_t(h_local) * args.band_stride + args.band_slot];

  __shared__ float s_max, s_total;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = args.col_begin; s < col_end; ++s) m = fmaxf(m, base[s].m[t]);
    float l = 0.0f;
    if (m != -INFINITY) {
      for (int s = args.col_begin; s < col_end; ++s) {
        if (base[s].m[t] == -INFINITY) continue;
        l += base[s].l[t] * expf(base[s].m[t] - m);
      }
    }
    s_max = m;
    s_total = l;
  }
  __syncthreads();

  float acc = 0.0f;
  if (s_max != -INFINITY) {
    for (int s = args.col_begin; s < col_end; ++s) {
      if (base[s].m[t] == -INFINITY) continue;
      acc += base[s].acc[t * kDim + d] * expf(base[s].m[t] - s_max);
    }
  }
  dst.acc[t * kDim + d] = acc;
  if (d == 0) {
    dst.m[t] = s_max;
    dst.l[t] = s_total;
  }
}

// One block per (head, position): the single-token norm_rope kernel, with the position of the row it handles.
__global__ __launch_bounds__(ATTN_HEAD_DIM) void qk_norm_rope_rows_kernel(const QkNormRopeRowsArgs args) {
  const int t = blockIdx.y;
  const int h = blockIdx.x;
  const bool is_q = h < ATTN_HEADS;
  float* x = is_q ? args.q + (std::size_t(t) * ATTN_HEADS + h) * ATTN_HEAD_DIM
                  : args.k + (std::size_t(t) * ATTN_KV_HEADS + (h - ATTN_HEADS)) * ATTN_HEAD_DIM;
  const float* weight = is_q ? args.q_weight : args.k_weight;
  const int d = threadIdx.x;

  __shared__ float warp_sums[ATTN_HEAD_DIM / 32];
  __shared__ float inv;

  const float v = x[d];
  float sum = v * v;
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int w = 0; w < ATTN_HEAD_DIM / 32; ++w) total += warp_sums[w];
    inv = rsqrtf(total / float(ATTN_HEAD_DIM) + args.eps);
  }
  __syncthreads();
  x[d] = v * inv * weight[d];
  __syncthreads();

  constexpr int kPairs = ROTARY_DIM / 2;
  if (d < kPairs) {
    const float angle = float(args.pos0 + t) * powf(ROTARY_THETA, -2.0f * float(d) / float(ROTARY_DIM));
    const float c = cosf(angle), s = sinf(angle);
    const float x0 = x[d];
    const float x1 = x[d + kPairs];
    x[d] = x0 * c - x1 * s;
    x[d + kPairs] = x0 * s + x1 * c;
  }
}

// cache_rows < 0 means "the cache covers the whole capacity", the plain single-cache case.
BatchAttnArgs normalised(const BatchAttnArgs& args) {
  if (args.cache_rows >= 0) return args;
  BatchAttnArgs a = args;
  a.cache_rows = args.max_pos;
  return a;
}

bool valid_core(const BatchAttnArgs& args) {
  if (!args.q || !args.k_new || !args.v_new || !args.k_cache || !args.v_cache || !args.out) return false;
  if (args.n < 1 || args.n > BATCH_MAX) return false;
  if (args.pos0 < 0 || args.pos0 + args.n > args.max_pos) return false;
  if (args.cache_rows < 0) return false;
  if (args.pos_begin < 0 || args.pos_begin + args.cache_rows > args.max_pos) return false;
  // A query block may sit outside this cache entirely — that is a band whose positions are all above the query,
  // and it contributes the empty state. What it may not do is read or write outside the cache, which the row
  // clip in the store and split kernels enforces.
  return args.head_begin >= 0 && args.head_count > 0 && args.head_count <= ATTN_HEADS - args.head_begin;
}

// The split path additionally needs somewhere to put one state per (head, split).
bool valid(const BatchAttnArgs& args) {
  return valid_core(args) && args.partial != nullptr && args.splits >= 1;
}

#define INGINE_BATCH_ATT_KV(N, KK)                                                                       \
  do {                                                                                                   \
    switch (args.kv_type_v) {                                                                            \
      case KvType::Bf16: batch_decode_split_kernel<N, KK, KvType::Bf16><<<grid, kDim, 0, stream>>>(args); break; \
      case KvType::Q8_0: batch_decode_split_kernel<N, KK, KvType::Q8_0><<<grid, kDim, 0, stream>>>(args); break; \
      default: batch_decode_split_kernel<N, KK, KvType::Q4_0><<<grid, kDim, 0, stream>>>(args); break;      \
    }                                                                                                    \
  } while (0)

#define INGINE_BATCH_ATT_FIRE(N)                                                                         \
  do {                                                                                                   \
    switch (args.kv_type_k) {                                                                            \
      case KvType::Bf16: INGINE_BATCH_ATT_KV(N, KvType::Bf16); break;                                     \
      case KvType::Q8_0: INGINE_BATCH_ATT_KV(N, KvType::Q8_0); break;                                     \
      default: INGINE_BATCH_ATT_KV(N, KvType::Q4_0); break;                                               \
    }                                                                                                    \
  } while (0)

#define INGINE_BATCH_STORE(TYPE, SIDE)                                                                   \
  do {                                                                                                   \
    batch_kv_store_kernel<TYPE><<<args.n * ATTN_KV_HEADS, kDim, 0, stream>>>(args, SIDE);                  \
  } while (0)

}  // namespace

namespace {

// The store plus the split kernel: what one card does for the columns it owns.
cudaError_t split_and_store(const BatchAttnArgs& args, cudaStream_t stream) {
  switch (args.kv_type_k) {
    case KvType::Bf16: INGINE_BATCH_STORE(KvType::Bf16, 0); break;
    case KvType::Q8_0: INGINE_BATCH_STORE(KvType::Q8_0, 0); break;
    default: INGINE_BATCH_STORE(KvType::Q4_0, 0); break;
  }
  switch (args.kv_type_v) {
    case KvType::Bf16: INGINE_BATCH_STORE(KvType::Bf16, 1); break;
    case KvType::Q8_0: INGINE_BATCH_STORE(KvType::Q8_0, 1); break;
    default: INGINE_BATCH_STORE(KvType::Q4_0, 1); break;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  const dim3 grid(args.head_count, args.splits);
  switch (args.n) {
    case 1: INGINE_BATCH_ATT_FIRE(1); break;
    case 2: INGINE_BATCH_ATT_FIRE(2); break;
    case 3: INGINE_BATCH_ATT_FIRE(3); break;
    default: INGINE_BATCH_ATT_FIRE(4); break;
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t batch_attention_split(const BatchAttnArgs& in, cudaStream_t stream) {
  const BatchAttnArgs args = normalised(in);
  if (!valid(args)) return cudaErrorInvalidValue;
  return split_and_store(args, stream);
}

cudaError_t batch_attention_combine(const BatchAttnArgs& in, cudaStream_t stream) {
  const BatchAttnArgs args = normalised(in);
  if (!valid(args)) return cudaErrorInvalidValue;
  const dim3 cgrid(args.head_count, args.n);
  switch (args.n) {
    case 1: batch_combine_kernel<1><<<cgrid, kDim, 0, stream>>>(args); break;
    case 2: batch_combine_kernel<2><<<cgrid, kDim, 0, stream>>>(args); break;
    case 3: batch_combine_kernel<3><<<cgrid, kDim, 0, stream>>>(args); break;
    default: batch_combine_kernel<4><<<cgrid, kDim, 0, stream>>>(args); break;
  }
  return cudaGetLastError();
}

cudaError_t batch_attention_band_states(const BatchAttnArgs& in, cudaStream_t stream) {
  const BatchAttnArgs args = normalised(in);
  if (!valid(args) || !args.band_states) return cudaErrorInvalidValue;
  const dim3 cgrid(args.head_count, args.n);
  switch (args.n) {
    case 1: batch_band_states_kernel<1><<<cgrid, kDim, 0, stream>>>(args); break;
    case 2: batch_band_states_kernel<2><<<cgrid, kDim, 0, stream>>>(args); break;
    case 3: batch_band_states_kernel<3><<<cgrid, kDim, 0, stream>>>(args); break;
    default: batch_band_states_kernel<4><<<cgrid, kDim, 0, stream>>>(args); break;
  }
  return cudaGetLastError();
}

cudaError_t batch_attention(const BatchAttnArgs& in, cudaStream_t stream) {
  const BatchAttnArgs args = normalised(in);
  if (!valid(args)) return cudaErrorInvalidValue;
  cudaError_t err = split_and_store(args, stream);
  if (err != cudaSuccess) return err;
  return batch_attention_combine(args, stream);
}

cudaError_t qk_norm_rope_rows(const QkNormRopeRowsArgs& args, cudaStream_t stream) {
  if (!args.q || !args.k || !args.q_weight || !args.k_weight) return cudaErrorInvalidValue;
  if (args.n < 1 || args.n > BATCH_MAX) return cudaErrorInvalidValue;
  const dim3 grid(ATTN_HEADS + ATTN_KV_HEADS, args.n);
  qk_norm_rope_rows_kernel<<<grid, ATTN_HEAD_DIM, 0, stream>>>(args);
  return cudaGetLastError();
}

namespace {

// Host mirror of the store kernel, per side, so the reference reads exactly the bytes the device wrote.
void store_side_cpu(const float* src, void* dst, KvType type, int row, int h, int cache_rows) {
  if (type == KvType::Bf16) {
    for (int d = 0; d < kDim; ++d) {
      const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + d;
      static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(src[h * kDim + d]);
    }
    return;
  }
  const std::uint64_t values = std::uint64_t(cache_rows) * ATTN_KV_HEADS * kDim;
  for (int block = 0; block < kDim / KV_BLOCK; ++block) {
    float vmax = 0.0f;
    for (int i = 0; i < KV_BLOCK; ++i) vmax = std::max(vmax, std::fabs(src[h * kDim + block * KV_BLOCK + i]));
    const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
    const std::size_t at = (std::size_t(row) * ATTN_KV_HEADS + h) * kDim + block * KV_BLOCK;
    if (type == KvType::Q8_0) {
      const_cast<__half*>(kv_q8_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
      for (int i = 0; i < KV_BLOCK; ++i) {
        kv_q8_values(dst)[at + i] = std::int8_t(std::lround(src[h * kDim + block * KV_BLOCK + i] / d));
      }
    } else {
      const_cast<__half*>(kv_q4_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
      for (int i = 0; i < KV_BLOCK; i += 2) {
        const int lo = int(std::lround(src[h * kDim + block * KV_BLOCK + i] / d)) + 8;
        const int hi = int(std::lround(src[h * kDim + block * KV_BLOCK + i + 1] / d)) + 8;
        kv_q4_values(dst)[(at + i) / 2] = std::uint8_t((lo & 0xF) | ((hi & 0xF) << 4));
      }
    }
  }
}

double read_cpu(const void* base, KvType type, std::size_t index, std::uint64_t values) {
  if (type == KvType::Bf16) return double(__bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]));
  if (type == KvType::Q4_0) {
    const std::uint8_t packed = kv_q4_values(base)[index / 2];
    const double q = double(int((packed >> (4 * (index & 1))) & 0xFu)) - 8.0;
    return q * double(__half2float(kv_q4_scales(base, values)[index / KV_BLOCK]));
  }
  return double(kv_q8_values(const_cast<void*>(base))[index]) *
         double(__half2float(kv_q8_scales(base, values)[index / KV_BLOCK]));
}

}  // namespace

void batch_attention_cpu(const BatchAttnArgs& in) {
  const BatchAttnArgs args = normalised(in);
  if (!valid_core(args)) return;
  const std::uint64_t values = std::uint64_t(args.cache_rows) * ATTN_KV_HEADS * kDim;
  for (int t = 0; t < args.n; ++t) {
    const int pos = args.pos0 + t;
    if (pos < args.pos_begin || pos >= args.pos_begin + args.cache_rows) continue;
    const float* kn = args.k_new + std::size_t(t) * kNewRow;
    const float* vn = args.v_new + std::size_t(t) * kNewRow;
    for (int h = 0; h < ATTN_KV_HEADS; ++h) {
      store_side_cpu(kn, args.k_cache, args.kv_type_k, pos - args.pos_begin, h, args.cache_rows);
      store_side_cpu(vn, args.v_cache, args.kv_type_v, pos - args.pos_begin, h, args.cache_rows);
    }
  }

  const double scale = 1.0 / std::sqrt(double(kDim));
  std::vector<double> score;
  for (int t = 0; t < args.n; ++t) {
    const int limit = args.pos0 + t;
    score.assign(std::size_t(limit) + 1, 0.0);
    for (int hh = 0; hh < args.head_count; ++hh) {
      const int h = args.head_begin + hh;
      const int kh = h / kGroup;
      const float* q = args.q + (std::size_t(t) * ATTN_HEADS + h) * kDim;
      double m = -INFINITY;
      for (int p = args.pos_begin; p <= limit; ++p) {
        double dot = 0.0;
        const std::size_t row = (std::size_t(p - args.pos_begin) * ATTN_KV_HEADS + kh) * kDim;
        for (int d = 0; d < kDim; ++d) {
          dot += double(q[d]) * read_cpu(args.k_cache, args.kv_type_k, row + d, values);
        }
        score[std::size_t(p)] = dot * scale;
        m = std::max(m, score[std::size_t(p)]);
      }
      double sum = 0.0;
      for (int p = args.pos_begin; p <= limit; ++p) {
        score[std::size_t(p)] = std::exp(score[std::size_t(p)] - m);
        sum += score[std::size_t(p)];
      }
      for (int d = 0; d < kDim; ++d) {
        double acc = 0.0;
        for (int p = args.pos_begin; p <= limit; ++p) {
          const std::size_t row = (std::size_t(p - args.pos_begin) * ATTN_KV_HEADS + kh) * kDim;
          acc += score[std::size_t(p)] * read_cpu(args.v_cache, args.kv_type_v, row + d, values);
        }
        args.out[(std::size_t(t) * ATTN_HEADS + h) * kDim + d] = float(acc / sum);
      }
    }
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
