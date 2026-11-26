#include "arch/qwen35moe/kernels/batch/batch_attn.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kDim = ATTN_HEAD_DIM;
constexpr int kGroup = ATTN_HEADS / ATTN_KV_HEADS;
constexpr int kNewRow = ATTN_KV_HEADS * ATTN_HEAD_DIM;  // one position of k_new / v_new
constexpr int kLaneDims = 8;                 // one 8-byte q8_0 read, or one 16-byte bf16 read
constexpr int kLanes = kDim / kLaneDims;     // 32: a whole warp covers one position
constexpr int kQ8Words = kLaneDims / 4;      // uint32 words of int8 per lane (2)
constexpr int kBfWords = kLaneDims / 2;      // uint32 words of bf16 per lane (4)
static_assert(kLanes == 32, "one warp must cover exactly one 256-dim cache row");

// Every lane needs the finished score, because each lane accumulates a different slice of the value vector.
__device__ __forceinline__ float warp_sum32(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float q8_byte(std::uint32_t w, int shift) { return float(std::int8_t(w >> shift)); }
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
__device__ __forceinline__ float dot_bf16(const std::uint32_t (&w)[8], const float (&q)[kLaneDims]) {
  float dot = 0.0f;
#pragma unroll
  for (int u = 0; u < USED; ++u) {
    dot = fmaf(q[u * 2], bf16_lo(w[u]), dot);
    dot = fmaf(q[u * 2 + 1], bf16_hi(w[u]), dot);
  }
  return dot;
}
// acc = acc * corr + p * v
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
__device__ __forceinline__ void accum_bf16(float (&acc)[kLaneDims], const std::uint32_t (&w)[8], float p,
                                           float corr) {
#pragma unroll
  for (int u = 0; u < kBfWords; ++u) {
    acc[u * 2] = fmaf(p, bf16_lo(w[u]), acc[u * 2] * corr);
    acc[u * 2 + 1] = fmaf(p, bf16_hi(w[u]), acc[u * 2 + 1] * corr);
  }
}

// This lane's eight dims of cache row `row`: one 8-byte read for q8_0, one 16-byte read for bf16. The other
// words of the staging array are untouched in the q8_0 instantiation, which is why it is passed by size.
template <KvType TYPE>
__device__ __forceinline__ void load_row(const void* base, std::size_t row, std::uint32_t (&w)[8]) {
  if constexpr (TYPE == KvType::Q8_0) {
    const uint2 r = *reinterpret_cast<const uint2*>(kv_q8_values(const_cast<void*>(base)) + row);
    w[0] = unsigned(r.x);
    w[1] = unsigned(r.y);
  } else {
    const uint4* p4 = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(base) + row);
    const uint4 a = p4[0];
    w[0] = a.x;
    w[1] = a.y;
    w[2] = a.z;
    w[3] = a.w;
  }
}

__device__ __forceinline__ float row_scale(const void* base, KvType type, std::uint64_t cache_values,
                                           std::size_t row) {
  if (type != KvType::Q8_0) return 1.0f;
  return __half2float(kv_q8_scales(base, cache_values)[row / KV_BLOCK]);
}

// Store the n new rows. grid = (n * ATTN_KV_HEADS, 2): the y index picks the cache side, so a block touches
// one type. One warp per 32-value q8_0 block (eight warps cover 256 dims), one thread per dim for bf16.
__global__ __launch_bounds__(kDim) void batch_kv_store_kernel(const BatchAttnArgs args) {
  const int t = blockIdx.x / ATTN_KV_HEADS;
  const int h = blockIdx.x % ATTN_KV_HEADS;
  const int side = blockIdx.y;
  const float* in = (side == 0 ? args.k_new : args.v_new) + std::size_t(t) * kNewRow + h * kDim;
  void* dst = side == 0 ? args.k_cache : args.v_cache;
  const KvType type = side == 0 ? args.kv_type_k : args.kv_type_v;
  const int pos = args.pos0 + t;

  if (type == KvType::Bf16) {
    const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + threadIdx.x;
    static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(in[threadIdx.x]);
    return;
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int base_d = warp * KV_BLOCK;
  const float v = in[base_d + lane];
  float vmax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
  vmax = __shfl_sync(0xffffffffu, vmax, 0);
  const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
  const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + base_d;
  const std::uint64_t values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;
  if (lane == 0) const_cast<__half*>(kv_q8_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
  kv_q8_values(dst)[at + lane] = std::int8_t(__float2int_rn(v / d));
}

// One warp per (head, split): a cache row is read once and feeds all n query states, which is the whole
// point of the batched path. The block is a single warp, so there is no barrier and no shared merge in the
// streaming loop — the warp is the reduction unit, and the splits are merged by the combine kernel.
template <int N, KvType KK, KvType KV>
__global__ __launch_bounds__(kLanes) void batch_decode_split_kernel(const BatchAttnArgs args) {
  const int split = blockIdx.y;
  const int h_local = blockIdx.x;
  const int h = args.head_begin + h_local;
  const int kh = h / kGroup;
  const int lane = threadIdx.x;
  const int d0 = lane * kLaneDims;
  const std::uint64_t cache_values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;

  const int count = args.pos0 + N;  // cache rows 0..pos0+N-1 are in scope
  const int per_split = (count + args.splits - 1) / args.splits;
  const int begin = split * per_split;
  const int end = min(count, begin + per_split);

  BatchSplitState& st = args.partial[std::size_t(h_local) * args.splits + split];
  if (begin >= end) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
      if (lane == 0) {
        st.m[t] = -INFINITY;
        st.l[t] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) st.acc[t * kDim + d0 + j] = 0.0f;
    }
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

  for (int p = begin; p < end; ++p) {
    const std::size_t row = (std::size_t(p) * ATTN_KV_HEADS + kh) * kDim + d0;
    std::uint32_t kw[8], vw[8];
    load_row<KK>(args.k_cache, row, kw);
    load_row<KV>(args.v_cache, row, vw);
    const float ks = row_scale(args.k_cache, KK, cache_values, row);
    const float vs = row_scale(args.v_cache, KV, cache_values, row);

#pragma unroll
    for (int t = 0; t < N; ++t) {
      if (p > args.pos0 + t) continue;  // causal: a query does not see the later rows of its own block
      float dot = 0.0f;
      if constexpr (KK == KvType::Q8_0) dot = dot_q8<kQ8Words>(kw, q_reg[t]) * ks;
      else dot = dot_bf16<kBfWords>(kw, q_reg[t]);
      dot = warp_sum32(dot);
      const float score = dot * scale;
      const float m_new = fmaxf(m[t], score);
      const float corr = expf(m[t] - m_new);
      const float p_ex = expf(score - m_new);
      // the q8_0 row scale folds into p, which keeps the value update at one multiply per dim
      const float pv = p_ex * vs;
      if constexpr (KV == KvType::Q8_0) accum_q8(acc[t], vw, pv, corr);
      else accum_bf16(acc[t], vw, pv, corr);
      l[t] = l[t] * corr + p_ex;
      m[t] = m_new;
    }
  }

#pragma unroll
  for (int t = 0; t < N; ++t) {
    if (lane == 0) {
      st.m[t] = m[t];
      st.l[t] = l[t];
    }
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) st.acc[t * kDim + d0 + j] = acc[t][j];
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

  __shared__ float s_max, s_total;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s < args.splits; ++s) m = fmaxf(m, base[s].m[t]);
    float l = 0.0f;
    if (m != -INFINITY) {
      for (int s = 0; s < args.splits; ++s) {
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
    for (int s = 0; s < args.splits; ++s) {
      if (base[s].m[t] == -INFINITY) continue;
      acc += base[s].acc[t * kDim + d] * expf(base[s].m[t] - s_max);
    }
  }
  args.out[(std::size_t(t) * ATTN_HEADS + h) * kDim + d] = acc / s_total;
}

bool valid_core(const BatchAttnArgs& args) {
  if (!args.q || !args.k_new || !args.v_new || !args.k_cache || !args.v_cache || !args.out) return false;
  if (args.n < 1 || args.n > BATCH_MAX) return false;
  if (args.pos0 < 0 || args.pos0 + args.n > args.max_pos) return false;
  return args.head_begin >= 0 && args.head_count > 0 && args.head_count <= ATTN_HEADS - args.head_begin;
}

// The split path additionally needs somewhere to put one state per (head, split).
bool valid(const BatchAttnArgs& args) {
  return valid_core(args) && args.partial != nullptr && args.splits >= 1;
}

#define INGINE_BATCH_ATT_FIRE(N)                                                                        \
  do {                                                                                                  \
    if (q8_k && q8_v) batch_decode_split_kernel<N, KvType::Q8_0, KvType::Q8_0><<<grid, kLanes, 0, stream>>>(args); \
    else if (q8_k) batch_decode_split_kernel<N, KvType::Q8_0, KvType::Bf16><<<grid, kLanes, 0, stream>>>(args);    \
    else if (q8_v) batch_decode_split_kernel<N, KvType::Bf16, KvType::Q8_0><<<grid, kLanes, 0, stream>>>(args);    \
    else batch_decode_split_kernel<N, KvType::Bf16, KvType::Bf16><<<grid, kLanes, 0, stream>>>(args);              \
  } while (0)

}  // namespace

cudaError_t batch_attention(const BatchAttnArgs& args, cudaStream_t stream) {
  if (!valid(args)) return cudaErrorInvalidValue;
  const bool q8_k = args.kv_type_k == KvType::Q8_0, q8_v = args.kv_type_v == KvType::Q8_0;
  const dim3 sgrid(args.n * ATTN_KV_HEADS, 2);
  batch_kv_store_kernel<<<sgrid, kDim, 0, stream>>>(args);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  const dim3 grid(args.head_count, args.splits);
  switch (args.n) {
    case 1: INGINE_BATCH_ATT_FIRE(1); break;
    case 2: INGINE_BATCH_ATT_FIRE(2); break;
    case 3: INGINE_BATCH_ATT_FIRE(3); break;
    default: INGINE_BATCH_ATT_FIRE(4); break;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  const dim3 cgrid(args.head_count, args.n);
  switch (args.n) {
    case 1: batch_combine_kernel<1><<<cgrid, kDim, 0, stream>>>(args); break;
    case 2: batch_combine_kernel<2><<<cgrid, kDim, 0, stream>>>(args); break;
    case 3: batch_combine_kernel<3><<<cgrid, kDim, 0, stream>>>(args); break;
    default: batch_combine_kernel<4><<<cgrid, kDim, 0, stream>>>(args); break;
  }
  return cudaGetLastError();
}

namespace {

// Host mirror of the store kernel, per side, so the reference reads exactly the bytes the device wrote.
// src is the whole row [ATTN_KV_HEADS, kDim] for this position; head h is at src + h * kDim, matching the
// device store kernel.
void store_side_cpu(const float* src, void* dst, KvType type, int pos, int h, int max_pos) {
  if (type == KvType::Bf16) {
    for (int d = 0; d < kDim; ++d) {
      const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + d;
      static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(src[h * kDim + d]);
    }
    return;
  }
  const std::uint64_t values = std::uint64_t(max_pos) * ATTN_KV_HEADS * kDim;
  for (int block = 0; block < kDim / KV_BLOCK; ++block) {
    float vmax = 0.0f;
    for (int i = 0; i < KV_BLOCK; ++i) {
      vmax = std::max(vmax, std::fabs(src[h * kDim + block * KV_BLOCK + i]));
    }
    const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
    const std::size_t at = (std::size_t(pos) * ATTN_KV_HEADS + h) * kDim + block * KV_BLOCK;
    const_cast<__half*>(kv_q8_scales(dst, values))[at / KV_BLOCK] = __float2half(d);
    for (int i = 0; i < KV_BLOCK; ++i) {
      kv_q8_values(dst)[at + i] = std::int8_t(std::lround(src[h * kDim + block * KV_BLOCK + i] / d));
    }
  }
}

double read_cpu(const void* base, KvType type, std::size_t index, std::uint64_t values) {
  if (type == KvType::Bf16) return double(__bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]));
  return double(kv_q8_values(const_cast<void*>(base))[index]) *
         double(__half2float(kv_q8_scales(base, values)[index / KV_BLOCK]));
}

}  // namespace

void batch_attention_cpu(const BatchAttnArgs& args) {
  if (!valid_core(args)) return;
  const std::uint64_t values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;
  for (int t = 0; t < args.n; ++t) {
    const int pos = args.pos0 + t;
    const float* kn = args.k_new + std::size_t(t) * kNewRow;
    const float* vn = args.v_new + std::size_t(t) * kNewRow;
    for (int h = 0; h < ATTN_KV_HEADS; ++h) {
      store_side_cpu(kn, args.k_cache, args.kv_type_k, pos, h, args.max_pos);
      store_side_cpu(vn, args.v_cache, args.kv_type_v, pos, h, args.max_pos);
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
      for (int p = 0; p <= limit; ++p) {
        double dot = 0.0;
        const std::size_t row = (std::size_t(p) * ATTN_KV_HEADS + kh) * kDim;
        for (int d = 0; d < kDim; ++d) {
          dot += double(q[d]) * read_cpu(args.k_cache, args.kv_type_k, row + d, values);
        }
        score[std::size_t(p)] = dot * scale;
        m = std::max(m, score[std::size_t(p)]);
      }
      double sum = 0.0;
      for (int p = 0; p <= limit; ++p) {
        score[std::size_t(p)] = std::exp(score[std::size_t(p)] - m);
        sum += score[std::size_t(p)];
      }
      for (int d = 0; d < kDim; ++d) {
        double acc = 0.0;
        for (int p = 0; p <= limit; ++p) {
          const std::size_t row = (std::size_t(p) * ATTN_KV_HEADS + kh) * kDim;
          acc += score[std::size_t(p)] * read_cpu(args.v_cache, args.kv_type_v, row + d, values);
        }
        args.out[(std::size_t(t) * ATTN_HEADS + h) * kDim + d] = float(acc / sum);
      }
    }
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
