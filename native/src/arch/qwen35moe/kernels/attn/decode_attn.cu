#include "arch/qwen35moe/kernels/attn/decode_attn.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "arch/qwen35moe/model/model_data.h"

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kDim = ATTN_HEAD_DIM;  // 256 threads, one per output dim
constexpr int kWarps = kDim / 32;    // 8 warps: one position each per step
constexpr int kGroup = ATTN_HEADS / ATTN_KV_HEADS;
static_assert(ATTN_HEADS % ATTN_KV_HEADS == 0, "query heads must divide evenly over KV heads");

// head_count > 0, so ATTN_HEADS - head_begin bounds the largest legal range without overflowing.
bool head_range_valid(const DecodeAttnArgs& args) {
  return args.head_begin >= 0 && args.head_count > 0 && args.head_count <= ATTN_HEADS - args.head_begin;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float kv_at(const void* base, KvType type, std::size_t index, std::uint64_t values) {
  if (type == KvType::Bf16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]);
  }
  return float(kv_q8_values(const_cast<void*>(base))[index]) *
         __half2float(kv_q8_scales(base, values)[index / KV_BLOCK]);
}

__global__ void kv_store_bf16_kernel(DecodeAttnArgs args) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  if (h >= ATTN_KV_HEADS || d >= kDim) return;
  const std::size_t at = (std::size_t(args.pos) * ATTN_KV_HEADS + h) * kDim + d;
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
  const std::size_t at = (std::size_t(args.pos) * ATTN_KV_HEADS + h) * kDim + base_d;

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

  const std::uint64_t values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;
  const std::size_t block = at / KV_BLOCK;
  if (lane == 0) {
    const_cast<__half*>(kv_q8_scales(args.k_cache, values))[block] = __float2half(kd);
    const_cast<__half*>(kv_q8_scales(args.v_cache, values))[block] = __float2half(vd);
  }
  kv_q8_values(args.k_cache)[at + lane] = std::int8_t(__float2int_rn(k / kd));
  kv_q8_values(args.v_cache)[at + lane] = std::int8_t(__float2int_rn(v / vd));
}

__global__ __launch_bounds__(kDim) void decode_kernel(DecodeAttnArgs args) {
  const int h = args.head_begin + blockIdx.x;  // absolute query head; the grid is the head range
  const int d = threadIdx.x;
  const int kh = h / kGroup;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint64_t cache_values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;

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

// Mixed K/V types: the first half of the block handles K's q8_0 blocks, the second half V's. Only used when the
// two sides differ; the uniform cases take the simpler kernels above.
__global__ __launch_bounds__(512) void kv_store_mixed_kernel(DecodeAttnArgs args) {
  const int half = threadIdx.x / 256;
  const int t = threadIdx.x % 256;
  const int h = blockIdx.x;
  const int warp = t >> 5;
  const int lane = t & 31;
  if (h >= ATTN_KV_HEADS) return;
  const float* src = half == 0 ? args.k_new : args.v_new;
  void* dst = half == 0 ? args.k_cache : args.v_cache;
  const KvType type = half == 0 ? args.kv_type_k : args.kv_type_v;
  if (type == KvType::Bf16) {
    const int d = t;
    if (d >= kDim) return;
    const std::size_t at = (std::size_t(args.pos) * ATTN_KV_HEADS + h) * kDim + d;
    static_cast<__nv_bfloat16*>(dst)[at] = __float2bfloat16(src[h * kDim + d]);
    return;
  }
  if (warp >= kDim / KV_BLOCK) return;
  const int base_d = warp * KV_BLOCK;
  float v = src[h * kDim + base_d + lane];
  float vmax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
  vmax = __shfl_sync(0xffffffffu, vmax, 0);
  const float d = vmax > 0.0f ? vmax / 127.0f : 1.0f;
  const std::size_t at = (std::size_t(args.pos) * ATTN_KV_HEADS + h) * kDim + base_d;
  const std::size_t block = at / KV_BLOCK;
  const std::uint64_t values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;
  if (lane == 0) const_cast<__half*>(kv_q8_scales(dst, values))[block] = __float2half(d);
  kv_q8_values(dst)[at + lane] = std::int8_t(__float2int_rn(v / d));
}

// Flash decode: one warp walks a strided subset of the split's positions and keeps its own running softmax
// state in registers — eight output dims per lane — so the score and the value update happen in the same pass
// with no shared score array and no per-chunk barriers. The eight warps of the block merge at the end.
//
// Per position a lane does two aligned 4-byte reads of K, one scale read, eight K multiplies for the score, then
// the same two reads and eight fused updates for V. That is the widest access the q8_0 layout allows, and it is
// what the depth cost needs: 2.85 GB of cache per token at 262k.
__global__ __launch_bounds__(kDim) void decode_split_kernel(DecodeAttnArgs args) {
  const int split = blockIdx.y;
  const int h = args.head_begin + blockIdx.x;
  const int kh = h / kGroup;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint64_t cache_values = std::uint64_t(args.max_pos) * ATTN_KV_HEADS * kDim;

  const int per_split = (args.pos + 1 + args.splits - 1) / args.splits;
  const int begin = split * per_split;
  const int end = min(args.pos + 1, begin + per_split);
  const std::size_t slot = std::size_t(blockIdx.x) * std::size_t(args.splits) + std::size_t(split);
  if (begin >= end) {
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

  const int d0 = lane * 8;  // this lane carries eight dims of the head
  float q_reg[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) q_reg[i] = args.q[h * kDim + d0 + i];

  const float scale = rsqrtf(float(kDim));
  float m = -INFINITY;
  float l = 0.0f;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int i = begin + warp; i < end; i += kWarps) {
    const std::size_t kbase = (std::size_t(i) * ATTN_KV_HEADS + kh) * kDim + d0;
    float dot = 0.0f;
    float v[8];
    if (args.kv_type_k == KvType::Q8_0) {
      const float ks = __half2float(kv_q8_scales(args.k_cache, cache_values)[kbase / KV_BLOCK]);
      const std::int8_t* kp = kv_q8_values(args.k_cache) + kbase;
      const std::int32_t a = *reinterpret_cast<const std::int32_t*>(kp);
      const std::int32_t b = *reinterpret_cast<const std::int32_t*>(kp + 4);
      dot = q_reg[0] * float(std::int8_t(a & 0xFF)) + q_reg[1] * float(std::int8_t((a >> 8) & 0xFF)) +
            q_reg[2] * float(std::int8_t((a >> 16) & 0xFF)) + q_reg[3] * float(std::int8_t((a >> 24) & 0xFF)) +
            q_reg[4] * float(std::int8_t(b & 0xFF)) + q_reg[5] * float(std::int8_t((b >> 8) & 0xFF)) +
            q_reg[6] * float(std::int8_t((b >> 16) & 0xFF)) + q_reg[7] * float(std::int8_t((b >> 24) & 0xFF));
      dot *= ks;
    } else {
      const auto* kp = static_cast<const __nv_bfloat16*>(args.k_cache) + kbase;
      const uint2 a = *reinterpret_cast<const uint2*>(kp);
      const uint2 b = *reinterpret_cast<const uint2*>(kp + 4);
      const float kv[8] = {__bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.x & 0xFFFFu))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.x >> 16))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.y & 0xFFFFu))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.y >> 16))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.x & 0xFFFFu))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.x >> 16))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.y & 0xFFFFu))),
                           __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.y >> 16)))};
#pragma unroll
      for (int j = 0; j < 8; ++j) dot += q_reg[j] * kv[j];
    }
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    const float score = dot * scale;

    const std::size_t vbase = (std::size_t(i) * ATTN_KV_HEADS + kh) * kDim + d0;
    if (args.kv_type_v == KvType::Q8_0) {
      const float vs = __half2float(kv_q8_scales(args.v_cache, cache_values)[vbase / KV_BLOCK]);
      const std::int8_t* vp = kv_q8_values(args.v_cache) + vbase;
      const std::int32_t a = *reinterpret_cast<const std::int32_t*>(vp);
      const std::int32_t b = *reinterpret_cast<const std::int32_t*>(vp + 4);
      v[0] = vs * float(std::int8_t(a & 0xFF));
      v[1] = vs * float(std::int8_t((a >> 8) & 0xFF));
      v[2] = vs * float(std::int8_t((a >> 16) & 0xFF));
      v[3] = vs * float(std::int8_t((a >> 24) & 0xFF));
      v[4] = vs * float(std::int8_t(b & 0xFF));
      v[5] = vs * float(std::int8_t((b >> 8) & 0xFF));
      v[6] = vs * float(std::int8_t((b >> 16) & 0xFF));
      v[7] = vs * float(std::int8_t((b >> 24) & 0xFF));
    } else {
      const auto* vp = static_cast<const __nv_bfloat16*>(args.v_cache) + vbase;
      const uint2 a = *reinterpret_cast<const uint2*>(vp);
      const uint2 b = *reinterpret_cast<const uint2*>(vp + 4);
      v[0] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.x & 0xFFFFu)));
      v[1] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.x >> 16)));
      v[2] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.y & 0xFFFFu)));
      v[3] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(a.y >> 16)));
      v[4] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.x & 0xFFFFu)));
      v[5] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.x >> 16)));
      v[6] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.y & 0xFFFFu)));
      v[7] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(b.y >> 16)));
    }

    const float m_new = fmaxf(m, score);
    const float corr = expf(m - m_new);
    const float p = expf(score - m_new);
    l = l * corr + p;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = acc[j] * corr + p * v[j];
    m = m_new;
  }

  if (lane == 0) {
    s_m[warp] = m;
    s_l[warp] = l;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) s_acc[warp][d0 + j] = acc[j];
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

cudaError_t decode_attention(const DecodeAttnArgs& args, cudaStream_t stream) {
  if (!args.q || !args.k_new || !args.v_new || !args.k_cache || !args.v_cache || !args.out) {
    return cudaErrorInvalidValue;
  }
  if (args.pos < 0 || args.pos >= args.max_pos) return cudaErrorInvalidValue;
  if (!head_range_valid(args)) return cudaErrorInvalidValue;
  if (args.kv_type_k == KvType::Bf16 && args.kv_type_v == KvType::Bf16) {
    kv_store_bf16_kernel<<<ATTN_KV_HEADS, kDim, 0, stream>>>(args);
  } else if (args.kv_type_k == KvType::Q8_0 && args.kv_type_v == KvType::Q8_0) {
    kv_store_q8_kernel<<<ATTN_KV_HEADS, 256, 0, stream>>>(args);
  } else {
    kv_store_mixed_kernel<<<ATTN_KV_HEADS, 2 * 256, 0, stream>>>(args);
  }
  if (args.partial && args.splits > 1) {
    const dim3 grid(args.head_count, args.splits);
    decode_split_kernel<<<grid, kDim, 0, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    combine_splits_kernel<<<args.head_count, kDim, 0, stream>>>(args);
  } else {
    decode_kernel<<<args.head_count, kDim, 0, stream>>>(args);
  }
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
  return float(kv_q8_values(const_cast<void*>(base))[index]) *
         __half2float(kv_q8_scales(base, values)[index / KV_BLOCK]);
}

}  // namespace

void decode_attention_cpu(const DecodeAttnArgs& args) {
  if (!head_range_valid(args)) return;
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
