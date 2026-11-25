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

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ void kv_store_kernel(DecodeAttnArgs args) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  if (h >= ATTN_KV_HEADS || d >= kDim) return;
  const std::size_t at = (std::size_t(args.pos) * ATTN_KV_HEADS + h) * kDim + d;
  args.k_cache[at] = __float2bfloat16(args.k_new[h * kDim + d]);
  args.v_cache[at] = __float2bfloat16(args.v_new[h * kDim + d]);
}

__global__ __launch_bounds__(kDim) void decode_kernel(DecodeAttnArgs args) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  const int kh = h / kGroup;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

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
      const __nv_bfloat16* k = args.k_cache + (std::size_t(base + i) * ATTN_KV_HEADS + kh) * kDim;
      float dot = 0.0f;
#pragma unroll
      for (int r = 0; r < 8; ++r) dot += q_reg[r] * __bfloat162float(k[lane + r * 32]);
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
      const __nv_bfloat16* v = args.v_cache + (std::size_t(base + i) * ATTN_KV_HEADS + kh) * kDim;
      acc += p * __bfloat162float(v[d]);
    }

    if (threadIdx.x == 0) {
      run_sum = run_sum * rescale + chunk_sum;
      run_max = m_new;
    }
    __syncthreads();
  }

  args.out[h * kDim + d] = acc / run_sum;
}

}  // namespace

cudaError_t decode_attention(const DecodeAttnArgs& args, cudaStream_t stream) {
  if (!args.q || !args.k_new || !args.v_new || !args.k_cache || !args.v_cache || !args.out) {
    return cudaErrorInvalidValue;
  }
  if (args.pos < 0 || args.pos >= args.max_pos) return cudaErrorInvalidValue;
  kv_store_kernel<<<ATTN_KV_HEADS, kDim, 0, stream>>>(args);
  decode_kernel<<<ATTN_HEADS, kDim, 0, stream>>>(args);
  return cudaGetLastError();
}

void decode_attention_cpu(const DecodeAttnArgs& args) {
  for (int h = 0; h < ATTN_KV_HEADS; ++h) {
    for (int d = 0; d < kDim; ++d) {
      const std::size_t at = (std::size_t(args.pos) * ATTN_KV_HEADS + h) * kDim + d;
      args.k_cache[at] = __float2bfloat16(args.k_new[h * kDim + d]);
      args.v_cache[at] = __float2bfloat16(args.v_new[h * kDim + d]);
    }
  }

  const double scale = 1.0 / std::sqrt(double(kDim));
  std::vector<double> q(ATTN_HEADS * kDim);
  for (int i = 0; i < ATTN_HEADS * kDim; ++i) q[std::size_t(i)] = double(args.q[i]);
  std::vector<double> score(args.pos + 1);
  for (int h = 0; h < ATTN_HEADS; ++h) {
    const int kh = h / kGroup;
    double m = -INFINITY;
    for (int i = 0; i <= args.pos; ++i) {
      double dot = 0.0;
      for (int d = 0; d < kDim; ++d) {
        const std::size_t at = (std::size_t(i) * ATTN_KV_HEADS + kh) * kDim + d;
        dot += q[h * kDim + d] * double(__bfloat162float(args.k_cache[at]));
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
        acc += score[i] * double(__bfloat162float(args.v_cache[at]));
      }
      args.out[h * kDim + d] = float(acc / sum);
    }
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
