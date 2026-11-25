#include "arch/qwen35moe/kernels/attn/rope_norm.h"

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kThreads = NORM_HEAD_DIM;
constexpr int kWarps = kThreads / 32;
constexpr int kPairs = ROTARY_DIM / 2;  // rotate_half pairs (i, i + ROTARY_DIM / 2)

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// One block per head; blocks 0..NORM_HEADS-1 are q, the rest are k.
__global__ __launch_bounds__(kThreads) void norm_rope_kernel(NormRopeArgs args) {
  const int h = blockIdx.x;
  const bool is_q = h < NORM_HEADS;
  float* x = is_q ? args.q + std::size_t(h) * NORM_HEAD_DIM : args.k + std::size_t(h - NORM_HEADS) * NORM_HEAD_DIM;
  const float* weight = is_q ? args.q_weight : args.k_weight;
  const int d = threadIdx.x;

  __shared__ float warp_sums[kWarps];
  __shared__ float inv;

  const float v = x[d];
  float sum = warp_sum(v * v);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) total += warp_sums[w];
    inv = rsqrtf(total / float(NORM_HEAD_DIM) + args.eps);
  }
  __syncthreads();
  x[d] = v * inv * weight[d];
  __syncthreads();

  // rotate_half on the leading ROTARY_DIM: pair (i, i + ROTARY_DIM / 2).
  if (d < kPairs) {
    const float angle = float(args.pos) * powf(ROTARY_THETA, -2.0f * float(d) / float(ROTARY_DIM));
    const float c = cosf(angle), s = sinf(angle);
    const float x0 = x[d];
    const float x1 = x[d + kPairs];
    x[d] = x0 * c - x1 * s;
    x[d + kPairs] = x0 * s + x1 * c;
  }
}

__global__ void output_gate_kernel(OutputGateArgs args) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= args.count) return;
  args.x[i] *= 1.0f / (1.0f + __expf(-args.gate[i]));
}


}  // namespace

cudaError_t qk_norm_rope(NormRopeArgs args, cudaStream_t stream) {
  if (!args.q || !args.k || !args.q_weight || !args.k_weight) return cudaErrorInvalidValue;
  norm_rope_kernel<<<NORM_HEADS + NORM_KV_HEADS, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void qk_norm_rope_cpu(NormRopeArgs args) {
  auto norm = [](float* x, const float* weight, float eps) {
    double sum = 0.0;
    for (int d = 0; d < NORM_HEAD_DIM; ++d) sum += double(x[d]) * double(x[d]);
    const double inv = 1.0 / std::sqrt(sum / double(NORM_HEAD_DIM) + double(eps));
    for (int d = 0; d < NORM_HEAD_DIM; ++d) x[d] = float(double(x[d]) * inv * double(weight[d]));
  };
  auto rope = [](float* x, int pos) {
    for (int i = 0; i < kPairs; ++i) {
      const double angle = double(pos) * std::pow(double(ROTARY_THETA), -2.0 * double(i) / double(ROTARY_DIM));
      const double c = std::cos(angle), s = std::sin(angle);
      const double x0 = x[i], x1 = x[i + kPairs];
      x[i] = float(x0 * c - x1 * s);
      x[i + kPairs] = float(x0 * s + x1 * c);
    }
  };

  for (int h = 0; h < NORM_HEADS; ++h) {
    float* x = args.q + std::size_t(h) * NORM_HEAD_DIM;
    norm(x, args.q_weight, args.eps);
    rope(x, args.pos);
  }
  for (int h = 0; h < NORM_KV_HEADS; ++h) {
    float* x = args.k + std::size_t(h) * NORM_HEAD_DIM;
    norm(x, args.k_weight, args.eps);
    rope(x, args.pos);
  }
}

cudaError_t output_gate(OutputGateArgs args, cudaStream_t stream) {
  if (!args.x || !args.gate || args.count <= 0) return cudaErrorInvalidValue;
  const int blocks = (args.count + 255) / 256;
  output_gate_kernel<<<blocks, 256, 0, stream>>>(args);
  return cudaGetLastError();
}

void output_gate_cpu(OutputGateArgs args) {
  for (int i = 0; i < args.count; ++i) {
    args.x[i] = float(double(args.x[i]) / (1.0 + std::exp(-double(args.gate[i]))));
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
