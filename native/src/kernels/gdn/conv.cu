#include "kernels/gdn/conv.h"

#include <cmath>
#include <cstddef>

namespace ingine::kernels::gdn {
namespace {

constexpr int kConvThreads = 256;
constexpr int kNormThreads = 128;  // one value head per block; v_dim is 128 in every Qwen3.x config
constexpr int kNormWarps = kNormThreads / 32;

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

// Time-major history keeps every wave of loads and stores coalesced: thread c of a wave reads
// history[j * conv_dim + c] for the same j.
__global__ void conv_silu_kernel(ConvStepArgs args) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= args.conv_dim) return;
  const int d = args.conv_dim;
  float* h0 = args.history;
  float* h1 = args.history + d;
  float* h2 = args.history + 2 * d;

  const float x0 = h0[c];
  const float x1 = h1[c];
  const float x2 = h2[c];
  const float xn = args.x[c];
  const float* w = args.weight + std::size_t(c) * CONV_K;
  const float acc = w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * xn;

  args.y[c] = silu(acc);
  h0[c] = x1;
  h1[c] = x2;
  h2[c] = xn;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ __launch_bounds__(kNormThreads) void gated_norm_kernel(GatedNormArgs args) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  const int base = h * args.v_dim;
  __shared__ float warp_sums[kNormWarps];
  __shared__ float inv;

  const float x = args.x[base + d];
  float sum = warp_sum(x * x);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kNormWarps; ++w) total += warp_sums[w];
    inv = rsqrtf(total / float(args.v_dim) + args.eps);
  }
  __syncthreads();

  const float z = args.z[base + d];
  args.x[base + d] = x * inv * args.weight[d] * silu(z);
}

}  // namespace

cudaError_t conv_silu_step(const ConvStepArgs& args, cudaStream_t stream) {
  if (args.conv_dim <= 0 || !args.x || !args.weight || !args.history || !args.y) return cudaErrorInvalidValue;
  const int blocks = (args.conv_dim + kConvThreads - 1) / kConvThreads;
  conv_silu_kernel<<<blocks, kConvThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void conv_silu_step_cpu(const ConvStepArgs& args) {
  for (int c = 0; c < args.conv_dim; ++c) {
    const float x0 = args.history[c];
    const float x1 = args.history[args.conv_dim + c];
    const float x2 = args.history[2 * args.conv_dim + c];
    const float xn = args.x[c];
    const float* w = args.weight + std::size_t(c) * CONV_K;
    const float acc = w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * xn;
    args.y[c] = acc / (1.0f + std::exp(-acc));
    args.history[c] = x1;
    args.history[args.conv_dim + c] = x2;
    args.history[2 * args.conv_dim + c] = xn;
  }
}

cudaError_t gated_norm(GatedNormArgs args, cudaStream_t stream) {
  if (args.v_heads <= 0 || args.v_dim != kNormThreads || !args.x || !args.z || !args.weight) {
    return cudaErrorInvalidValue;
  }
  gated_norm_kernel<<<args.v_heads, kNormThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void gated_norm_cpu(GatedNormArgs args) {
  for (int h = 0; h < args.v_heads; ++h) {
    const int base = h * args.v_dim;
    double sum = 0.0;
    for (int d = 0; d < args.v_dim; ++d) {
      const double x = args.x[base + d];
      sum += x * x;
    }
    const double inv = 1.0 / std::sqrt(sum / double(args.v_dim) + double(args.eps));
    for (int d = 0; d < args.v_dim; ++d) {
      const double x = args.x[base + d];
      const double z = args.z[base + d];
      const double gate = z / (1.0 + std::exp(-z));
      args.x[base + d] = float(x * inv * double(args.weight[d]) * gate);
    }
  }
}

}  // namespace ingine::kernels::gdn
