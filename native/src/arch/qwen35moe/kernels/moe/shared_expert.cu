#include "arch/qwen35moe/kernels/moe/shared_expert.h"

#include <cmath>

namespace ingine::arch::qwen35moe::kernels {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// One block: 256 threads cover the 2048-wide dot and then the same 2048-wide scale, so the scalar never
// leaves the block.
__global__ __launch_bounds__(256) void scale_sigmoid_dot_kernel(float* y, const __nv_bfloat16* w, const float* x,
                                                                int n) {
  __shared__ float warp_sums[8];
  __shared__ float gate;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += __bfloat162float(w[i]) * x[i];
  acc = warp_sum(acc);
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int k = 0; k < 8; ++k) total += warp_sums[k];
    gate = 1.0f / (1.0f + __expf(-total));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] *= gate;
}

}  // namespace

cudaError_t scale_sigmoid_dot(float* y, const __nv_bfloat16* w, const float* x, int n, cudaStream_t stream) {
  if (!y || !w || !x || n <= 0) return cudaErrorInvalidValue;
  scale_sigmoid_dot_kernel<<<1, 256, 0, stream>>>(y, w, x, n);
  return cudaGetLastError();
}

void scale_sigmoid_dot_cpu(float* y, const __nv_bfloat16* w, const float* x, int n) {
  double dot = 0.0;
  for (int i = 0; i < n; ++i) dot += double(__bfloat162float(w[i])) * double(x[i]);
  const double gate = 1.0 / (1.0 + std::exp(-dot));
  for (int i = 0; i < n; ++i) y[i] = float(double(y[i]) * gate);
}

cudaError_t shared_expert(const SharedExpertArgs& args, cudaStream_t stream) {
  if (!args.gate_weight || !args.x || !args.y || !args.scratch_h || !args.scratch_up) {
    return cudaErrorInvalidValue;
  }
  quant::Nvfp4GemvArgs gate = args.gate;
  gate.y = args.scratch_h;
  quant::Nvfp4GemvArgs up = args.up;
  up.y = args.scratch_up;
  quant::Nvfp4GemvArgs down = args.down;
  down.x = args.scratch_h;
  down.y = args.y;

  cudaError_t err = quant::nvfp4_gemv(gate, stream);
  if (err != cudaSuccess) return err;
  err = quant::nvfp4_gemv(up, stream);
  if (err != cudaSuccess) return err;
  err = silu_mul(args.scratch_h, args.scratch_h, args.scratch_up, MOE_INTER, stream);
  if (err != cudaSuccess) return err;
  err = quant::nvfp4_gemv(down, stream);
  if (err != cudaSuccess) return err;
  return scale_sigmoid_dot(args.y, args.gate_weight, args.x, MOE_HIDDEN, stream);
}

}  // namespace ingine::arch::qwen35moe::kernels
