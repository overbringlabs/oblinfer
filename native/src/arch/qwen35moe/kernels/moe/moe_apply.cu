#include "arch/qwen35moe/kernels/moe/moe_apply.h"

#include <cmath>

namespace ingine::arch::qwen35moe::kernels {
namespace {

__global__ void silu_mul_kernel(float* out, const float* g, const float* u, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = g[i];
  out[i] = (v / (1.0f + __expf(-v))) * u[i];
}

__global__ void axpy_kernel(float* y, const float* x, float a, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] += a * x[i];
}

}  // namespace

cudaError_t silu_mul(float* out, const float* g, const float* u, int n, cudaStream_t stream) {
  if (!out || !g || !u || n <= 0) return cudaErrorInvalidValue;
  silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, g, u, n);
  return cudaGetLastError();
}

cudaError_t axpy(float* y, const float* x, float a, int n, cudaStream_t stream) {
  if (!y || !x || n <= 0) return cudaErrorInvalidValue;
  axpy_kernel<<<(n + 255) / 256, 256, 0, stream>>>(y, x, a, n);
  return cudaGetLastError();
}

void silu_mul_cpu(float* out, const float* g, const float* u, int n) {
  for (int i = 0; i < n; ++i) {
    const double v = g[i];
    out[i] = float(v / (1.0 + std::exp(-v)) * double(u[i]));
  }
}

cudaError_t moe_apply(const MoeApplyArgs& args, cudaStream_t stream) {
  if (!args.x || !args.ids || !args.weights || !args.experts || !args.y || !args.scratch_h ||
      !args.scratch_up || !args.scratch_down) {
    return cudaErrorInvalidValue;
  }
  bool first = true;
  for (int k = 0; k < MOE_TOP_K; ++k) {
    const MoeExpert& e = args.experts[args.ids[k]];

    // h_k = silu(gate_k x) * (up_k x)
    quant::Nvfp4GemvArgs gate = e.gate;
    gate.y = args.scratch_h;
    quant::Nvfp4GemvArgs up = e.up;
    up.y = args.scratch_up;
    cudaError_t err = quant::nvfp4_gemv(gate, stream);
    if (err != cudaSuccess) return err;
    err = quant::nvfp4_gemv(up, stream);
    if (err != cudaSuccess) return err;
    err = silu_mul(args.scratch_h, args.scratch_h, args.scratch_up, MOE_INTER, stream);
    if (err != cudaSuccess) return err;

    quant::Nvfp4GemvArgs down = e.down;
    down.x = args.scratch_h;
    down.y = args.scratch_down;
    err = quant::nvfp4_gemv(down, stream);
    if (err != cudaSuccess) return err;

    if (first) {
      err = cudaMemsetAsync(args.y, 0, MOE_HIDDEN * sizeof(float), stream);
      if (err != cudaSuccess) return err;
      first = false;
    }
    err = axpy(args.y, args.scratch_down, args.weights[k], MOE_HIDDEN, stream);
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

}  // namespace ingine::arch::qwen35moe::kernels
