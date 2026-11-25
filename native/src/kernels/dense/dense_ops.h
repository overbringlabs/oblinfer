// Unquantized dense operations at decode width: the bf16 GEMV used by the GDN scalars and the router-adjacent
// projections, the RMSNorm that surrounds every layer, and the embedding row lookup.
//
// Activations are FP32 throughout, matching the other pack kernels; the weights stay in their stored BF16.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ingine::kernels::dense {

// y[rows] = W[rows, cols] @ x[cols], W in BF16, x and y in FP32.
cudaError_t gemv_bf16(const __nv_bfloat16* w, const float* x, float* y, int rows, int cols, cudaStream_t stream);
void gemv_bf16_cpu(const __nv_bfloat16* w, const float* x, float* y, int rows, int cols);

// y[n] = x[n] * rsqrt(mean(x^2) + eps) * weight[n]. `y` may alias `x`.
cudaError_t rms_norm(const float* x, const float* weight, float* y, int n, float eps, cudaStream_t stream);
void rms_norm_cpu(const float* x, const float* weight, float* y, int n, float eps);

// out[n] = float(table[row, 0..n))
cudaError_t embedding_row(const __nv_bfloat16* table, int row, float* out, int n, cudaStream_t stream);
void embedding_row_cpu(const __nv_bfloat16* table, int row, float* out, int n);

}  // namespace ingine::kernels::dense
