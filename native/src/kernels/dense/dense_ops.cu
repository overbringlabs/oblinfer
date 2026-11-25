#include "kernels/dense/dense_ops.h"

#include <cmath>
#include <cstddef>

namespace ingine::kernels::dense {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerBlock = kThreads / 32;

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ __launch_bounds__(kThreads) void gemv_bf16_kernel(const __nv_bfloat16* w, const float* x, float* y,
                                                            int rows, int cols) {
  const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* wrow = w + std::size_t(row) * cols;
  float acc = 0.0f;
  for (int c = lane; c < cols; c += 32) acc += __bfloat162float(wrow[c]) * x[c];
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

__global__ __launch_bounds__(kThreads) void rms_norm_kernel(const float* x, const float* weight, float* y, int n,
                                                           float eps) {
  __shared__ float warp_sums[kThreads / 32];
  __shared__ float inv;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += kThreads) sum += x[i] * x[i];
  sum = warp_sum(sum);
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int w = 0; w < kThreads / 32; ++w) total += warp_sums[w];
    inv = rsqrtf(total / float(n) + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += kThreads) y[i] = x[i] * inv * weight[i];
}

__global__ __launch_bounds__(kThreads) void argmax_kernel(const float* x, int n, int* index, float* value) {
  __shared__ float warp_values[kThreads / 32];
  __shared__ int warp_indices[kThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float best = -INFINITY;
  int best_i = 0;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    if (x[i] > best) {
      best = x[i];
      best_i = i;
    }
  }
  // Warp reduction keeping the lower index on ties.
  for (int off = 16; off > 0; off >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, best, off);
    const int other_i = __shfl_down_sync(0xffffffffu, best_i, off);
    if (other > best || (other == best && other_i < best_i)) {
      best = other;
      best_i = other_i;
    }
  }
  if (lane == 0) {
    warp_values[warp] = best;
    warp_indices[warp] = best_i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float top = warp_values[0];
    int top_i = warp_indices[0];
    for (int w = 1; w < kThreads / 32; ++w) {
      if (warp_values[w] > top || (warp_values[w] == top && warp_indices[w] < top_i)) {
        top = warp_values[w];
        top_i = warp_indices[w];
      }
    }
    *index = top_i;
    if (value) *value = top;
  }
}

__global__ void embedding_row_kernel(const __nv_bfloat16* table, int row, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __bfloat162float(table[std::size_t(row) * n + i]);
}

}  // namespace

cudaError_t gemv_bf16(const __nv_bfloat16* w, const float* x, float* y, int rows, int cols, cudaStream_t stream) {
  if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  gemv_bf16_kernel<<<(rows + kRowsPerBlock - 1) / kRowsPerBlock, kThreads, 0, stream>>>(w, x, y, rows, cols);
  return cudaGetLastError();
}

void gemv_bf16_cpu(const __nv_bfloat16* w, const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    double acc = 0.0;
    const __nv_bfloat16* wrow = w + std::size_t(r) * cols;
    for (int c = 0; c < cols; ++c) acc += double(__bfloat162float(wrow[c])) * double(x[c]);
    y[r] = float(acc);
  }
}

cudaError_t rms_norm(const float* x, const float* weight, float* y, int n, float eps, cudaStream_t stream) {
  if (!x || !weight || !y || n <= 0) return cudaErrorInvalidValue;
  rms_norm_kernel<<<1, kThreads, 0, stream>>>(x, weight, y, n, eps);
  return cudaGetLastError();
}

void rms_norm_cpu(const float* x, const float* weight, float* y, int n, float eps) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += double(x[i]) * double(x[i]);
  const double inv = 1.0 / std::sqrt(sum / double(n) + double(eps));
  for (int i = 0; i < n; ++i) y[i] = float(double(x[i]) * inv * double(weight[i]));
}

cudaError_t embedding_row(const __nv_bfloat16* table, int row, float* out, int n, cudaStream_t stream) {
  if (!table || !out || row < 0 || n <= 0) return cudaErrorInvalidValue;
  embedding_row_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(table, row, out, n);
  return cudaGetLastError();
}

void embedding_row_cpu(const __nv_bfloat16* table, int row, float* out, int n) {
  for (int i = 0; i < n; ++i) out[i] = __bfloat162float(table[std::size_t(row) * n + i]);
}

cudaError_t argmax(const float* x, int n, int* index, float* value, cudaStream_t stream) {
  if (!x || !index || n <= 0) return cudaErrorInvalidValue;
  argmax_kernel<<<1, kThreads, 0, stream>>>(x, n, index, value);
  return cudaGetLastError();
}

int argmax_cpu(const float* x, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

}  // namespace ingine::kernels::dense
