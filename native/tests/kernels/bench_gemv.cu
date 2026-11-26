// GEMV design bench: the real shapes, several inner-loop variants side by side, plus a pure-read kernel as
// the bandwidth ceiling. Prints realised GB/s so a design is chosen by measurement, not by arithmetic.
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ uint32_t f16x2_e4m3x2(uint16_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
  return r;
}

__device__ __forceinline__ uint32_t f16x2_e2m1x2(uint8_t b) {
  uint32_t r;
  const uint32_t w = b;
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %1;\ncvt.rn.f16x2.e2m1x2 %0, b0;\n}" : "=r"(r) : "r"(w));
  return r;
}

__device__ __forceinline__ float2 halves(uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

__device__ __forceinline__ float e4m3_to_f(uint8_t code) {
  const int s = (code >> 7) & 1, e = (code >> 3) & 0xF, m = code & 7;
  if (e == 0xF && m == 7) return 0.0f;
  const float v = e == 0 ? float(m) * exp2f(-9.0f) : (1.0f + float(m) / 8.0f) * exp2f(float(e - 7));
  return s ? -v : v;
}

// ---- ceiling -------------------------------------------------------------------------------------------
__global__ void read_only_kernel(const uint4* p, std::size_t n4, float* out) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  uint32_t acc = 0;
  for (; i < n4; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0xdeadbeefu) *out = 1.0f;
}

// ---- FP8 variants --------------------------------------------------------------------------------------
// A: current design, one warp per row, two codes per lane per step.
__global__ __launch_bounds__(kThreads) void fp8_warp_row_scalar(const uint8_t* w, const float* x, float* y, int rows,
                                                                int cols) {
  const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint8_t* wr = w + std::size_t(row) * cols;
  float acc = 0.0f;
  for (int j = lane; j < cols / 2; j += 32) {
    const uint16_t code = uint16_t(wr[2 * j]) | (uint16_t(wr[2 * j + 1]) << 8);
    const float2 v = halves(f16x2_e4m3x2(code));
    acc += v.x * x[2 * j] + v.y * x[2 * j + 1];
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// B: one warp per row, 16 bytes per lane per step, explicit scalars (no local array).
__global__ __launch_bounds__(kThreads) void fp8_warp_row_wide(const uint8_t* w, const float* x, float* y, int rows,
                                                              int cols) {
  const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint8_t* wr = w + std::size_t(row) * cols;
  const int chunks = cols / 16;
  float acc = 0.0f;
  for (int c = lane; c < chunks; c += 32) {
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(wr + std::size_t(c) * 16);
    const float* xp = x + std::size_t(c) * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t word = wp[i];
      const float2 lo = halves(f16x2_e4m3x2(uint16_t(word & 0xffffu)));
      const float2 hi = halves(f16x2_e4m3x2(uint16_t(word >> 16)));
      acc += lo.x * xp[i * 4 + 0] + lo.y * xp[i * 4 + 1] + hi.x * xp[i * 4 + 2] + hi.y * xp[i * 4 + 3];
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// C: one block per row, K split across 256 threads (8 bytes per thread per step), block reduction.
__global__ __launch_bounds__(kThreads) void fp8_block_row(const uint8_t* w, const float* x, float* y, int rows,
                                                          int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const uint8_t* wr = w + std::size_t(row) * cols;
  float acc = 0.0f;
  for (int j = threadIdx.x; j < cols / 2; j += kThreads) {
    const uint16_t code = uint16_t(wr[2 * j]) | (uint16_t(wr[2 * j + 1]) << 8);
    const float2 v = halves(f16x2_e4m3x2(code));
    acc += v.x * x[2 * j] + v.y * x[2 * j + 1];
  }
  acc = warp_sum(acc);
  __shared__ float warp_sums[kThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < kThreads / 32; ++i) total += warp_sums[i];
    y[row] = total;
  }
}

// D: 4 rows per warp (8 lanes each), x broadcast through shuffles, 16 bytes per lane per step.
__global__ __launch_bounds__(kThreads) void fp8_quad_row(const uint8_t* w, const float* x, float* y, int rows,
                                                         int cols) {
  const int warp = (blockIdx.x * kThreads + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int sub = lane >> 3;   // 0..3: which of the warp's four rows
  const int l = lane & 7;      // 0..7: lane inside the row group
  const int row = warp * 4 + sub;
  if (row >= rows) return;
  const uint8_t* wr = w + std::size_t(row) * cols;
  const int chunks = cols / 16;
  float acc = 0.0f;
  for (int c = l; c < chunks; c += 8) {
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(wr + std::size_t(c) * 16);
    const float* xp = x + std::size_t(c) * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t word = wp[i];
      const float2 lo = halves(f16x2_e4m3x2(uint16_t(word & 0xffffu)));
      const float2 hi = halves(f16x2_e4m3x2(uint16_t(word >> 16)));
      acc += lo.x * xp[i * 4 + 0] + lo.y * xp[i * 4 + 1] + hi.x * xp[i * 4 + 2] + hi.y * xp[i * 4 + 3];
    }
  }
  // Reduce the 8 lanes of this row group.
  acc += __shfl_down_sync(0xffffffffu, acc, 4);
  acc += __shfl_down_sync(0xffffffffu, acc, 2);
  acc += __shfl_down_sync(0xffffffffu, acc, 1);
  if (l == 0) y[row] = acc;
}

// ---- NVFP4 variants ------------------------------------------------------------------------------------
// A: current design, one warp per row, one byte (2 codes) per lane per step.
__global__ __launch_bounds__(kThreads) void nvfp4_warp_row_scalar(const uint8_t* w, const uint8_t* s, const float* x,
                                                                  float* y, int rows, int cols) {
  const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint8_t* wr = w + std::size_t(row) * (cols / 2);
  const uint8_t* sr = s + std::size_t(row) * (cols / 16);
  float acc = 0.0f;
  for (int j = lane; j < cols / 2; j += 32) {
    const float2 v = halves(f16x2_e2m1x2(wr[j]));
    acc += (v.x * x[2 * j] + v.y * x[2 * j + 1]) * e4m3_to_f(sr[j / 8]);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// B: one warp per row, 16 bytes per lane per step, scale per two words.
__global__ __launch_bounds__(kThreads) void nvfp4_warp_row_wide(const uint8_t* w, const uint8_t* s, const float* x,
                                                                float* y, int rows, int cols) {
  const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint8_t* wr = w + std::size_t(row) * (cols / 2);
  const uint8_t* sr = s + std::size_t(row) * (cols / 16);
  const int chunks = (cols / 2) / 16;
  float acc = 0.0f;
  for (int c = lane; c < chunks; c += 32) {
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(wr + std::size_t(c) * 16);
    const float* xp = x + std::size_t(c) * 32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float block = e4m3_to_f(sr[c * 2 + (i >> 1)]);
      const uint32_t word = wp[i];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 p = halves(f16x2_e2m1x2(uint8_t(word >> (8 * j))));
        acc += (p.x * xp[i * 8 + j * 2] + p.y * xp[i * 8 + j * 2 + 1]) * block;
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// C: 4 rows per warp.
__global__ __launch_bounds__(kThreads) void nvfp4_quad_row(const uint8_t* w, const uint8_t* s, const float* x,
                                                           float* y, int rows, int cols) {
  const int warp = (blockIdx.x * kThreads + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int sub = lane >> 3;
  const int l = lane & 7;
  const int row = warp * 4 + sub;
  if (row >= rows) return;
  const uint8_t* wr = w + std::size_t(row) * (cols / 2);
  const uint8_t* sr = s + std::size_t(row) * (cols / 16);
  const int chunks = (cols / 2) / 16;
  float acc = 0.0f;
  for (int c = l; c < chunks; c += 8) {
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(wr + std::size_t(c) * 16);
    const float* xp = x + std::size_t(c) * 32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float block = e4m3_to_f(sr[c * 2 + (i >> 1)]);
      const uint32_t word = wp[i];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 p = halves(f16x2_e2m1x2(uint8_t(word >> (8 * j))));
        acc += (p.x * xp[i * 8 + j * 2] + p.y * xp[i * 8 + j * 2 + 1]) * block;
      }
    }
  }
  acc += __shfl_down_sync(0xffffffffu, acc, 4);
  acc += __shfl_down_sync(0xffffffffu, acc, 2);
  acc += __shfl_down_sync(0xffffffffu, acc, 1);
  if (l == 0) y[row] = acc;
}

template <typename Fn>
float measure(int iters, Fn&& fn, double bytes) {
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  fn();  // warm
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("launch failed: %s (kernels target sm_120a)\n", cudaGetErrorString(err));
    return -1.0f;
  }
  cudaDeviceSynchronize();
  cudaEventRecord(t0);
  for (int i = 0; i < iters; ++i) fn();
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  if (cudaGetLastError() != cudaSuccess) ms = -1.0f;
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / iters;
}

}  // namespace

int main() {
  int devs = 0;
  if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
    std::printf("skip: no CUDA device\n");
    return 0;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);

  const int fp8_rows = 8192, fp8_cols = 2048;   // q_proj / in_proj_qkv shape
  const int nv_rows = 512, nv_cols = 2048;      // one expert's gate_proj
  const int nvd_rows = 2048, nvd_cols = 512;    // one expert's down_proj
  std::printf("device %s, %d MiB, sm_%d%d\n", prop.name, int(prop.totalGlobalMem >> 20), prop.major, prop.minor);

  // ---- ceiling
  {
    const std::size_t bytes = std::size_t(512) << 20;
    uint8_t* d = nullptr;
    cudaMalloc(&d, bytes);
    cudaMemset(d, 1, bytes);
    float* out = nullptr;
    cudaMalloc(&out, 4);
    const int blocks = 1024;
    const float ms = measure(20, [&] { read_only_kernel<<<blocks, kThreads>>>(reinterpret_cast<const uint4*>(d), bytes / 16, out); },
                             double(bytes));
    std::printf("read-only ceiling: %.1f GB/s\n", ms > 0 ? double(bytes) / (ms * 1e-3) / 1e9 : 0.0);
    cudaFree(d);
    cudaFree(out);
  }

  // ---- fp8
  {
    const std::size_t wb = std::size_t(fp8_rows) * fp8_cols;
    uint8_t* w = nullptr;
    float *x = nullptr, *y = nullptr;
    cudaMalloc(&w, wb);
    cudaMalloc(&x, fp8_cols * 4);
    cudaMalloc(&y, fp8_rows * 4);
    cudaMemset(w, 0x38, wb);
    cudaMemset(x, 0, fp8_cols * 4);
    const double gb = double(wb) / 1e9;
    struct V {
      const char* name;
      void (*fn)(const uint8_t*, const float*, float*, int, int);
      int blocks;
    } variants[] = {
        {"fp8 warp-row scalar (current)", fp8_warp_row_scalar, (fp8_rows + 7) / 8},
        {"fp8 warp-row 16B", fp8_warp_row_wide, (fp8_rows + 7) / 8},
        {"fp8 block-row", fp8_block_row, fp8_rows},
        {"fp8 quad-row (4 rows/warp)", fp8_quad_row, (fp8_rows + 3) / 4 / 8},
    };
    for (const V& v : variants) {
      const float ms = measure(50, [&] { v.fn<<<v.blocks, kThreads>>>(w, x, y, fp8_rows, fp8_cols); }, gb);
      std::printf("%-32s %7.3f ms  %6.1f GB/s\n", v.name, ms, ms > 0 ? gb / (ms * 1e-3) : 0.0);
    }
    cudaFree(w);
    cudaFree(x);
    cudaFree(y);
  }

  // ---- fp8 at DRAM size (the engine streams ~2.2 GB per token, so L2-resident numbers are misleading)
  {
    const int rows = 32768, cols = 2048;  // 64 MiB of weights
    uint8_t* w = nullptr;
    float *x = nullptr, *y = nullptr;
    cudaMalloc(&w, std::size_t(rows) * cols);
    cudaMalloc(&x, cols * 4);
    cudaMalloc(&y, rows * 4);
    cudaMemset(w, 0x38, std::size_t(rows) * cols);
    cudaMemset(x, 0, cols * 4);
    const double gb = double(std::size_t(rows) * cols) / 1e9;
    struct V {
      const char* name;
      void (*fn)(const uint8_t*, const float*, float*, int, int);
      int blocks;
    } variants[] = {
        {"fp8 DRAM 64MB warp-row scalar", fp8_warp_row_scalar, (rows + 7) / 8},
        {"fp8 DRAM 64MB block-row", fp8_block_row, rows},
        {"fp8 DRAM 64MB quad-row", fp8_quad_row, (rows + 3) / 4 / 8},
    };
    for (const V& v : variants) {
      const float ms = measure(10, [&] { v.fn<<<v.blocks, kThreads>>>(w, x, y, rows, cols); }, gb);
      std::printf("%-32s %7.3f ms  %6.1f GB/s\n", v.name, ms, ms > 0 ? gb / (ms * 1e-3) : 0.0);
    }
    cudaFree(w);
    cudaFree(x);
    cudaFree(y);
  }

  // ---- nvfp4
  {
    const std::size_t wb = std::size_t(nv_rows) * nv_cols / 2;
    const std::size_t sb = std::size_t(nv_rows) * nv_cols / 16;
    uint8_t *w = nullptr, *s = nullptr;
    float *x = nullptr, *y = nullptr;
    cudaMalloc(&w, wb);
    cudaMalloc(&s, sb);
    cudaMalloc(&x, nv_cols * 4);
    cudaMalloc(&y, nv_rows * 4);
    cudaMemset(w, 0x11, wb);
    cudaMemset(s, 0x38, sb);
    cudaMemset(x, 0, nv_cols * 4);
    const double gb = double(wb + sb) / 1e9;
    struct V {
      const char* name;
      void (*fn)(const uint8_t*, const uint8_t*, const float*, float*, int, int);
      int blocks;
    } variants[] = {
        {"nvfp4 warp-row scalar (current)", nvfp4_warp_row_scalar, (nv_rows + 7) / 8},
        {"nvfp4 warp-row 16B", nvfp4_warp_row_wide, (nv_rows + 7) / 8},
        {"nvfp4 quad-row (4 rows/warp)", nvfp4_quad_row, (nv_rows + 3) / 4 / 8},
    };
    for (const V& v : variants) {
      const float ms = measure(200, [&] { v.fn<<<v.blocks, kThreads>>>(w, s, x, y, nv_rows, nv_cols); }, gb);
      std::printf("%-32s %7.3f ms  %6.1f GB/s\n", v.name, ms, ms > 0 ? gb / (ms * 1e-3) : 0.0);
    }
    cudaFree(w);
    cudaFree(s);
    cudaFree(x);
    cudaFree(y);
  }
  return 0;
}
