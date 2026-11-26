#include "arch/qwen35moe/kernels/batch/batch_gemv.h"

#include <cmath>
#include <cstddef>

#include "arch/qwen35moe/kernels/batch/batch_dequant.cuh"
#include "kernels/quant/gemv.h"

namespace ingine::arch::qwen35moe::kernels {
namespace {

namespace bd = batch::detail;
namespace quant = ingine::kernels::quant;

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
// Two weight rows per warp: the n activation windows of a step are staged once and feed both rows, which
// halves the load-instruction count of the inner loop.
constexpr int kRowsPerWarp = 2;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kLanePairs = bd::kColsPerLane / 2;  // four float2 of activations per lane

// A lane owns eight consecutive columns: 8 bytes of e4m3, 4 bytes of packed e2m1 (one block scale), 16
// bytes of bf16. One warp step therefore covers 256 columns and the eight decoded weight values line up
// with the four staged float2 of each activation row. The weight row is read once for all n rows.
// The checkpoint's dense tensors are placed by size, so a weight base is not guaranteed to be vector
// aligned (kernels/quant/gemv.cu reads bytes for exactly this reason). ALIGNED picks the 8-byte load when
// the base allows it and byte pairs otherwise; the launcher tests the pointer.
template <int N, bool ALIGNED>
__global__ __launch_bounds__(kThreads) void fp8_gemv_batch_kernel(const Fp8GemvBatchArgs args) {
  const int row0 = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;
  const int lane = threadIdx.x & 31;
  const int c0 = lane * bd::kColsPerLane;
  const int cols = args.cols;
  const std::uint8_t* w[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) w[r] = args.w + std::size_t(row0 + r) * cols;

  float acc[kRowsPerWarp][N];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int t = 0; t < N; ++t) acc[r][t] = 0.0f;

  for (int step = 0; step < cols / bd::kWarpCols; ++step) {
    const int c = c0 + step * bd::kWarpCols;
    float2 xv[N][kLanePairs];
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float2* xp = reinterpret_cast<const float2*>(args.x + std::size_t(t) * cols + c);
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) xv[t][p] = xp[p];
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      std::uint16_t pair[kLanePairs];
      if constexpr (ALIGNED) {
        const uint2 word = *reinterpret_cast<const uint2*>(w[r] + c);
        const std::uint32_t raw[2] = {word.x, word.y};
        const std::uint16_t* p16 = reinterpret_cast<const std::uint16_t*>(raw);
#pragma unroll
        for (int q = 0; q < kLanePairs; ++q) pair[q] = p16[q];
      } else {
#pragma unroll
        for (int q = 0; q < kLanePairs; ++q)
          pair[q] = std::uint16_t(w[r][c + 2 * q]) | std::uint16_t(std::uint16_t(w[r][c + 2 * q + 1]) << 8);
      }
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) {
        const float2 wv = bd::e4m3x2_decode(pair[p]);  // columns c+2p (low byte) and c+2p+1
#pragma unroll
        for (int t = 0; t < N; ++t) acc[r][t] += wv.x * xv[t][p].x + wv.y * xv[t][p].y;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float total = bd::warp_sum(acc[r][t]);
      if (lane == 0) args.y[std::size_t(t) * args.rows + row0 + r] = total * args.scale;
    }
  }
}

template <int N, bool ALIGNED>
__global__ __launch_bounds__(kThreads) void nvfp4_gemv_batch_kernel(const Nvfp4GemvBatchArgs args) {
  INGINE_BATCH_E2M1_TABLE(s_lut, kThreads);
  const int row0 = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;
  const int lane = threadIdx.x & 31;
  const int c0 = lane * bd::kColsPerLane;
  const int cols = args.cols;
  const std::uint8_t* w[kRowsPerWarp];
  const std::uint8_t* s[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    w[r] = args.w + std::size_t(row0 + r) * (cols / 2);
    s[r] = args.scales + std::size_t(row0 + r) * (cols / 16);
  }

  float acc[kRowsPerWarp][N];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int t = 0; t < N; ++t) acc[r][t] = 0.0f;

  for (int step = 0; step < cols / bd::kWarpCols; ++step) {
    const int c = c0 + step * bd::kWarpCols;
    float2 xv[N][kLanePairs];
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float2* xp = reinterpret_cast<const float2*>(args.x + std::size_t(t) * cols + c);
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) xv[t][p] = xp[p];
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      std::uint32_t word;
      if constexpr (ALIGNED) {
        word = *reinterpret_cast<const std::uint32_t*>(w[r] + c / 2);
      } else {
        const std::uint8_t* wb = w[r] + c / 2;
        word = std::uint32_t(wb[0]) | (std::uint32_t(wb[1]) << 8) | (std::uint32_t(wb[2]) << 16) |
               (std::uint32_t(wb[3]) << 24);
      }
      // Eight columns never straddle a 16-column block, so one block scale covers this lane's whole step.
      const float block = bd::e4m3_decode(s[r][c / 16]);
      std::uint32_t dec[4];
      bd::deq4w(word, lut, dec);
      float part[N];
#pragma unroll
      for (int t = 0; t < N; ++t) part[t] = 0.0f;
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) {
        const float2 wv = bd::halves(dec[p]);  // byte p holds columns c+2p (low nibble) and c+2p+1
#pragma unroll
        for (int t = 0; t < N; ++t) part[t] += wv.x * xv[t][p].x + wv.y * xv[t][p].y;
      }
#pragma unroll
      for (int t = 0; t < N; ++t) acc[r][t] += block * part[t];
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float total = bd::warp_sum(acc[r][t]);
      if (lane == 0) args.y[std::size_t(t) * args.rows + row0 + r] = total * args.scale;
    }
  }
}

template <int N, bool ALIGNED>
__global__ __launch_bounds__(kThreads) void bf16_gemv_batch_kernel(const Bf16GemvBatchArgs args) {
  const int row0 = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;
  const int lane = threadIdx.x & 31;
  const int c0 = lane * bd::kColsPerLane;
  const int cols = args.cols;
  const __nv_bfloat16* w[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) w[r] = args.w + std::size_t(row0 + r) * cols;

  float acc[kRowsPerWarp][N];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int t = 0; t < N; ++t) acc[r][t] = 0.0f;

  for (int step = 0; step < cols / bd::kWarpCols; ++step) {
    const int c = c0 + step * bd::kWarpCols;
    float2 xv[N][kLanePairs];
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float2* xp = reinterpret_cast<const float2*>(args.x + std::size_t(t) * cols + c);
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) xv[t][p] = xp[p];
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      std::uint32_t raw[kLanePairs];
      if constexpr (ALIGNED) {
        const uint4 word = *reinterpret_cast<const uint4*>(w[r] + c);
        raw[0] = word.x;
        raw[1] = word.y;
        raw[2] = word.z;
        raw[3] = word.w;
      } else {
        const std::uint8_t* wb = reinterpret_cast<const std::uint8_t*>(w[r] + c);
#pragma unroll
        for (int q = 0; q < kLanePairs; ++q)
          raw[q] = std::uint32_t(wb[4 * q]) | (std::uint32_t(wb[4 * q + 1]) << 8) |
                   (std::uint32_t(wb[4 * q + 2]) << 16) | (std::uint32_t(wb[4 * q + 3]) << 24);
      }
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) {
        const float2 wv = bd::bfloats(raw[p]);  // one uint32 = two bf16 = columns c+2p, c+2p+1
#pragma unroll
        for (int t = 0; t < N; ++t) acc[r][t] += wv.x * xv[t][p].x + wv.y * xv[t][p].y;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
      const float total = bd::warp_sum(acc[r][t]);
      if (lane == 0) args.y[std::size_t(t) * args.rows + row0 + r] = total;
    }
  }
}

__global__ __launch_bounds__(kThreads) void silu_mul_batch_kernel(float* g, const float* u, int total) {
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= total) return;
  const float v = g[i];
  g[i] = (v / (1.0f + __expf(-v))) * u[i];
}

#define INGINE_BATCH_DISPATCH(kernel, n, aligned)                                       \
  if (aligned) {                                                                        \
    switch (n) {                                                                        \
      case 1: kernel<1, true><<<blocks, kThreads, 0, stream>>>(args); break;             \
      case 2: kernel<2, true><<<blocks, kThreads, 0, stream>>>(args); break;             \
      case 3: kernel<3, true><<<blocks, kThreads, 0, stream>>>(args); break;             \
      default: kernel<4, true><<<blocks, kThreads, 0, stream>>>(args); break;            \
    }                                                                                   \
  } else {                                                                              \
    switch (n) {                                                                        \
      case 1: kernel<1, false><<<blocks, kThreads, 0, stream>>>(args); break;             \
      case 2: kernel<2, false><<<blocks, kThreads, 0, stream>>>(args); break;             \
      case 3: kernel<3, false><<<blocks, kThreads, 0, stream>>>(args); break;             \
      default: kernel<4, false><<<blocks, kThreads, 0, stream>>>(args); break;            \
    }                                                                                   \
  }

}  // namespace

cudaError_t fp8_gemv_batch(const Fp8GemvBatchArgs& args, cudaStream_t stream) {
  if (!args.w || !args.x || !args.y) return cudaErrorInvalidValue;
  if (args.rows <= 0 || args.n < 1 || args.n > BATCH_MAX) return cudaErrorInvalidValue;
  if (args.rows % kRowsPerBlock || args.cols % bd::kWarpCols) return cudaErrorInvalidValue;
  const int blocks = args.rows / kRowsPerBlock;
  INGINE_BATCH_DISPATCH(fp8_gemv_batch_kernel, args.n, (reinterpret_cast<std::uintptr_t>(args.w) & 7u) == 0)
  return cudaGetLastError();
}

cudaError_t nvfp4_gemv_batch(const Nvfp4GemvBatchArgs& args, cudaStream_t stream) {
  if (!args.w || !args.scales || !args.x || !args.y) return cudaErrorInvalidValue;
  if (args.rows <= 0 || args.n < 1 || args.n > BATCH_MAX) return cudaErrorInvalidValue;
  if (args.rows % kRowsPerBlock || args.cols % bd::kWarpCols) return cudaErrorInvalidValue;
  const int blocks = args.rows / kRowsPerBlock;
  INGINE_BATCH_DISPATCH(nvfp4_gemv_batch_kernel, args.n, (reinterpret_cast<std::uintptr_t>(args.w) & 3u) == 0)
  return cudaGetLastError();
}

cudaError_t bf16_gemv_batch(const Bf16GemvBatchArgs& args, cudaStream_t stream) {
  if (!args.w || !args.x || !args.y) return cudaErrorInvalidValue;
  if (args.rows <= 0 || args.n < 1 || args.n > BATCH_MAX) return cudaErrorInvalidValue;
  if (args.rows % kRowsPerBlock || args.cols % bd::kWarpCols) return cudaErrorInvalidValue;
  const int blocks = args.rows / kRowsPerBlock;
  INGINE_BATCH_DISPATCH(bf16_gemv_batch_kernel, args.n,
                        (reinterpret_cast<std::uintptr_t>(args.w) & 15u) == 0)
  return cudaGetLastError();
}

cudaError_t silu_mul_batch(float* g, const float* u, int count, cudaStream_t stream) {
  if (!g || !u || count <= 0) return cudaErrorInvalidValue;
  silu_mul_batch_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(g, u, count);
  return cudaGetLastError();
}

void fp8_gemv_batch_cpu(const Fp8GemvBatchArgs& args) {
  for (int t = 0; t < args.n; ++t) {
    const float* x = args.x + std::size_t(t) * args.cols;
    for (int r = 0; r < args.rows; ++r) {
      const std::uint8_t* w = args.w + std::size_t(r) * args.cols;
      double acc = 0.0;
      for (int c = 0; c < args.cols; ++c) acc += double(quant::fp8_e4m3_to_float(w[c])) * double(x[c]);
      args.y[std::size_t(t) * args.rows + r] = float(acc * double(args.scale));
    }
  }
}

void nvfp4_gemv_batch_cpu(const Nvfp4GemvBatchArgs& args) {
  for (int t = 0; t < args.n; ++t) {
    const float* x = args.x + std::size_t(t) * args.cols;
    for (int r = 0; r < args.rows; ++r) {
      const std::uint8_t* w = args.w + std::size_t(r) * (args.cols / 2);
      const std::uint8_t* s = args.scales + std::size_t(r) * (args.cols / 16);
      double acc = 0.0;
      for (int c = 0; c < args.cols; ++c) {
        const std::uint8_t byte = w[c / 2];
        const std::uint8_t code = (c % 2) ? std::uint8_t(byte >> 4) : std::uint8_t(byte & 0xF);
        acc += double(quant::e2m1_to_float(code)) * double(quant::fp8_e4m3_to_float(s[c / 16])) * double(x[c]);
      }
      args.y[std::size_t(t) * args.rows + r] = float(acc * double(args.scale));
    }
  }
}

void bf16_gemv_batch_cpu(const Bf16GemvBatchArgs& args) {
  for (int t = 0; t < args.n; ++t) {
    const float* x = args.x + std::size_t(t) * args.cols;
    for (int r = 0; r < args.rows; ++r) {
      const __nv_bfloat16* w = args.w + std::size_t(r) * args.cols;
      double acc = 0.0;
      for (int c = 0; c < args.cols; ++c) acc += double(__bfloat162float(w[c])) * double(x[c]);
      args.y[std::size_t(t) * args.rows + r] = float(acc);
    }
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
