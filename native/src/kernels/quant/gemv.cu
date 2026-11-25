#include "kernels/quant/gemv.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>

namespace ingine::kernels::quant {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerBlock = kThreads / 32;

// e4m3fn: bias 7, subnormal mant * 2^-9, 0x7F/0xFF NaN. e2m1: the eight magnitudes of the format.
__host__ __device__ __forceinline__ float e4m3_to_float(std::uint8_t code) {
  const int sign = (code >> 7) & 1;
  const int exp = (code >> 3) & 0xF;
  const int mant = code & 0x7;
  if (exp == 0xF && mant == 0x7) return 0.0f;
  float value;
  if (exp == 0) {
    value = std::ldexp(float(mant), -9);
  } else {
    value = std::ldexp(1.0f + float(mant) / 8.0f, exp - 7);
  }
  return sign ? -value : value;
}

__host__ __device__ __forceinline__ float e2m1_code_to_float(std::uint8_t code) {
  // sign(1) exp(2) mant(1): magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6
  const float mag = code & 0x1 ? (code & 0x4 ? ((code & 0x2) ? 6.0f : 3.0f) : ((code & 0x2) ? 1.5f : 0.5f))
                               : (code & 0x4 ? ((code & 0x2) ? 4.0f : 2.0f) : ((code & 0x2) ? 1.0f : 0.0f));
  return (code & 0x8) ? -mag : mag;
}

// PTX decodes: one 16-bit word holds two e4m3 codes (low byte first); one byte holds two e2m1 codes (even
// column in the low nibble).
__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(std::uint16_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
  return r;
}

__device__ __forceinline__ uint32_t f16x2_from_e2m1x2(std::uint8_t b) {
  uint32_t r;
  const std::uint32_t w = b;  // the byte lands in b0 of the little-endian word
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %1;\ncvt.rn.f16x2.e2m1x2 %0, b0;\n}"
      : "=r"(r)
      : "r"(w));
  return r;
}

__device__ __forceinline__ float2 halves(uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ __launch_bounds__(kThreads) void fp8_gemv_kernel(Fp8GemvArgs args) {
  const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
  if (row >= args.rows) return;
  const int lane = threadIdx.x & 31;
  const int pairs = args.cols / 2;
  const std::uint8_t* w = args.w + std::size_t(row) * args.cols;
  float acc = 0.0f;
  for (int j = lane; j < pairs; j += 32) {
    const std::uint16_t code = std::uint16_t(w[2 * j]) | (std::uint16_t(w[2 * j + 1]) << 8);
    const float2 v = halves(f16x2_from_e4m3x2(code));
    acc += v.x * args.x[2 * j] + v.y * args.x[2 * j + 1];
  }
  acc = warp_sum(acc);
  if (lane == 0) args.y[row] = acc * args.scale;
}

__global__ __launch_bounds__(kThreads) void nvfp4_gemv_kernel(Nvfp4GemvArgs args) {
  const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
  if (row >= args.rows) return;
  const int lane = threadIdx.x & 31;
  const int bytes = args.cols / 2;
  const std::uint8_t* w = args.w + std::size_t(row) * bytes;
  const std::uint8_t* s = args.scales + std::size_t(row) * (args.cols / 16);
  float acc = 0.0f;
  for (int j = lane; j < bytes; j += 32) {
    const float2 v = halves(f16x2_from_e2m1x2(w[j]));
    const float block = e4m3_to_float(s[j / 8]);
    acc += (v.x * args.x[2 * j] + v.y * args.x[2 * j + 1]) * block;
  }
  acc = warp_sum(acc);
  if (lane == 0) args.y[row] = acc * args.scale;
}

}  // namespace

float fp8_e4m3_to_float(std::uint8_t code) { return e4m3_to_float(code); }
float e2m1_to_float(std::uint8_t code) { return e2m1_code_to_float(code); }

cudaError_t fp8_gemv(const Fp8GemvArgs& args, cudaStream_t stream) {
  if (!args.w || !args.x || !args.y || args.rows <= 0 || args.cols <= 0 || args.cols % 2) {
    return cudaErrorInvalidValue;
  }
  fp8_gemv_kernel<<<(args.rows + kRowsPerBlock - 1) / kRowsPerBlock, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void fp8_gemv_cpu(const Fp8GemvArgs& args) {
  for (int r = 0; r < args.rows; ++r) {
    double acc = 0.0;
    const std::uint8_t* w = args.w + std::size_t(r) * args.cols;
    for (int c = 0; c < args.cols; ++c) acc += double(e4m3_to_float(w[c])) * double(args.x[c]);
    args.y[r] = float(acc * double(args.scale));
  }
}

cudaError_t nvfp4_gemv(const Nvfp4GemvArgs& args, cudaStream_t stream) {
  if (!args.w || !args.scales || !args.x || !args.y || args.rows <= 0 || args.cols <= 0 || args.cols % 16) {
    return cudaErrorInvalidValue;
  }
  nvfp4_gemv_kernel<<<(args.rows + kRowsPerBlock - 1) / kRowsPerBlock, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void nvfp4_gemv_cpu(const Nvfp4GemvArgs& args) {
  for (int r = 0; r < args.rows; ++r) {
    const std::uint8_t* w = args.w + std::size_t(r) * (args.cols / 2);
    const std::uint8_t* s = args.scales + std::size_t(r) * (args.cols / 16);
    double acc = 0.0;
    for (int c = 0; c < args.cols; ++c) {
      const std::uint8_t byte = w[c / 2];
      const std::uint8_t code = (c % 2) ? std::uint8_t(byte >> 4) : std::uint8_t(byte & 0xF);
      acc += double(e2m1_code_to_float(code)) * double(e4m3_to_float(s[c / 16])) * double(args.x[c]);
    }
    args.y[r] = float(acc * double(args.scale));
  }
}

}  // namespace ingine::kernels::quant
