// Element-decode primitives shared by the batch (verify-width) kernels in this directory.
//
// They mirror what `kernels/quant/gemv.cu` and `kernels/moe/fused_experts.cu` do file-locally, kept here
// because those files do not export them and are owned by other agents. One batch lane owns eight columns,
// which is 8 bytes of e4m3, 4 bytes of packed e2m1 and 16 bytes of bf16, so every load is a full vector
// register and one 16-column block scale always covers two lanes.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ingine::arch::qwen35moe::kernels::batch {
namespace detail {

// Columns decoded by one lane in one step. A 16-column block scale therefore covers exactly two lanes.
inline constexpr int kColsPerLane = 8;
inline constexpr int kWarpCols = kColsPerLane * 32;  // 256

// e4m3 pair: cvt.rn.f16x2.e4m3x2 is sm_89+. The checkpoint keeps scales in the finite positive range; the
// NaN codes 0x7F/0xFF decode to 0, matching the reference, because the hardware cvt would propagate NaN.
__device__ __forceinline__ float2 e4m3x2_decode(std::uint16_t pair) {
  std::uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  const float lo = __low2float(v), hi = __high2float(v);
  return make_float2(lo != lo ? 0.0f : lo, hi != hi ? 0.0f : hi);
}

// Single block scale: the pair decode leaves 0 (byte 0x00) in the unused half.
__device__ __forceinline__ float e4m3_decode(std::uint8_t code) {
  return e4m3x2_decode(std::uint16_t(code)).x;
}

__device__ __forceinline__ float2 halves(std::uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

// Two packed BF16 -> two floats. Distinct from halves(), which decodes the FP16 output of the quantized
// pair conversions.
__device__ __forceinline__ float2 bfloats(std::uint32_t h2) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&h2);
  return make_float2(__bfloat162float(__low2bfloat16(v)), __bfloat162float(__high2bfloat16(v)));
}

// Sum over a whole warp leaving the result in every lane: the batched kernels need the same score in all
// 32 lanes, because each lane accumulates a different slice of the value vector.
__device__ __forceinline__ float warp_all_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// e2m1 pairs: sm_100a/sm_120a decode a byte directly, so one unpack of a 32-bit word feeds four decodes.
// Other targets (plain sm_120, the sm_89 syntax probe) go through a shared-memory table of the same exact
// f16 pairs.
#if defined(__CUDA_ARCH__) &&                                                                             \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) ||                         \
     defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL) ||                         \
     defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                         \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000))
__device__ __forceinline__ void deq4w(std::uint32_t word, const std::uint32_t*, std::uint32_t* out) {
  asm("{\n.reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(word));
}
#define INGINE_BATCH_E2M1_TABLE(name, threads) const std::uint32_t* const lut = nullptr
#else
__device__ __forceinline__ void deq4w(std::uint32_t word, const std::uint32_t* lut, std::uint32_t* out) {
  out[0] = lut[word & 0xFFu];
  out[1] = lut[(word >> 8) & 0xFFu];
  out[2] = lut[(word >> 16) & 0xFFu];
  out[3] = lut[word >> 24];
}

// Byte `i` holds columns 2i (low nibble) and 2i+1 (high nibble); the table folds a byte's two nibbles into
// one f16x2 pair, sign bit included.
__device__ __forceinline__ void init_e2m1_lut(std::uint32_t* lut, int threads) {
  static constexpr std::uint32_t mag[8] = {0x0000u, 0x3800u, 0x3C00u, 0x3E00u,
                                           0x4000u, 0x4200u, 0x4400u, 0x4600u};
  for (int n = threadIdx.x; n < 256; n += threads) {
    const std::uint32_t lo = mag[n & 7] | ((n & 8u) << 12);
    const std::uint32_t hi = mag[(n >> 4) & 7] | ((n & 0x80u) << 8);
    lut[n] = lo | (hi << 16);
  }
}

#define INGINE_BATCH_E2M1_TABLE(name, threads)                                                             \
  __shared__ std::uint32_t name[256];                                                                      \
  ::ingine::arch::qwen35moe::kernels::batch::detail::init_e2m1_lut(name, threads);                         \
  __syncthreads();                                                                                         \
  const std::uint32_t* const lut = name
#endif

}  // namespace detail
}  // namespace ingine::arch::qwen35moe::kernels::batch
