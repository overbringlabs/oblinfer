#include "arch/qwen35moe/kernels/moe/shared_expert.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

// 128-thread blocks, one row per warp: 128 blocks for the 512-wide gate/up and 512 for the 2048-wide down.
// The generic `quant::nvfp4_gemv` path this replaces runs one warp per row with byte-granular loads and an
// 8-row block, which on a 0.5 MB shared-expert projection reached only 64 blocks on a 70-SM card and about
// 90 GB/s; the wide loads plus the extra blocks are what the routed fused kernel already uses.
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kColsPerLane = 8;                    // one uint32 of packed e2m1
constexpr int kWarpCols = kColsPerLane * 32;       // 256
constexpr int kGateSteps = MOE_HIDDEN / kWarpCols;  // 8
constexpr int kGateRowsPerBlock = kWarps;
constexpr int kGateBlocks = MOE_INTER / kGateRowsPerBlock;  // 128, plus one for the gate scalar

constexpr int kDownCols = MOE_INTER / 32;  // 16: two uint32 of weights, one block scale per lane
constexpr int kDownRowsPerBlock = kWarps;
constexpr int kDownBlocks = MOE_HIDDEN / kDownRowsPerBlock;  // 512

static_assert(MOE_HIDDEN % kWarpCols == 0, "the gate/up column split must be exact");
static_assert(MOE_INTER % kWarpCols == 0, "the gate/up split must cover the down input width");
static_assert(kDownCols == 16, "the down kernel reads two uint32 of weights and four float4 of h per lane");
static_assert(MOE_INTER % kGateRowsPerBlock == 0 && MOE_HIDDEN % kDownRowsPerBlock == 0, "row split exact");

// NVFP4 decoders, same as fused_experts.cu keeps file-local: the checkpoint's scales stay in the finite
// positive range, and the 0x7F/0xFF NaN codes decode to 0 so a NaN never enters the dot.
__device__ __forceinline__ float2 e4m3x2_decode(std::uint16_t pair) {
  std::uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  const float lo = __low2float(v), hi = __high2float(v);
  return make_float2(lo != lo ? 0.0f : lo, hi != hi ? 0.0f : hi);
}

__device__ __forceinline__ float2 halves(std::uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

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
#define INGINE_SHARED_E2M1_TABLE(name) const std::uint32_t* const lut = nullptr
#else
__device__ __forceinline__ void deq4w(std::uint32_t word, const std::uint32_t* lut, std::uint32_t* out) {
  out[0] = lut[word & 0xFFu];
  out[1] = lut[(word >> 8) & 0xFFu];
  out[2] = lut[(word >> 16) & 0xFFu];
  out[3] = lut[word >> 24];
}

__device__ __forceinline__ void init_e2m1_lut(std::uint32_t* lut) {
  static constexpr std::uint32_t mag[8] = {0x0000u, 0x3800u, 0x3C00u, 0x3E00u,
                                           0x4000u, 0x4200u, 0x4400u, 0x4600u};
  for (int n = threadIdx.x; n < 256; n += kThreads) {
    const std::uint32_t lo = mag[n & 7] | ((n & 8u) << 12);
    const std::uint32_t hi = mag[(n >> 4) & 7] | ((n & 0x80u) << 8);
    lut[n] = lo | (hi << 16);
  }
}

#define INGINE_SHARED_E2M1_TABLE(name)                                                                   \
  __shared__ std::uint32_t name[256];                                                                    \
  init_e2m1_lut(name);                                                                                   \
  __syncthreads();                                                                                       \
  const std::uint32_t* const lut = name
#endif

// h[i] = silu(gate_scale2 * gate_i·x) * (up_scale2 * up_i·x), for all 512 shared-expert rows.
//
// The extra block (index kGateBlocks) owns the sigmoid gate scalar: it is a plain 2048-long dot against the
// same x, and the down step needs it as a factor. The up projection never leaves registers, so the caller's
// `scratch_up` is unused here and slot 0 of it carries the scalar across the two launches.
__global__ __launch_bounds__(kThreads) void shared_gate_up_kernel(const SharedExpertArgs args) {
  INGINE_SHARED_E2M1_TABLE(s_lut);
  const int lane = threadIdx.x & 31;
  if (blockIdx.x == kGateBlocks) {
    __shared__ float s_part[kWarps];
    const __nv_bfloat16* w = args.gate_weight;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < MOE_HIDDEN; i += kThreads) acc += __bfloat162float(w[i]) * args.x[i];
    acc = warp_sum(acc);
    if (lane == 0) s_part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int k = 0; k < kWarps; ++k) total += s_part[k];
      args.scratch_up[0] = 1.0f / (1.0f + __expf(-total));
    }
    return;
  }

  const int row = blockIdx.x * kGateRowsPerBlock + (threadIdx.x >> 5);
  const int col = lane * kColsPerLane;
  const std::uint8_t* gw = args.gate.w + std::size_t(row) * (MOE_HIDDEN / 2);
  const std::uint8_t* uw = args.up.w + std::size_t(row) * (MOE_HIDDEN / 2);
  const std::uint16_t* gs = reinterpret_cast<const std::uint16_t*>(args.gate.scales) +
                            std::size_t(row) * (MOE_HIDDEN / 32);
  const std::uint16_t* us = reinterpret_cast<const std::uint16_t*>(args.up.scales) +
                            std::size_t(row) * (MOE_HIDDEN / 32);
  float ag = 0.0f, au = 0.0f;
#pragma unroll
  for (int step = 0; step < kGateSteps; ++step) {
    const int c = col + step * kWarpCols;
    const float2* x = reinterpret_cast<const float2*>(args.x + c);
    std::uint32_t g4[4], u4[4];
    deq4w(*reinterpret_cast<const std::uint32_t*>(gw + c / 2), lut, g4);
    deq4w(*reinterpret_cast<const std::uint32_t*>(uw + c / 2), lut, u4);
    const float2 gsc = e4m3x2_decode(gs[c / 32]);
    const float2 usc = e4m3x2_decode(us[c / 32]);
    float pg = 0.0f, pu = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 gv = halves(g4[i]);
      const float2 uv = halves(u4[i]);
      pg += gv.x * x[i].x + gv.y * x[i].y;
      pu += uv.x * x[i].x + uv.y * x[i].y;
    }
    ag += pg * ((c / 16) & 1 ? gsc.y : gsc.x);
    au += pu * ((c / 16) & 1 ? usc.y : usc.x);
  }
  ag = warp_sum(ag) * args.gate.scale;
  au = warp_sum(au) * args.up.scale;
  if (lane == 0) args.scratch_h[row] = (ag / (1.0f + __expf(-ag))) * au;
}

// y[r] = sigmoid(gate_weight·x) * down_scale2 * down_r·h, with h kept in the scratch the step above wrote.
__global__ __launch_bounds__(kThreads) void shared_down_kernel(const SharedExpertArgs args) {
  INGINE_SHARED_E2M1_TABLE(s_lut);
  const float gate = args.scratch_up[0];
  const int row = blockIdx.x * kDownRowsPerBlock + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int col = lane * kDownCols;
  const std::uint8_t* drow = args.down.w + std::size_t(row) * (MOE_INTER / 2);
  const std::uint32_t* dw = reinterpret_cast<const std::uint32_t*>(drow + col / 2);
  const std::uint32_t dlo = dw[0], dhi = dw[1];
  const std::uint16_t* srow = reinterpret_cast<const std::uint16_t*>(
      args.down.scales + std::size_t(row) * (MOE_INTER / 16));
  const float4* hp = reinterpret_cast<const float4*>(args.scratch_h + col);
  const float4 h0 = hp[0], h1 = hp[1], h2 = hp[2], h3 = hp[3];
  const float hv[16] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w,
                        h2.x, h2.y, h2.z, h2.w, h3.x, h3.y, h3.z, h3.w};
  std::uint32_t d4[4], e4[4];
  deq4w(dlo, lut, d4);
  deq4w(dhi, lut, e4);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 dv = halves(d4[i]);
    const float2 ev = halves(e4[i]);
    acc += dv.x * hv[2 * i] + dv.y * hv[2 * i + 1];
    acc += ev.x * hv[8 + 2 * i] + ev.y * hv[8 + 2 * i + 1];
  }
  const float2 sc = e4m3x2_decode(srow[(col / 16) >> 1]);
  acc *= (col / 16) & 1 ? sc.y : sc.x;
  acc = warp_sum(acc);
  if (lane == 0) args.y[row] = gate * (acc * args.down.scale);
}

// The shape the fused path needs: the checkpoint's shared expert is one routed-expert-sized block. Dense
// arena slots are packed byte-tight, so the vector loads have to be checked before they are used: the weight
// rows take uint32 loads and the block scales uint16 ones.
bool fused_path(const SharedExpertArgs& a) {
  const bool shapes = a.gate.rows == MOE_INTER && a.gate.cols == MOE_HIDDEN && a.up.rows == MOE_INTER &&
                      a.up.cols == MOE_HIDDEN && a.down.rows == MOE_HIDDEN && a.down.cols == MOE_INTER;
  const auto bits = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p); };
  const bool weights = (bits(a.gate.w) | bits(a.up.w) | bits(a.down.w)) % 4 == 0;
  const bool scales =
      (bits(a.gate.scales) | bits(a.up.scales) | bits(a.down.scales) | bits(a.gate_weight)) % 2 == 0;
  const bool vectors = (bits(a.x) | bits(a.scratch_h)) % 16 == 0;
  return shapes && weights && scales && vectors;
}

cudaError_t shared_expert_generic(const SharedExpertArgs& args, cudaStream_t stream) {
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

}  // namespace

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
  if (!fused_path(args)) return shared_expert_generic(args, stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  shared_gate_up_kernel<<<kGateBlocks + 1, kThreads, 0, stream>>>(args);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  shared_down_kernel<<<kDownBlocks, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace ingine::arch::qwen35moe::kernels
