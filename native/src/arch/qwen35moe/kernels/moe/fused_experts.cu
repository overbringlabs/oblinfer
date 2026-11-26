#include "arch/qwen35moe/kernels/moe/fused_experts.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

// 128-thread blocks keep the gate/up grid at 512 blocks (2 rows per warp) and the down grid at 512
// blocks (4 rows), the shape measured fastest at decode on sm_89 and the one that leaves the most
// per-SM block granularity when a pack spreads experts over both cards.
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

// A lane owns 8 adjacent columns per step: one uint32 carries the four e2m1 bytes and four float2 carry
// the activations. Block scales cover 16 columns, so two lanes share one scale byte; reading it as a
// uint16 decodes both halves with one cvt.rn.f16x2.e4m3x2.
constexpr int kColChunk = 8;
constexpr int kWarpCols = kColChunk * 32;
constexpr int kGateUpSteps = MOE_HIDDEN / kWarpCols;
constexpr int kDownCols = MOE_INTER / 32;  // a down warp crosses the whole row in one step per slot

// Gate and up share a launch: one warp takes the same rows of both projections and interleaves
// kGateUpRowsPerWarp rows, so each x window is loaded once and feeds every row of the group. The epilogue
// also applies the SiLU and the multiply, because both dot results for a row are already in this warp's
// registers: scratch_gate comes out holding h, the down step reads it straight away, and the third launch
// disappears. scratch_up still receives the up projection, which is what the reference path materialises.
constexpr int kGateUpRowsPerWarp = 2;
constexpr int kGateUpRowsPerBlock = kWarps * kGateUpRowsPerWarp;
constexpr int kGateUpGroups = MOE_INTER / kGateUpRowsPerBlock;
constexpr int kGateUpBlocks = MOE_TOP_K * kGateUpGroups;

// Down: one warp owns one output row and folds all eight slots into one register total, so the weighting
// and the reduction never leave the warp, y needs no zeroing pass, and the lane partials go through a
// single warp reduction per row instead of one per slot.
constexpr int kDownRowsPerBlock = kWarps;
constexpr int kDownBlocks = MOE_HIDDEN / kDownRowsPerBlock;

static_assert(MOE_HIDDEN % kWarpCols == 0 && MOE_INTER % kWarpCols == 0, "column chunk must divide both widths");
static_assert(MOE_INTER % kGateUpRowsPerBlock == 0, "gate/up row split must be exact");
static_assert(MOE_HIDDEN % kDownRowsPerBlock == 0, "down row split must be exact");
static_assert(kDownCols == 16, "the down kernel reads one uint2 of weights and four float4 of h per lane");

// NVFP4 decoders, mirroring the file-local ones in kernels/quant/gemv.cu, which its header does not export.
// e4m3 block scales: cvt.rn.f16x2.e4m3x2 is sm_89+ and every arch this file is built for. The checkpoint
// keeps scales in the finite positive range; the NaN codes 0x7F/0xFF decode to 0 below, matching the
// reference, because the hardware cvt would propagate NaN into the dot.
__device__ __forceinline__ float2 e4m3x2_decode(std::uint16_t pair) {
  std::uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  float lo = __low2float(v), hi = __high2float(v);
  return make_float2(lo != lo ? 0.0f : lo, hi != hi ? 0.0f : hi);
}

// e2m1 pairs: sm_100a/sm_120a decode a byte directly (F2FP.F16.E2M1.UNPACK_B with a byte selector, so
// one unpack feeds four decodes with no shifts). Other targets — plain sm_120, sm_89 probe builds — cannot
// compile that instruction and go through a shared-memory table holding the same exact f16 pairs.
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
// Declares the table pointer the decode helpers take; unused where the hardware instruction exists.
#define INGINE_E2M1_TABLE(name) const std::uint32_t* const lut = nullptr
#else
__device__ __forceinline__ void deq4w(std::uint32_t word, const std::uint32_t* lut, std::uint32_t* out) {
  out[0] = lut[word & 0xFFu];
  out[1] = lut[(word >> 8) & 0xFFu];
  out[2] = lut[(word >> 16) & 0xFFu];
  out[3] = lut[word >> 24];
}

// Byte `i` of `word` holds columns 2i (low nibble) and 2i+1 (high nibble); the table folds both nibbles
// of a byte into one f16x2, including the e2m1 sign bit.
__device__ __forceinline__ void init_e2m1_lut(std::uint32_t* lut) {
  static constexpr std::uint32_t mag[8] = {0x0000u, 0x3800u, 0x3C00u, 0x3E00u,
                                           0x4000u, 0x4200u, 0x4400u, 0x4600u};
  for (int n = threadIdx.x; n < 256; n += kThreads) {
    const std::uint32_t lo = mag[n & 7] | ((n & 8u) << 12);
    const std::uint32_t hi = mag[(n >> 4) & 7] | ((n & 0x80u) << 8);
    lut[n] = lo | (hi << 16);
  }
}

// Declares and fills the table on archs without the decode instruction.
#define INGINE_E2M1_TABLE(name)                                                                                \
  __shared__ std::uint32_t name[256];                                                                          \
  init_e2m1_lut(name);                                                                                         \
  __syncthreads();                                                                                             \
  const std::uint32_t* const lut = name
#endif

__device__ __forceinline__ float2 halves(std::uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Dot of one lane's 8-column chunk: the four packed uint32 hold eight e2m1 pairs against four float2.
__device__ __forceinline__ void chunk_pair_dot(std::uint32_t gc, std::uint32_t uc, const std::uint32_t* lut,
                                               const float2* x, float& pg, float& pu) {
  std::uint32_t gw[4], uw[4];
  deq4w(gc, lut, gw);
  deq4w(uc, lut, uw);
  pg = 0.0f;
  pu = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 gv = halves(gw[i]);
    const float2 uv = halves(uw[i]);
    pg += gv.x * x[i].x + gv.y * x[i].y;
    pu += uv.x * x[i].x + uv.y * x[i].y;
  }
}

struct DownSlot {
  const std::uint8_t* w;  // down rows of the expert named by ids[k]
  const std::uint8_t* s;  // its block scales
  const float* h;         // [MOE_INTER] of slot k
  float scale;            // weight_scale_2 * router weight
};

__global__ __launch_bounds__(kThreads, 6) void fused_gate_up_kernel(const FusedMoeArgs args) {
  INGINE_E2M1_TABLE(s_lut);
  const int k = blockIdx.x / kGateUpGroups;
  const int row0 = (blockIdx.x % kGateUpGroups) * kGateUpRowsPerBlock;

  // Uniform broadcast load: the expert id is resolved on the device and never reaches the host.
  const int eid = args.ids[k];
  const FusedExpertLayer& L = args.layers[eid];
  const int rowg = row0 + (threadIdx.x >> 5) * kGateUpRowsPerWarp;
  if (L.stride_bytes == 0) {  // this expert lives in another card's arena: contribute zeros for its rows
    float* zg = args.scratch_gate + std::size_t(k) * MOE_INTER;
    float* zu = args.scratch_up + std::size_t(k) * MOE_INTER;
    if ((threadIdx.x & 31) == 0) {
      for (int r = 0; r < kGateUpRowsPerWarp; ++r) {
        zg[rowg + r] = 0.0f;
        zu[rowg + r] = 0.0f;
      }
    }
    return;
  }
  const std::uint8_t* wb = L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes);
  const std::uint8_t* sb = L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes);
  const std::uint8_t* gw = wb + L.gate_weight_off;
  const std::uint8_t* uw = wb + L.up_weight_off;
  const std::uint8_t* gs = sb + L.gate_scale_off;
  const std::uint8_t* us = sb + L.up_scale_off;
  float* outg = args.scratch_gate + std::size_t(k) * MOE_INTER;
  float* outu = args.scratch_up + std::size_t(k) * MOE_INTER;

  const int lane = threadIdx.x & 31;
  const int col = lane * kColChunk;
  float part_g[kGateUpRowsPerWarp], part_u[kGateUpRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kGateUpRowsPerWarp; ++r) {
    part_g[r] = 0.0f;
    part_u[r] = 0.0f;
  }
#pragma unroll
  for (int step = 0; step < kGateUpSteps; ++step) {
    const int c = col + step * kWarpCols;
    // The x window and the scale-block half are the same for every row of the group: load once.
    const float2* x = reinterpret_cast<const float2*>(args.x + c);
#pragma unroll
    for (int r = 0; r < kGateUpRowsPerWarp; ++r) {
      const int row = rowg + r;
      const std::uint32_t gc = *reinterpret_cast<const std::uint32_t*>(gw + std::size_t(row) * (MOE_HIDDEN / 2) + c / 2);
      const std::uint32_t uc = *reinterpret_cast<const std::uint32_t*>(uw + std::size_t(row) * (MOE_HIDDEN / 2) + c / 2);
      const float2 gsc = e4m3x2_decode(reinterpret_cast<const std::uint16_t*>(gs + std::size_t(row) * (MOE_HIDDEN / 16))[c / 32]);
      const float2 usc = e4m3x2_decode(reinterpret_cast<const std::uint16_t*>(us + std::size_t(row) * (MOE_HIDDEN / 16))[c / 32]);
      float pg, pu;
      chunk_pair_dot(gc, uc, lut, x, pg, pu);
      part_g[r] += pg * ((c / 16) & 1 ? gsc.y : gsc.x);
      part_u[r] += pu * ((c / 16) & 1 ? usc.y : usc.x);
    }
  }
#pragma unroll
  for (int r = 0; r < kGateUpRowsPerWarp; ++r) {
    const float ag = warp_sum(part_g[r]) * L.gate_scale2;
    const float au = warp_sum(part_u[r]) * L.up_scale2;
    if (lane == 0) {
      // h = silu(gate) * up: the same expression and the same rounded operands the separate step used.
      outg[rowg + r] = (ag / (1.0f + __expf(-ag))) * au;
      outu[rowg + r] = au;
    }
  }
}

__global__ __launch_bounds__(kThreads) void fused_down_kernel(const FusedMoeArgs args) {
  INGINE_E2M1_TABLE(s_lut);
  __shared__ int s_ids[MOE_TOP_K];
  __shared__ DownSlot s_slot[MOE_TOP_K];
  if (threadIdx.x < MOE_TOP_K) s_ids[threadIdx.x] = args.ids[threadIdx.x];
  __syncthreads();
  if (threadIdx.x < MOE_TOP_K) {
    const int eid = s_ids[threadIdx.x];
    const FusedExpertLayer& L = args.layers[eid];
    const bool here = L.stride_bytes != 0;
    s_slot[threadIdx.x] = {here ? L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes) + L.down_weight_off
                               : nullptr,
                           here ? L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes) + L.down_scale_off
                                : nullptr,
                           args.scratch_gate + std::size_t(threadIdx.x) * MOE_INTER,
                           here ? L.down_scale2 * args.weights[threadIdx.x] : 0.0f};
  }
  __syncthreads();

  // One 16-column chunk per lane spans the whole 512-wide row, so each slot is one step: two weight
  // bytes, one scale half and four float4 of h. The lane keeps its piece of every slot's dot in one
  // register and the row reduces once at the end.
  const int row = blockIdx.x * kDownRowsPerBlock + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int col = lane * kDownCols;
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < MOE_TOP_K; ++k) {
    const DownSlot d = s_slot[k];
    if (d.w == nullptr) continue;  // expert hosted on the other card
    const uint2 code = *reinterpret_cast<const uint2*>(d.w + std::size_t(row) * (MOE_INTER / 2) + col / 2);
    const float2 sc = e4m3x2_decode(
        reinterpret_cast<const std::uint16_t*>(d.s)[std::size_t(row) * (MOE_INTER / 32) + ((col / 16) >> 1)]);
    const float scale = (col / 16) & 1 ? sc.y : sc.x;
    const float4* hp = reinterpret_cast<const float4*>(d.h + col);
    const float4 h0 = hp[0], h1 = hp[1], h2 = hp[2], h3 = hp[3];
    const float hv[16] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w,
                          h2.x, h2.y, h2.z, h2.w, h3.x, h3.y, h3.z, h3.w};
    std::uint32_t gw[4], uw[4];
    deq4w(code.x, lut, gw);
    deq4w(code.y, lut, uw);
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 gv = halves(gw[i]);
      const float2 uv = halves(uw[i]);
      acc += gv.x * hv[2 * i] + gv.y * hv[2 * i + 1];
      acc += uv.x * hv[8 + 2 * i] + uv.y * hv[8 + 2 * i + 1];
    }
    total = __fmaf_rn(d.scale * scale, acc, total);
  }
  total = warp_sum(total);
  if (lane == 0) args.y[row] = total;
}

}  // namespace

cudaError_t fused_moe_apply(const FusedMoeArgs& args, cudaStream_t stream) {
  if (!args.layers || !args.x || !args.ids || !args.weights || !args.y || !args.scratch_gate ||
      !args.scratch_up || !args.scratch_down) {
    return cudaErrorInvalidValue;
  }
  fused_gate_up_kernel<<<kGateUpBlocks, kThreads, 0, stream>>>(args);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  fused_down_kernel<<<kDownBlocks, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void fused_moe_apply_cpu(const FusedMoeArgs& args) {
  auto gemv = [](const std::uint8_t* w, const std::uint8_t* s, double scale, const float* x, double* y, int rows,
                 int cols) {
    for (int r = 0; r < rows; ++r) {
      const std::uint8_t* wr = w + std::size_t(r) * (cols / 2);
      const std::uint8_t* sr = s + std::size_t(r) * (cols / 16);
      double acc = 0.0;
      for (int c = 0; c < cols; ++c) {
        const std::uint8_t byte = wr[c / 2];
        const std::uint8_t code = (c % 2) ? std::uint8_t(byte >> 4) : std::uint8_t(byte & 0xF);
        acc += double(quant::e2m1_to_float(code)) * double(quant::fp8_e4m3_to_float(sr[c / 16])) * double(x[c]);
      }
      y[r] = acc * scale;
    }
  };

  double g[MOE_INTER], u[MOE_INTER];
  for (int k = 0; k < MOE_TOP_K; ++k) {
    const int eid = args.ids[k];
    const FusedExpertLayer& L = args.layers[eid];
    if (L.stride_bytes == 0) {  // hosted on the other card, as the CUDA kernel sees it
      for (int i = 0; i < MOE_INTER; ++i) {
        args.scratch_gate[std::size_t(k) * MOE_INTER + i] = 0.0f;
        args.scratch_up[std::size_t(k) * MOE_INTER + i] = 0.0f;
      }
      for (int i = 0; i < MOE_HIDDEN; ++i) args.scratch_down[std::size_t(k) * MOE_HIDDEN + i] = 0.0f;
      continue;
    }
    const std::uint8_t* wb = L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes);
    const std::uint8_t* sb = L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes);
    gemv(wb + L.gate_weight_off, sb + L.gate_scale_off, double(L.gate_scale2), args.x, g, MOE_INTER, MOE_HIDDEN);
    gemv(wb + L.up_weight_off, sb + L.up_scale_off, double(L.up_scale2), args.x, u, MOE_INTER, MOE_HIDDEN);

    float h[MOE_INTER];
    for (int i = 0; i < MOE_INTER; ++i) {
      const float v = float(g[i] / (1.0 + std::exp(-g[i])) * u[i]);
      h[i] = v;
      args.scratch_gate[std::size_t(k) * MOE_INTER + i] = v;
      args.scratch_up[std::size_t(k) * MOE_INTER + i] = float(u[i]);
    }
    // The CUDA down kernel keeps this step in registers; materialising it is what scratch_down is for.
    double d[MOE_HIDDEN];
    gemv(wb + L.down_weight_off, sb + L.down_scale_off, double(L.down_scale2), h, d, MOE_HIDDEN, MOE_INTER);
    const double wk = args.weights[k];
    for (int i = 0; i < MOE_HIDDEN; ++i) {
      args.scratch_down[std::size_t(k) * MOE_HIDDEN + i] = float(wk * d[i]);
    }
  }
  for (int i = 0; i < MOE_HIDDEN; ++i) {
    double acc = 0.0;
    for (int k = 0; k < MOE_TOP_K; ++k) acc += double(args.scratch_down[std::size_t(k) * MOE_HIDDEN + i]);
    args.y[i] = float(acc);
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
