#include "arch/qwen35moe/kernels/moe/fused_experts.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// A lane owns 8 adjacent columns per step: one uint32 carries the four e2m1 bytes and four float2 carry the
// activations. Block scales cover 16 columns, so two lanes share one scale byte (a broadcast).
constexpr int kColChunk = 8;
constexpr int kWarpCols = kColChunk * 32;
constexpr int kGateUpSteps = MOE_HIDDEN / kWarpCols;
constexpr int kDownSteps = MOE_INTER / kWarpCols;

// Gate and up share a launch: one warp takes the same row of both projections, so each x window is loaded
// once and used by both dot products.
constexpr int kRowsPerBlock = 8;
constexpr int kGateUpGroups = MOE_INTER / kRowsPerBlock;
constexpr int kGateUpBlocks = MOE_TOP_K * kGateUpGroups;
constexpr int kGateUpRowsPerWarp = kRowsPerBlock / kWarps;

// Down: one warp owns one output row and folds all eight slots in registers, so the weighting and the
// reduction never leave the warp and y needs no zeroing pass.
constexpr int kDownBlocks = MOE_HIDDEN / kWarps;

static_assert(MOE_HIDDEN % kWarpCols == 0 && MOE_INTER % kWarpCols == 0, "column chunk must divide both widths");
static_assert(MOE_INTER % kRowsPerBlock == 0 && kRowsPerBlock % kWarps == 0, "gate/up row split must be exact");
static_assert(MOE_HIDDEN % kWarps == 0, "down row split must be exact");

// NVFP4 decoders, mirroring the file-local ones in kernels/quant/gemv.cu, which its header does not export.
__device__ __forceinline__ float e4m3_decode(std::uint8_t code) {
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

// Byte `i` of `word` holds columns 2i (low nibble) and 2i+1 (high nibble).
__device__ __forceinline__ std::uint32_t f16x2_from_e2m1x2(std::uint32_t word, int i) {
  const std::uint32_t w = word >> (8 * i);  // the byte lands in b0 of the little-endian word
  std::uint32_t r;
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %1;\ncvt.rn.f16x2.e2m1x2 %0, b0;\n}"
      : "=r"(r)
      : "r"(w));
  return r;
}

__device__ __forceinline__ float2 halves(std::uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Dot of one lane's 8-column chunk against the four e2m1 bytes of `code`.
__device__ __forceinline__ float chunk_dot(std::uint32_t code, const float2* x) {
  float part = 0.0f;
#pragma unroll
  for (int i = 0; i < kColChunk / 2; ++i) {
    const float2 v = halves(f16x2_from_e2m1x2(code, i));
    const float2 xv = x[i];
    part += v.x * xv.x + v.y * xv.y;
  }
  return part;
}

struct DownSlot {
  const std::uint8_t* w;  // down rows of the expert named by ids[k]
  const std::uint8_t* s;  // its block scales
  const float* h;         // [MOE_INTER] of slot k
  float scale;            // weight_scale_2 * router weight
};

__global__ __launch_bounds__(kThreads) void fused_gate_up_kernel(const FusedMoeArgs args) {
  const int k = blockIdx.x / kGateUpGroups;
  const int row0 = (blockIdx.x % kGateUpGroups) * kRowsPerBlock;

  // Uniform broadcast load: the expert id is resolved on the device and never reaches the host.
  const int eid = args.ids[k];
  const FusedExpertLayer& L = args.layers[eid];
  if (L.stride_bytes == 0) {  // this expert lives in another card's arena: contribute zeros for its rows
    float* zg = args.scratch_gate + std::size_t(k) * MOE_INTER;
    float* zu = args.scratch_up + std::size_t(k) * MOE_INTER;
    for (int r = row0 + (threadIdx.x >> 5) * kGateUpRowsPerWarp, i = 0; i < kGateUpRowsPerWarp; ++r, ++i) {
      if (r < MOE_INTER && threadIdx.x % 32 == 0) {
        zg[r] = 0.0f;
        zu[r] = 0.0f;
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
#pragma unroll
  for (int r = 0; r < kGateUpRowsPerWarp; ++r) {
    const int row = row0 + (threadIdx.x >> 5) * kGateUpRowsPerWarp + r;
    const std::uint8_t* gwr = gw + std::size_t(row) * (MOE_HIDDEN / 2);
    const std::uint8_t* uwr = uw + std::size_t(row) * (MOE_HIDDEN / 2);
    const std::uint8_t* gsr = gs + std::size_t(row) * (MOE_HIDDEN / 16);
    const std::uint8_t* usr = us + std::size_t(row) * (MOE_HIDDEN / 16);
    float ag = 0.0f, au = 0.0f;
#pragma unroll
    for (int step = 0; step < kGateUpSteps; ++step) {
      const int c = col + step * kWarpCols;
      const float2* x = reinterpret_cast<const float2*>(args.x + c);
      const std::uint32_t gc = *reinterpret_cast<const std::uint32_t*>(gwr + c / 2);
      const std::uint32_t uc = *reinterpret_cast<const std::uint32_t*>(uwr + c / 2);
      ag += chunk_dot(gc, x) * e4m3_decode(gsr[c / 16]);
      au += chunk_dot(uc, x) * e4m3_decode(usr[c / 16]);
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
      outg[row] = ag * L.gate_scale2;
      outu[row] = au * L.up_scale2;
    }
  }
}

// h_k = silu(gate_k) * up_k for every slot, in place over the gate scratch.
__global__ __launch_bounds__(kThreads) void fused_silu_mul_kernel(float* h, const float* u, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = h[i];
  h[i] = (v / (1.0f + __expf(-v))) * u[i];
}

__global__ __launch_bounds__(kThreads) void fused_down_kernel(const FusedMoeArgs args) {
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

  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int col = lane * kColChunk;
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < MOE_TOP_K; ++k) {
    const DownSlot d = s_slot[k];
    if (d.w == nullptr) continue;  // expert hosted on the other card
    const std::uint8_t* wr = d.w + std::size_t(row) * (MOE_INTER / 2);
    const std::uint8_t* sr = d.s + std::size_t(row) * (MOE_INTER / 16);
    float acc = 0.0f;
#pragma unroll
    for (int step = 0; step < kDownSteps; ++step) {
      const int c = col + step * kWarpCols;
      const std::uint32_t code = *reinterpret_cast<const std::uint32_t*>(wr + c / 2);
      acc += chunk_dot(code, reinterpret_cast<const float2*>(d.h + c)) * e4m3_decode(sr[c / 16]);
    }
    total += d.scale * warp_sum(acc);
  }
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

  fused_silu_mul_kernel<<<(MOE_TOP_K * MOE_INTER + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      args.scratch_gate, args.scratch_up, MOE_TOP_K * MOE_INTER);
  err = cudaGetLastError();
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
