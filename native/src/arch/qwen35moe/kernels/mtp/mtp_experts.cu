#include "arch/qwen35moe/kernels/mtp/mtp_experts.h"

#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

constexpr int kGateThreads = 128;
constexpr int kGateWarps = kGateThreads / 32;
constexpr int kLaneCols = 8;                       // one uint4 of bf16 weights per lane
constexpr int kWarpCols = kLaneCols * 32;          // 256
constexpr int kGateSteps = MOE_HIDDEN / kWarpCols;  // 8
constexpr int kGateRowBlocks = MOE_INTER / kGateWarps;

constexpr int kDownThreads = 256;
constexpr int kDownWarps = kDownThreads / 32;  // one warp per router slot
constexpr int kDownSteps = MOE_INTER / kWarpCols;
constexpr int kDownBlocks = MOE_HIDDEN / kDownWarps;

static_assert(MOE_HIDDEN % kWarpCols == 0, "the gate/up column split must be exact");
static_assert(MOE_INTER % kWarpCols == 0, "the down column split must be exact");
static_assert(MOE_INTER % kGateWarps == 0 && MOE_HIDDEN % kDownWarps == 0, "row split exact");
static_assert(kDownWarps == MOE_TOP_K, "the down kernel needs one warp per router slot");

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ void bf16x8(const uint4& w, float (&v)[kLaneCols]) {
  const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(words[i] & 0xFFFFu)));
    v[2 * i + 1] = __bfloat162float(__ushort_as_bfloat16(std::uint16_t(words[i] >> 16)));
  }
}

// h[slot * MOE_INTER + row] = silu(gate_row . x) * (up_row . x), one warp per row, both projections of the row in
// the same warp so the intermediate never touches memory.
__global__ __launch_bounds__(kGateThreads) void gate_up_kernel(const Bf16MoeArgs args) {
  const int slot = blockIdx.x / kGateRowBlocks;
  const int row = (blockIdx.x % kGateRowBlocks) * kGateWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int expert = args.ids[slot];
  const __nv_bfloat16* g = args.gate_up + (std::size_t(expert) * 2 * MOE_INTER + row) * MOE_HIDDEN;
  const __nv_bfloat16* u = g + std::size_t(MOE_INTER) * MOE_HIDDEN;

  float ag = 0.0f, au = 0.0f;
#pragma unroll
  for (int step = 0; step < kGateSteps; ++step) {
    const int c = lane * kLaneCols + step * kWarpCols;
    const uint4 gw = *reinterpret_cast<const uint4*>(g + c);
    const uint4 uw = *reinterpret_cast<const uint4*>(u + c);
    const float4 x0 = *reinterpret_cast<const float4*>(args.x + c);
    const float4 x1 = *reinterpret_cast<const float4*>(args.x + c + 4);
    const float xv[kLaneCols] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    float gv[kLaneCols], uv[kLaneCols];
    bf16x8(gw, gv);
    bf16x8(uw, uv);
#pragma unroll
    for (int i = 0; i < kLaneCols; ++i) {
      ag = fmaf(gv[i], xv[i], ag);
      au = fmaf(uv[i], xv[i], au);
    }
  }
  ag = warp_sum(ag);
  au = warp_sum(au);
  if (lane == 0) args.scratch_h[slot * MOE_INTER + row] = (ag / (1.0f + __expf(-ag))) * au;
}

// y[row] = sum_slot w_slot * (down_slot_row . h_slot). One warp owns one output row and walks all eight slots
// itself: every row the block owns is written, the reduction order is fixed, and the eight warps of a block read
// the same h_slot.
__global__ __launch_bounds__(kDownThreads) void down_kernel(const Bf16MoeArgs args) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kDownWarps + warp;

  float total = 0.0f;
#pragma unroll
  for (int slot = 0; slot < MOE_TOP_K; ++slot) {
    const float* h = args.scratch_h + std::size_t(slot) * MOE_INTER;
    const __nv_bfloat16* d = args.down + (std::size_t(args.ids[slot]) * MOE_HIDDEN + row) * MOE_INTER;
    float acc = 0.0f;
#pragma unroll
    for (int step = 0; step < kDownSteps; ++step) {
      const int c = lane * kLaneCols + step * kWarpCols;
      const float4 h0 = *reinterpret_cast<const float4*>(h + c);
      const float4 h1 = *reinterpret_cast<const float4*>(h + c + 4);
      const float hv[kLaneCols] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
      const uint4 dw = *reinterpret_cast<const uint4*>(d + c);
      float dv[kLaneCols];
      bf16x8(dw, dv);
#pragma unroll
      for (int i = 0; i < kLaneCols; ++i) acc = fmaf(dv[i], hv[i], acc);
    }
    total += warp_sum(acc) * args.weights[slot];
  }
  if (lane == 0) args.y[row] = total;
}

}  // namespace

cudaError_t bf16_moe_apply(const Bf16MoeArgs& args, cudaStream_t stream) {
  if (!args.gate_up || !args.down || !args.x || !args.ids || !args.weights || !args.y || !args.scratch_h) {
    return cudaErrorInvalidValue;
  }
  gate_up_kernel<<<MOE_TOP_K * kGateRowBlocks, kGateThreads, 0, stream>>>(args);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  down_kernel<<<kDownBlocks, kDownThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace ingine::arch::qwen35moe::kernels
