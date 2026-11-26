#include "arch/qwen35moe/kernels/batch/batch_moe.h"

#include <cmath>
#include <cstddef>

#include "arch/qwen35moe/kernels/batch/batch_dequant.cuh"
#include "arch/qwen35moe/kernels/moe/router.h"
#include "kernels/quant/gemv.h"

namespace ingine::arch::qwen35moe::kernels {
namespace {

namespace bd = batch::detail;

// Same 128-thread shape as the single-token fused kernel: four warps, two output rows per warp, so one block
// covers eight rows of the projection and the gate/up grid is 64 groups per unique expert.
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = 2;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kGateUpGroups = MOE_INTER / kRowsPerBlock;  // 64
constexpr int kDownGroups = MOE_HIDDEN / kRowsPerBlock;   // 256
constexpr int kLanePairs = bd::kColsPerLane / 2;
constexpr int kGateUpSteps = MOE_HIDDEN / bd::kWarpCols;  // 8
constexpr int kDownSteps = MOE_INTER / bd::kWarpCols;     // 2

static_assert(MOE_INTER % kRowsPerBlock == 0 && MOE_HIDDEN % kRowsPerBlock == 0, "row split must be exact");
static_assert(MOE_HIDDEN % bd::kWarpCols == 0 && MOE_INTER % bd::kWarpCols == 0, "column split must be exact");

// One block, one thread over at most BATCH_PAIRS pairs: the union is tiny and first-seen order makes the
// plan deterministic, which a parallel atomic version would not be.
__global__ __launch_bounds__(kThreads) void batch_dedup_kernel(const BatchMoeArgs args) {
  __shared__ int slot_of[ROUTER_EXPERTS];
  for (int e = threadIdx.x; e < ROUTER_EXPERTS; e += kThreads) slot_of[e] = -1;
  for (int i = threadIdx.x; i < BATCH_PAIRS * BATCH_MAX; i += kThreads) args.plan.pair_of[i] = -1;
  for (int i = threadIdx.x; i < BATCH_PAIRS; i += kThreads) args.plan.rows[i] = 0;
  __syncthreads();

  const int total = args.n * MOE_TOP_K;
  if (threadIdx.x != 0) return;
  int unique = 0;
  for (int d = 0; d < total; ++d) {
    const int eid = args.ids[d];
    int u = slot_of[eid];
    if (u < 0) {
      u = unique++;
      slot_of[eid] = u;
      args.plan.unique_ids[u] = eid;
    }
    const int row = args.plan.rows[u]++;
    args.plan.pair_unique[d] = u;
    args.plan.pair_of[u * BATCH_MAX + row] = d;
    if (row >= BATCH_MAX) break;  // ids repeat more often than the plan can hold: a caller bug, not data
  }
  *args.plan.unique_count = unique;
}

// gate/up for one unique expert: the weight row is read once and dotted against every token that selected it.
template <int N>
__global__ __launch_bounds__(kThreads) void batch_gate_up_kernel(const BatchMoeArgs args) {
  INGINE_BATCH_E2M1_TABLE(s_lut, kThreads);
  const int u = blockIdx.x / kGateUpGroups;
  if (u >= *args.plan.unique_count) return;
  const int rows = args.plan.rows[u];
  if (rows == 0 || rows > N) return;
  const int eid = args.plan.unique_ids[u];
  const BatchExpertBlock& L = args.layers[eid];
  const int lane = threadIdx.x & 31;
  const int c0 = lane * bd::kColsPerLane;
  const int row0 = (blockIdx.x % kGateUpGroups) * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;

  int pair[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const int p = args.plan.pair_of[u * BATCH_MAX + j];
    pair[j] = p < 0 ? 0 : p;
  }

  if (L.stride_bytes == 0) {  // expert hosted on the other card: contribute zeros for its rows
    if (lane == 0) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        if (j >= rows) break;
        args.scratch_gate[std::size_t(pair[j]) * MOE_INTER + row0] = 0.0f;
        args.scratch_gate[std::size_t(pair[j]) * MOE_INTER + row0 + 1] = 0.0f;
        args.scratch_up[std::size_t(pair[j]) * MOE_INTER + row0] = 0.0f;
        args.scratch_up[std::size_t(pair[j]) * MOE_INTER + row0 + 1] = 0.0f;
      }
    }
    return;
  }

  const std::uint8_t* wb = L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes);
  const std::uint8_t* sb = L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes);
  const std::uint8_t* gw = wb + L.gate_w;
  const std::uint8_t* uw = wb + L.up_w;
  const std::uint8_t* gs = sb + L.gate_s;
  const std::uint8_t* us = sb + L.up_s;

  float acc_g[kRowsPerWarp][N], acc_u[kRowsPerWarp][N];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int j = 0; j < N; ++j) acc_g[r][j] = acc_u[r][j] = 0.0f;

  for (int step = 0; step < kGateUpSteps; ++step) {
    const int c = c0 + step * bd::kWarpCols;
    // The activation windows are the same for both weight rows: one staging per step serves the group. Rows
    // past this expert's group are held as zeros so the unrolled dot never reads an unset register.
    float2 xv[N][kLanePairs];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float2* xp =
          reinterpret_cast<const float2*>(args.x + std::size_t(pair[j] / MOE_TOP_K) * MOE_HIDDEN + c);
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) xv[j][p] = j < rows ? xp[p] : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int row = row0 + r;
      const std::uint32_t gword = *reinterpret_cast<const std::uint32_t*>(gw + std::size_t(row) * (MOE_HIDDEN / 2) + c / 2);
      const std::uint32_t uword = *reinterpret_cast<const std::uint32_t*>(uw + std::size_t(row) * (MOE_HIDDEN / 2) + c / 2);
      const float gsc = bd::e4m3_decode(gs[std::size_t(row) * (MOE_HIDDEN / 16) + c / 16]);
      const float usc = bd::e4m3_decode(us[std::size_t(row) * (MOE_HIDDEN / 16) + c / 16]);
      std::uint32_t gdec[4], udec[4];
      bd::deq4w(gword, lut, gdec);
      bd::deq4w(uword, lut, udec);
      float pg[N], pu[N];
#pragma unroll
      for (int j = 0; j < N; ++j) pg[j] = pu[j] = 0.0f;
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) {
        const float2 gv = bd::halves(gdec[p]);
        const float2 uv = bd::halves(udec[p]);
#pragma unroll
        for (int j = 0; j < N; ++j) {
          pg[j] += gv.x * xv[j][p].x + gv.y * xv[j][p].y;
          pu[j] += uv.x * xv[j][p].x + uv.y * xv[j][p].y;
        }
      }
#pragma unroll
      for (int j = 0; j < N; ++j) {
        acc_g[r][j] += gsc * pg[j];
        acc_u[r][j] += usc * pu[j];
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      if (j >= rows) break;
      const float tg = bd::warp_sum(acc_g[r][j]) * L.gate_scale2;
      const float tu = bd::warp_sum(acc_u[r][j]) * L.up_scale2;
      if (lane == 0) {
        const std::size_t at = std::size_t(pair[j]) * MOE_INTER + row0 + r;
        args.scratch_gate[at] = tg;
        args.scratch_up[at] = tu;
      }
    }
  }
}

// down for one unique expert: the weight row is read once for the group, the router weight folds in here.
template <int N>
__global__ __launch_bounds__(kThreads) void batch_down_kernel(const BatchMoeArgs args) {
  INGINE_BATCH_E2M1_TABLE(s_lut, kThreads);
  const int u = blockIdx.x / kDownGroups;
  if (u >= *args.plan.unique_count) return;
  const int rows = args.plan.rows[u];
  if (rows == 0 || rows > N) return;
  const int eid = args.plan.unique_ids[u];
  const BatchExpertBlock& L = args.layers[eid];
  const int lane = threadIdx.x & 31;
  const int c0 = lane * bd::kColsPerLane;
  const int row0 = (blockIdx.x % kDownGroups) * kRowsPerBlock + (threadIdx.x >> 5) * kRowsPerWarp;

  int pair[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const int p = args.plan.pair_of[u * BATCH_MAX + j];
    pair[j] = p < 0 ? 0 : p;
  }

  if (L.stride_bytes == 0) {
    if (lane == 0) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        if (j >= rows) break;
        args.scratch_down[std::size_t(pair[j]) * MOE_HIDDEN + row0] = 0.0f;
        args.scratch_down[std::size_t(pair[j]) * MOE_HIDDEN + row0 + 1] = 0.0f;
      }
    }
    return;
  }

  const std::uint8_t* wb = L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes);
  const std::uint8_t* sb = L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes);
  const std::uint8_t* dw = wb + L.down_w;
  const std::uint8_t* ds = sb + L.down_s;

  float acc[kRowsPerWarp][N];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int j = 0; j < N; ++j) acc[r][j] = 0.0f;

  for (int step = 0; step < kDownSteps; ++step) {
    const int c = c0 + step * bd::kWarpCols;
    float2 xv[N][kLanePairs];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float2* xp =
          reinterpret_cast<const float2*>(args.scratch_gate + std::size_t(pair[j]) * MOE_INTER + c);
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) xv[j][p] = j < rows ? xp[p] : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int row = row0 + r;
      const std::uint32_t word =
          *reinterpret_cast<const std::uint32_t*>(dw + std::size_t(row) * (MOE_INTER / 2) + c / 2);
      const float blk = bd::e4m3_decode(ds[std::size_t(row) * (MOE_INTER / 16) + c / 16]);
      std::uint32_t dec[4];
      bd::deq4w(word, lut, dec);
      float part[N];
#pragma unroll
      for (int j = 0; j < N; ++j) part[j] = 0.0f;
#pragma unroll
      for (int p = 0; p < kLanePairs; ++p) {
        const float2 wv = bd::halves(dec[p]);
#pragma unroll
        for (int j = 0; j < N; ++j) part[j] += wv.x * xv[j][p].x + wv.y * xv[j][p].y;
      }
#pragma unroll
      for (int j = 0; j < N; ++j) acc[r][j] += blk * part[j];
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      if (j >= rows) break;
      const float total = bd::warp_sum(acc[r][j]) * L.down_scale2;
      if (lane == 0) {
        const int d = pair[j];
        args.scratch_down[std::size_t(d) * MOE_HIDDEN + row0 + r] = total * args.weights[d];
      }
    }
  }
}

__global__ __launch_bounds__(256) void batch_reduce_kernel(const BatchMoeArgs args) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  const int total = args.n * MOE_HIDDEN;
  if (i >= total) return;
  const int t = i / MOE_HIDDEN;
  const int d = i - t * MOE_HIDDEN;
  float sum = 0.0f;
  const float* base = args.scratch_down + std::size_t(t * MOE_TOP_K) * MOE_HIDDEN + d;
#pragma unroll
  for (int k = 0; k < MOE_TOP_K; ++k) sum += base[std::size_t(k) * MOE_HIDDEN];
  args.y[i] = sum;
}

// One NVFP4 GEMV in FP64 over the checkpoint's bytes: the oracle for every expert projection here.
template <typename X>
void expert_row(const std::uint8_t* w, const std::uint8_t* s, double scale, const X* x, double* y, int rows,
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
}

}  // namespace

void batch_moe_plan_cpu(const int* ids, int n, BatchMoePlanHost* out) {
  int slot_of[ROUTER_EXPERTS];
  for (int e = 0; e < ROUTER_EXPERTS; ++e) slot_of[e] = -1;
  out->unique_count = 0;
  for (int u = 0; u < BATCH_PAIRS; ++u) {
    out->rows[u] = 0;
    for (int j = 0; j < BATCH_MAX; ++j) out->pair_of[u * BATCH_MAX + j] = -1;
  }
  const int total = n * MOE_TOP_K;
  for (int d = 0; d < total; ++d) {
    const int eid = ids[d];
    int u = slot_of[eid];
    if (u < 0) {
      u = out->unique_count++;
      slot_of[eid] = u;
      out->unique_ids[u] = eid;
    }
    const int row = out->rows[u]++;
    out->pair_unique[d] = u;
    out->pair_row[d] = row;
    out->pair_of[u * BATCH_MAX + row] = d;
  }
}

cudaError_t batch_moe_apply(const BatchMoeArgs& args, cudaStream_t stream) {
  if (!args.layers || !args.x || !args.ids || !args.weights || !args.y) return cudaErrorInvalidValue;
  if (!args.scratch_gate || !args.scratch_up || !args.scratch_down) return cudaErrorInvalidValue;
  if (!args.plan.unique_ids || !args.plan.unique_count || !args.plan.pair_unique || !args.plan.pair_of ||
      !args.plan.rows) {
    return cudaErrorInvalidValue;
  }
  if (args.n < 1 || args.n > BATCH_MAX) return cudaErrorInvalidValue;

  batch_dedup_kernel<<<1, kThreads, 0, stream>>>(args);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  const int gate_up_blocks = BATCH_PAIRS * kGateUpGroups;
  const int down_blocks = BATCH_PAIRS * kDownGroups;
  switch (args.n) {
    case 1: batch_gate_up_kernel<1><<<gate_up_blocks, kThreads, 0, stream>>>(args); break;
    case 2: batch_gate_up_kernel<2><<<gate_up_blocks, kThreads, 0, stream>>>(args); break;
    case 3: batch_gate_up_kernel<3><<<gate_up_blocks, kThreads, 0, stream>>>(args); break;
    default: batch_gate_up_kernel<4><<<gate_up_blocks, kThreads, 0, stream>>>(args); break;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = silu_mul_batch(args.scratch_gate, args.scratch_up, args.n * MOE_TOP_K * MOE_INTER, stream);
  if (err != cudaSuccess) return err;

  switch (args.n) {
    case 1: batch_down_kernel<1><<<down_blocks, kThreads, 0, stream>>>(args); break;
    case 2: batch_down_kernel<2><<<down_blocks, kThreads, 0, stream>>>(args); break;
    case 3: batch_down_kernel<3><<<down_blocks, kThreads, 0, stream>>>(args); break;
    default: batch_down_kernel<4><<<down_blocks, kThreads, 0, stream>>>(args); break;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  batch_reduce_kernel<<<(args.n * MOE_HIDDEN + 255) / 256, 256, 0, stream>>>(args);
  return cudaGetLastError();
}

void batch_moe_apply_cpu(const BatchMoeArgs& args, const BatchMoePlanHost& host) {
  double g[MOE_INTER], u[MOE_INTER], h[MOE_INTER], d[MOE_HIDDEN];
  for (int t = 0; t < args.n; ++t) {
    for (int i = 0; i < MOE_HIDDEN; ++i) args.y[std::size_t(t) * MOE_HIDDEN + i] = 0.0f;
    for (int k = 0; k < MOE_TOP_K; ++k) {
      const int p = t * MOE_TOP_K + k;
      const int eid = args.ids[p];
      const BatchExpertBlock& L = args.layers[eid];
      const float wk = args.weights[p];
      if (L.stride_bytes == 0) continue;  // other card, as the device sees it
      const std::uint8_t* wb = L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes);
      const std::uint8_t* sb = L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes);
      expert_row(wb + L.gate_w, sb + L.gate_s, double(L.gate_scale2),
                 args.x + std::size_t(t) * MOE_HIDDEN, g, MOE_INTER, MOE_HIDDEN);
      expert_row(wb + L.up_w, sb + L.up_s, double(L.up_scale2),
                 args.x + std::size_t(t) * MOE_HIDDEN, u, MOE_INTER, MOE_HIDDEN);
      for (int i = 0; i < MOE_INTER; ++i) h[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
      expert_row(wb + L.down_w, sb + L.down_s, double(L.down_scale2), h, d, MOE_HIDDEN, MOE_INTER);
      for (int i = 0; i < MOE_HIDDEN; ++i) {
        args.y[std::size_t(t) * MOE_HIDDEN + i] += float(wk * d[i]);
      }
    }
  }
  (void)host;
}

}  // namespace ingine::arch::qwen35moe::kernels
