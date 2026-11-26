#include "arch/qwen35moe/kernels/moe/router.h"

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

// Phase 1: one warp per expert, each lane covering 64 of the 2048 columns, warp-reduced. The previous
// single-block version gave every thread a full 2048-long dot, which cost hundreds of microseconds per layer.
// The lane-stride 32 column order is what the FP64 reference and the llama.cpp anchor agree with; a 16-byte
// vector load was tried and measured 3.8 us -> 2.6 us per launch (about 0.7% of the token) but regroups the
// sum, so it stays out.
constexpr int kRouterThreads = 256;
constexpr int kExpertsPerBlock = kRouterThreads / 32;

__global__ __launch_bounds__(kRouterThreads) void router_logits_kernel(const __nv_bfloat16* w, const float* x,
                                                                      float* logits) {
  const int expert = blockIdx.x * kExpertsPerBlock + (threadIdx.x >> 5);
  if (expert >= ROUTER_EXPERTS) return;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* row = w + std::size_t(expert) * ROUTER_HIDDEN;
  float acc = 0.0f;
  for (int i = lane; i < ROUTER_HIDDEN; i += 32) acc += __bfloat162float(row[i]) * x[i];
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) logits[expert] = acc;
}

// `a` is chosen before `b`: higher logit wins, an equal logit falls to the lower expert id, and a NaN key
// loses to everything (the caller maps NaN to -INFINITY). Same total order as the ascending scan with a
// strict `>` that the reference and the previous single-thread kernel used.
__device__ __forceinline__ bool beats(float a, int ia, float b, int ib) {
  return a > b || (a == b && ia < ib);
}

// Phase 2: top-k over the 256 logits, one block with one thread per expert. Every thread finds its own rank
// — how many experts beat it — and the eight ranked below ROUTER_TOP_K write their slot, so the ids land in
// the same descending order the serial scan produced. That scan was one thread walking a 256-byte `used[]`
// array in local memory 2048 times with no overlap, which cost 95 us per layer; this is one coalesced read
// of the logits plus a shared-memory rank sweep.
//
// Thread 0 then reduces the eight selected logits on its own, in slot order, so `m`, `sum` and every weight
// come out bit-identical to the previous kernel.
__global__ __launch_bounds__(ROUTER_EXPERTS) void router_select_kernel(const float* logits, int* ids,
                                                                      float* weights) {
  __shared__ __align__(16) float s_key[ROUTER_EXPERTS];
  __shared__ float s_pick[ROUTER_TOP_K];
  const int e = threadIdx.x;
  const float v = logits[e];
  s_key[e] = v != v ? -INFINITY : v;
  __syncthreads();

  const float k = s_key[e];
  int rank = 0;
#pragma unroll 4
  for (int b = 0; b < ROUTER_EXPERTS; b += 4) {
    const float4 q = reinterpret_cast<const float4*>(s_key)[b >> 2];
    rank += beats(q.x, b + 0, k, e) + beats(q.y, b + 1, k, e) + beats(q.z, b + 2, k, e) +
           beats(q.w, b + 3, k, e);
  }
  if (rank < ROUTER_TOP_K) {
    ids[rank] = e;
    s_pick[rank] = v;
  }
  __syncthreads();

  if (e != 0) return;
  float m = -INFINITY;
  for (int r = 0; r < ROUTER_TOP_K; ++r) m = fmaxf(m, s_pick[r]);
  float sum = 0.0f;
  for (int r = 0; r < ROUTER_TOP_K; ++r) sum += expf(s_pick[r] - m);
  for (int r = 0; r < ROUTER_TOP_K; ++r) weights[r] = expf(s_pick[r] - m) / sum;
}

}  // namespace

cudaError_t route(RouterArgs args, cudaStream_t stream) {
  if (!args.weight || !args.x || !args.ids || !args.weights) return cudaErrorInvalidValue;
  if (!args.logits) return cudaErrorInvalidValue;
  const auto grid = (ROUTER_EXPERTS + kExpertsPerBlock - 1) / kExpertsPerBlock;
  router_logits_kernel<<<grid, kRouterThreads, 0, stream>>>(args.weight, args.x, args.logits);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  router_select_kernel<<<1, ROUTER_EXPERTS, 0, stream>>>(args.logits, args.ids, args.weights);
  return cudaGetLastError();
}

void route_cpu(RouterArgs args) {
  double logits[ROUTER_EXPERTS];
  for (int e = 0; e < ROUTER_EXPERTS; ++e) {
    const __nv_bfloat16* w = args.weight + std::size_t(e) * ROUTER_HIDDEN;
    double acc = 0.0;
    for (int i = 0; i < ROUTER_HIDDEN; ++i) acc += double(__bfloat162float(w[i])) * double(args.x[i]);
    logits[e] = acc;
  }
  bool used[ROUTER_EXPERTS] = {};
  int best[ROUTER_TOP_K];
  for (int k = 0; k < ROUTER_TOP_K; ++k) {
    int pick = -1;
    double best_value = -1e300;
    for (int i = 0; i < ROUTER_EXPERTS; ++i) {
      if (!used[i] && logits[i] > best_value) {
        best_value = logits[i];
        pick = i;
      }
    }
    if (pick < 0) {
      for (int i = 0; i < ROUTER_EXPERTS; ++i) {
        if (!used[i]) {
          pick = i;
          break;
        }
      }
    }
    used[pick] = true;
    best[k] = pick;
  }
  double m = -1e300;
  for (int k = 0; k < ROUTER_TOP_K; ++k) m = std::max(m, logits[best[k]]);
  double sum = 0.0;
  for (int k = 0; k < ROUTER_TOP_K; ++k) sum += std::exp(logits[best[k]] - m);
  for (int k = 0; k < ROUTER_TOP_K; ++k) {
    args.ids[k] = best[k];
    args.weights[k] = float(std::exp(logits[best[k]] - m) / sum);
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
