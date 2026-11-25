#include "arch/qwen35moe/kernels/moe/router.h"

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

// Phase 1: one warp per expert, each lane covering 64 of the 2048 columns, warp-reduced. The previous
// single-block version gave every thread a full 2048-long dot, which cost hundreds of microseconds per layer.
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

// Phase 2: top-k and the softmax over the selected logits, single thread over 256 floats.
__global__ void router_select_kernel(const float* logits, int* ids, float* weights) {
  bool used[ROUTER_EXPERTS] = {};
  int best[ROUTER_TOP_K];
  for (int k = 0; k < ROUTER_TOP_K; ++k) {
    int pick = -1;
    float best_value = -INFINITY;
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
  float m = -INFINITY;
  for (int k = 0; k < ROUTER_TOP_K; ++k) m = fmaxf(m, logits[best[k]]);
  float sum = 0.0f;
  for (int k = 0; k < ROUTER_TOP_K; ++k) {
    sum += expf(logits[best[k]] - m);
    ids[k] = best[k];
  }
  for (int k = 0; k < ROUTER_TOP_K; ++k) weights[k] = expf(logits[best[k]] - m) / sum;
}

}  // namespace

cudaError_t route(RouterArgs args, cudaStream_t stream) {
  if (!args.weight || !args.x || !args.ids || !args.weights) return cudaErrorInvalidValue;
  if (!args.logits) return cudaErrorInvalidValue;
  router_logits_kernel<<<(ROUTER_EXPERTS + kExpertsPerBlock - 1) / kExpertsPerBlock, kRouterThreads, 0, stream>>>(
      args.weight, args.x, args.logits);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  router_select_kernel<<<1, 1, 0, stream>>>(args.logits, args.ids, args.weights);
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
