#include "arch/qwen35moe/kernels/moe/router.h"

#include <cmath>
#include <cstddef>

namespace ingine::arch::qwen35moe::kernels {
namespace {

// One block of 256 threads: thread e owns expert e, so the router needs no cross-thread reduction for the
// logits. Selection and softmax are single-thread over 256 shared floats, which is cheaper than a parallel
// top-k at decode width.
__global__ __launch_bounds__(ROUTER_EXPERTS) void route_kernel(RouterArgs args) {
  __shared__ float logits[ROUTER_EXPERTS];
  const int e = threadIdx.x;

  const __nv_bfloat16* w = args.weight + std::size_t(e) * ROUTER_HIDDEN;
  float acc = 0.0f;
#pragma unroll 8
  for (int i = 0; i < ROUTER_HIDDEN; ++i) acc += __bfloat162float(w[i]) * args.x[i];
  logits[e] = acc;
  __syncthreads();

  if (e == 0) {
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
      used[pick] = true;
      best[k] = pick;
    }
    float m = -INFINITY;
    for (int k = 0; k < ROUTER_TOP_K; ++k) m = fmaxf(m, logits[best[k]]);
    float sum = 0.0f;
    for (int k = 0; k < ROUTER_TOP_K; ++k) {
      sum += expf(logits[best[k]] - m);
      args.ids[k] = best[k];
    }
    for (int k = 0; k < ROUTER_TOP_K; ++k) args.weights[k] = expf(logits[best[k]] - m) / sum;
  }
}

}  // namespace

cudaError_t route(RouterArgs args, cudaStream_t stream) {
  if (!args.weight || !args.x || !args.ids || !args.weights) return cudaErrorInvalidValue;
  route_kernel<<<1, ROUTER_EXPERTS, 0, stream>>>(args);
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
