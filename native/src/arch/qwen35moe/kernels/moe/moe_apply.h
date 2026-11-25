// Routed-expert path of the Qwen3.6-35B-A3B MoE layers at decode width, correctness first:
//   h_k = silu(gate_k @ x) * (up_k @ x)        per selected expert k
//   y   = sum_k w_k * (down_k @ h_k)           w_k from the router
//
// Each expert's three projections are the checkpoint's NVFP4 tensors (kernels/quant/gemv.h). This runs them
// one call at a time from the host, which costs a launch per step; the speed pass fuses the eight experts into
// one token-major kernel once the fused variant has an oracle. Nothing here synchronises the stream, so the
// whole sequence is still capture-friendly apart from the host loop over the eight ids.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/quant/gemv.h"

namespace ingine::arch::qwen35moe::kernels {

namespace quant = ingine::kernels::quant;

inline constexpr int MOE_TOP_K = 8;
inline constexpr int MOE_INTER = 512;
inline constexpr int MOE_HIDDEN = 2048;

struct MoeExpert {
  quant::Nvfp4GemvArgs gate;  // [MOE_INTER, MOE_HIDDEN]
  quant::Nvfp4GemvArgs up;    // [MOE_INTER, MOE_HIDDEN]
  quant::Nvfp4GemvArgs down;  // [MOE_HIDDEN, MOE_INTER]
};

struct MoeApplyArgs {
  const float* x = nullptr;                                     // [MOE_HIDDEN]
  const int* ids = nullptr;                                     // [MOE_TOP_K] host array
  const float* weights = nullptr;                               // [MOE_TOP_K] host array
  const MoeExpert* experts = nullptr;                           // host array indexed by expert id
  float* y = nullptr;                                           // [MOE_HIDDEN], overwritten
  float* scratch_h = nullptr;                                   // [MOE_INTER]
  float* scratch_up = nullptr;                                  // [MOE_INTER]
  float* scratch_down = nullptr;                                // [MOE_HIDDEN]
};

cudaError_t moe_apply(const MoeApplyArgs& args, cudaStream_t stream);

// out[i] = silu(g[i]) * u[i]
cudaError_t silu_mul(float* out, const float* g, const float* u, int n, cudaStream_t stream);
// y[i] += a * x[i]
cudaError_t axpy(float* y, const float* x, float a, int n, cudaStream_t stream);

void silu_mul_cpu(float* out, const float* g, const float* u, int n);

}  // namespace ingine::arch::qwen35moe::kernels
