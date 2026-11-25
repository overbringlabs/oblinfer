// Gates of the Qwen3.x gated delta-rule linear attention, per value head:
//   beta = sigmoid(b)
//   g    = -exp(A_log) * softplus(a + dt_bias)      softplus threshold 20, as in the HF reference
// Inputs `a` and `b` come from the pack's in_proj_a / in_proj_b; A_log and dt_bias are stored FP32.
#pragma once

#include <cuda_runtime.h>

namespace ingine::kernels::gdn {

struct GdnGateArgs {
  const float* a = nullptr;        // [heads]
  const float* b = nullptr;        // [heads]
  const float* dt_bias = nullptr;  // [heads]
  const float* a_log = nullptr;    // [heads]
  float* beta = nullptr;           // [heads]
  float* g = nullptr;              // [heads]
  int heads = 0;
};

cudaError_t gdn_gates(const GdnGateArgs& args, cudaStream_t stream);
void gdn_gates_cpu(const GdnGateArgs& args);

}  // namespace ingine::kernels::gdn
