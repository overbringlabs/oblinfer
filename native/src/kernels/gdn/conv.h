// Decode-step pieces of the Qwen3.x gated delta-rule linear attention that surround the recurrence in
// gdn.h: the causal conv1d (kernel 4) with its SiLU, and the gated RMSNorm that follows the delta rule.
//
// Reference order (llama.cpp src/models/qwen3next.cpp, HF Qwen3Next):
//   mix   = in_proj_qkv(h)                      [conv_dim = 2 * key_dim + value_dim]
//   y     = silu(conv1d(mix))                   causal, left-padded with kernel - 1 zeros
//   q|k|v = y split at key_dim and key_dim * 2   (q and k are l2-normalised inside gdn.h)
//   out_h = rms_norm(delta_rule_out_h, weight[value_dim]) * silu(z_h)
//
// Activations stay FP32 here, matching gdn.h's interface; a bf16 variant is a speed pass, not a semantics
// change. Both entry points allocate nothing and synchronise nothing, so both are graph-capturable.
#pragma once

#include <cuda_runtime.h>

namespace ingine::kernels::gdn {

inline constexpr int CONV_K = 4;  // linear_conv_kernel_dim in every Qwen3.x config

struct ConvStepArgs {
  const float* x = nullptr;       // [conv_dim] current projection output
  const float* weight = nullptr;  // [conv_dim, CONV_K], k fastest
  float* history = nullptr;       // [CONV_K - 1, conv_dim], time-major, oldest row first; updated in place
  float* y = nullptr;             // [conv_dim] silu(conv1d) output
  int conv_dim = 0;
};

// y = silu(sum_k weight[c][k] * x[t - CONV_K + 1 + k]); the history rolls forward one step.
cudaError_t conv_silu_step(const ConvStepArgs& args, cudaStream_t stream);

// Same maths on the host, for tests and the golden path.
void conv_silu_step_cpu(const ConvStepArgs& args);

struct GatedNormArgs {
  float* x = nullptr;             // [v_heads * v_dim], updated in place
  const float* z = nullptr;       // [v_heads * v_dim] gate projection
  const float* weight = nullptr;  // [v_dim], one RMSNorm weight shared by every value head
  int v_heads = 0;
  int v_dim = 0;
  float eps = 1e-6f;
};

// x_h = rms_norm(x_h, weight) * silu(z_h), per value head h.
cudaError_t gated_norm(GatedNormArgs args, cudaStream_t stream);
void gated_norm_cpu(GatedNormArgs args);

}  // namespace ingine::kernels::gdn
