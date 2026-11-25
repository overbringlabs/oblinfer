// Decode step (S = 1) of the Qwen3.8 gated delta-rule linear attention. Math: docs/arch/gdn-recurrent.md.
//
// Per value head h, with k head kh = h / (v_heads / GDN_K_HEADS) (the repeat_interleave of the grouped heads):
//
//   q_hat = l2norm(q[kh]) / sqrt(GDN_K_DIM),  k_hat = l2norm(k[kh]),  l2norm(x) = x * rsqrt(sum(x^2) + 1e-6)
//   S_h  *= exp(g[h])
//   kv_h  = S_h^T k_hat
//   delta = (v_h - kv_h) * beta[h]
//   S_h  += k_hat delta^T
//   out_h = S_h^T q_hat
//
// `state` is FP32 [v_heads, GDN_K_DIM, GDN_V_DIM] and updated in place. conv1d, the gated RMSNorm over `out`
// and out_proj are not part of this step.
#pragma once

#include <cuda_runtime.h>

namespace ingine::arch::qwen38::kernels {

// 27B/9B/4B text_config: linear_num_key_heads 16, linear_key_head_dim 128, linear_value_head_dim 128.
inline constexpr int GDN_K_HEADS = 16;
inline constexpr int GDN_K_DIM = 128;
inline constexpr int GDN_V_DIM = 128;

struct GdnStepArgs {
  const float* q = nullptr;     // [GDN_K_HEADS, GDN_K_DIM]
  const float* k = nullptr;     // [GDN_K_HEADS, GDN_K_DIM]
  const float* v = nullptr;     // [v_heads, GDN_V_DIM]
  const float* beta = nullptr;  // [v_heads], sigmoid(b) already applied
  const float* g = nullptr;     // [v_heads], log decay: -exp(A_log) * softplus(a + dt_bias)
  float* state = nullptr;       // [v_heads, GDN_K_DIM, GDN_V_DIM], FP32, updated in place
  float* out = nullptr;         // [v_heads, GDN_V_DIM]
  int v_heads = 0;              // 48 (27B) or 32 (9B/4B); GDN_K_HEADS must divide it
};

// One decode step. The kernel is instantiated for 48 and 32 value heads only; any other v_heads returns
// cudaErrorInvalidValue. Allocates nothing and synchronises nothing, so it is CUDA-graph capturable.
cudaError_t gdn_recurrent_step(const GdnStepArgs& args, cudaStream_t stream);

// Host reference of the same loop in FP64, used by tests: reads q/k/v/beta/g from `args` (args.state and
// args.out are unused) and writes `state` [v_heads, GDN_K_DIM, GDN_V_DIM] and `out` [v_heads, GDN_V_DIM],
// both FP64.
void gdn_recurrent_step_cpu(const GdnStepArgs& args, double* state, double* out);

}  // namespace ingine::arch::qwen38::kernels
