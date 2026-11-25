// Router of the Qwen3.6-35B-A3B MoE layers at decode width: one token, 256 experts, top 8.
//
// The checkpoint keeps the router in bf16 (`mlp.gate.weight`, excluded from quantization). HF computes a
// softmax over all 256 logits, keeps the top 8 and renormalises them; that equals a softmax over the top-8
// logits, which is what this kernel does. Ties resolve to the lower expert id on both sides.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ingine::arch::qwen35moe::kernels {

inline constexpr int ROUTER_EXPERTS = 256;
inline constexpr int ROUTER_TOP_K = 8;
inline constexpr int ROUTER_HIDDEN = 2048;

struct RouterArgs {
  const __nv_bfloat16* weight = nullptr;  // [ROUTER_EXPERTS, ROUTER_HIDDEN]
  const float* x = nullptr;               // [ROUTER_HIDDEN]
  int* ids = nullptr;                     // [ROUTER_TOP_K], descending logit
  float* weights = nullptr;               // [ROUTER_TOP_K], softmax over the selected logits
};

cudaError_t route(RouterArgs args, cudaStream_t stream);
void route_cpu(RouterArgs args);

}  // namespace ingine::arch::qwen35moe::kernels
