// Shared expert of the Qwen3.6-35B-A3B MoE layers at decode width:
//   y = sigmoid(shared_expert_gate @ x) * down(silu(gate_expert(x)) * up_expert(x))
// The three projections are the checkpoint's NVFP4 tensors; the gate is a bf16 vector [hidden].
//
// The fused path is two launches: gate/up with the SiLU applied in the same warp that holds both dot
// results, then down with the sigmoid gate folded into the row scale. `scratch_up` is therefore not used
// for the up projection; slot 0 of it carries the gate scalar between the two launches. Shapes or pointer
// alignment the fused kernels cannot use (dense arena slots are packed byte-tight) fall back to the generic
// `quant::nvfp4_gemv` sequence.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/moe/moe_apply.h"

namespace ingine::arch::qwen35moe::kernels {

struct SharedExpertArgs {
  quant::Nvfp4GemvArgs gate;         // [MOE_INTER, MOE_HIDDEN]
  quant::Nvfp4GemvArgs up;           // [MOE_INTER, MOE_HIDDEN]
  quant::Nvfp4GemvArgs down;         // [MOE_HIDDEN, MOE_INTER]
  const __nv_bfloat16* gate_weight = nullptr;  // [MOE_HIDDEN], shared_expert_gate
  const float* x = nullptr;                    // [MOE_HIDDEN]
  float* y = nullptr;                          // [MOE_HIDDEN], overwritten
  float* scratch_h = nullptr;                  // [MOE_INTER]
  float* scratch_up = nullptr;                 // [MOE_INTER]
};

cudaError_t shared_expert(const SharedExpertArgs& args, cudaStream_t stream);

// y = sigmoid<w, x> * y, in place.
cudaError_t scale_sigmoid_dot(float* y, const __nv_bfloat16* w, const float* x, int n, cudaStream_t stream);
void scale_sigmoid_dot_cpu(float* y, const __nv_bfloat16* w, const float* x, int n);

}  // namespace ingine::arch::qwen35moe::kernels
