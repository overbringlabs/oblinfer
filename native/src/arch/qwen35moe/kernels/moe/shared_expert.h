// Shared expert of the Qwen3.6-35B-A3B MoE layers at decode width:
//   y = sigmoid(shared_expert_gate @ x) * down(silu(gate_expert(x)) * up_expert(x))
// The three projections are the checkpoint's NVFP4 tensors; the gate is a bf16 vector [hidden]. The scalar
// sigmoid is computed by one block that then scales the output in place, so the whole path is two launches
// after the GEMVs.
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
