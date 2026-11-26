// Routed-expert path of the MTP block. The checkpoint ships the drafter's experts unquantised, packed as
//   gate_up_proj [EXPERTS, 2 * MOE_INTER, MOE_HIDDEN]   gate rows first, then up rows
//   down_proj    [EXPERTS, MOE_HIDDEN, MOE_INTER]
// while kernels/moe is NVFP4-only. This is the BF16 sibling of fused_moe_apply: same math, same three-launch
// shape, ids and weights read on the device so nothing about the routing reaches the host.
//   h_k = silu(gate_k @ x) * (up_k @ x)        per router slot k
//   y   = sum_k w_k * (down_k @ h_k)           w_k from the router
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/moe/moe_apply.h"

namespace ingine::arch::qwen35moe::kernels {

struct Bf16MoeArgs {
  const __nv_bfloat16* gate_up = nullptr;  // [EXPERTS, 2 * MOE_INTER, MOE_HIDDEN]
  const __nv_bfloat16* down = nullptr;     // [EXPERTS, MOE_HIDDEN, MOE_INTER]
  const float* x = nullptr;                // [MOE_HIDDEN]
  const int* ids = nullptr;                // [MOE_TOP_K], read on the device
  const float* weights = nullptr;          // [MOE_TOP_K], read on the device, already normalised
  float* y = nullptr;                      // [MOE_HIDDEN], overwritten
  float* scratch_h = nullptr;              // [MOE_TOP_K * MOE_INTER], gate/up output of every slot
};

// Two launches: gate/up with the SiLU and the multiply folded in, then down with the weighted reduction.
cudaError_t bf16_moe_apply(const Bf16MoeArgs& args, cudaStream_t stream);

}  // namespace ingine::arch::qwen35moe::kernels
