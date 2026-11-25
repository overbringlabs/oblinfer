// Token-major routed-expert path: one launch per projection over all eight selected experts, replacing the
// per-expert launch loop of moe_apply.cu. Same math:
//   h_k = silu(gate_k @ x) * (up_k @ x)        per router slot k
//   y   = sum_k w_k * (down_k @ h_k)           w_k from the router
//
// The router's ids and weights are device arrays and are read inside the kernels, so nothing about the
// routing reaches the host and the whole path is three launches (gate+up, silu, down) with no synchronisation
// and no host loop: stream-capture friendly.
//
// Weights come from the packed expert block of model/placement.h: the artifacts of expert `e` live at
// `w_base + e * stride_bytes + <projection offset>` (and the same for `s_base`), so one array entry per
// expert id describes the whole block and a pack that spreads experts over arenas overrides the entry.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/moe/moe_apply.h"

namespace ingine::arch::qwen35moe::kernels {

// One expert's artifacts inside a packed block. Offsets are relative to the start of that expert's stride
// and must be 4-byte aligned: the kernels read the weights as uint32 (four e2m1 bytes) and the activations
// as float2.
struct FusedExpertLayer {
  const std::uint8_t* w_base = nullptr;  // start of this layer's packed expert block (weights)
  const std::uint8_t* s_base = nullptr;  // start of this layer's block scales (same stride layout)
  int stride_bytes = 0;                  // bytes per expert in w_base
  int scale_stride_bytes = 0;            // bytes per expert in s_base
  int gate_weight_off = 0, up_weight_off = 0, down_weight_off = 0;
  int gate_scale_off = 0, up_scale_off = 0, down_scale_off = 0;
  float gate_scale2 = 1.0f, up_scale2 = 1.0f, down_scale2 = 1.0f;  // per-tensor weight_scale_2
};

struct FusedMoeArgs {
  const FusedExpertLayer* layers = nullptr;  // one entry per expert id, indexed by args.ids
  const float* x = nullptr;                  // [MOE_HIDDEN]
  const int* ids = nullptr;                  // [MOE_TOP_K], read on the device
  const float* weights = nullptr;            // [MOE_TOP_K], read on the device, already normalised
  float* y = nullptr;                        // [MOE_HIDDEN], overwritten (never accumulated into)
  float* scratch_gate = nullptr;             // [MOE_TOP_K * MOE_INTER], becomes h after the silu step
  float* scratch_up = nullptr;               // [MOE_TOP_K * MOE_INTER]
  float* scratch_down = nullptr;             // [MOE_TOP_K * MOE_HIDDEN]; written by the reference only, the
                                             // CUDA down kernel reduces the slots in registers
};

// Three launches: gate/up for every slot, silu-mul, then down with the weighted reduction.
cudaError_t fused_moe_apply(const FusedMoeArgs& args, cudaStream_t stream);

// Same math in FP64 over the same bytes. Reads args.ids and args.weights as host arrays; `y` is overwritten.
void fused_moe_apply_cpu(const FusedMoeArgs& args);

}  // namespace ingine::arch::qwen35moe::kernels
