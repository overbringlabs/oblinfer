// Gated-DeltaNet at verify width. Two separate questions live here, because the layer has two separate costs:
//
//   projections   — `in_proj_qkv`, `in_proj_z`, `out_proj` (FP8, 33.6 MB per layer) and the bf16 `in_proj_a/b`.
//                   They are ordinary GEMVs, so batch_gemv.h covers them: one weight read serves n positions.
//                   This is the term that dominates the GDN cost (about 1.0 GB of the 2.2 GB a token moves).
//   recurrence    — the FP32 state is [v_heads, 128, 128] = 2 MB per layer, 629 MB over the 30 linear layers,
//                   and every position must advance it in order, so it cannot be batched across positions. It
//                   can be *staged*: one block keeps its slice of the state in registers and runs the n updates
//                   back to back, which turns n reads and n writes of the whole state into one of each.
//
// gdn_recurrent_steps is that staged kernel: the block owns (value head, 32 value dims), each of its 128
// threads holds 32 state elements, and the n positions run in registers with two shared reductions each
// (the k-dot and the q-dot). Math per position is the same order as kernels/gdn/gdn.cu: decay, kv from the
// decayed state, rank-1 update, out from the updated state. The reduction tree differs (four row groups
// instead of one serial walk of 128), so results agree to fp32 rounding, not bitwise.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/gdn/gdn.h"

namespace ingine::arch::qwen35moe::kernels {

struct GdnStepsArgs {
  const float* q = nullptr;     // [n, GDN_K_HEADS, GDN_K_DIM]
  const float* k = nullptr;     // [n, GDN_K_HEADS, GDN_K_DIM]
  const float* v = nullptr;     // [n, v_heads, GDN_V_DIM]
  const float* beta = nullptr;  // [n, v_heads], sigmoid applied
  const float* g = nullptr;     // [n, v_heads], log decay
  float* state = nullptr;       // [v_heads, GDN_K_DIM, GDN_V_DIM] FP32, updated in place
  float* out = nullptr;         // [n, v_heads, GDN_V_DIM]
  int v_heads = 0;              // 32 for this pack
  int n = 0;                    // 1..BATCH_MAX
};

// Runs positions 0..n-1 in order, holding the state in registers between them. Allocates nothing,
// synchronises nothing.
cudaError_t gdn_recurrent_steps(const GdnStepsArgs& args, cudaStream_t stream);

// FP64 reference of the same loop: starts from `state`, updates it in place, writes `out` [n, v_heads,
// GDN_V_DIM]. Mirrors kernels/gdn/gdn.cu's reference, unrolled over positions.
void gdn_recurrent_steps_cpu(const GdnStepsArgs& args, double* state, double* out);

}  // namespace ingine::arch::qwen35moe::kernels
