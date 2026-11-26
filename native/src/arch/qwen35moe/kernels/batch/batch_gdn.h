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

// The three per-position passes that surround the recurrence, batched for the same reason: one launch for the
// whole block instead of one per position. Each reproduces the single-token kernel of kernels/gdn/ exactly,
// position by position, so a row of the batched call equals the single-row kernel's output.

// The causal conv1d with its SiLU over n positions: the history rolls forward n times inside the block, so one
// launch advances it once per position and writes n output rows. Row t of `y` is what conv_silu_step produces
// for position t.
struct ConvStepsArgs {
  const float* x = nullptr;       // [n, conv_dim]
  const float* weight = nullptr;  // [conv_dim, CONV_K], k fastest
  float* history = nullptr;       // [CONV_K - 1, conv_dim], updated to the state after position n - 1
  float* y = nullptr;             // [n, conv_dim]
  int conv_dim = 0;
  int n = 0;
};
cudaError_t conv_silu_steps(const ConvStepsArgs& args, cudaStream_t stream);

// Contiguous q/k/v out of the conv output, whose rows carry q, k and v back to back (or in the per-key-head
// grouping some exporters keep), so the recurrence reads [rows, key_heads * key_dim] without a stride.
struct GatherQkvRowsArgs {
  const float* qkv = nullptr;  // [rows, 2 * key_heads * key_dim + value_heads * value_dim]
  float* q = nullptr;          // [rows, key_heads * key_dim]
  float* k = nullptr;          // same
  float* v = nullptr;          // [rows, value_heads * value_dim]
  int key_heads = 0;
  int key_dim = 0;
  int value_heads = 0;
  int value_dim = 0;
  int rows = 0;
  bool grouped = false;  // per key head [q, k, v] instead of [all q | all k | all v]
};
cudaError_t gather_qkv_rows(const GatherQkvRowsArgs& args, cudaStream_t stream);

// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias), for `rows` positions of `heads` value heads.
struct GdnGatesRowsArgs {
  const float* a = nullptr;        // [rows, heads]
  const float* b = nullptr;        // [rows, heads]
  const float* dt_bias = nullptr;  // [heads]
  const float* a_log = nullptr;    // [heads]
  float* beta = nullptr;           // [rows, heads]
  float* g = nullptr;              // [rows, heads]
  int heads = 0;
  int rows = 0;
};
cudaError_t gdn_gates_rows(const GdnGatesRowsArgs& args, cudaStream_t stream);

// x_row = rms_norm(x_row, weight) * silu(z_row) per value head, one block per (position, head).
struct GatedNormRowsArgs {
  float* x = nullptr;             // [rows, heads * v_dim], updated in place
  const float* z = nullptr;       // [rows, heads * v_dim]
  const float* weight = nullptr;  // [v_dim], shared by every head
  int heads = 0;
  int v_dim = 0;                  // 128, one thread per dim
  int rows = 0;
  float eps = 1e-6f;
};
cudaError_t gated_norm_rows(const GatedNormRowsArgs& args, cudaStream_t stream);

// FP64 reference of the same loop: starts from `state`, updates it in place, writes `out` [n, v_heads,
// GDN_V_DIM]. Mirrors kernels/gdn/gdn.cu's reference, unrolled over positions.
void gdn_recurrent_steps_cpu(const GdnStepsArgs& args, double* state, double* out);

}  // namespace ingine::arch::qwen35moe::kernels
