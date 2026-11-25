// Per-head RMSNorm and partial rotary for the Qwen3.x full-attention layers, at decode width.
//
// q and k arrive from their projections, are normalised per head over head_dim with their own learned
// weight, then rotated on the leading 64 of 256 dims (partial_rotary_factor 0.25, theta 1e7). Everything
// after the rotary window passes through untouched. Interleaved mrope uses three position streams; for text
// all three are the same position, which makes it the 1-D rotary implemented here.
//
// The output gate (sigmoid of the second half of q_proj, multiplied into the attention output) is a separate
// one-line kernel because o_proj consumes its result.
#pragma once

#include <cuda_runtime.h>

namespace ingine::arch::qwen35moe::kernels {

inline constexpr int NORM_HEADS = 16;      // q heads
inline constexpr int NORM_KV_HEADS = 2;    // k heads
inline constexpr int NORM_HEAD_DIM = 256;
inline constexpr int ROTARY_DIM = 64;      // head_dim * partial_rotary_factor
inline constexpr float ROTARY_THETA = 1e7f;

struct NormRopeArgs {
  float* q = nullptr;                 // [NORM_HEADS, NORM_HEAD_DIM], updated in place
  float* k = nullptr;                 // [NORM_KV_HEADS, NORM_HEAD_DIM], updated in place
  const float* q_weight = nullptr;    // [NORM_HEAD_DIM]
  const float* k_weight = nullptr;    // [NORM_HEAD_DIM]
  int pos = 0;
  float eps = 1e-6f;
};

cudaError_t qk_norm_rope(NormRopeArgs args, cudaStream_t stream);
void qk_norm_rope_cpu(NormRopeArgs args);

// q_proj emits, per head, [query(head_dim), gate(head_dim)]; the head loop of the reference chunks the last
// dimension. This splits the flat projection into a contiguous query buffer and a contiguous gate buffer.
struct SplitQGateArgs {
  const float* qgate = nullptr;  // [heads * 2 * head_dim]
  float* q = nullptr;            // [heads * head_dim]
  float* gate = nullptr;         // [heads * head_dim]
  int heads = 0;
  int head_dim = 0;
};

cudaError_t split_q_gate(const SplitQGateArgs& args, cudaStream_t stream);
void split_q_gate_cpu(const SplitQGateArgs& args);

struct OutputGateArgs {
  float* x = nullptr;            // [heads * head_dim], updated in place
  const float* gate = nullptr;   // same length, sigmoid applied here
  int count = 0;
};

cudaError_t output_gate(OutputGateArgs args, cudaStream_t stream);
void output_gate_cpu(OutputGateArgs args);

}  // namespace ingine::arch::qwen35moe::kernels
