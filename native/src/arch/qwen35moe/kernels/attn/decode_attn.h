// Decode attention of the Qwen3.x full-attention layers (head_dim 256, 2 KV heads, 8 query heads per KV
// head). This is the per-layer cost that dominates decode at long context: at 262k it reads the whole KV
// cache of 10 layers, so the loop is chunked and every wave of cache reads is coalesced across the head
// dimension.
//
// The kernel boundary is after q/k norm and rotary: callers pass q [HEADS, HEAD_DIM] and the new k/v
// [KV_HEADS, HEAD_DIM] in FP32, and the cache stores them as BF16 in a position-major layout
// [max_pos, KV_HEADS, HEAD_DIM]. FP8 caches follow once this path is verified.
//
// One block per query head, one thread per output dimension. Within a chunk, one warp computes one
// position's score by a 32-lane reduction over the head dimension; the block then updates the running
// softmax state and accumulates p * v into the thread's output element.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ingine::arch::qwen35moe::kernels {

inline constexpr int ATTN_HEADS = 16;
inline constexpr int ATTN_KV_HEADS = 2;
inline constexpr int ATTN_HEAD_DIM = 256;
inline constexpr int ATTN_CHUNK = 512;

struct DecodeAttnArgs {
  const float* q = nullptr;                     // [ATTN_HEADS, ATTN_HEAD_DIM], post q_norm and rotary
  const float* k_new = nullptr;                 // [ATTN_KV_HEADS, ATTN_HEAD_DIM], post k_norm and rotary
  const float* v_new = nullptr;                 // [ATTN_KV_HEADS, ATTN_HEAD_DIM]
  __nv_bfloat16* k_cache = nullptr;             // [max_pos, ATTN_KV_HEADS, ATTN_HEAD_DIM]
  __nv_bfloat16* v_cache = nullptr;             // same shape
  float* out = nullptr;                         // [ATTN_HEADS, ATTN_HEAD_DIM], pre output gate and o_proj
  int pos = 0;                                  // position of k_new/v_new; attends over 0..pos
  int max_pos = 0;                              // cache capacity
};

// Stores k_new/v_new at `pos`, then attends over 0..pos. Allocates nothing, synchronises nothing.
cudaError_t decode_attention(const DecodeAttnArgs& args, cudaStream_t stream);

// FP64 reference over the same BF16 cache bytes: stores the new rows, then attends. Used by tests.
void decode_attention_cpu(const DecodeAttnArgs& args);

}  // namespace ingine::arch::qwen35moe::kernels
