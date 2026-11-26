// Verify-width full attention: n query positions against one KV cache, so the cache is streamed once for
// the whole block instead of once per position.
//
// Structure follows kernels/attn/decode_attn.cu's flash-decode split (that file is owned by another agent, so
// this is a separate instantiation of the same idea rather than a shared kernel):
//   grid (query head, position split); one warp per block walks the positions of its split, eight dims per
//   lane, so a warp covers a full 256-dim head row in one coalesced 8-byte-per-lane load and holds n online
//   softmax states (m, l, acc) in registers. One warp per block also means no shared-memory merge and no
//   barrier in the streaming loop: the block *is* the reduction unit.
//   A combine kernel merges the splits per (head, query) with a log-sum-exp over the split maxima.
//
// Causality for a verify block at absolute positions pos0..pos0+n-1: query t attends 0..pos0+t, so the last
// n-1 cache rows are masked out for the earlier queries. The check is per position and uniform across the
// warp, so it costs a predicate, not a divergence.
//
// Same contract as the pack's other attention entry points: FP32 q / k_new / v_new in, cache bytes in `KvType`
// (BF16 or q8_0, position-major [max_pos, ATTN_KV_HEADS, ATTN_HEAD_DIM]), no allocation, no synchronisation,
// and a head range so the two cards can each attend the KV heads they host.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"

namespace ingine::arch::qwen35moe::kernels {

// One warp's partial state for a whole verify block: n online softmax states over one position split.
struct BatchSplitState {
  float m[BATCH_MAX];
  float l[BATCH_MAX];
  float acc[BATCH_MAX * ATTN_HEAD_DIM];
};

struct BatchAttnArgs {
  const float* q = nullptr;        // [n, ATTN_HEADS, ATTN_HEAD_DIM], post q_norm and rotary
  const float* k_new = nullptr;    // [n, ATTN_KV_HEADS, ATTN_HEAD_DIM]
  const float* v_new = nullptr;    // [n, ATTN_KV_HEADS, ATTN_HEAD_DIM]
  void* k_cache = nullptr;         // [max_pos, ATTN_KV_HEADS, ATTN_HEAD_DIM] in kv_type_k
  void* v_cache = nullptr;         // same, in kv_type_v
  KvType kv_type_k = KvType::Bf16;
  KvType kv_type_v = KvType::Bf16;
  float* out = nullptr;            // [n, ATTN_HEADS, ATTN_HEAD_DIM]
  int pos0 = 0;                    // absolute position of row 0 of the block
  int max_pos = 0;                 // cache capacity
  int head_begin = 0;
  int head_count = ATTN_HEADS;
  BatchSplitState* partial = nullptr;  // [head_count, splits]; required (this path is always split-K)
  int splits = 1;
  int n = 0;                       // 1..BATCH_MAX
};

// Stores the n cache rows, then attends them. `partial` must hold head_count * splits states and
// splits >= 1. Returns cudaErrorInvalidValue for an out-of-range head window or a position past the cache.
cudaError_t batch_attention(const BatchAttnArgs& args, cudaStream_t stream);

// FP64 reference over the same cache bytes: stores the n rows the way the store kernel does, then attends
// every (query, head) pair. Writes `out` for the head window only.
void batch_attention_cpu(const BatchAttnArgs& args);

}  // namespace ingine::arch::qwen35moe::kernels
