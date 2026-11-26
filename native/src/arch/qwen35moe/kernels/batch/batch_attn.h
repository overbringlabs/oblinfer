// Verify-width full attention: n query positions against one KV cache, so the cache is streamed once for
// the whole block instead of once per position.
//
// The block shape is the shipped decode kernel's (kernels/attn/decode_attn.cu), because that shape is what
// measured 135 GB/s on this cache layout: eight warps per block, sixteen dims per lane, two position streams
// per warp, so a warp step advances sixteen positions and every load is a full 16-byte access. The earlier
// one-warp-per-split version of this file underfilled the card and reached 57 GB/s. What the batch version
// adds is n accumulator sets in registers: one cache row feeds all n queries, and each lane keeps
// (m, l, acc[16]) for every position of the block.
//
// Split-K is kept: grid (query head, position split), one BatchSplitState per (head, split) holding n online
// softmax states, and a combine kernel that merges the splits per (head, query).
//
// Causality for a verify block at absolute positions pos0..pos0+n-1: query t attends 0..pos0+t, so the last
// n-1 cache rows are masked out for the earlier queries. The check is per position and uniform across the
// half-warp, so it costs a predicate, not a divergence.
//
// A cache may be a band of a larger one: `pos_begin` is the global position of the cache's row 0 and
// `cache_rows` its length (the block-scale array of the block formats starts after that many rows' values).
//
// Same contract as the pack's other attention entry points: FP32 q / k_new / v_new in, cache bytes in `KvType`
// (BF16, q8_0 or q4_0, position-major [rows, ATTN_KV_HEADS, ATTN_HEAD_DIM]), no allocation, no synchronisation,
// and a head range so the two cards can each attend the KV heads they host.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/attn/rope_norm.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"

namespace ingine::arch::qwen35moe::kernels {

// One split's state for a whole verify block: n online softmax states over one position split.
struct BatchSplitState {
  float m[BATCH_MAX];
  float l[BATCH_MAX];
  float acc[BATCH_MAX * ATTN_HEAD_DIM];
};

struct BatchAttnArgs {
  const float* q = nullptr;        // [n, ATTN_HEADS, ATTN_HEAD_DIM], post q_norm and rotary
  const float* k_new = nullptr;    // [n, ATTN_KV_HEADS, ATTN_HEAD_DIM]
  const float* v_new = nullptr;    // [n, ATTN_KV_HEADS, ATTN_HEAD_DIM]
  void* k_cache = nullptr;         // [cache_rows, ATTN_KV_HEADS, ATTN_HEAD_DIM] in kv_type_k
  void* v_cache = nullptr;         // same, in kv_type_v
  KvType kv_type_k = KvType::Bf16;
  KvType kv_type_v = KvType::Bf16;
  float* out = nullptr;            // [n, ATTN_HEADS, ATTN_HEAD_DIM]
  int pos0 = 0;                    // absolute position of row 0 of the block
  int max_pos = 0;                 // capacity the block may reach (pos0 + n <= max_pos)
  int pos_begin = 0;               // global position of the cache's own row 0 (a band offset; 0 for a full cache)
  int cache_rows = -1;             // rows the cache holds; -1 means max_pos
  int head_begin = 0;
  int head_count = ATTN_HEADS;
  BatchSplitState* partial = nullptr;  // [head_count, splits]; required (this path is always split-K)
  int splits = 1;
  int n = 0;                       // 1..BATCH_MAX
  // The flash columns this launch owns. A two-band cache runs one launch per band, each over the columns it
  // holds whole, so a column never has its state written by two devices.
  int col_begin = 0;
  int col_end = -1;                // -1 means `splits`
  // Where batch_attention_band_states puts this launch's merge of its columns: two states per head, indexed by
  // band. `band_stride` is the number of states per head (2 for a two-band cache) and `band_slot` this band's.
  BatchSplitState* band_states = nullptr;
  int band_stride = 2;
  int band_slot = 0;
};

// Stores the n cache rows, then attends them: the split kernel over [col_begin, col_end) and the combine that
// merges every column into `out`. `partial` must hold head_count * splits states and splits >= 1. Returns
// cudaErrorInvalidValue for an out-of-range head window or a position past the cache.
cudaError_t batch_attention(const BatchAttnArgs& args, cudaStream_t stream);

// The two halves, for a cache that is split across cards: each card stores, runs its own columns and merges
// them (batch_attention_band_states), and the host joins the bands' states with one more combine over the
// `band_states` array at `band_stride` splits.
cudaError_t batch_attention_split(const BatchAttnArgs& args, cudaStream_t stream);
cudaError_t batch_attention_combine(const BatchAttnArgs& args, cudaStream_t stream);
cudaError_t batch_attention_band_states(const BatchAttnArgs& args, cudaStream_t stream);

// FP64 reference over the same cache bytes: stores the n rows the way the store kernel does, then attends
// every (query, head) pair. Writes `out` for the head window only.
void batch_attention_cpu(const BatchAttnArgs& args);

// Per-head RMSNorm and partial rotary for n positions: row t of q/k is normalised with position pos0 + t, so
// the whole block is one launch instead of one per position. Same arithmetic as kernels/attn/rope_norm.cu.
struct QkNormRopeRowsArgs {
  float* q = nullptr;                 // [n, ATTN_HEADS, ATTN_HEAD_DIM], updated in place
  float* k = nullptr;                 // [n, ATTN_KV_HEADS, ATTN_HEAD_DIM], updated in place
  const float* q_weight = nullptr;    // [ATTN_HEAD_DIM], the (1 + w) convention already applied
  const float* k_weight = nullptr;    // same
  int pos0 = 0;
  int n = 0;
  float eps = 1e-6f;
};
cudaError_t qk_norm_rope_rows(const QkNormRopeRowsArgs& args, cudaStream_t stream);

}  // namespace ingine::arch::qwen35moe::kernels
