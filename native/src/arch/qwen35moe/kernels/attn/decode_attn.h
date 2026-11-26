// Decode attention of the Qwen3.x full-attention layers (head_dim 256, 2 KV heads, 8 query heads per KV
// head). This is the per-layer cost that dominates decode at long context: at 262k it reads the whole KV
// cache of 10 layers, so the loop is chunked and every wave of cache reads is coalesced across the head
// dimension.
//
// The kernel boundary is after q/k norm and rotary: callers pass q [HEADS, HEAD_DIM] and the new k/v
// [KV_HEADS, HEAD_DIM] in FP32, and the cache stores them as BF16 in a position-major layout
// [max_pos, KV_HEADS, HEAD_DIM]. FP8 caches follow once this path is verified.
//
// One block per query head in the requested head range, one thread per output dimension. Within a chunk,
// one warp computes one position's score by a 32-lane reduction over the head dimension; the block then
// updates the running softmax state and accumulates p * v into the thread's output element.
//
// The head range lets the two cards each run the query heads whose KV rows live on them: a launch over
// [head_begin, head_begin + head_count) reads only the KV heads those query heads group onto and writes
// only those rows of `out`, so ranges on separate devices compose into one output tensor.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ingine::arch::qwen35moe::kernels {

inline constexpr int ATTN_HEADS = 16;
inline constexpr int ATTN_KV_HEADS = 2;
inline constexpr int ATTN_HEAD_DIM = 256;
inline constexpr int ATTN_CHUNK = 512;
inline constexpr int kMaxSplits = 64;

// KV cache element formats, selectable per side like llama-server's -ctk/-ctv. q8_0 is llama.cpp's block
// format: 32 int8 values behind one fp16 scale, 34 bytes per block, 8 blocks per 256-dim head.
enum class KvType : int { Bf16 = 0, Q8_0 = 1 };

inline constexpr int KV_BLOCK = 32;

// Q8_0 keeps two parallel arrays in one allocation: `values` int8 first, then one __half scale per 32-value
// block. The interleaved 34-byte block put the values at byte offset 2, which forbids 32-bit reads; splitting
// the arrays keeps both 16-byte aligned (the value count is a multiple of 16 for every supported context).
// The split kernel loads sixteen bytes per lane, so callers must hand it a 16-byte aligned base for either
// format; a sub-allocation has to keep that alignment.
inline std::uint64_t kv_cache_bytes(KvType type, int max_pos) {
  const std::uint64_t values = std::uint64_t(max_pos) * ATTN_KV_HEADS * ATTN_HEAD_DIM;
  return type == KvType::Bf16 ? values * 2 : values + (values / KV_BLOCK) * sizeof(__half);
}

__host__ __device__ inline std::int8_t* kv_q8_values(void* base) { return static_cast<std::int8_t*>(base); }
__host__ __device__ inline const __half* kv_q8_scales(const void* base, std::uint64_t values) {
  return reinterpret_cast<const __half*>(static_cast<const std::int8_t*>(base) + values);
}

// One position split's online-softmax state, written by a split block and merged by the combine kernel. A split
// block runs sixteen dims per lane, so each warp holds two position streams and folds them into one state in
// registers before the eight warps of the block merge through shared memory: one SplitState per block.
struct SplitState {
  float m;                      // running max of the split
  float l;                      // running denominator
  float acc[ATTN_HEAD_DIM];     // running numerator, unnormalised
};

struct DecodeAttnArgs {
  const float* q = nullptr;                     // [ATTN_HEADS, ATTN_HEAD_DIM], post q_norm and rotary
  const float* k_new = nullptr;                 // [ATTN_KV_HEADS, ATTN_HEAD_DIM], post k_norm and rotary
  const float* v_new = nullptr;                 // [ATTN_KV_HEADS, ATTN_HEAD_DIM]
  void* k_cache = nullptr;                      // [max_pos, ATTN_KV_HEADS, ATTN_HEAD_DIM] in `kv_type`
  void* v_cache = nullptr;                      // same layout
  KvType kv_type_k = KvType::Bf16;
  KvType kv_type_v = KvType::Bf16;
  float* out = nullptr;                         // [ATTN_HEADS, ATTN_HEAD_DIM], pre output gate and o_proj
  int pos = 0;                                  // position of k_new/v_new; attends over 0..pos
  int max_pos = 0;                              // cache capacity
  int head_begin = 0;                           // first query head this launch handles
  int head_count = ATTN_HEADS;                  // query heads this launch handles
  // Flash decoding: when `partial` is set the positions are split into `splits` ranges, each block writes its
  // SplitState to partial[(head - head_begin) * splits + split], and a combine kernel merges them. Leave null
  // to run the single-block-per-head path (used by the reference tests).
  SplitState* partial = nullptr;
  int splits = 0;
};

// Stores k_new/v_new at `pos`, then attends over 0..pos for the query heads [head_begin, head_begin +
// head_count), writing `out` at those absolute head offsets. The store always covers every KV head, so a
// partial range still leaves the cache consistent. Allocates nothing, synchronises nothing. Returns
// cudaErrorInvalidValue for a range outside [0, ATTN_HEADS) or an empty range, before launching anything.
cudaError_t decode_attention(const DecodeAttnArgs& args, cudaStream_t stream);

// FP64 reference over the same BF16 cache bytes: stores the new rows, then attends the head range. Used by
// tests; an invalid range is refused the same way, with no write to any buffer.
void decode_attention_cpu(const DecodeAttnArgs& args);

}  // namespace ingine::arch::qwen35moe::kernels
