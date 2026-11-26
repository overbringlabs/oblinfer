// Decode attention of the Qwen3.x full-attention layers (head_dim 256, 2 KV heads, 8 query heads per KV
// head). This is the per-layer cost that dominates decode at long context: at 262k it reads the whole KV
// cache of 10 layers, so the loop is chunked and every wave of cache reads is coalesced across the head
// dimension.
//
// The kernel boundary is after q/k norm and rotary: callers pass q [HEADS, HEAD_DIM] and the new k/v
// [KV_HEADS, HEAD_DIM] in FP32, and the cache stores them in a position-major layout
// [max_pos, KV_HEADS, HEAD_DIM] in one of the KvType formats below.
//
// One block per query head in the requested head range, one thread per output dimension. Within a chunk,
// one warp computes one position's score by a 32-lane reduction over the head dimension; the block then
// updates the running softmax state and accumulates p * v into the thread's output element.
//
// The head range lets the two cards each run the query heads whose KV rows live on them: a launch over
// [head_begin, head_begin + head_count) reads only the KV heads those query heads group onto and writes
// only those rows of `out`, so ranges on separate devices compose into one output tensor.
#pragma once

#include <cmath>
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
// format: 32 int8 values behind one fp16 scale, 8 blocks per 256-dim head. q4_0 is the same 32-value block at
// four bits: sixteen bytes of unsigned nibbles with llama.cpp's -8 offset, so a stored value reconstructs as
// `d * (q - 8)`, behind the same one fp16 scale per block.
enum class KvType : int { Bf16 = 0, Q8_0 = 1, Q4_0 = 2 };

inline constexpr int KV_BLOCK = 32;

// Every block format keeps two parallel arrays in one allocation: the values first, then one __half scale per
// 32-value block. The interleaved block llama.cpp ships put the scale between two runs of values, which
// forbids wide aligned reads; splitting the arrays keeps both 16-byte aligned (the value count is a multiple
// of 16 for every supported context). q4_0's value array is half as long as q8_0's, one byte per two values.
// The split kernel reads aligned 8- or 16-byte windows per lane, so callers must hand it a 16-byte aligned
// base for every format; a sub-allocation (a band) has to keep that alignment, which block-aligned rows do.
inline std::uint64_t kv_cache_bytes(KvType type, int max_pos) {
  const std::uint64_t values = std::uint64_t(max_pos) * ATTN_KV_HEADS * ATTN_HEAD_DIM;
  const std::uint64_t scales = (values / KV_BLOCK) * sizeof(__half);
  if (type == KvType::Bf16) return values * 2;
  if (type == KvType::Q4_0) return values / 2 + scales;
  return values + scales;
}

// Bytes the value array occupies for `values` values; the block scales start right after it.
__host__ __device__ inline std::uint64_t kv_value_bytes(KvType type, std::uint64_t values) {
  return type == KvType::Q4_0 ? values / 2 : values;
}

__host__ __device__ inline std::int8_t* kv_q8_values(void* base) { return static_cast<std::int8_t*>(base); }
__host__ __device__ inline const __half* kv_q8_scales(const void* base, std::uint64_t values) {
  return reinterpret_cast<const __half*>(static_cast<const std::int8_t*>(base) + values);
}
// q4_0: byte j of a block holds values 2j (low nibble) and 2j + 1 (high nibble), so a lane's sixteen dims are
// eight consecutive bytes and the pair of lanes that share one 32-value block read one aligned 16-byte
// window between them.
__host__ __device__ inline std::uint8_t* kv_q4_values(void* base) { return static_cast<std::uint8_t*>(base); }
__host__ __device__ inline std::uint8_t* kv_q4_values(const void* base) {
  return const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(base));
}
__host__ __device__ inline const __half* kv_q4_scales(const void* base, std::uint64_t values) {
  return reinterpret_cast<const __half*>(static_cast<const std::uint8_t*>(base) + values / 2);
}

// One value's unsigned nibble, llama.cpp's convention: q = clamp(rn(v / d) + 8, 0, 15) and the value reads
// back as d * (q - 8). Round-to-nearest-even is deliberate: it is what the store kernel's __float2int_rn
// does, so a host-encoded row is byte-identical to the row the device writes.
__host__ __device__ inline int kv_q4_nibble(float v, float d) {
  const int q = (d > 0.0f ? int(::rintf(v / d)) : 0) + 8;
  return q < 0 ? 0 : (q > 15 ? 15 : q);
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
  void* k_cache = nullptr;                      // [kv_cache_rows, ATTN_KV_HEADS, ATTN_HEAD_DIM] in `kv_type`
  void* v_cache = nullptr;                      // same layout
  KvType kv_type_k = KvType::Bf16;
  KvType kv_type_v = KvType::Bf16;
  float* out = nullptr;                         // [ATTN_HEADS, ATTN_HEAD_DIM], pre output gate and o_proj
  int pos = 0;                                  // position of k_new/v_new; attends over 0..pos
  int max_pos = 0;                              // cache capacity
  int head_begin = 0;                           // first query head this launch handles
  int head_count = ATTN_HEADS;                  // query heads this launch handles
  // Position band (see kv_bands.h): the launch reads global positions [pos_begin, min(pos_end, pos + 1)) and
  // k_cache/v_cache hold exactly the band's rows, band-local, so global position p is row p - pos_begin.
  // The defaults describe one un-banded cache: pos_end < 0 means "up to and including pos" and sizes the cache
  // by max_pos, which is what every pre-banding caller passes, so its results stay bit for bit unchanged.
  int pos_begin = 0;
  int pos_end = -1;
  // Flash decoding: when `partial` is set the positions are cut into `splits` columns of equal length, each
  // block writes its SplitState to partial[(head - head_begin) * splits + col], and a combine kernel merges
  // them. Leave null to run the single-block-per-head path (used by the reference tests).
  //
  // The column grid depends on pos and splits alone, never on which band runs it: `col_begin` is the first
  // column of that grid this launch covers, and the launch writes columns [col_begin, col_begin + band).
  // Several bands can therefore share one row per head without moving a boundary, and each column's state is
  // bit for bit what a single un-banded launch would have written.
  SplitState* partial = nullptr;
  int splits = 0;
  int col_begin = 0;
};

// Rows the cache pointers span: the band width when a band is set, otherwise the whole capacity. The scale
// array of a block format starts after the value array of exactly these rows, and q4_0 packs two values per
// byte, so every cache read derives its scale base from this count and kv_value_bytes of its type.
__host__ __device__ inline int kv_cache_rows(const DecodeAttnArgs& args) {
  return args.pos_end >= 0 ? args.pos_end - args.pos_begin : args.max_pos;
}

// Stores k_new/v_new at `pos`, then attends over 0..pos for the query heads [head_begin, head_begin +
// head_count), writing `out` at those absolute head offsets. The store always covers every KV head, so a
// partial range still leaves the cache consistent. A band window is honoured: the store goes to row
// `pos - pos_begin` and is skipped when `pos` falls outside the band. Allocates nothing, synchronises
// nothing. Returns cudaErrorInvalidValue for a range outside [0, ATTN_HEADS) or an empty range, or for a
// malformed band, before launching anything.
cudaError_t decode_attention(const DecodeAttnArgs& args, cudaStream_t stream);

// The three passes of the flash path, exposed separately so kv_bands.cu can run one band per device and merge
// on a single card. Each takes the same args as decode_attention and launches nothing else. `band_splits` is
// the number of SplitState columns this launch writes, starting at the column args.partial already points at;
// args.splits stays the row stride of that buffer and must be at least band_splits. The store is a no-op when
// args.pos lies outside [args.pos_begin, args.pos_end), so a caller can hand the same args to every band.
// decode_split never touches args.out, which lets a peer band run without a pointer into another device's
// memory. Allocations and cross-device traffic are the caller's job.
cudaError_t decode_kv_store(const DecodeAttnArgs& args, cudaStream_t stream);
cudaError_t decode_split(const DecodeAttnArgs& args, int band_splits, cudaStream_t stream);
cudaError_t decode_combine(const DecodeAttnArgs& args, cudaStream_t stream);

// FP64 reference over the same cache bytes: stores the new rows, then attends the head range. Used by tests;
// an invalid range is refused the same way, with no write to any buffer. It models one un-banded cache, so
// pos_begin/pos_end must stay at their defaults (a band's own bytes are a contiguous slice of the un-banded
// cache, which is how the banded tests feed it).
void decode_attention_cpu(const DecodeAttnArgs& args);

}  // namespace ingine::arch::qwen35moe::kernels
