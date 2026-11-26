// K/V cache of the drafter's single full-attention layer, in a selectable element format.
//
// decode_attn.cu ships BF16 and Q8_0 caches for the trunk's ten full layers and has no Q4_0 instantiation; the
// drafter's default cache is Q4_0, so this file adds that format on the *same* column grid and the *same*
// SplitState layout. A drafter's proposal only costs acceptance, not correctness, and at 262144 positions Q4_0
// is a quarter of the BF16 bytes: 75.5 MB per side against 268.4 MB.
//
// Q4_0 layout, per side, in one allocation like the Q8_0 cache: the values array first, then one __half scale
// per 32-value block, so both arrays stay 16-byte and 8-byte aligned respectively.
//   values  sixteen bytes of nibbles per 32 values; byte b of a block holds value 2b in its low nibble and
//           value 2b + 1 in its high nibble (ggml's q4_0 packing), so a lane's sixteen dims are one eight-byte
//           load and never straddle a block.
//   decode  x = scale * (nibble - 8)
//   store   scale = max|x| / 8; nibble = clamp(round(x / scale) + 8, 0, 15)
// 16.5 bytes per 32 values: half of Q8_0, a quarter of BF16.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"

namespace ingine::arch::qwen35moe::kernels {

// Cache element format of the drafter's layer, selectable like -ctkd/-ctvd. Q4_0 is the default.
enum class MtpKvType : int { Q4_0 = 0, Q8_0 = 1, Bf16 = 2 };

inline constexpr int MTP_KV_BLOCK = 32;

const char* mtp_kv_type_name(MtpKvType type);

// Bytes of one side's cache for `max_pos` positions (ATTN_KV_HEADS * ATTN_HEAD_DIM values each). Zero for an
// unknown type or a non-positive capacity.
std::uint64_t mtp_kv_bytes(MtpKvType type, int max_pos);

__host__ __device__ inline std::uint8_t* mtp_q4_values(void* base) { return static_cast<std::uint8_t*>(base); }
__host__ __device__ inline const std::uint8_t* mtp_q4_values(const void* base) {
  return static_cast<const std::uint8_t*>(base);
}
// The scale array starts after `values / 2` bytes: every supported capacity holds a multiple of 64 values.
__host__ __device__ inline const __half* mtp_q4_scales(const void* base, std::uint64_t values) {
  return reinterpret_cast<const __half*>(static_cast<const std::uint8_t*>(base) + values / 2);
}

// Stores args.k_new/args.v_new at args.pos in Q4_0. Same contract as decode_kv_store: every KV head is written,
// nothing is allocated or synchronised, and a position outside the [pos_begin, pos_end) window is skipped.
cudaError_t mtp_kv_store_q4(const DecodeAttnArgs& args, cudaStream_t stream);

// The Q4_0 counterpart of decode_split: writes `band_splits` SplitState columns of the grid decode_split cuts
// from (pos, splits), for the query heads [head_begin, head_begin + head_count), and touches nothing else.
// Merge the columns with decode_combine (decode_attn.cu): the state layout is the shipped one.
cudaError_t mtp_decode_split_q4(const DecodeAttnArgs& args, int band_splits, cudaStream_t stream);

}  // namespace ingine::arch::qwen35moe::kernels
