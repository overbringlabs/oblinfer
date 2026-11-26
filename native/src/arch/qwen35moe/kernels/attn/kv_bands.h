// Position-band KV cache for the Qwen3.6-35B-A3B full-attention layers.
//
// A decode token at 262k reads 2.72 GB of q8_0 cache. With one cache on one card that read costs the whole
// transfer on card 0 (896 GB/s) while card 1 (448 GB/s) idles, and there is no P2P path between them (PCIe
// 5.0 x8/x8, no NVLink), so the only way to stream the cache from both cards is to keep disjoint position
// ranges on each: card 0 takes the first share of the positions and card 1 the rest, sized by bandwidth, and
// both read their own bytes at their own rate in parallel.
//
// A band is a complete cache of its own length in the layout decode_attn.h defines, addressed band-local:
// global position p is row p - pos_begin, and the band's block-scale array starts after its own value array
// (at that many bytes for q8_0, half as many for q4_0).
//
// The band boundary does not cut a flash-decode column. Columns are cut from (pos, splits) alone, so the
// column that straddles the boundary would otherwise need rows from both cards; instead band 1 mirrors the
// last `overlap` rows below the boundary, `overlap` sized to the widest column the depth rule can produce,
// and that column runs whole on card 1. Every column's SplitState is then bit for bit what an un-banded
// launch at the same splits would have written, so banded decode is exact rather than nearly equal, and the
// ids a banded decoder generates are the ids an un-banded one generates.
//
// Two shapes of the plan exist. When both bands share a device the split blocks write their columns straight
// into the merged SplitState row: no staging, no events. When they do not, the peer card cannot read the host
// card's q, k_new, v_new and cannot write its SplitStates there, so both cross pinned host memory with
// events, the way the decoder's MoE hop does.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"

namespace ingine::arch::qwen35moe {

// Bandwidth ratio card 0 : card 1 of the target box (RTX 5070 Ti 896 GB/s, RTX 5060 Ti 448 GB/s). It only
// places the band boundary; QWEN35MOE_KV_BAND_RATIO overrides it at init, because the ratio worth using is a
// measured quantity, not the spec sheet's.
inline constexpr float KV_BAND_RATIO = 2.0f;
// The decoder's split rule: one flash column per this many positions until kMaxSplits caps it. It sets the
// widest column, and so the mirror width, as max(this, ceil(max_pos / kMaxSplits)).
inline constexpr int KV_BAND_SPLIT_POSITIONS = 2048;

struct KvBand {
  int device = -1;    // CUDA device holding this band
  int pos_begin = 0;  // global position of this band's row 0
  int pos_end = 0;    // one past the last global position this band holds bytes for
  void* k = nullptr;  // [pos_end - pos_begin, ATTN_KV_HEADS, ATTN_HEAD_DIM] in the side's KvType
  void* v = nullptr;  // same
};

struct KvBands {
  KvBand bands[2] = {};
  int count = 1;      // 1 when the plan collapses (one device, or a capacity too small to split)
  int max_pos = 0;    // positions the bands cover together
  int cut = 0;        // band 0 owns and stores positions [0, cut)
  int overlap = 0;    // rows band 1 mirrors below the cut, so no column straddles cards
};

// First band's length in global positions: max_pos * ratio/(1+ratio) rounded up to a multiple of KV_BLOCK,
// clamped so both bands keep at least one block. Returns max_pos when the capacity cannot hold two bands or
// the ratio is not positive.
int kv_band_cut(int max_pos, float ratio = KV_BAND_RATIO);

// Rows band 1 mirrors below `cut`: the widest column the depth rule can produce, namely
// max(split_positions, ceil(max_pos / kMaxSplits)) rounded up to a KV_BLOCK, capped at the boundary so a
// band never mirrors more than band 0 holds. 0 when the plan does not split.
int kv_band_overlap(int max_pos, int cut, int split_positions = KV_BAND_SPLIT_POSITIONS);

// Bytes of one cache side of `positions` positions, i.e. of one band. A row is ATTN_KV_HEADS *
// ATTN_HEAD_DIM = 512 values, a whole number of KV_BLOCK blocks, so a band that starts and ends on a row
// holds a contiguous slice of the un-banded cache in both arrays (values, then scales), in every format.
std::size_t kv_band_bytes(int positions, kernels::KvType type);

// Allocates the bands on `dev0`/`dev1`. With two different devices band 0 owns [0, cut) and band 1 owns
// [cut - overlap, max_pos), where cut comes from `ratio` unless a non-zero `cut` is passed. The same device
// twice, or a cut >= max_pos, gives one band covering everything: the plain path. Nothing leaks on failure;
// both bands end null. The types are per side, like -ctk/-ctv.
bool kv_bands_init(KvBands* bands, int max_pos, int dev0, int dev1, kernels::KvType k_type, kernels::KvType v_type,
                   std::string* error, float ratio = KV_BAND_RATIO, int cut = 0);
void kv_bands_free(KvBands* bands);

// The band that stores global position `pos`: band 1 owns [cut - overlap, max_pos), which includes the
// mirrored tail band 0 also holds, because the column straddling the boundary is read from card 1.
const KvBand* kv_band_for(const KvBands& bands, int pos);

// True when every band lives on the first band's device, i.e. the plan needs no cross-card staging.
bool kv_bands_local(const KvBands& bands);

// Leading flash columns that run whole on band 0, given the global grid (pos, splits) and the boundary
// geometry: the columns whose end is at or before `cut`. The rest start at or after cut - overlap and run on
// band 1. Returns `splits` when the query never reached the boundary and 0 when even the first column belongs
// to card 1 because band 1 mirrors back to the start. Returns -1 when the column is wider than the mirror and
// some column would need rows from both cards, which kv_band_overlap is meant to make unreachable. A
// single-band plan takes the whole grid.
int kv_band_columns_at(int cut, int overlap, int pos, int splits);
inline int kv_band_columns(const KvBands& bands, int pos, int splits) {
  return bands.count < 2 ? splits : kv_band_columns_at(bands.cut, bands.overlap, pos, splits);
}

// Host-side staging for a banded launch across two devices: the peer card's q, k_new, v_new and its slice of
// the SplitState rows. Sizes are fixed by ATTN_HEADS and kMaxSplits, so nothing is allocated per step and one
// exchange serves every layer and every token. There is one slot per direction, guarded by the consumer's
// event, so a launch and the next one's staging alternate rather than overlap.
struct KvBandExchange {
  int host_device = -1;   // device that owns q, out and the merged SplitStates
  int peer_device = -1;   // device the second band lives on
  float* q = nullptr;     // peer-side [ATTN_HEADS, ATTN_HEAD_DIM]
  float* k_new = nullptr;  // peer-side [ATTN_KV_HEADS, ATTN_HEAD_DIM]
  float* v_new = nullptr;
  kernels::SplitState* scratch = nullptr;  // peer-side [ATTN_HEADS][kMaxSplits], same layout as `partial`
  std::uint8_t* pinned = nullptr;          // query staging, then SplitState staging

  // Recorded on the host device: the query has reached pinned memory / the peer's states have been taken.
  cudaEvent_t q_ready = nullptr;
  cudaEvent_t states_consumed = nullptr;
  // Recorded on the peer device: the peer has taken the query / its states have reached pinned memory.
  cudaEvent_t states_ready = nullptr;
  cudaEvent_t q_consumed = nullptr;
  bool q_pending = false;       // a previous call's query copy has not been consumed yet
  bool states_pending = false;  // a previous call's states have not been merged yet
};

// Allocates the peer-side buffers on `peer_device` and the pinned staging on the host, and creates each event
// with the device that records it made current. Leaves `host_device` current. Not needed for a local plan.
bool kv_band_exchange_init(KvBandExchange* exch, int host_device, int peer_device, std::string* error);
void kv_band_exchange_free(KvBandExchange* exch);

// Store the new row, run the flash split kernel per band over the columns that band holds whole, and merge
// all `splits` columns into args.out on the first band's device.
//
// `args` supplies q, k_new, v_new, out, pos, max_pos, the cache element types and the head range; its cache
// pointers, pos_begin/pos_end/partial/col_begin are overridden per band. Band 0 must start at position 0 and
// the bands must cover max_pos. Requires splits in [1, kMaxSplits] and an exchange unless the plan is local.
// A band with no column to run is not launched; it still stores, so the mirror stays warm for the crossing.
//
// Streams: band 0 and the merge run on `s0` (the host device's default stream when null), the second band on
// `s1`, which must belong to that band's device when the plan is not local. Nothing is synchronised, so the
// caller may pipeline layers, but it must not free or reuse `partial`, the bands or the exchange before s0
// (and, for the exchange, the next call) completes. The function switches the host thread's current device,
// so it is not something to capture into a CUDA graph: keep banded decode eager, or splice a child graph per
// device the way the MoE hop does.
bool decode_attention_bands(const kernels::DecodeAttnArgs& args, const KvBands& bands,
                            kernels::SplitState* partial, int splits, cudaStream_t s0, cudaStream_t s1,
                            std::string* error, KvBandExchange* exch = nullptr);

}  // namespace ingine::arch::qwen35moe
