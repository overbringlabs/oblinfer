// Position-band KV cache: the band plan, its allocation, and the two-card launch that reads the bands in
// parallel. See kv_bands.h for why band 1 mirrors a tail of band 0's rows: it keeps every flash column whole
// on one card, which is what makes a banded result bit-identical to the un-banded one.
#include "arch/qwen35moe/kernels/attn/kv_bands.h"

#include <algorithm>

namespace ingine::arch::qwen35moe {

using kernels::ATTN_HEAD_DIM;
using kernels::ATTN_HEADS;
using kernels::ATTN_KV_HEADS;
using kernels::DecodeAttnArgs;
using kernels::KvType;
using kernels::SplitState;

namespace {

constexpr int kMaxSplits = kernels::kMaxSplits;
constexpr int kBlock = kernels::KV_BLOCK;

bool fail(std::string* error, const std::string& what) {
  if (error) *error = what;
  return false;
}

bool ok(cudaError_t e, const char* what, std::string* error) {
  if (e == cudaSuccess) return true;
  return fail(error, std::string(what) + ": " + cudaGetErrorString(e));
}

int round_block(int rows) {
  return (rows + kBlock - 1) / kBlock * kBlock;
}

constexpr std::size_t kQBytes = std::size_t(ATTN_HEADS) * ATTN_HEAD_DIM * sizeof(float);
constexpr std::size_t kKvBytes = std::size_t(ATTN_KV_HEADS) * ATTN_HEAD_DIM * sizeof(float);
// The peer's SplitState mirror: one full [ATTN_HEADS][kMaxSplits] block, so a peer column lands at the same
// index it would on the host card and the staging copy only skips the columns band 0 already filled.
constexpr std::size_t kStateBytes = std::size_t(ATTN_HEADS) * kMaxSplits * sizeof(SplitState);

// Pinned staging: the query block and the new KV row (fixed sizes, so nothing is allocated per step), then
// the SplitState mirror. Every offset is a multiple of 4, which is what float and SplitState need; the value
// arrays the split kernel reads live in the device allocations, not here.
constexpr std::size_t kPinnedBytes = kQBytes + 2 * kKvBytes + kStateBytes;

float* pinned_q(const KvBandExchange& e) { return reinterpret_cast<float*>(e.pinned); }
float* pinned_k(const KvBandExchange& e) { return reinterpret_cast<float*>(e.pinned + kQBytes); }
float* pinned_v(const KvBandExchange& e) { return reinterpret_cast<float*>(e.pinned + kQBytes + kKvBytes); }
SplitState* pinned_states(const KvBandExchange& e) {
  return reinterpret_cast<SplitState*>(e.pinned + kQBytes + 2 * kKvBytes);
}

}  // namespace

int kv_band_cut(int max_pos, float ratio) {
  if (max_pos <= 0 || !(ratio > 0.0f)) return max_pos;
  if (max_pos < 2 * kBlock) return max_pos;  // two block-aligned bands do not fit
  const int want = int(std::int64_t(double(max_pos) * ratio / (1.0 + ratio) + 0.5));
  const int hi = (max_pos - kBlock) / kBlock * kBlock;  // band 1 keeps at least one block
  return std::max(kBlock, std::min(round_block(want), hi));
}

int kv_band_overlap(int max_pos, int cut, int split_positions) {
  if (max_pos <= 0 || cut <= 0 || cut >= max_pos) return 0;
  // The widest column the decoder's split rule can produce: one per split_positions below the cap, and
  // ceil(max_pos / kMaxSplits) at it.
  const int by_cap = (max_pos + kMaxSplits - 1) / kMaxSplits;
  const int by_rule = split_positions > 0 ? split_positions : by_cap;
  return std::min(round_block(std::max(by_cap, by_rule)), cut);
}

std::size_t kv_band_bytes(int positions, KvType type) {
  if (positions <= 0) return 0;
  return std::size_t(kernels::kv_cache_bytes(type, positions));
}

bool kv_bands_init(KvBands* bands, int max_pos, int dev0, int dev1, KvType k_type, KvType v_type,
                   std::string* error, float ratio, int cut) {
  if (!bands) return fail(error, "kv_bands_init: null bands");
  *bands = KvBands{};
  if (max_pos <= 0) return fail(error, "kv_bands_init: max_pos must be positive");
  bands->max_pos = max_pos;
  // An explicit cut wins, including on one device: that is how the band arithmetic gets covered without a
  // second card.
  int want = cut > 0 ? std::min(cut, max_pos) : (dev1 == dev0 ? max_pos : kv_band_cut(max_pos, ratio));
  bands->cut = std::max(1, std::min(want, max_pos));
  bands->overlap = bands->cut >= max_pos ? 0 : kv_band_overlap(max_pos, bands->cut);

  auto alloc_band = [&](KvBand& band, int dev, int begin, int end) -> bool {
    band.device = dev;
    band.pos_begin = begin;
    band.pos_end = end;
    if (!ok(cudaSetDevice(dev), "kv_bands_init: set device", error)) return false;
    void* k = nullptr;
    void* v = nullptr;
    if (!ok(cudaMalloc(&k, kv_band_bytes(end - begin, k_type)), "kv_bands_init: k cache", error)) return false;
    if (!ok(cudaMalloc(&v, kv_band_bytes(end - begin, v_type)), "kv_bands_init: v cache", error)) {
      cudaFree(k);
      band = KvBand{};
      return false;
    }
    // Zeroed: an unwritten row must read as a zero value and a zero scale, not as allocation leftovers.
    if (!ok(cudaMemset(k, 0, kv_band_bytes(end - begin, k_type)), "kv_bands_init: zero k", error) ||
        !ok(cudaMemset(v, 0, kv_band_bytes(end - begin, v_type)), "kv_bands_init: zero v", error)) {
      cudaFree(k);
      cudaFree(v);
      band = KvBand{};
      return false;
    }
    band.k = k;
    band.v = v;
    return true;
  };

  if (!alloc_band(bands->bands[0], dev0, 0, bands->cut)) {
    kv_bands_free(bands);
    return false;
  }
  if (bands->cut >= max_pos) {
    bands->count = 1;
    cudaSetDevice(dev0);
    return true;
  }
  // Band 1 starts `overlap` rows early: the column straddling the boundary runs whole on it.
  if (!alloc_band(bands->bands[1], dev1, bands->cut - bands->overlap, max_pos)) {
    kv_bands_free(bands);
    return false;
  }
  bands->count = 2;
  cudaSetDevice(dev0);
  return true;
}

void kv_bands_free(KvBands* bands) {
  if (!bands) return;
  for (KvBand& band : bands->bands) {
    if (band.device < 0) continue;
    cudaSetDevice(band.device);
    if (band.k) cudaFree(band.k);
    if (band.v) cudaFree(band.v);
    band = KvBand{};
  }
  *bands = KvBands{};
}

const KvBand* kv_band_for(const KvBands& bands, int pos) {
  if (pos < 0 || pos >= bands.max_pos) return nullptr;
  if (bands.count == 2 && pos >= bands.cut - bands.overlap) return &bands.bands[1];
  return &bands.bands[0];
}

bool kv_bands_local(const KvBands& bands) {
  return bands.count < 2 || bands.bands[1].device == bands.bands[0].device;
}

int kv_band_columns_at(int cut, int overlap, int pos, int splits) {
  if (splits < 1) return 1;
  const int per = (pos + 1 + splits - 1) / splits;
  // The leading columns that end at or before the boundary belong to band 0.
  int c1 = std::min(splits, cut / per);
  // The rest belong to band 1 and start at c1 * per, which must be a row band 1 actually holds.
  if (c1 < splits && c1 * per < cut - overlap) return -1;
  return c1;
}

bool kv_band_exchange_init(KvBandExchange* exch, int host_device, int peer_device, std::string* error) {
  if (!exch) return fail(error, "kv_band_exchange_init: null exchange");
  *exch = KvBandExchange{};
  exch->host_device = host_device;
  exch->peer_device = peer_device;
  bool good = true;

  // The peer's buffers and the events recorded on them are created with device 1 current; the host-side
  // events with device 0 current, exactly as the decoder's MoE exchange does.
  good = ok(cudaSetDevice(peer_device), "kv_band_exchange_init: set peer", error);
  good = ok(cudaMalloc(reinterpret_cast<void**>(&exch->q), kQBytes), "exchange q", error) && good;
  good = ok(cudaMalloc(reinterpret_cast<void**>(&exch->k_new), kKvBytes), "exchange k_new", error) && good;
  good = ok(cudaMalloc(reinterpret_cast<void**>(&exch->v_new), kKvBytes), "exchange v_new", error) && good;
  good = ok(cudaMalloc(reinterpret_cast<void**>(&exch->scratch), kStateBytes), "exchange scratch", error) && good;
  good = ok(cudaEventCreateWithFlags(&exch->q_consumed, cudaEventDisableTiming), "event q_consumed", error) && good;
  good = ok(cudaEventCreateWithFlags(&exch->states_ready, cudaEventDisableTiming), "event states_ready", error) &&
          good;
  good = ok(cudaSetDevice(host_device), "kv_band_exchange_init: set host", error) && good;
  good = ok(cudaEventCreateWithFlags(&exch->q_ready, cudaEventDisableTiming), "event q_ready", error) && good;
  good = ok(cudaEventCreateWithFlags(&exch->states_consumed, cudaEventDisableTiming), "event states_consumed",
            error) && good;
  good = ok(cudaHostAlloc(reinterpret_cast<void**>(&exch->pinned), kPinnedBytes, cudaHostAllocDefault),
            "exchange pinned", error) && good;

  if (!good) {
    kv_band_exchange_free(exch);  // ok() already left the reason in *error
    return false;
  }
  return true;
}

void kv_band_exchange_free(KvBandExchange* exch) {
  if (!exch) return;
  if (exch->peer_device >= 0) {
    cudaSetDevice(exch->peer_device);
    if (exch->q) cudaFree(exch->q);
    if (exch->k_new) cudaFree(exch->k_new);
    if (exch->v_new) cudaFree(exch->v_new);
    if (exch->scratch) cudaFree(exch->scratch);
    if (exch->q_consumed) cudaEventDestroy(exch->q_consumed);
    if (exch->states_ready) cudaEventDestroy(exch->states_ready);
  }
  if (exch->host_device >= 0) cudaSetDevice(exch->host_device);
  if (exch->q_ready) cudaEventDestroy(exch->q_ready);
  if (exch->states_consumed) cudaEventDestroy(exch->states_consumed);
  if (exch->pinned) cudaFreeHost(exch->pinned);
  *exch = KvBandExchange{};
}

bool decode_attention_bands(const DecodeAttnArgs& args, const KvBands& bands, SplitState* partial, int splits,
                            cudaStream_t s0, cudaStream_t s1, std::string* error, KvBandExchange* exch) {
  const int count = bands.count;
  if (!args.q || !args.k_new || !args.v_new || !args.out) {
    return fail(error, "decode_attention_bands: q, k_new, v_new and out are required");
  }
  if (!partial) return fail(error, "decode_attention_bands: partial is required");
  if (count < 1 || count > 2) return fail(error, "decode_attention_bands: bands are not initialised");
  if (args.head_begin < 0 || args.head_count < 1 || args.head_begin + args.head_count > ATTN_HEADS) {
    return fail(error, "decode_attention_bands: head range outside [0, ATTN_HEADS)");
  }
  if (args.pos < 0 || args.pos >= bands.max_pos) return fail(error, "decode_attention_bands: position outside");
  if (splits < 1 || splits > kMaxSplits) return fail(error, "decode_attention_bands: split count out of range");
  if (bands.bands[0].pos_begin != 0 || !bands.bands[0].k || !bands.bands[0].v) {
    return fail(error, "decode_attention_bands: band 0 must start at 0 and own a cache");
  }
  if (bands.bands[count - 1].pos_end != bands.max_pos) {
    return fail(error, "decode_attention_bands: the bands must cover max_pos");
  }
  const bool cross = count == 2 && !kv_bands_local(bands);
  if (cross && !exch) return fail(error, "decode_attention_bands: a two-card plan needs a KvBandExchange");
  if (cross && (exch->host_device != bands.bands[0].device || exch->peer_device != bands.bands[1].device)) {
    return fail(error, "decode_attention_bands: the exchange does not match the band devices");
  }
  if (cross && !s1) return fail(error, "decode_attention_bands: a two-card plan needs a peer stream");

  // Columns [0, c1) end at or before the boundary and run on band 0; columns [c1, splits) start inside band
  // 1's mirror and run there. c1 == splits means the query never crossed the boundary, c1 == 0 that even the
  // first column belongs to card 1 because band 1 mirrors back to the start.
  const int c1 = kv_band_columns(bands, args.pos, splits);
  if (c1 < 0 || c1 > splits) return fail(error, "decode_attention_bands: no column boundary fits the mirror");
  const int cols1 = splits - c1;

  int prev_device = 0;
  if (!ok(cudaGetDevice(&prev_device), "decode_attention_bands: get device", error)) return false;
  struct Restore {
    int device;
    ~Restore() { cudaSetDevice(device); }
  } restore{prev_device};

  // Card 1 needs the new row (its mirror and its band are stored there) and the query (to score against it).
  // It cannot read card 0's memory, so both cross through pinned host memory, ordered by events.
  const bool peer_store = count == 2 && args.pos >= bands.bands[1].pos_begin;
  const bool peer_work = cols1 > 0 || peer_store;
  if (cross && peer_work) {
    if (!ok(cudaSetDevice(exch->host_device), "decode_attention_bands: set host", error)) return false;
    if (exch->q_pending && !ok(cudaStreamWaitEvent(s0, exch->q_consumed, 0), "wait q consumed", error)) return false;
    if (!ok(cudaMemcpyAsync(pinned_k(*exch), args.k_new, kKvBytes, cudaMemcpyDeviceToHost, s0), "stage k_new",
            error))
      return false;
    if (!ok(cudaMemcpyAsync(pinned_v(*exch), args.v_new, kKvBytes, cudaMemcpyDeviceToHost, s0), "stage v_new",
            error))
      return false;
    if (cols1 > 0 &&
        !ok(cudaMemcpyAsync(pinned_q(*exch), args.q, kQBytes, cudaMemcpyDeviceToHost, s0), "stage q", error)) {
      return false;
    }
    if (!ok(cudaEventRecord(exch->q_ready, s0), "record q_ready", error)) return false;
    exch->q_pending = true;
  }

  for (int b = 0; b < count; ++b) {
    const KvBand& band = bands.bands[b];
    if (b == 1 && !peer_work) continue;  // nothing to store and nothing to read
    if (!ok(cudaSetDevice(band.device), "decode_attention_bands: set band device", error)) return false;
    const cudaStream_t stream = (b == 0 || !cross) ? s0 : s1;
    DecodeAttnArgs a = args;
    a.k_cache = band.k;
    a.v_cache = band.v;
    a.max_pos = bands.max_pos;
    a.pos_begin = band.pos_begin;
    a.pos_end = band.pos_end;
    a.splits = splits;
    a.col_begin = b == 0 ? 0 : c1;
    a.partial = partial;
    if (b == 1 && cross) {
      if (!ok(cudaStreamWaitEvent(s1, exch->q_ready, 0), "wait q ready", error)) return false;
      if (exch->states_pending &&
          !ok(cudaStreamWaitEvent(s1, exch->states_consumed, 0), "wait states consumed", error))
        return false;
      if (!ok(cudaMemcpyAsync(exch->k_new, pinned_k(*exch), kKvBytes, cudaMemcpyHostToDevice, s1),
              "k_new to peer", error))
        return false;
      if (!ok(cudaMemcpyAsync(exch->v_new, pinned_v(*exch), kKvBytes, cudaMemcpyHostToDevice, s1),
              "v_new to peer", error))
        return false;
      if (cols1 > 0 &&
          !ok(cudaMemcpyAsync(exch->q, pinned_q(*exch), kQBytes, cudaMemcpyHostToDevice, s1), "q to peer", error)) {
        return false;
      }
      if (!ok(cudaEventRecord(exch->q_consumed, s1), "record q_consumed", error)) return false;
      a.q = exch->q;  // never dereferenced when cols1 == 0: the store does not read q
      a.k_new = exch->k_new;
      a.v_new = exch->v_new;
      a.out = nullptr;  // a band launch never writes the output
      a.partial = exch->scratch;
    }
    if (!ok(kernels::decode_kv_store(a, stream), "decode_kv_store", error)) return false;
    const int cols = b == 0 ? c1 : cols1;
    if (cols > 0 && !ok(kernels::decode_split(a, cols, stream), "decode_split", error)) return false;

    if (b == 1 && cross && cols1 > 0) {
      // The peer's mirror has the host row layout, so one pitched copy moves its columns across as they are.
      const std::size_t width = std::size_t(cols1) * sizeof(SplitState);
      const std::size_t pitch = std::size_t(splits) * sizeof(SplitState);
      if (!ok(cudaMemcpy2DAsync(pinned_states(*exch) + c1, pitch, exch->scratch + c1, pitch, width,
                                std::size_t(args.head_count), cudaMemcpyDeviceToHost, s1),
              "stage peer states", error))
        return false;
      if (!ok(cudaEventRecord(exch->states_ready, s1), "record states_ready", error)) return false;
      exch->states_pending = true;
    }
  }

  if (!ok(cudaSetDevice(bands.bands[0].device), "decode_attention_bands: set host for the merge", error)) {
    return false;
  }
  if (cross && cols1 > 0) {
    if (!ok(cudaStreamWaitEvent(s0, exch->states_ready, 0), "wait states ready", error)) return false;
    const std::size_t width = std::size_t(cols1) * sizeof(SplitState);
    const std::size_t pitch = std::size_t(splits) * sizeof(SplitState);
    if (!ok(cudaMemcpy2DAsync(partial + c1, pitch, pinned_states(*exch) + c1, pitch, width,
                              std::size_t(args.head_count), cudaMemcpyHostToDevice, s0),
            "merge peer states", error))
      return false;
    if (!ok(cudaEventRecord(exch->states_consumed, s0), "record states_consumed", error)) return false;
  }

  // One combine over all splits columns on the host card. It reads partial and writes out only, so it needs
  // no cache pointers and no band window.
  DecodeAttnArgs merged = args;
  merged.k_cache = nullptr;
  merged.v_cache = nullptr;
  merged.max_pos = bands.max_pos;
  merged.pos_begin = 0;
  merged.pos_end = -1;
  merged.col_begin = 0;
  merged.partial = partial;
  merged.splits = splits;
  return ok(kernels::decode_combine(merged, s0), "decode_combine", error);
}

}  // namespace ingine::arch::qwen35moe
