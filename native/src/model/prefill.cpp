// model.Model.prefill for one chunk of T tokens, wired as fast.FastDecoder.decode (block states B0, B1, B2[p],
// E[p], partial / write pairs, message mixes) with T rows instead of M:
//   embed -> per visit: [message mix] -> attention read (mix) [-> message gates] -> taps -> projections
//   (cuBLASLt over the BF16 rows, FP8 rows dequantized per visit as weights.dequant) -> attn_prep (q/rel post-
//   processing, key/value convs, cache or sliding rows) -> rel_logits -> attention (+ gate) -> wo -> attention
//   conv -> MLP read (mix) [-> gate] -> router GEMM + top-8 -> grouped experts -> combine -> MLP conv
//   -> final read (mix) -> taps -> head of the last row.
// The mixes and the embedding are ModelCore's decode kernels (per-row CTAs, any row count). Ablation of the
// attention / MLP writes happens inside the mixes that consume them, as in the decode step.
//
// Two devices: a layer whose experts live on the second device runs its grouped expert GEMMs there
// (expert_hop), a 64/64 split runs half on each (expert_split_hop), and with the KV-head split a full-attention
// visit runs one launch per device over its own KV cache shard (full_attn_split). Every value the second device
// reads is a copy of this device's, so a hop is bit-identical to the one-device path; without a second device
// nothing is copied or waited on.
#include "model/prefill.h"

#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {
namespace pk = kernels::prefill;
namespace dense = kernels::dense;

namespace {

constexpr int H = HIDDEN;

// --kv-split: the second device's 3 full-attention KV heads and the proj columns of their query/rel/gate rows
constexpr int kVkHere = KV_HEADS - KV_SPLIT_SECOND_HEADS;  // 5
constexpr int kVkThere = KV_SPLIT_SECOND_HEADS;            // 3
constexpr int kVkQ = kVkThere * 3;                         // its 9 query heads

// kv_layer_'s per-full-layer slices: q_scale (all 24 heads, absolute indexing), r_scale, rel_proj (distance
// addressed) and the static-conv rows of the shard's heads (compact)
constexpr std::size_t kVkLayerWeights = std::size_t(HEADS) * HEAD_DIM * 4 + std::size_t(HEADS) * D_REL * 4 +
                                        std::size_t(D_REL) * REL_EXTENT * 4 +
                                        2 * std::size_t(CONV_K) * kVkThere * HEAD_DIM * 4;

// The proj columns of the shard's heads: contiguous within a row, so the packet mirrors a proj row and the
// second device addresses its slices exactly as it would the whole row (Decoder::full_attn_split's list).
struct VkSlice {
  int col, len;
};
constexpr std::array<VkSlice, 8> kVkSlices = {{
    {SEG_BEGIN[SEG_Q] + kVkHere * 3 * HEAD_DIM, kVkQ * HEAD_DIM},
    {SEG_BEGIN[SEG_GATE] + kVkHere * 3 * HEAD_DIM, kVkQ * HEAD_DIM},
    {SEG_BEGIN[SEG_REL] + kVkHere * 3 * D_REL, kVkQ * D_REL},
    {SEG_BEGIN[SEG_SINK] + kVkHere * 3, kVkQ},
    {SEG_BEGIN[SEG_K] + kVkHere * HEAD_DIM, kVkThere * HEAD_DIM},
    {SEG_BEGIN[SEG_V] + kVkHere * HEAD_DIM, kVkThere * HEAD_DIM},
    {SEG_BEGIN[SEG_K_DYN] + kVkHere * CONV_K, kVkThere * CONV_K},
    {SEG_BEGIN[SEG_V_DYN] + kVkHere * CONV_K, kVkThere * CONV_K},
}};

// Makes `device` current for the scope, restoring the caller's on exit (the hop's device-1 copies and
// launches). Never used inside stream capture: a split visit is not capturable.
struct BindDevice {
  explicit BindDevice(int device) {
    CUDA_CHECK(cudaGetDevice(&saved));
    if (saved != device) CUDA_CHECK(cudaSetDevice(device));
  }
  ~BindDevice() { (void)cudaSetDevice(saved); }
  BindDevice(const BindDevice&) = delete;
  BindDevice& operator=(const BindDevice&) = delete;

 private:
  int saved = 0;
};

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const __nv_bfloat16* bf16p(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }
const __nv_bfloat16* bf16p(const void* p) { return static_cast<const __nv_bfloat16*>(p); }

// fast.Buffers.blocks
std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

}  // namespace

Prefiller::Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream,
                     const DeviceContext* second, bool kv_split)
    : w_(w), abl_(abl), max_chunk_(max_chunk), stream_(stream), gemm_(std::make_unique<pk::Gemm>()),
      second_(second), kv_split_(kv_split) {
  if (max_chunk < 1) throw std::invalid_argument("Prefiller: max_chunk must be positive");
  if (kv_split_ && second_ == nullptr)
    throw std::invalid_argument("Prefiller: the KV-head split needs the second device");
  // A layer's experts that are not all here make its visit hop: a whole layer there (split 0) runs its grouped
  // GEMMs there, a 64/64 split runs half on each device. Either way the grouped GEMMs address their tensors'
  // rows by the local expert id.
  bool remote = false, shard = false;
  for (const Layer& l : w.layers) {
    remote = remote || l.experts.split == 0;
    shard = shard || (l.experts.split != EXPERTS && l.experts.split != 0);
  }
  if ((remote || shard) && second_ == nullptr)
    throw std::invalid_argument("Prefiller: a layer's experts are on the second device, which is not open");
  int nfull = 0;
  for (const Layer& l : w.layers) nfull += l.full ? 1 : 0;
  const int L = WINDOW + max_chunk;
  kpos_rows_ = (L + pk::kAttnKeyTile - 1) / pk::kAttnKeyTile * pk::kAttnKeyTile;
  if (kpos_rows_ / pk::kAttnKeyTile > pk::kAttnMaxTiles)
    throw std::invalid_argument("Prefiller: max_chunk above " +
                                std::to_string(pk::kAttnMaxTiles * pk::kAttnKeyTile - WINDOW));
  const std::size_t C = std::size_t(max_chunk);
  auto alloc = [&](std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  };
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto bf16 = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };
  B0_ = f32(C * H), B1_ = f32(C * H);
  for (int p = 0; p < PASSES; ++p) B2_[p] = f32(C * H), E_[p] = f32(C * H);
  hid_ = f32(C * H), part_ = f32(C * H), y_ = f32(C * H), y2_ = f32(C * H);
  xin_ = bf16(3 * C * H), xm_ = bf16(C * H), xf_ = bf16(C * H);
  mixf_ = f32(3 * C * H), msgs_ = f32(4 * C * H), mixm_ = f32(C * H);
  proj_ = bf16(C * PROJ_ROWS_FULL);
  q_ = bf16(C * HEADS * HEAD_DIM), rel16_ = bf16(C * HEADS * D_REL), rel_ = bf16(C * HEADS * REL_EXTENT);
  krows_ = bf16(std::size_t(kpos_rows_) * KV_DIM), vrows_ = bf16(std::size_t(kpos_rows_) * KV_DIM);
  kpos_ = i32(kpos_rows_);
  bounds_ = static_cast<int4*>(alloc(std::size_t(kpos_rows_ / pk::kAttnKeyTile) * sizeof(int4)));
  attn_out_ = bf16(C * HEADS * HEAD_DIM), wo_out_ = bf16(C * H);
  g16_ = bf16(C * GATE_RANK);
  // message-gate scratch lives in buffers dead while the gates run: gin_ in rel_ (written by rel_logits after
  // the attention-side gates, dead before the MLP-side gate), fpre_/gpre_ in down_ (written by the experts after
  // both gates)
  static_assert(3 * 2 * HIDDEN * 2 <= HEADS * REL_EXTENT * 2 && 2 * HIDDEN * 4 <= TOP_K * HIDDEN * 2);
  router_logits_ = f32(C * EXPERTS), wts_ = f32(C * TOP_K);
  idx_ = i32(C * TOP_K);
  counts_ = i32(EXPERTS), tiles_n_ = i32(1), pair_token_ = i32(C * TOP_K), pair_dst_ = i32(C * TOP_K);
  tiles_capacity_ = pk::moe_tiles_capacity(max_chunk);
  tiles_ = static_cast<int4*>(alloc(std::size_t(tiles_capacity_) * sizeof(int4)));
  act_ = bf16(C * TOP_K * INTER), down_ = bf16(C * TOP_K * H), moe_sum_ = bf16(C * H);
  gin_ = rel_;
  fpre_ = reinterpret_cast<float*>(down_), gpre_ = fpre_ + C * H;
  wscratch_ = w.dense_fp8 ? bf16(std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM)) : nullptr;
  ids_ = static_cast<std::int64_t*>(alloc(C * 8));
  for (auto& b : storage_) b.fill(0);  // e.g. gathered sliding rows past ring + chunk: never keys, kept finite

  if (second_ == nullptr) return;
  hop_stream_ = second_->stream;
  CUDA_CHECK(cudaEventCreateWithFlags(&hop_ready_, cudaEventDisableTiming));
  if (remote || shard) {
    // The hop's staging is sized for the whole chunk: xm out (bf16 [T, H]), the router's tile and pair lists
    // out, the combined rows back (bf16 [T, H]).
    hop_x_ = MappedHostBuffer(C * H * 2);
    hop_sum_ = MappedHostBuffer(C * H * 2);
    hop_work_ = MappedHostBuffer(4 + std::size_t(tiles_capacity_) * sizeof(int4) + 3 * C * TOP_K * 4);
    const BindDevice bind(second_->device);
    CUDA_CHECK(cudaEventCreateWithFlags(&hop_back_, cudaEventDisableTiming));
    hop_hi_.device = second_->device;
    hop_hi_.buffers.emplace_back(C * H * 2);                        // xm
    hop_hi_.buffers.emplace_back(4);                                // tiles_n
    hop_hi_.buffers.emplace_back(std::size_t(tiles_capacity_) * sizeof(int4));  // tiles
    hop_hi_.buffers.emplace_back(C * TOP_K * 4);                    // pair_token
    hop_hi_.buffers.emplace_back(C * TOP_K * 4);                    // pair_dst
    hop_hi_.buffers.emplace_back(C * TOP_K * 4);                    // wts
    hop_hi_.buffers.emplace_back(C * TOP_K * INTER * 2);            // act
    hop_hi_.buffers.emplace_back(C * TOP_K * H * 2);                // per-slot rows
    hop_hi_.buffers.emplace_back(C * H * 2);                        // combined rows
  }
  if (shard) {  // a 64/64 split returns the second device's per-slot rows instead of one combined block; they
    // are merged here, so hop_rows0_ is a primary allocation: made outside the bind above
    hop_rows_ = MappedHostBuffer(C * TOP_K * H * 2);
    hop_rows0_ = DeviceBuffer(C * TOP_K * H * 2);
  }
  if (kv_split_) {
    const std::size_t CC = std::size_t(kVkThere) * HEAD_DIM;
    std::size_t in_bytes = 2 * std::size_t(CONV_K - 1) * CC * 4;
    for (const VkSlice& s : kVkSlices) in_bytes += C * std::size_t(s.len) * 2;
    kv_in_ = MappedHostBuffer(in_bytes);
    kv_out_ = MappedHostBuffer(C * kVkQ * HEAD_DIM * 2 + 2 * std::size_t(CONV_K - 1) * CC * 4);
    CUDA_CHECK(cudaEventCreateWithFlags(&kv_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&kv_consumed_, cudaEventDisableTiming));
    {
      const BindDevice bind(second_->device);
      CUDA_CHECK(cudaEventCreateWithFlags(&kv_loaded_, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&kv_back_, cudaEventDisableTiming));
      // full width by absolute head (q, rel, rel_logits, out): only the shard's heads are written, so the
      // kernels index them exactly as on this device
      kv_hi_.device = second_->device;
      kv_hi_.buffers.emplace_back(C * PROJ_ROWS_FULL * 2);      // proj rows (the shard's columns)
      kv_hi_.buffers.emplace_back(C * HEADS * HEAD_DIM * 2);    // q
      kv_hi_.buffers.emplace_back(C * HEADS * D_REL * 2);        // rel
      kv_hi_.buffers.emplace_back(C * HEADS * REL_EXTENT * 2);   // rel_logits
      kv_hi_.buffers.emplace_back(C * HEADS * HEAD_DIM * 2);    // attention out
      kv_hi_.buffers.emplace_back(2 * std::size_t(CONV_K - 1) * CC * 4);  // conv state of the shard's heads
      kv_hi_.buffers.emplace_back(std::size_t(nfull) * kVkLayerWeights);  // per-layer slices
    }
    layer_weights_second();
  }
}

Prefiller::~Prefiller() {
  if (second_ == nullptr) return;
  {
    const BindDevice bind(second_->device);  // the second device's events and storage
    if (hop_back_) (void)cudaEventDestroy(hop_back_);
    if (kv_loaded_) (void)cudaEventDestroy(kv_loaded_);
    if (kv_back_) (void)cudaEventDestroy(kv_back_);
    hop_hi_.release();
    kv_hi_.release();
  }
  if (hop_ready_) (void)cudaEventDestroy(hop_ready_);
  if (kv_ready_) (void)cudaEventDestroy(kv_ready_);
  if (kv_consumed_) (void)cudaEventDestroy(kv_consumed_);
}

void Prefiller::layer_weights_second() {
  // The split's per-layer slices are immutable, so they cross the bus once at construction (synchronously:
  // this is not a token's path).
  const int here = kVkHere;
  std::vector<float> host;
  auto append = [&](const float* src, std::size_t floats) {
    const std::size_t at = host.size();
    host.resize(at + floats);
    CUDA_CHECK(cudaMemcpy(host.data() + at, src, floats * 4, cudaMemcpyDeviceToHost));
  };
  auto append_rows = [&](const float* src, int rows, int width, int pitch) {
    for (int r = 0; r < rows; ++r) append(src + std::size_t(r) * pitch, std::size_t(width));
  };
  for (const Layer& l : w_.layers) {
    if (!l.full) continue;
    append(f32p(l.q_scale), std::size_t(HEADS) * HEAD_DIM);
    append(f32p(l.r_scale), std::size_t(HEADS) * D_REL);
    append(f32p(l.rel_proj), std::size_t(D_REL) * REL_EXTENT);
    append_rows(f32p(l.k_static) + here * HEAD_DIM, CONV_K, kVkThere * HEAD_DIM, KV_DIM);
    append_rows(f32p(l.v_static) + here * HEAD_DIM, CONV_K, kVkThere * HEAD_DIM, KV_DIM);
  }
  {
    const BindDevice bind(second_->device);
    CUDA_CHECK(cudaMemcpy(kv_hi_.buffers[6].p, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  }
  const std::size_t qb = std::size_t(HEADS) * HEAD_DIM * 4, rb = std::size_t(HEADS) * D_REL * 4,
                    pb = std::size_t(D_REL) * REL_EXTENT * 4, kb = std::size_t(CONV_K) * kVkThere * HEAD_DIM * 4;
  int o = 0;
  for (const Layer& l : w_.layers) {
    if (!l.full) continue;
    char* base = static_cast<char*>(kv_hi_.buffers[6].p) + std::size_t(o++) * kVkLayerWeights;
    kv_layer_[l.idx].q_scale = reinterpret_cast<const float*>(base);
    kv_layer_[l.idx].r_scale = reinterpret_cast<const float*>(base + qb);
    kv_layer_[l.idx].rel_proj = reinterpret_cast<const float*>(base + qb + rb);
    kv_layer_[l.idx].k_w = reinterpret_cast<const float*>(base + qb + rb + pb);
    kv_layer_[l.idx].v_w = reinterpret_cast<const float*>(base + qb + rb + pb + kb);
  }
}

std::size_t Prefiller::device_bytes() const {
  std::size_t n = 0;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

const float* const* Prefiller::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const float* const*>(it->second.p);
}

void Prefiller::run(State& st, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
                    std::span<const EmbedRows> rows) {
  if (T < 1 || T > max_chunk_) throw std::invalid_argument("Prefiller::run: T outside 1..max_chunk");
  if ((st.kv_heads_hi != 0) != kv_split_)
    throw std::logic_error("Prefiller::run: the state's KV-head split does not match this Prefiller's");
  if (pos0 < 0 || pos0 + T > st.max_ctx)
    throw std::invalid_argument("context " + std::to_string(pos0 + T) + " exceeds max_ctx " +
                                std::to_string(st.max_ctx));
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;  // one [T, H] slab of a [G, T, H] buffer
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);
  __nv_bfloat16* tap_out = taps ? static_cast<__nv_bfloat16*>(taps->data) : nullptr;

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;  // the detector of the same visit
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  // B rows of a Dense matrix as BF16: the matrix itself, or its FP8 rows dequantized (weights.dequant)
  auto bf16_rows = [&](const Dense& d) -> const __nv_bfloat16* {
    if (!d.fp8()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    return wscratch_;
  };
  // message_gate for G gates: h_g, msg_g fp32 [T, H] -> out bf16 [G, T, H] = bf16(rms_norm(gated) * norm_w[g])
  auto gates = [&](const pk::GateRows& rows_, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                   const float* norm_w, __nv_bfloat16* out) {
    pk::gate_inputs(rows_, G, T, gin_, s);
    for (int g = 0; g < G; ++g) {
      const __nv_bfloat16* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
      gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
      gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                 GATE_RANK, s);
      gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                 GATE_RANK, s);
      pk::gate_apply(rows_.h[g], rows_.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                     s);
    }
  };

  pk::widen_ids(tokens, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  // image tokens: the vision tower's rows take the place of the placeholder's embedding (model.forward's
  // masked_scatter into the normed embeddings); every later read of the embedding block sees them
  for (const EmbedRows& r : rows) {
    if (r.row < 0 || r.count < 0 || r.row + r.count > T) throw std::invalid_argument("Prefiller::run: embedding rows outside the chunk");
    pk::put_rows(r.src, r.count, B0_ + std::size_t(r.row) * H, s);
  }
  int prev_v = -1;
  int tap = 0;

  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer, p = VISITS[v].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
      base = part_, write = y2_;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + 4 * std::size_t(pv);
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
      if (i == 0) write = nullptr;
    }
    // ---- attention read(s)
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
      pk::GateRows rows_;
      for (int g = 0; g < 3; ++g) rows_.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows_.msg[g] = msgs_ + g * slab;
      gates(rows_, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3),
            xin_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }
    if (tap < 3 && v == TAP_VISITS[tap]) {
      if (tap_out) pk::store_tap(hid, T, tap_out + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
      ++tap;
    }

    // ---- projections (fast.proj_layout): one GEMM for a single input, else per input segment
    const int R = proj_rows(layer.full);
    const __nv_bfloat16* wp = bf16_rows(layer.wproj);
    const bool single = !layer.full && !msgs;
    if (single) {
      gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
    } else {
      struct Part {
        int begin, end, input;
      };
      const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                            {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                            {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                            {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                            {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
      for (const Part& part : parts) {
        if (part.begin >= R) break;
        gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                   R, false, T, part.end - part.begin, H, s);
      }
    }

    // ---- attention
    pk::AttnPrepParams ap;
    ap.proj = proj_, ap.ld = R, ap.T = T, ap.pos0 = pos0, ap.full = layer.full;
    ap.q_scale = f32p(layer.q_scale), ap.r_scale = f32p(layer.r_scale);
    ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
    ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
    ap.conv_k = st.conv_k_at(v), ap.conv_v = st.conv_v_at(v);
    ap.q = q_, ap.rel = rel16_;
    // the KV-head split caches the last 3 of the 8 heads on the second device: a full visit then writes and
    // attends this device's [0, kv_heads); the sliding rings and the one-device layout keep all 8 here
    const int kv = layer.full ? st.kv_heads : KV_HEADS;
    ap.head_count = kv, ap.cache_heads = kv;
    pk::AttnParams at;
    at.q = q_, at.T = T, at.pos0 = pos0, at.rel = rel_, at.proj = proj_, at.ld = R, at.out = attn_out_;
    at.head_count = kv, at.cache_heads = kv;
    if (layer.full) {
      const int fi = FULL_INDEX[v];
      const std::size_t es = st.kv_fp8 ? 1 : 2, off = std::size_t(pos0) * kv * HEAD_DIM * es;
      void* kc = st.k_full_at(fi).data;
      void* vc = st.v_full_at(fi).data;
      ap.k_out = static_cast<char*>(kc) + off, ap.v_out = static_cast<char*>(vc) + off, ap.out_fp8 = st.kv_fp8;
      at.k = kc, at.v = vc, at.kv_fp8 = st.kv_fp8, at.kpos = nullptr, at.bounds = nullptr, at.L = pos0 + T;
      at.window = 0;
      if (st.kv_heads_hi != 0) {
        full_attn_split(st, layer, ap, at, T, pos0, v, s);
      } else {
        pk::attn_prep(ap, s);
        pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
        pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);
        pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
        pk::attention(at, s);
      }
    } else {
      const int ri = RING_INDEX[v];
      auto* k_ring = static_cast<__nv_bfloat16*>(st.k_ring_at(ri).data);
      auto* v_ring = static_cast<__nv_bfloat16*>(st.v_ring_at(ri).data);
      std::int64_t* ring_pos = st.ring_pos_at(ri);
      // keys = cat(ring, chunk) as model.prefill reads them, before the ring takes the chunk's last rows
      pk::ring_gather(k_ring, v_ring, ring_pos, pos0, T, krows_, vrows_, kpos_, kpos_rows_, s);
      ap.k_out = krows_ + std::size_t(WINDOW) * KV_DIM, ap.v_out = vrows_ + std::size_t(WINDOW) * KV_DIM;
      ap.out_fp8 = false;
      pk::attn_prep(ap, s);
      pk::ring_update(krows_, vrows_, pos0, T, k_ring, v_ring, ring_pos, s);
      pk::key_tile_bounds(kpos_, WINDOW + T, bounds_, s);
      at.k = krows_, at.v = vrows_, at.kv_fp8 = false, at.kpos = kpos_, at.bounds = bounds_, at.L = WINDOW + T;
      at.window = WINDOW;
      pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
      pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);
      pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
      pk::attention(at, s);
    }

    // ---- output projection + post-attention conv
    const __nv_bfloat16* wo = bf16_rows(layer.wo);
    gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_out_, H, false, T, H, HEADS * HEAD_DIM, s);
    pk::static_conv(wo_out_, H, T, H, f32p(layer.attn_conv), st.conv_attn_at(v), y_, s);
    pk::conv_state_update(st.conv_attn_at(v), wo_out_, H, T, H, s);

    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_, f32p(layer.mlp_query), 1, nullptr,
          nullptr, mixm_);
      pk::GateRows rows_;
      rows_.h[0] = mixm_, rows_.msg[0] = msgs_ + 3 * slab;
      gates(rows_, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1),
            xm_);
    } else {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_, f32p(layer.mlp_query), 1,
          f32p(layer.mlp_norm1), xm_, nullptr);
    }

    // ---- routed experts (model._moe_grouped) + post-MLP conv
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
    pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_route(idx_, T, work, s);
    const Experts& ex = layer.experts;
    if (ex.split == 0) {  // EP: every expert of this layer is on the second device
      expert_hop(ex, T, s);
    } else if (ex.split != EXPERTS) {  // 64/64: each device runs the grouped GEMMs of its half
      expert_split_hop(ex, T, s);
    } else {
      const pk::MoeExperts me{ex.fmt == ExpertFormat::NVFP4 ? 0 : 1, 0, EXPERTS, ex.w13.data, ex.w13_scale.data,
                              f32p(ex.w13_factor), ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor)};
      pk::moe_gate_up(me, xm_, T, work, act_, s);
      pk::moe_down(me, act_, T, work, down_, s);
      pk::moe_combine(down_, wts_, T, moe_sum_, s);
    }
    pk::static_conv(moe_sum_, H, T, H, f32p(layer.mlp_conv), st.conv_mlp_at(v), y2_, s);
    pk::conv_state_update(st.conv_mlp_at(v), moe_sum_, H, T, H, s);
    prev_v = v;
  }

  // ---- final read, taps, head of the last row
  mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H, mlp_w + 4 * std::size_t(prev_v), nullptr,
      f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr);
  if (tap_out)
    CUDA_CHECK(cudaMemcpy2DAsync(tap_out + std::size_t(NUM_TAPS - 1) * H, std::size_t(NUM_TAPS) * H * 2, xf_,
                                 std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, s));
  pk::head_logits(xf_ + std::size_t(T - 1) * H, w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB, logits, s);
  CUDA_CHECK(cudaGetLastError());
  st.length = pos0 + T;
}

void Prefiller::expert_hop(const Experts& ex, int T, cudaStream_t stream) {
  // EP (plans/20261124_ingine-vs-maya-ds.md §4.2): the layer's experts are on the second device and no peer
  // access is enabled, so the visit's expert work crosses over pinned host staging. This device's router
  // output — the tile list and the expert-sorted pair lists — and xm go over; the second device runs the
  // grouped gate/up and down GEMMs and the slot-ordered combine, and one bf16 [T, H] block of combined rows
  // comes back for the post-MLP conv here. Every value it touches is a copy of a GPU-0 value and the combine
  // still sums the slots in order, so the result is bit-identical to the one-device path.
  if (second_ == nullptr) throw std::logic_error("expert_hop: this Prefiller has no second device");
  const std::size_t xb = std::size_t(T) * H * 2, pb = std::size_t(T) * TOP_K * 4;
  const std::size_t tb = std::size_t(tiles_capacity_) * sizeof(int4);
  char* work = static_cast<char*>(hop_work_.data());
  char* pair_token = work + 4 + tb;
  char* pair_dst = pair_token + pb;
  char* wts = pair_dst + pb;
  CUDA_CHECK(cudaMemcpyAsync(hop_x_.data(), xm_, xb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(work, tiles_n_, 4, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(work + 4, tiles_, tb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(pair_token, pair_token_, pb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(pair_dst, pair_dst_, pb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(wts, wts_, pb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(hop_ready_, stream));
  {
    const BindDevice bind(second_->device);
    __nv_bfloat16* x1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[0].p);
    int* tiles_n1 = static_cast<int*>(hop_hi_.buffers[1].p);
    int4* tiles1 = static_cast<int4*>(hop_hi_.buffers[2].p);
    int* pair_token1 = static_cast<int*>(hop_hi_.buffers[3].p);
    int* pair_dst1 = static_cast<int*>(hop_hi_.buffers[4].p);
    float* wts1 = static_cast<float*>(hop_hi_.buffers[5].p);
    __nv_bfloat16* act1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[6].p);
    __nv_bfloat16* rows1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[7].p);
    __nv_bfloat16* sum1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[8].p);
    CUDA_CHECK(cudaStreamWaitEvent(hop_stream_, hop_ready_, 0));
    CUDA_CHECK(cudaMemcpyAsync(x1, hop_x_.data(), xb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(tiles_n1, work, 4, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(tiles1, work + 4, tb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(pair_token1, pair_token, pb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(pair_dst1, pair_dst, pb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(wts1, wts, pb, cudaMemcpyHostToDevice, hop_stream_));
    const pk::MoeWork work1{nullptr, tiles_n1, tiles1, pair_token1, pair_dst1};
    const pk::MoeExperts me1{ex.fmt == ExpertFormat::NVFP4 ? 0 : 1, 0, EXPERTS, ex.w13_hi.data,
                             ex.w13_scale_hi.data, f32p(ex.w13_factor), ex.w2_hi.data, ex.w2_scale_hi.data,
                             f32p(ex.w2_factor)};
    pk::moe_gate_up(me1, x1, T, work1, act1, hop_stream_);
    pk::moe_down(me1, act1, T, work1, rows1, hop_stream_);
    pk::moe_combine(rows1, wts1, T, sum1, hop_stream_);
    CUDA_CHECK(cudaMemcpyAsync(hop_sum_.data(), sum1, xb, cudaMemcpyDeviceToHost, hop_stream_));
    CUDA_CHECK(cudaEventRecord(hop_back_, hop_stream_));
  }
  CUDA_CHECK(cudaStreamWaitEvent(stream, hop_back_, 0));
  CUDA_CHECK(cudaMemcpyAsync(moe_sum_, hop_sum_.data(), xb, cudaMemcpyHostToDevice, stream));
}

void Prefiller::expert_split_hop(const Experts& ex, int T, cudaStream_t stream) {
  // 64/64 (ExpertResidency split 64): experts [0, split) run their grouped gate/up and down GEMMs here and
  // [split, 128) on the second device, so both GDDR buses stream the routed pairs. This device's router output
  // (the tile and expert-sorted pair lists) and xm go over; the second device's per-pair rows come back and
  // moe_rows_merge writes them into this device's, in place. moe_combine then sums the 8 slots in order here,
  // so the chunk is bit-identical to the one-device path. The kernels skip tiles of the other shard.
  if (second_ == nullptr) throw std::logic_error("expert_split_hop: this Prefiller has no second device");
  const int split = ex.split, there = EXPERTS - split;
  const int fmt = ex.fmt == ExpertFormat::NVFP4 ? 0 : 1;
  const std::size_t xb = std::size_t(T) * H * 2, pb = std::size_t(T) * TOP_K * 4;
  const std::size_t tb = std::size_t(tiles_capacity_) * sizeof(int4);
  const std::size_t rb = std::size_t(T) * TOP_K * H * 2;
  char* work = static_cast<char*>(hop_work_.data());
  char* pair_token = work + 4 + tb;
  char* pair_dst = pair_token + pb;
  char* wts = pair_dst + pb;
  CUDA_CHECK(cudaMemcpyAsync(hop_x_.data(), xm_, xb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(work, tiles_n_, 4, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(work + 4, tiles_, tb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(pair_token, pair_token_, pb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(pair_dst, pair_dst_, pb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(wts, wts_, pb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(hop_ready_, stream));
  {
    const BindDevice bind(second_->device);
    __nv_bfloat16* x1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[0].p);
    int* tiles_n1 = static_cast<int*>(hop_hi_.buffers[1].p);
    int4* tiles1 = static_cast<int4*>(hop_hi_.buffers[2].p);
    int* pair_token1 = static_cast<int*>(hop_hi_.buffers[3].p);
    int* pair_dst1 = static_cast<int*>(hop_hi_.buffers[4].p);
    float* wts1 = static_cast<float*>(hop_hi_.buffers[5].p);
    __nv_bfloat16* act1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[6].p);
    __nv_bfloat16* rows1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[7].p);
    CUDA_CHECK(cudaStreamWaitEvent(hop_stream_, hop_ready_, 0));
    CUDA_CHECK(cudaMemcpyAsync(x1, hop_x_.data(), xb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(tiles_n1, work, 4, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(tiles1, work + 4, tb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(pair_token1, pair_token, pb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(pair_dst1, pair_dst, pb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(wts1, wts, pb, cudaMemcpyHostToDevice, hop_stream_));
    const pk::MoeWork work1{nullptr, tiles_n1, tiles1, pair_token1, pair_dst1};
    const pk::MoeExperts me1{fmt, split, there, ex.w13_hi.data, ex.w13_scale_hi.data, f32p(ex.w13_factor_hi),
                             ex.w2_hi.data, ex.w2_scale_hi.data, f32p(ex.w2_factor_hi)};
    pk::moe_gate_up(me1, x1, T, work1, act1, hop_stream_);
    pk::moe_down(me1, act1, T, work1, rows1, hop_stream_);
    CUDA_CHECK(cudaMemcpyAsync(hop_rows_.data(), rows1, rb, cudaMemcpyDeviceToHost, hop_stream_));
    CUDA_CHECK(cudaEventRecord(hop_back_, hop_stream_));
  }
  // this device's half runs while the second device's does
  {
    const pk::MoeExperts me0{fmt, 0, split, ex.w13.data, ex.w13_scale.data, f32p(ex.w13_factor),
                             ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor)};
    const pk::MoeWork work0{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_gate_up(me0, xm_, T, work0, act_, stream);
    pk::moe_down(me0, act_, T, work0, down_, stream);
  }
  CUDA_CHECK(cudaStreamWaitEvent(stream, hop_back_, 0));
  CUDA_CHECK(cudaMemcpyAsync(hop_rows0_.p, hop_rows_.data(), rb, cudaMemcpyHostToDevice, stream));
  pk::moe_rows_merge(down_, static_cast<const __nv_bfloat16*>(hop_rows0_.p), idx_, T, split, stream);
  pk::moe_combine(down_, wts_, T, moe_sum_, stream);
}

void Prefiller::full_attn_split(State& st, const Layer& layer, pk::AttnPrepParams& ap, pk::AttnParams& at, int T,
                                int pos0, int v, cudaStream_t stream) {
  // One full-attention visit over both devices (Decoder::full_attn_split's protocol): this device runs KV
  // heads [0, here) against its cache shard, the second device [here, 8) against its own, each launch the
  // unchanged kernel — per KV head it owns Q/K/V completely, so merging is the disjoint query-head output
  // rows. The shard's proj columns (mirrored into a full-width row, as the decode step stages them), its conv
  // state and the layer's scale/static slices go over; its 9 query-head rows and its conv state come back,
  // the state into this device's rows [here * 128, ...) of the full-width conv state, where the decode step
  // keeps it too.
  if (second_ == nullptr || !kv_split_) throw std::logic_error("full_attn_split: no KV-head split");
  const int here = st.kv_heads, there = st.kv_heads_hi;
  const std::size_t CC = std::size_t(there) * HEAD_DIM;  // channels of the shard's cache / conv rows
  const int q0 = here * 3;                                // its first query head
  const int R = PROJ_ROWS_FULL;
  const int fi = FULL_INDEX[v];
  const std::size_t es = st.kv_fp8 ? 1 : 2;
  char* in = static_cast<char*>(kv_in_.data());
  char* back = static_cast<char*>(kv_out_.data());
  std::array<std::size_t, kVkSlices.size()> offs{};
  std::size_t slice_bytes = 0;
  for (std::size_t i = 0; i < kVkSlices.size(); ++i) {
    offs[i] = slice_bytes;
    slice_bytes += std::size_t(T) * std::size_t(kVkSlices[i].len) * 2;
  }
  float* conv_host = reinterpret_cast<float*>(in + slice_bytes);
  const std::size_t out_bytes = std::size_t(T) * std::size_t(kVkQ) * HEAD_DIM * 2;
  float* conv_back = reinterpret_cast<float*>(back + out_bytes);

  // this device's slices and the shard's conv state (cols here * 128.. of the full-width rows) into the
  // packet, once the second device is done with the previous one
  CUDA_CHECK(cudaStreamWaitEvent(stream, kv_loaded_, 0));
  for (std::size_t i = 0; i < kVkSlices.size(); ++i)
    CUDA_CHECK(cudaMemcpy2DAsync(in + offs[i], std::size_t(kVkSlices[i].len) * 2, proj_ + kVkSlices[i].col,
                                 std::size_t(R) * 2, std::size_t(kVkSlices[i].len) * 2, T, cudaMemcpyDeviceToHost,
                                 stream));
  CUDA_CHECK(cudaMemcpy2DAsync(conv_host, CC * 4, st.conv_k_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4,
                               CC * 4, CONV_K - 1, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(conv_host + std::size_t(CONV_K - 1) * CC, CC * 4,
                               st.conv_v_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4, CC * 4, CONV_K - 1,
                               cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(kv_ready_, stream));

  {
    const BindDevice bind(second_->device);
    char* proj1 = static_cast<char*>(kv_hi_.buffers[0].p);
    __nv_bfloat16* q1 = static_cast<__nv_bfloat16*>(kv_hi_.buffers[1].p);
    __nv_bfloat16* rel16_1 = static_cast<__nv_bfloat16*>(kv_hi_.buffers[2].p);
    __nv_bfloat16* rel_1 = static_cast<__nv_bfloat16*>(kv_hi_.buffers[3].p);
    __nv_bfloat16* out1 = static_cast<__nv_bfloat16*>(kv_hi_.buffers[4].p);
    float* conv1 = static_cast<float*>(kv_hi_.buffers[5].p);
    CUDA_CHECK(cudaStreamWaitEvent(hop_stream_, kv_ready_, 0));
    for (std::size_t i = 0; i < kVkSlices.size(); ++i)
      CUDA_CHECK(cudaMemcpy2DAsync(proj1 + std::size_t(kVkSlices[i].col) * 2, std::size_t(R) * 2, in + offs[i],
                                   std::size_t(kVkSlices[i].len) * 2, std::size_t(kVkSlices[i].len) * 2, T,
                                   cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(conv1, conv_host, 2 * std::size_t(CONV_K - 1) * CC * 4, cudaMemcpyHostToDevice,
                               hop_stream_));
    CUDA_CHECK(cudaEventRecord(kv_loaded_, hop_stream_));

    const KvLayer& wl = kv_layer_[layer.idx];
    const std::size_t koff = std::size_t(pos0) * CC * es;
    void* kc1 = static_cast<char*>(st.k_full_hi_at(fi).data) + koff;
    void* vc1 = static_cast<char*>(st.v_full_hi_at(fi).data) + koff;
    pk::AttnPrepParams a1 = ap;
    a1.proj = reinterpret_cast<const __nv_bfloat16*>(proj1), a1.ld = R;
    a1.q_scale = wl.q_scale, a1.r_scale = wl.r_scale;
    a1.k_w = wl.k_w, a1.v_w = wl.v_w;
    a1.conv_k = conv1, a1.conv_v = conv1 + std::size_t(CONV_K - 1) * CC;
    a1.q = q1, a1.rel = rel16_1;
    a1.k_out = kc1, a1.v_out = vc1;
    a1.head_begin = here, a1.head_count = there, a1.cache_heads = there;
    pk::attn_prep(a1, hop_stream_);
    pk::rel_logits(rel16_1, wl.rel_proj, T, rel_1, hop_stream_, q0, kVkQ);

    pk::AttnParams b1 = at;
    b1.q = q1, b1.rel = rel_1, b1.proj = a1.proj, b1.ld = R, b1.out = out1;
    b1.k = st.k_full_hi_at(fi).data, b1.v = st.v_full_hi_at(fi).data;
    b1.head_begin = here, b1.head_count = there, b1.cache_heads = there;
    pk::attention(b1, hop_stream_);
    pk::conv_state_update(conv1, a1.proj + SEG_BEGIN[SEG_K] + here * HEAD_DIM, R, T, int(CC), hop_stream_);
    pk::conv_state_update(conv1 + std::size_t(CONV_K - 1) * CC, a1.proj + SEG_BEGIN[SEG_V] + here * HEAD_DIM, R, T,
                          int(CC), hop_stream_);

    // back: the shard's query-head rows and its conv state, once this device read the previous ones
    CUDA_CHECK(cudaStreamWaitEvent(hop_stream_, kv_consumed_, 0));
    CUDA_CHECK(cudaMemcpy2DAsync(back, std::size_t(kVkQ) * HEAD_DIM * 2, out1 + std::size_t(q0) * HEAD_DIM,
                                 std::size_t(HEADS) * HEAD_DIM * 2, std::size_t(kVkQ) * HEAD_DIM * 2, T,
                                 cudaMemcpyDeviceToHost, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(conv_back, conv1, 2 * std::size_t(CONV_K - 1) * CC * 4, cudaMemcpyDeviceToHost,
                               hop_stream_));
    CUDA_CHECK(cudaEventRecord(kv_back_, hop_stream_));
  }

  // this device's heads run while the second device's do
  pk::attn_prep(ap, stream);
  pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, here * HEAD_DIM, stream);
  pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, here * HEAD_DIM, stream);
  pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, stream);
  pk::attention(at, stream);
  CUDA_CHECK(cudaStreamWaitEvent(stream, kv_back_, 0));
  CUDA_CHECK(cudaMemcpy2DAsync(attn_out_ + std::size_t(q0) * HEAD_DIM, std::size_t(HEADS) * HEAD_DIM * 2, back,
                               std::size_t(kVkQ) * HEAD_DIM * 2, std::size_t(kVkQ) * HEAD_DIM * 2, T,
                               cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(st.conv_k_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4, conv_back, CC * 4, CC * 4,
                               CONV_K - 1, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(st.conv_v_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4,
                               conv_back + std::size_t(CONV_K - 1) * CC, CC * 4, CC * 4, CONV_K - 1,
                               cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(kv_consumed_, stream));
}

}  // namespace ingine::model
