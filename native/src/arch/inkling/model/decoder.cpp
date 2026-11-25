// fast.FastDecoder.decode with fast.KernelOps: the fixed kernel sequence of one decode step for M
// consecutive tokens (48 visits: attention prologue mix (+ message gates), proj, attn, proj_conv, MLP
// prologue mix (+ gate), router, moe_up, moe_down; then the final read, norm and head).
#include "model/runtime.h"

#include "core/device.h"
#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"

#include <algorithm>
#include <stdexcept>

namespace ingine::model {
namespace dense = kernels::dense;
namespace moe = kernels::moe;
namespace attn = kernels::attn;

namespace {

dense::Matrix matrix(const Dense& d) {
  return dense::Matrix{d.w.data, static_cast<const float*>(d.scale.data), d.rows()};
}

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
float* f32m(const Tensor& t) { return static_cast<float*>(t.data); }
__nv_bfloat16* bf16m(const Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); }

// fast.proj_layout(full, single_input) as an int32 [S, 3] table (row_begin, row_end, input)
std::vector<int> proj_layout(bool full, bool single_input) {
  std::vector<int> t;
  for (int s = 0; s < num_segs(full); ++s) {
    t.push_back(SEG_BEGIN[s]);
    t.push_back(SEG_BEGIN[s + 1]);
    t.push_back(single_input ? 0 : SEG_INPUT[s]);
  }
  return t;
}

DeviceBuffer upload_ints(const std::vector<int>& v) {
  DeviceBuffer b(v.size() * sizeof(int));
  b.copy_from_host(v.data(), v.size() * sizeof(int));
  return b;
}

// The expert hop's staging is sized for the widest call width, not per width: the device streams run one hop
// at a time.
constexpr int kHopMaxM = 8;

// KV-head split staging (bytes), from the fixed shapes: proj slices mirrored into a full-width row, the conv
// state of the shard's heads, pos, and the shard's query-head output.
constexpr int kVkHeads = KV_SPLIT_SECOND_HEADS;
constexpr int kVkQ = kVkHeads * 3;  // query heads of those KV heads
constexpr std::size_t kVkProj = std::size_t(kHopMaxM) * PROJ_ROWS_FULL * 4;
constexpr std::size_t kVkConv = 2 * std::size_t(CONV_K - 1) * kVkHeads * HEAD_DIM * 4;
constexpr std::size_t kVkOut = std::size_t(kHopMaxM) * kVkQ * HEAD_DIM * 2;
constexpr std::size_t kVkIn = kVkProj + kVkConv + sizeof(std::int64_t);
constexpr std::size_t kVkOutHost = kVkOut + kVkConv;
// q_scale (all 24 heads, absolute indexing), r_scale, rel_proj (distance-addressed), and the compact
// static-conv rows of the shard's heads, for every full layer
constexpr std::size_t kVkLayerWeights = std::size_t(HEADS) * HEAD_DIM * 4 + std::size_t(HEADS) * D_REL * 4 +
                                       std::size_t(D_REL) * REL_EXTENT * 4 +
                                       2 * std::size_t(CONV_K) * kVkHeads * HEAD_DIM * 4;

// Makes `device` current for the scope, restoring the caller's on exit (the hop's device-1 copies and
// launches). Never used inside stream capture: a split visit's second-device half is captured on its own.
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

}  // namespace

Decoder::Decoder(const Weights& w, const Ablation& abl, State& st, bool pdl, const DeviceContext* second)
    : w_(w), abl_(abl), st_(st), pdl_(pdl), attn_ws_(attn::workspace_bytes(8)), second_(second) {
  attn_ws_.fill(0);  // zeroed once; the kernel resets its counter itself
  segments_[0] = upload_ints(proj_layout(false, true));
  segments_[1] = upload_ints(proj_layout(false, false));
  full_segments_ = upload_ints(proj_layout(true, false));
  kv_split_ = st_.kv_heads_hi > 0 && second_ != nullptr;
  if (kv_split_) {
    int nfull = 0;
    for (const Layer& l : w.layers) nfull += l.full ? 1 : 0;
    kv_.stream = second_->stream;
    kv_.ws_bytes = attn::workspace_bytes(8);
    kv_.in = MappedHostBuffer(kVkIn);
    kv_.out_host = MappedHostBuffer(kVkOutHost);
    kv_.w_host = MappedHostBuffer(std::size_t(nfull) * kVkLayerWeights);
    CUDA_CHECK(cudaEventCreateWithFlags(&kv_.ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&kv_.consumed, cudaEventDisableTiming));
    {
      const BindDevice bind(second_->device);
      kv_hi_.device = second_->device;
      kv_hi_.buffers.emplace_back(kVkProj);
      kv_hi_.buffers.emplace_back(kVkConv);
      kv_hi_.buffers.emplace_back(sizeof(std::int64_t));
      kv_hi_.buffers.emplace_back(kVkOut);
      kv_hi_.buffers.emplace_back(kv_.ws_bytes);
      kv_hi_.buffers.emplace_back(std::size_t(nfull) * kVkLayerWeights);
      CUDA_CHECK(cudaEventCreateWithFlags(&kv_.loaded, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&kv_.back, cudaEventDisableTiming));
      CUDA_CHECK(cudaMemset(kv_hi_.buffers[4].p, 0, kv_.ws_bytes));  // counters: the kernel resets its own
    }
    kv_.proj = kv_hi_.buffers[0].p;
    kv_.conv = kv_hi_.buffers[1].p;
    kv_.pos = kv_hi_.buffers[2].p;
    kv_.out = kv_hi_.buffers[3].p;
    kv_.ws = kv_hi_.buffers[4].p;
    kv_.weights = kv_hi_.buffers[5].p;
    int o = 0;
    for (const Layer& l : w.layers) {
      if (!l.full) continue;
      char* base = static_cast<char*>(kv_.weights) + std::size_t(o++) * kVkLayerWeights;
      auto take = [&](std::size_t bytes) {
        const float* p = reinterpret_cast<const float*>(base);
        base += bytes;
        return p;
      };
      const int l0 = int(l.idx);
      kv_.layer[l0].q_scale = take(std::size_t(HEADS) * HEAD_DIM * 4);
      kv_.layer[l0].r_scale = take(std::size_t(HEADS) * D_REL * 4);
      kv_.layer[l0].rel_proj = take(std::size_t(D_REL) * REL_EXTENT * 4);
      kv_.layer[l0].k_w = take(std::size_t(CONV_K) * kVkHeads * HEAD_DIM * 4);
      kv_.layer[l0].v_w = take(std::size_t(CONV_K) * kVkHeads * HEAD_DIM * 4);
    }
  }
  if (second_ == nullptr) return;
  // Nothing is allocated unless some layer really keeps experts there (`--second-gpu` alone keeps today's
  // one-device step, and then not even the staging below exists). A whole layer there and a 64/64 split share
  // the staging; only the split needs the second device's shard rows on this one.
  bool remote = false, shard = false;
  for (const Layer& l : w.layers) {
    remote = remote || l.experts.split == 0;
    shard = shard || (l.experts.split != 0 && l.experts.split != EXPERTS);
  }
  if (!remote && !shard) return;
  // The hop (plans/20261124_hop-measure.md): portable pinned staging, one stream and one event pair. Each
  // event belongs to the device that records it: hop_ready_ to this one, hop_back_ to the second.
  hop_stream_ = second_->stream;
  hop_x_ = MappedHostBuffer(std::size_t(kHopMaxM) * HIDDEN * 2);            // bf16 [m, H]
  hop_idx_ = MappedHostBuffer(std::size_t(kHopMaxM) * TOP_K * 4);           // int32 [m, 8]
  hop_rows_ = MappedHostBuffer(std::size_t(kHopMaxM) * TOP_K * HIDDEN * 2);  // bf16 [m, 8, H]
  hop_rows0_ = DeviceBuffer(std::size_t(kHopMaxM) * TOP_K * HIDDEN * 2);
  // The split's rows are merged here, so this one is a primary allocation: made before the bind below, which
  // makes the second device current for the hop's own staging.
  if (shard) hop_rows1_ = DeviceBuffer(std::size_t(kHopMaxM) * TOP_K * HIDDEN * 2);
  CUDA_CHECK(cudaEventCreateWithFlags(&hop_ready_, cudaEventDisableTiming));
  const BindDevice bind(second_->device);
  CUDA_CHECK(cudaEventCreateWithFlags(&hop_back_, cudaEventDisableTiming));
  hop_hi_.device = second_->device;
  hop_hi_.buffers.emplace_back(std::size_t(kHopMaxM) * HIDDEN * 2);          // xm
  hop_hi_.buffers.emplace_back(std::size_t(kHopMaxM) * TOP_K * 4);           // idx
  hop_hi_.buffers.emplace_back(std::size_t(kHopMaxM) * TOP_K * INTER * 2);   // act
  hop_hi_.buffers.emplace_back(std::size_t(kHopMaxM) * TOP_K * HIDDEN * 2);  // per-slot rows
}

Decoder::~Decoder() {
  // Before the second device goes: the child graphs were captured on it (their nodes' kernels live there).
  for (auto* slot : {&hop_child_, &kv_child_})
    for (auto& [m, per_visit] : *slot)
      for (cudaGraph_t& g : per_visit)
        if (g) cudaGraphDestroy(g);
  if (second_ == nullptr) return;
  {
    const BindDevice bind(second_->device);  // the second device's events and storage
    if (hop_back_) (void)cudaEventDestroy(hop_back_);
    if (kv_.loaded) (void)cudaEventDestroy(kv_.loaded);
    if (kv_.back) (void)cudaEventDestroy(kv_.back);
    hop_hi_.release();
    kv_hi_.release();
  }
  if (hop_ready_) (void)cudaEventDestroy(hop_ready_);
  if (kv_.ready) (void)cudaEventDestroy(kv_.ready);
  if (kv_.consumed) (void)cudaEventDestroy(kv_.consumed);
}

const std::int64_t* Decoder::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    // built on first use, i.e. in the eager warm-up before capture (KernelOps._block_ptrs)
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const std::int64_t*>(it->second.p);
}

cudaGraph_t Decoder::capture_child(const std::function<void()>& body) {
  // Called with the second device current and no capture active on this thread: its kernels need that
  // device, and cudaSetDevice is not allowed inside a capture.
  CUDA_CHECK(cudaStreamBeginCapture(hop_stream_, cudaStreamCaptureModeThreadLocal));
  try {
    body();
  } catch (...) {
    cudaGraph_t out = nullptr;
    cudaStreamEndCapture(hop_stream_, &out);
    if (out) cudaGraphDestroy(out);
    throw;
  }
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(hop_stream_, &graph));
  return graph;
}

void Decoder::keep_child(std::map<int, std::array<cudaGraph_t, NUM_VISITS>>& slot, int m, int v,
                         cudaGraph_t graph) {
  cudaGraph_t& held = slot[m][v];
  if (held) cudaGraphDestroy(held);
  held = graph;
}

cudaGraph_t Decoder::child_at(const std::map<int, std::array<cudaGraph_t, NUM_VISITS>>& slot, int m, int v) const {
  auto it = slot.find(m);
  if (it == slot.end() || it->second[v] == nullptr)
    throw std::logic_error("decode capture: no second-device graph for width " + std::to_string(m) + ", visit " +
                           std::to_string(v));
  return it->second[v];
}

cudaGraphNode_t Decoder::fork_child(cudaStream_t stream, cudaGraph_t child) {
  cudaStreamCaptureStatus status;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  std::size_t n = 0;
  CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, &edges, &n));
  if (status != cudaStreamCaptureStatusActive) throw std::logic_error("fork_child: the stream is not capturing");
  cudaGraphNode_t node = nullptr;
  CUDA_CHECK(cudaGraphAddChildGraphNode(&node, graph, deps, n, child));
  return node;
}

void Decoder::join_child(cudaStream_t stream, cudaGraphNode_t node) {
  CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1, cudaStreamAddCaptureDependencies));
}

void Decoder::decode(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream, Capture capture) {
  struct Restore {
    Capture& slot;
    Capture saved;
    ~Restore() { slot = saved; }
  } restore{capture_, capture_};
  capture_ = capture;
  const int m = b.m;
  const int H = HIDDEN;
  const bool pdl = pdl_;
  const Weights& w = w_;
  const Ablation& abl = abl_;
  State& st = st_;
  const std::size_t slab = std::size_t(m) * H;  // one [m, H] slab of a [G, m, H] buffer

  auto mix_params = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                        const float* abl_r, const float* abl_w, float* partial_out, const Tensor* queries,
                        const float* norm_w, __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : reinterpret_cast<const float* const*>(block_table(blocks));
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    // KernelOps.mix: the ablation applies to `write` only
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;  // the detector of the same visit
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries ? f32p(*queries) : nullptr;
    p.Q = queries ? queries->ne[1] : 1;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = m;
    return p;
  };

  dense::embed(static_cast<const std::int64_t*>(b.token.data), m, static_cast<const __nv_bfloat16*>(w.embed.data),
               f32p(w.embed_norm), f32m(b.B0), pdl, stream);
  int prev_v = -1;
  int tap = 0;
  const float* attn_dirs = f32p(abl.attn_dirs);
  const float* attn_w = f32p(abl.attn_w);
  const float* mlp_dirs = f32p(abl.mlp_dirs);
  const float* mlp_w = f32p(abl.mlp_w);
  __nv_bfloat16* taps = bf16m(b.taps);

  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer, p = VISITS[v].pass;
    const Layer& layer = w.layers[l];
    const bool loop = is_loop_layer(l);
    const bool msgs = loop && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = b.blocks(l, p);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = f32m(b.B0), base = f32p(b.B0), write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = f32m(b.B1), base = f32p(b.B1), write = nullptr;
    } else {
      hid = l == 6 ? f32m(b.B1) : l == 12 ? f32m(b.B2[p]) : l == 18 ? f32m(b.E[2]) : f32m(b.hid);
      base = f32p(b.part), write = f32p(b.y2);
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + 4 * std::size_t(pv);
    dense::MixParams msg_mix;  // message visits: launched together with the attention read (independent of it)
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? f32p(b.part) : f32p(b.E[p - 1]);
      const float* src_write = i == 0 ? f32p(b.y2) : nullptr;
      // messages from pass p-1: its layer-12 block and its final hidden (E[p-1], completed at i == 0)
      const Tensor mq = rm(f32m(w.message_queries) + std::size_t(i) * 4 * H, DType::FP32, {4, H});
      msg_mix = mix_params({f32p(b.B2[p - 1])}, src_base, src_write, r_prev, w_prev,
                           i == 0 ? f32m(b.E[p - 1]) : nullptr, &mq, nullptr, nullptr, f32m(b.msgs));
      if (i == 0) write = nullptr;  // pass p-1's last write went into E[p-1]; layer 6 restarts from B1
    }
    // ---- attention read(s); at a tap visit the same launch writes b.taps[:, tap] = bf16(hid)
    const Tensor* queries = layer.attn_query.data ? &layer.attn_query : nullptr;
    float* partial_out = hid != base ? hid : nullptr;
    dense::MixParams ap =
        msgs ? mix_params(blocks, base, write, r_prev, w_prev, partial_out, queries, nullptr, nullptr, f32m(b.mixf))
             : mix_params(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr,
                          f32p(layer.attn_norm_mix), bf16m(b.xin), nullptr);
    if (tap < 3 && v == TAP_VISITS[tap]) {
      ap.new_bf16 = taps + std::size_t(tap) * H;
      ap.new_bf16_stride = std::int64_t(NUM_TAPS) * H;
      ++tap;
    }
    if (msgs) dense::mix_pair(ap, msg_mix, pdl, stream);
    else dense::mix(ap, pdl, stream);
    if (msgs)  // sliding layers mix one query: all three gates read b.mixf[0]
      dense::message_gates(f32p(b.mixf), !layer.full, f32p(b.msgs), 3, m,
                           static_cast<const __nv_bfloat16*>(layer.qkv_gate_down.data), layer.qkv_gate_forget16.data,
                           layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3), bf16m(b.xin), nullptr, pdl,
                           stream);
    // ---- attention
    const bool single = !layer.full && !msgs;
    const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
    const Tensor& proj_out = layer.full ? b.proj_full : b.proj_sliding;
    const int R = proj_rows(layer.full);
    dense::proj(bf16m(b.xin), m, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                f32m(proj_out), pdl, stream);
    // b.hist_k/hist_v[v] (K and V segments of proj_out) are copied by the MLP-side mix below
    attn::DecodeArgs a;
    a.proj = f32p(proj_out);
    a.M = m;
    a.q_scale = f32p(layer.q_scale);
    a.r_scale = f32p(layer.r_scale);
    a.rel_proj = f32p(layer.rel_proj);
    a.conv_k_state = st.conv_k_at(v);
    a.conv_v_state = st.conv_v_at(v);
    a.pos = pos;
    a.workspace = attn_ws_.p;
    a.workspace_size = attn_ws_.bytes;
    a.out = bf16m(b.attn_out);
    if (layer.full) {
      const int fi = FULL_INDEX[v];
      a.k_conv = f32p(layer.k_static);
      a.v_conv = f32p(layer.v_static);
      a.k_cache = st.k_full_at(fi).data;
      a.v_cache = st.v_full_at(fi).data;
      a.cache_fp8 = st.kv_fp8;
      a.cache_rows = st.max_ctx;
      a.kpos = nullptr;
      a.window = 0;
      a.log_scaling = true;
      a.head_begin = 0;
      a.head_count = st.kv_heads;  // 8, or 5 with --kv-split
      a.cache_heads = st.kv_heads;
      a.out_head0 = 0;
      a.out_stride = HEADS * HEAD_DIM;
    } else {
      const int ri = RING_INDEX[v];
      a.k_conv = f32p(layer.k_conv);
      a.v_conv = f32p(layer.v_conv);
      a.k_cache = st.k_ring_at(ri).data;
      a.v_cache = st.v_ring_at(ri).data;
      a.cache_fp8 = false;
      a.cache_rows = WINDOW;
      a.kpos = st.ring_pos_at(ri);
      a.window = WINDOW;
      a.log_scaling = false;
    }
    if (layer.full && kv_split_) full_attn_split(layer, b, m, v, a, stream);
    else attn::attn_decode(a, stream);
    dense::proj_conv(bf16m(b.attn_out), m, matrix(layer.wo), f32p(layer.attn_conv), st.conv_attn_at(v), f32m(b.y),
                     history ? f32m(b.hist_attn) + std::size_t(v) * m * H : nullptr, pdl, stream);
    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    dense::MixParams mp =
        msgs ? mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), f32m(b.part),
                          &layer.mlp_query, nullptr, nullptr, f32m(b.mixm))
             : mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), f32m(b.part),
                          &layer.mlp_query, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr);
    if (history) {  // b.hist_k/hist_v[v] = proj_out[:, K/V segment]
      int c = 0;
      for (auto [dst, s] : {std::pair{&b.hist_k, SEG_K}, std::pair{&b.hist_v, SEG_V}})
        mp.copies[c++] = dense::MixRowCopy{f32p(proj_out) + SEG_BEGIN[s], R, f32m(*dst) + std::size_t(v) * m * KV_DIM,
                                           KV_DIM, KV_DIM};
    }
    dense::mix(mp, pdl, stream);
    if (msgs)
      dense::message_gates(f32p(b.mixm), false, f32p(b.msgs) + 3 * slab, 1, m,
                           static_cast<const __nv_bfloat16*>(layer.mlp1_gate_down.data), layer.mlp1_gate_forget16.data,
                           layer.mlp1_gate_gain16.data, false, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr, pdl,
                           stream);
    dense::router(bf16m(b.xm), m, layer.router_bf16.data, false, f32p(layer.router_bias),
                  static_cast<int*>(b.idx.data), f32m(b.wts), pdl, stream);
    const Experts& ex = layer.experts;
    const moe::Format fmt = ex.fmt == ExpertFormat::NVFP4 ? moe::Format::NVFP4 : moe::Format::MXFP8;
    float* conv_hist = history ? f32m(b.hist_mlp) + std::size_t(v) * m * H : nullptr;
    if (ex.split == 0) {  // EP: every expert of this layer is on the second device
      expert_hop(ex, b, m, v, f32p(layer.mlp_conv), st.conv_mlp_at(v), conv_hist, stream);
    } else if (ex.split != EXPERTS) {  // 64/64: both devices stream their half of the routed experts
      expert_split_hop(ex, b, m, v, f32p(layer.mlp_conv), st.conv_mlp_at(v), conv_hist, stream);
    } else {
      moe::moe_up(bf16m(b.xm), m, static_cast<const int*>(b.idx.data), ex.w13.data, ex.w13_scale.data,
                  f32p(ex.w13_factor), fmt, bf16m(b.act), pdl, stream);
      // with PDL, moe_down prefetches weights before waiting: valid because moe_up always precedes it
      moe::moe_down(bf16m(b.act), m, static_cast<const int*>(b.idx.data), f32p(b.wts), ex.w2.data, ex.w2_scale.data,
                    f32p(ex.w2_factor), fmt, f32p(layer.mlp_conv), st.conv_mlp_at(v), f32m(b.y2), conv_hist, pdl,
                    stream);
    }
    prev_v = v;
  }
  // ---- final read; the same launch writes b.taps[:, 3] = b.xf
  const Tensor final_norm1 = rm(w.final_norm.data, DType::FP32, {1, H});
  dense::MixParams fp = mix_params({f32p(b.B0), f32p(b.B1), f32p(b.B2[2]), f32p(b.E[2])}, f32p(b.part), f32p(b.y2),
                                   mlp_dirs + std::size_t(prev_v) * H, mlp_w + 4 * std::size_t(prev_v), nullptr, &w.final_query,
                                   f32p(final_norm1), bf16m(b.xf), nullptr);
  fp.out_copy = taps + std::size_t(tap) * H;
  fp.out_copy_stride = std::int64_t(NUM_TAPS) * H;
  dense::mix(fp, pdl, stream);
  dense::head(bf16m(b.xf), m, matrix(w.unembed), f32m(b.logits), pdl, stream);
}

void Decoder::expert_hop(const Experts& ex, Buffers& b, int m, int v, const float* conv_w, float* conv_state,
                         float* conv_hist, cudaStream_t stream) {
  // EP step 4 (plans/20261124_ingine-vs-maya-ds.md §4.2/§7.4): the layer's experts are on the second device
  // and no peer access is enabled, so the visit's expert work goes over pinned host staging, the protocol
  // apps/hop measured at 27.1 us per round trip. GPU0 sends xm (bf16 [m, 2304], 4.6 KiB at m = 1) and idx;
  // the second device runs moe_up and moe_down's expert pass; one BF16 row per slot (36 KiB at m = 1) comes
  // back; GPU0 does the slot-ordered FP32 weighted sum and the post-MLP conv here. Every value GPU1 touches
  // is a copy of a GPU-0 value, so the result is bit-identical to the one-device path.
  if (second_ == nullptr) throw std::logic_error("expert_hop: this decoder has no second device");
  const moe::Format fmt = ex.fmt == ExpertFormat::NVFP4 ? moe::Format::NVFP4 : moe::Format::MXFP8;
  const std::size_t xb = std::size_t(m) * HIDDEN * 2;
  const std::size_t ib = std::size_t(m) * TOP_K * 4;
  const std::size_t rb = std::size_t(m) * TOP_K * HIDDEN * 2;
  __nv_bfloat16* xm1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[0].p);
  int* idx1 = static_cast<int*>(hop_hi_.buffers[1].p);
  __nv_bfloat16* act1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[2].p);
  __nv_bfloat16* rows1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[3].p);
  CUDA_CHECK(cudaMemcpyAsync(hop_x_.data(), bf16m(b.xm), xb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(hop_idx_.data(), b.idx.data, ib, cudaMemcpyDeviceToHost, stream));
  auto second_half = [&] {
    CUDA_CHECK(cudaMemcpyAsync(xm1, hop_x_.data(), xb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(idx1, hop_idx_.data(), ib, cudaMemcpyHostToDevice, hop_stream_));
    moe::moe_up(xm1, m, idx1, ex.w13_hi.data, ex.w13_scale_hi.data, f32p(ex.w13_factor), fmt, act1, false,
                hop_stream_);
    moe::moe_down_rows(act1, m, idx1, ex.w2_hi.data, ex.w2_scale_hi.data, f32p(ex.w2_factor), fmt, rows1, false,
                       hop_stream_);
    CUDA_CHECK(cudaMemcpyAsync(hop_rows_.data(), rows1, rb, cudaMemcpyDeviceToHost, hop_stream_));
  };
  if (capture_ == Capture::kStep) {  // the visit's hop is the captured second-device graph
    join_child(stream, fork_child(stream, child_at(hop_child_, m, v)));
  } else if (capture_ == Capture::kHopChildren) {
    const BindDevice bind(second_->device);
    keep_child(hop_child_, m, v, capture_child(second_half));
  } else {
    CUDA_CHECK(cudaEventRecord(hop_ready_, stream));
    {
      const BindDevice bind(second_->device);
      CUDA_CHECK(cudaStreamWaitEvent(hop_stream_, hop_ready_, 0));
      second_half();
      CUDA_CHECK(cudaEventRecord(hop_back_, hop_stream_));
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, hop_back_, 0));
  }
  CUDA_CHECK(cudaMemcpyAsync(hop_rows0_.p, hop_rows_.data(), rb, cudaMemcpyHostToDevice, stream));
  moe::moe_sum(static_cast<const __nv_bfloat16*>(hop_rows0_.p), m, f32p(b.wts), conv_w, conv_state, f32m(b.y2),
               conv_hist, stream);
}

void Decoder::expert_split_hop(const Experts& ex, Buffers& b, int m, int v, const float* conv_w, float* conv_state,
                               float* conv_hist, cudaStream_t stream) {
  // 64/64 (ExpertResidency split 64): experts [0, split) stream here and [split, 128) on the second device, so
  // both GDDR buses read the routed 8 per token. Each device runs moe_up and moe_down's expert pass for the
  // slots that chose its experts (the kernels skip the other shard, and address their rows by the local id);
  // the second device's per-slot rows come back, moe_rows_merge writes them into this device's row buffer and
  // moe_sum sums all 8 slots in k order with the post-MLP conv. The rows are the same values the fused epilogue
  // would read, so the result is bit-identical to the one-device path.
  if (second_ == nullptr) throw std::logic_error("expert_split_hop: this decoder has no second device");
  const int split = ex.split, there = EXPERTS - split;
  const moe::Format fmt = ex.fmt == ExpertFormat::NVFP4 ? moe::Format::NVFP4 : moe::Format::MXFP8;
  const std::size_t xb = std::size_t(m) * HIDDEN * 2;
  const std::size_t ib = std::size_t(m) * TOP_K * 4;
  const std::size_t rb = std::size_t(m) * TOP_K * HIDDEN * 2;
  const int* idx = static_cast<const int*>(b.idx.data);
  __nv_bfloat16* xm1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[0].p);
  int* idx1 = static_cast<int*>(hop_hi_.buffers[1].p);
  __nv_bfloat16* act1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[2].p);
  __nv_bfloat16* rows1 = static_cast<__nv_bfloat16*>(hop_hi_.buffers[3].p);
  __nv_bfloat16* rows0 = static_cast<__nv_bfloat16*>(hop_rows0_.p);
  __nv_bfloat16* rows_hi = static_cast<__nv_bfloat16*>(hop_rows1_.p);

  CUDA_CHECK(cudaMemcpyAsync(hop_x_.data(), bf16m(b.xm), xb, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(hop_idx_.data(), b.idx.data, ib, cudaMemcpyDeviceToHost, stream));
  auto second_half = [&] {
    CUDA_CHECK(cudaMemcpyAsync(xm1, hop_x_.data(), xb, cudaMemcpyHostToDevice, hop_stream_));
    CUDA_CHECK(cudaMemcpyAsync(idx1, hop_idx_.data(), ib, cudaMemcpyHostToDevice, hop_stream_));
    moe::moe_up(xm1, m, idx1, ex.w13_hi.data, ex.w13_scale_hi.data, f32p(ex.w13_factor_hi), fmt, act1, false,
                hop_stream_, split, there);
    moe::moe_down_rows(act1, m, idx1, ex.w2_hi.data, ex.w2_scale_hi.data, f32p(ex.w2_factor_hi), fmt, rows1, false,
                       hop_stream_, split, there);
    CUDA_CHECK(cudaMemcpyAsync(hop_rows_.data(), rows1, rb, cudaMemcpyDeviceToHost, hop_stream_));
  };
  // The hop forks after this device's input copies and joins after its own half, so the two devices' expert
  // passes overlap as they do eagerly.
  cudaGraphNode_t hop_node = nullptr;
  if (capture_ == Capture::kStep) {
    hop_node = fork_child(stream, child_at(hop_child_, m, v));
  } else if (capture_ == Capture::kHopChildren) {
    const BindDevice bind(second_->device);
    keep_child(hop_child_, m, v, capture_child(second_half));
  } else {
    CUDA_CHECK(cudaEventRecord(hop_ready_, stream));
    {
      const BindDevice bind(second_->device);
      CUDA_CHECK(cudaStreamWaitEvent(hop_stream_, hop_ready_, 0));
      second_half();
      CUDA_CHECK(cudaEventRecord(hop_back_, hop_stream_));
    }
  }
  // this device's half runs while the second device's does
  moe::moe_up(bf16m(b.xm), m, idx, ex.w13.data, ex.w13_scale.data, f32p(ex.w13_factor), fmt, bf16m(b.act), false,
              stream, 0, split);
  moe::moe_down_rows(bf16m(b.act), m, idx, ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor), fmt, rows0, false,
                     stream, 0, split);
  if (capture_ == Capture::kStep) {
    join_child(stream, hop_node);
  } else if (capture_ != Capture::kHopChildren) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, hop_back_, 0));
  }
  CUDA_CHECK(cudaMemcpyAsync(rows_hi, hop_rows_.data(), rb, cudaMemcpyHostToDevice, stream));
  moe::moe_rows_merge(rows0, rows_hi, idx, m, split, stream);
  moe::moe_sum(rows0, m, f32p(b.wts), conv_w, conv_state, f32m(b.y2), conv_hist, stream);
}

void Decoder::full_attn_split(const Layer& layer, Buffers& b, int m, int v, attn::DecodeArgs a0,
                              cudaStream_t stream) {
  // One full-attention visit over both devices (plans/20261124_ingine-vs-maya-ds.md §7.5): this device runs
  // KV heads [0, here) against its cache shard, the second device runs [here, 8) against its own. Each launch
  // is the unchanged kernel: per KV head it owns Q/K/V completely, so merging is just the disjoint
  // query-head output rows. Proj rows for the shard's heads, its conv state and pos go over pinned staging;
  // 9 x 128 fp32 (m = 1) comes back. GPU0's attention runs while the second device's does.
  if (!kv_split_) throw std::logic_error("full_attn_split: no KV-head split");
  const int here = st_.kv_heads, there = st_.kv_heads_hi;
  const int q_begin = here * 3, q_count = there * 3;  // 3 query heads per KV head
  const int R = PROJ_ROWS_FULL;
  const int CC = there * HEAD_DIM;
  const int K = CONV_K - 1;
  const std::size_t proj_elems = std::size_t(kHopMaxM) * R;
  const float* proj = a0.proj;
  float* in = static_cast<float*>(kv_.in.data());
  __nv_bfloat16* out_host = static_cast<__nv_bfloat16*>(kv_.out_host.data());
  float* out_conv = reinterpret_cast<float*>(static_cast<char*>(kv_.out_host.data()) + kVkOut);
  float* conv = in + proj_elems;
  std::int64_t* pos_host = reinterpret_cast<std::int64_t*>(conv + 2 * std::size_t(K) * CC);

  struct Slice {
    int col, len;  // proj columns [col, col + len) of this visit's heads, contiguous within a row
  };
  const Slice slices[] = {{SEG_BEGIN[SEG_Q] + q_begin * HEAD_DIM, q_count * HEAD_DIM},
                          {SEG_BEGIN[SEG_GATE] + q_begin * HEAD_DIM, q_count * HEAD_DIM},
                          {SEG_BEGIN[SEG_REL] + q_begin * D_REL, q_count * D_REL},
                          {SEG_BEGIN[SEG_SINK] + q_begin, q_count},
                          {SEG_BEGIN[SEG_K] + here * HEAD_DIM, there * HEAD_DIM},
                          {SEG_BEGIN[SEG_V] + here * HEAD_DIM, there * HEAD_DIM},
                          {SEG_BEGIN[SEG_K_DYN] + here * CONV_K, there * CONV_K},
                          {SEG_BEGIN[SEG_V_DYN] + here * CONV_K, there * CONV_K}};
  // The packet mirrors a proj row, so the second device can address its slices exactly as it would the whole
  // row. Its previous reader must be done before it is rewritten: eagerly the event pair orders that, in a
  // captured step the previous visit's child node already precedes these copies.
  if (capture_ == Capture::kNone) CUDA_CHECK(cudaStreamWaitEvent(stream, kv_.loaded, 0));
  for (const Slice& s : slices)
    CUDA_CHECK(cudaMemcpy2DAsync(in + s.col, std::size_t(R) * 4, proj + s.col, std::size_t(R) * 4,
                                 std::size_t(s.len) * 4, m, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(conv, std::size_t(CC) * 4, st_.conv_k_at(v) + here * HEAD_DIM,
                               std::size_t(KV_DIM) * 4, std::size_t(CC) * 4, K, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(conv + std::size_t(K) * CC, std::size_t(CC) * 4,
                               st_.conv_v_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4, std::size_t(CC) * 4, K,
                               cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(pos_host, a0.pos, sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream));
  if (!kv_.weights_ready) {  // immutable per-layer slices, once: q/r scales, rel_proj, static conv rows
    // A captured half holds no upload (it changes the graph topology): the eager warm-up must have run first.
    if (capture_ != Capture::kNone)
      throw std::logic_error("full_attn_split: the second device's weight slices must be uploaded before capture");
    char* w = static_cast<char*>(kv_.w_host.data());
    const std::size_t qb = std::size_t(HEADS) * HEAD_DIM * 4, rb = std::size_t(HEADS) * D_REL * 4,
                      pb = std::size_t(D_REL) * REL_EXTENT * 4, kb = std::size_t(CONV_K) * CC * 4;
    int o = 0;
    for (const Layer& l : w_.layers) {
      if (!l.full) continue;
      char* base = w + std::size_t(o++) * kVkLayerWeights;
      CUDA_CHECK(cudaMemcpyAsync(base, f32p(l.q_scale), qb, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(base + qb, f32p(l.r_scale), rb, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(base + qb + rb, f32p(l.rel_proj), pb, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpy2DAsync(base + qb + rb + pb, std::size_t(CC) * 4,
                                   f32p(l.k_static) + here * HEAD_DIM, std::size_t(KV_DIM) * 4, std::size_t(CC) * 4,
                                   CONV_K, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpy2DAsync(base + qb + rb + pb + kb, std::size_t(CC) * 4,
                                   f32p(l.v_static) + here * HEAD_DIM, std::size_t(KV_DIM) * 4, std::size_t(CC) * 4,
                                   CONV_K, cudaMemcpyDeviceToHost, stream));
    }
  }

  auto packet_h2d = [&] {
    CUDA_CHECK(cudaMemcpyAsync(kv_.proj, in, std::size_t(m) * R * 4, cudaMemcpyHostToDevice, kv_.stream));
    CUDA_CHECK(cudaMemcpyAsync(kv_.conv, conv, 2 * std::size_t(K) * CC * 4, cudaMemcpyHostToDevice, kv_.stream));
    CUDA_CHECK(cudaMemcpyAsync(kv_.pos, pos_host, sizeof(std::int64_t), cudaMemcpyHostToDevice, kv_.stream));
    if (!kv_.weights_ready) {
      const std::size_t nfull = std::size_t(std::count_if(w_.layers.begin(), w_.layers.end(),
                                                           [](const Layer& l) { return l.full; }));
      CUDA_CHECK(cudaMemcpyAsync(kv_.weights, kv_.w_host.data(), nfull * kVkLayerWeights, cudaMemcpyHostToDevice,
                                 kv_.stream));
      kv_.weights_ready = true;  // ordered before every later kernel on this stream
    }
  };
  auto shard_attn = [&] {
    const auto& wl = kv_.layer[layer.idx];
    attn::DecodeArgs a1;
    a1.proj = static_cast<const float*>(kv_.proj);
    a1.M = m;
    a1.q_scale = wl.q_scale;
    a1.r_scale = wl.r_scale;
    a1.rel_proj = wl.rel_proj;
    a1.k_conv = wl.k_w;
    a1.v_conv = wl.v_w;
    a1.conv_k_state = static_cast<float*>(kv_.conv);
    a1.conv_v_state = static_cast<float*>(kv_.conv) + std::size_t(K) * CC;
    a1.k_cache = st_.k_full_hi_at(FULL_INDEX[v]).data;
    a1.v_cache = st_.v_full_hi_at(FULL_INDEX[v]).data;
    a1.cache_fp8 = st_.kv_fp8;
    a1.cache_rows = st_.max_ctx;
    a1.kpos = nullptr;
    a1.pos = static_cast<const std::int64_t*>(kv_.pos);
    a1.window = 0;
    a1.log_scaling = true;
    a1.workspace = kv_.ws;
    a1.workspace_size = kv_.ws_bytes;
    a1.out = static_cast<__nv_bfloat16*>(kv_.out);
    a1.head_begin = here;
    a1.head_count = there;
    a1.cache_heads = there;
    a1.out_head0 = 0;
    a1.out_stride = q_count * HEAD_DIM;
    attn::attn_decode(a1, kv_.stream);
  };
  auto rows_back = [&] {
    CUDA_CHECK(cudaMemcpy2DAsync(out_host, std::size_t(q_count) * HEAD_DIM * 2, kv_.out,
                                 std::size_t(q_count) * HEAD_DIM * 2, std::size_t(q_count) * HEAD_DIM * 2, m,
                                 cudaMemcpyDeviceToHost, kv_.stream));
    CUDA_CHECK(cudaMemcpyAsync(out_conv, kv_.conv, 2 * std::size_t(K) * CC * 4, cudaMemcpyDeviceToHost,
                               kv_.stream));
  };

  // The hop forks after this visit's packet and joins after this device's own heads, so the two devices'
  // attention runs concurrently as it does eagerly.
  cudaGraphNode_t kv_node = nullptr;
  if (capture_ == Capture::kStep) {
    kv_node = fork_child(stream, child_at(kv_child_, m, v));
  } else if (capture_ == Capture::kNone) {
    CUDA_CHECK(cudaEventRecord(kv_.ready, stream));
  }

  // this device's heads, concurrently with the second device's
  a0.head_begin = 0;
  a0.head_count = here;
  a0.cache_heads = here;
  a0.out_head0 = 0;
  a0.out_stride = HEADS * HEAD_DIM;
  attn::attn_decode(a0, stream);

  if (capture_ == Capture::kStep) {
    join_child(stream, kv_node);
  } else if (capture_ == Capture::kHopChildren) {
    const BindDevice bind(second_->device);
    keep_child(kv_child_, m, v, capture_child([&] {
                  packet_h2d();
                  shard_attn();
                  rows_back();
                }));
  } else {
    const BindDevice bind(second_->device);
    CUDA_CHECK(cudaStreamWaitEvent(kv_.stream, kv_.ready, 0));
    packet_h2d();
    CUDA_CHECK(cudaEventRecord(kv_.loaded, kv_.stream));
    shard_attn();
    CUDA_CHECK(cudaStreamWaitEvent(kv_.stream, kv_.consumed, 0));  // this device is done with the last result
    rows_back();
    CUDA_CHECK(cudaEventRecord(kv_.back, kv_.stream));
  }
  if (capture_ == Capture::kNone) CUDA_CHECK(cudaStreamWaitEvent(stream, kv_.back, 0));
  CUDA_CHECK(cudaMemcpy2DAsync(bf16m(b.attn_out) + q_begin * HEAD_DIM, std::size_t(HEADS) * HEAD_DIM * 2, out_host,
                               std::size_t(q_count) * HEAD_DIM * 2, std::size_t(q_count) * HEAD_DIM * 2, m,
                               cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(st_.conv_k_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4, out_conv,
                               std::size_t(CC) * 4, std::size_t(CC) * 4, K, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpy2DAsync(st_.conv_v_at(v) + here * HEAD_DIM, std::size_t(KV_DIM) * 4,
                               out_conv + std::size_t(K) * CC, std::size_t(CC) * 4, std::size_t(CC) * 4, K,
                               cudaMemcpyHostToDevice, stream));
  if (capture_ == Capture::kNone) CUDA_CHECK(cudaEventRecord(kv_.consumed, stream));
}

}  // namespace ingine::model
