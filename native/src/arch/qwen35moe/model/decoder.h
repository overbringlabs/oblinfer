// Decoder of the Qwen3.6-35B-A3B pack: one token per step, greedy sampling.
//
// Weights come from a loaded TextArena (uploader.h): the `placed` table gives every tensor's device span and
// offset, so the driver only resolves names once at init. Activations are FP32; KV caches are BF16, q8_0 or
// q4_0 (selectable per side) in a position-major layout per full-attention layer; the GDN state and conv
// history live on the device for the linear layers.
//
// This is the correctness path: the router ids and weights are copied to the host each layer because the
// MoE driver takes host arrays. The speed pass removes that with a device-side routing loop.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/attn/kv_bands.h"

#include "arch/qwen35moe/kernels/batch/batch_attn.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"
#include "arch/qwen35moe/kernels/batch/batch_moe.h"
#include "arch/qwen35moe/kernels/moe/fused_experts.h"
#include "arch/qwen35moe/kernels/moe/router.h"
#include "arch/qwen35moe/model/decode_graph.h"
#include "arch/qwen35moe/model/uploader.h"

namespace ingine::arch::qwen35moe {

class Decoder {
 public:
  Decoder();
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  // `max_pos` bounds both the KV caches and the positions the router may see.
  bool init(const TextArena& arena, int max_pos, std::string* error);

  // Runs one token at `pos` (0-based, strictly increasing) and writes the greedy next token.
  bool step(int token, int pos, int* next_token, std::string* error);

  // ---- verify width: n positions in one pass over the weights (spec-decode K2/K3) ----
  //
  // Runs ids[0..n-1] at positions pos..pos+n-1 and writes the greedy next token of each position, so a
  // speculative verify costs one pass over the weights instead of n. n is 1..BATCH_MAX (4).
  //
  // n = 1 is the single-token step, which is exact by construction and faster; the batch kernels order their
  // reductions differently (eight contiguous columns per lane instead of stride-32 column pairs), so a batch
  // pass at n = 1 could never be bit-identical to it. QWEN35MOE_BATCH_AT_ONE=1 forces the batch path at n = 1
  // for measurement and for the tolerance test.
  //
  // Not graph-capturable: the hop carries n rows, and the batch path switches devices inside the layer loop.
  // It needs one KV cache per full-attention layer, so QWEN35MOE_KV_BANDS=0 at capacities where the bands
  // would split (the plan's K5 owns the two-band staging).
  bool step_batch(const int* ids, int n, int pos, int* next_tokens, std::string* error);
  // Logits of position t of the last step_batch (VOCAB values), or null when they were not kept: the block
  // reads back one argmax per position, and the full logits only under QWEN35MOE_BATCH_LOGITS=1.
  const float* batch_logits(int t) const;
  // The trunk residual of position t of the last step_batch (HIDDEN values, after the last layer and before the
  // final norm), kept under the same flag. Separating it from the logits is what tells a rounding difference
  // that the lm_head amplifies apart from one the router picks up as a near tie.
  const float* batch_residual(int t) const;

  // GDN rollback. A rejected draft cannot recompute the linear-attention state, so the accept loop copies the
  // recurrent and conv state aside before a verify step and restores it when a draft is rejected. Trunk KV
  // needs no rollback: a rejected position's rows are simply overwritten, because the position counter only
  // advances over accepted tokens.
  bool checkpoint_gdn(std::string* error);
  bool restore_gdn(std::string* error);
  // Host copy of the trunk residual the last step() left (HIDDEN floats), for comparing a batch pass against
  // the sequential path. A measurement path: it forces a device-to-host copy.
  bool read_residual(std::vector<float>* out, std::string* error);

  // Host copy of the last step's logits.
  const std::vector<float>& logits() const { return logits_host_; }

  // ---- the per-token graph of decode_graph.h: one captured step per attention-split bucket ----

  // Allocates what a capture needs (the two step-input ints, the card-0 capture stream) and captures the
  // card-1 halves of every MoE layer once. Idempotent; a graph bucket is built from it.
  bool prepare_graph(std::string* error);
  int max_pos() const { return max_pos_; }
  // KV cache formats per cache: trunk (-ctk/-ctv) and drafter (-ctkd/-ctvd). The drafter module does not exist
  // yet, so its pair carries only the setting and its default (K4V4); no drafter cache is allocated.
  kernels::KvType kv_type_k() const { return kv_type_k_; }
  kernels::KvType kv_type_v() const { return kv_type_v_; }
  kernels::KvType kv_type_k_drafter() const { return kv_type_k_drafter_; }
  kernels::KvType kv_type_v_drafter() const { return kv_type_v_drafter_; }
  // The bucket `pos` falls in, and one position inside a bucket to capture the step at.
  int splits_at(int pos) const;
  int capture_pos(int splits) const;
  // Card 1's half of MoE layer `l`: H2D of the staged routing, the local experts, D2H of the result. Null on a
  // single card, or before prepare_graph.
  cudaGraph_t hop_graph(int l) const;
  cudaStream_t graph_stream() const { return cap0_; }
  int* token_dev() const { return token_dev_; }
  int* pos_dev() const { return pos_dev_; }
  // The graph body: one token from the embedding read out of token_dev_ through the lm_head GEMV, enqueued on
  // the capture stream with the position-bearing launches registered.
  void enqueue_graph(int pos, DecodeGraphCapture* cap);

 private:
  struct ExpertRow {
    struct Args {
      const std::uint8_t* w = nullptr;
      const std::uint8_t* scales = nullptr;
      float scale = 1.0f;
    };
    Args gate, up, down;
    int span = 0;  // which device holds this expert
  };

  struct FusedTable {
    std::vector<kernels::FusedExpertLayer> host;  // LAYERS * EXPERTS, indexed [layer * EXPERTS + expert]
    kernels::FusedExpertLayer* device = nullptr;
  };

  void* ptr(const std::string& name) const;
  // FP32 view of a BF16 checkpoint tensor (norms, A_log, dt_bias, conv1d): converted once at first use.
  // `plus_one` applies the Qwen3_5MoeRMSNorm convention y = x/rms * (1 + w); the GDN gated norm uses w.
  const float* f32(const std::string& name, bool plus_one = false);
  float scalar(const std::string& name, float fallback) const;
  void fp8_gemv(const std::string& base, const float* x, float* y, int rows, int cols);
  void nvfp4_gemv(const std::string& base, const float* x, float* y, int rows, int cols);

  void layer(int l, int pos);
  void linear_layer(int l);
  void full_layer(int l, int pos);
  void moe(int l);

  // ---- the batch path's private half: same layers, n rows ----
  void ensure_batch();
  void ensure_gdn_checkpoint();
  void batch_body(const int* ids, int n, int pos);
  void batch_linear_layer(int l, int n);
  void batch_full_layer(int l, int pos, int n);
  void batch_moe(int l, int n);
  // The full-attention layer's attention at verify width, one cache or two.
  void batch_attention_local(int slot, int pos, int n);
  void batch_attention_bands(int slot, int pos, int n);

  // One token's kernels on `s_`: the embedding, the LAYERS layers, the final norm and the lm_head GEMV. The
  // eager path passes the token id as an argument; a capture reads it from token_dev_ so one graph serves every
  // token, and registers the position-bearing launches so the replay can rewrite them. Both issue the same
  // kernels in the same order. The host readback stays in step().
  void body(int token, int pos, bool captured);
  // Wraps a launch whose by-value argument carries the position: a plain call when not capturing.
  template <typename Args, typename Launch>
  void at(const Args& args, std::size_t pos_offset, Launch&& launch);
  // Stream-ordered equivalents of the synchronous CUDA calls. A cudaMemset or cudaMemcpy issued during a
  // capture enqueues nothing and reports no error, so the captured path must not use them.
  void clear(void* p, std::size_t bytes);
  void copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind);
  // Blocking device-to-host read of the step outputs, ordered against `s_`.
  void read_back(void* dst, const void* src, std::size_t bytes);
  // The captured step for `pos`'s bucket, building it on first use. Null when the graph is off, not yet
  // primed, or unavailable for this bucket.
  DecodeGraph* graph_for(int pos);

  // One flash-decode split per this many positions; `splits` caps it and decides the graph bucket.
  static constexpr int kSplitPositions = 2048;
  // With QWEN35MOE_ATTN_SPLITS_DEPTH, the column width below the cap: 16 query heads x 16 columns is 256
  // blocks, four waves of the 5070 Ti's 70 SMs, where the fixed split width leaves a shallow cache on one.
  static constexpr int kSplitPositionsFine = 128;

  const TextArena* arena_ = nullptr;
  std::unordered_map<std::string, const PlacedTensor*> placed_;
  std::unordered_map<std::string, float> scalars_;
  std::vector<std::vector<ExpertRow>> experts_;
  std::unordered_map<std::string, float*> f32_;
  int max_pos_ = 0;
  int last_pos_ = 0;

  // activations and small states
  float* x_ = nullptr;
  float* h_ = nullptr;    // normed hidden, MoE input
  float* h2_ = nullptr;   // post-attention norm output
  float* y_ = nullptr;    // projection output before the residual add
  float* qkv_ = nullptr;  // conv input
  float* qkv2_ = nullptr; // conv output
  float* z_ = nullptr;    // GDN gate projection
  float* a_ = nullptr;
  float* b_ = nullptr;
  float* beta_ = nullptr;
  float* g_ = nullptr;
  float* gout_ = nullptr;   // GDN output before the gated norm
  float* q_lin_ = nullptr;  // gathered q/k/v when the export keeps the per-key-head grouping
  float* k_lin_ = nullptr;
  float* v_lin_ = nullptr;
  bool grouped_qkv_ = false;
  float* qgate_ = nullptr;  // q_proj rows: query then gate
  float* k_ = nullptr;
  float* v_ = nullptr;
  float* attn_ = nullptr;
  float* q_attn_ = nullptr;   // contiguous query half of q_proj
  float* gate_attn_ = nullptr;  // contiguous gate half of q_proj
  float* moe_out_ = nullptr;
  float* shared_out_ = nullptr;
  float* shared_h_ = nullptr;
  float* shared_up_ = nullptr;
  float* logits_ = nullptr;

  kernels::SplitState* kv_partial_ = nullptr;  // [ATTN_HEADS * kMaxSplits]
  KvBands kv_bands_[NUM_FULL_LAYERS] = {};  // one plan per full-attention layer; count 1 is the plain cache
  KvBandExchange kv_exchange_ = {};          // peer staging for a two-card plan, shared by every layer
  bool kv_bands_on_ = false;       // QWEN35MOE_KV_BANDS=1 and a two-card plan that fitted
  int kv_band_min_pos_ = 0;        // capacity below which the plan is not split (measured break-even)
  float kv_band_ratio_ = KV_BAND_RATIO;  // card 0 : card 1 share of the positions
  bool attn_splits_depth_ = false;       // QWEN35MOE_ATTN_SPLITS_DEPTH=1: depth-scaled column width
  kernels::KvType kv_type_k_ = kernels::KvType::Bf16;  // -ctk
  kernels::KvType kv_type_v_ = kernels::KvType::Bf16;  // -ctv
  kernels::KvType kv_type_k_drafter_ = kernels::KvType::Q4_0;  // -ctkd, for the drafter's own cache
  kernels::KvType kv_type_v_drafter_ = kernels::KvType::Q4_0;  // -ctvd
  float* gdn_state_ = nullptr;   // [NUM_LINEAR_LAYERS, LIN_VALUE_HEADS, LIN_KEY_DIM, LIN_VALUE_DIM]
  float* conv_state_ = nullptr;  // [NUM_LINEAR_LAYERS, CONV_K - 1, LIN_CONV_DIM]

  int dev0_ = 0;
  int dev1_ = 1;
  float* peer_x_ = nullptr;    // device 1: MoE input
  float* peer_h_ = nullptr;    // device 1: expert intermediate
  float* peer_up_ = nullptr;   // device 1: expert intermediate
  float* peer_out_ = nullptr;  // device 1: expert output
  float* pinned_in_ = nullptr;
  float* pinned_out_ = nullptr;

  FusedTable fused_[2];
  float* fscratch_gate_[2] = {};
  float* fscratch_up_[2] = {};
  float* fscratch_down_[2] = {};
  int* peer_ids_ = nullptr;
  float* peer_weights_ = nullptr;

  // Double-buffered pinned exchange: x + ids + weights out, y back. One slot per parity of the layer index.
  struct Exchange {
    float x[HIDDEN];
    int ids[8];
    float weights[8];
    float y[HIDDEN];
  };
  Exchange* pinned_ = nullptr;             // [2]
  int exch_ = 0;
  cudaStream_t s1_ = nullptr;
  cudaEvent_t ev_x_[2] = {};               // card 0 records, card 1 waits
  cudaEvent_t ev_x_free_[2] = {};          // card 1 records, card 0 waits
  cudaEvent_t ev_y_[2] = {};               // card 1 records, card 0 waits
  cudaEvent_t ev_y_free_[2] = {};          // card 0 records, card 1 waits
  __nv_bfloat16* peer_router_ = nullptr;  // [LAYERS][ROUTER_EXPERTS][HIDDEN]: the router replicated on card 1

  float* router_logits_ = nullptr;
  int* ids_dev_ = nullptr;
  float* weights_dev_ = nullptr;
  int* argmax_dev_ = nullptr;
  std::vector<float> logits_host_;

  // Graph decode. `s_` is the stream the step's kernels go to: the legacy default stream for the eager path,
  // cap0_ while capturing and replaying. `cap_` is non-null only inside a capture.
  bool use_graph_ = false;     // QWEN35MOE_GRAPH=1; cleared for good by the first capture failure
  bool graph_reported_ = false;
  bool primed_ = false;  // one eager step first: it loads every module a capture is not allowed to load
  cudaStream_t s_ = nullptr;
  cudaStream_t cap0_ = nullptr;
  DecodeGraphCapture* cap_ = nullptr;
  int* token_dev_ = nullptr;
  int* pos_dev_ = nullptr;
  std::vector<cudaGraph_t> hop_graphs_;  // [LAYERS], captured on device 1
  std::unordered_map<int, DecodeGraph> graphs_;

  // ------------------------------------------------------------------ verify-width state (step_batch)
  //
  // Every activation is [BATCH_MAX, width], row-major and contiguous, so row t is position t and a flat
  // launch over n * width covers the block. Allocated on the first step_batch: a single-token run never
  // touches them.
  struct BatchActs {
    float* x = nullptr;       // [n, HIDDEN], the residual stream
    float* h = nullptr;       // [n, HIDDEN], normed mixer input, then the normed lm_head input
    float* h2 = nullptr;      // [n, HIDDEN], normed MoE input
    float* y = nullptr;       // [n, HIDDEN], mixer output
    float* moe = nullptr;     // [n, HIDDEN], routed-expert output (both cards merged)
    float* shared = nullptr;  // [n, HIDDEN], shared-expert output
    float* sh = nullptr;      // [n, SHARED_INTER], shared gate, then the silu product
    float* su = nullptr;      // [n, SHARED_INTER], shared up
    float* qkv = nullptr;     // [n, LIN_CONV_DIM], conv input
    float* qkv2 = nullptr;    // [n, LIN_CONV_DIM], conv output
    float* z = nullptr;       // [n, LIN_VALUE_DIM_TOT]
    float* a = nullptr;       // [n, LIN_VALUE_HEADS]
    float* b = nullptr;
    float* beta = nullptr;
    float* g = nullptr;
    float* gout = nullptr;    // [n, LIN_VALUE_DIM_TOT]
    float* q_lin = nullptr;   // [n, LIN_KEY_DIM_TOT], gathered q/k/v for the recurrence
    float* k_lin = nullptr;
    float* v_lin = nullptr;   // [n, LIN_VALUE_DIM_TOT]
    float* qgate = nullptr;   // [n, Q_PROJ_ROWS]
    float* q = nullptr;       // [n, Q_DIM]
    float* gate = nullptr;    // [n, Q_DIM]
    float* attn = nullptr;    // [n, Q_DIM]
    float* k = nullptr;       // [n, KV_DIM]
    float* v = nullptr;       // [n, KV_DIM]
    float* logits = nullptr;  // [n, VOCAB]
    kernels::BatchSplitState* partial = nullptr;  // [HEADS, kMaxSplits]
    kernels::BatchSplitState* band_states = nullptr;  // [HEADS, 2]: the two bands' merges of their own columns
    float* router_logits = nullptr;               // [n, ROUTER_EXPERTS]
    int* ids = nullptr;        // [n, ROUTER_TOP_K]
    float* weights = nullptr;  // [n, ROUTER_TOP_K]
    int* argmax = nullptr;     // [n]
    int* dev_ids = nullptr;    // [n], the block's token ids for the embedding rows
  };

  // The MoE hop at verify width: one pinned slot per parity of the layer index, n rows of hidden state plus the
  // n x top-k routing, and n rows back.
  struct BatchExchange {
    float x[kernels::BATCH_MAX][HIDDEN];
    int ids[kernels::BATCH_MAX][kernels::ROUTER_TOP_K];
    float weights[kernels::BATCH_MAX][kernels::ROUTER_TOP_K];
    float y[kernels::BATCH_MAX][HIDDEN];
  };

  struct BatchCard {
    float* gate = nullptr;   // [BATCH_PAIRS, MOE_INTER] after the silu step
    float* up = nullptr;     // [BATCH_PAIRS, MOE_INTER]
    float* down = nullptr;   // [BATCH_PAIRS, HIDDEN]
    int* unique_ids = nullptr;
    int* unique_count = nullptr;
    int* pair_unique = nullptr;
    int* pair_of = nullptr;
    int* rows = nullptr;
    kernels::BatchExpertBlock* layers = nullptr;  // [LAYERS * EXPERTS]
    void* peer_x = nullptr;   // card 1 only: the block's hidden states
    void* peer_y = nullptr;   // card 1 only: its expert partial
    int* peer_ids = nullptr;  // card 1 only
    float* peer_weights = nullptr;
  };

  BatchActs ba_;
  BatchCard batch_[2];
  BatchExchange* bpinned_ = nullptr;  // [2]

  // Two-band attention at verify width: the peer card runs the flash columns its band owns, merges them into
  // one state per head, and only those states cross (66 KB per layer) — not the per-column states.
  struct BatchBand {
    float* q = nullptr;   // peer-side [n, HEADS, HEAD_DIM]
    float* k = nullptr;   // peer-side [n, KV_HEADS, HEAD_DIM]
    float* v = nullptr;
    float* out = nullptr;             // peer-side, unused: the combine runs on the host card
    kernels::BatchSplitState* partial = nullptr;       // [HEADS, kMaxSplits]
    kernels::BatchSplitState* band_states = nullptr;   // [HEADS, 2]
    float* pinned = nullptr;          // host: q, then k, then v, then the peer's HEADS band states
    cudaEvent_t q_ready = nullptr;      // host card records, peer waits
    cudaEvent_t q_consumed = nullptr;   // peer records, host card waits
    cudaEvent_t states_ready = nullptr; // peer records, host card waits
    cudaEvent_t states_consumed = nullptr;  // host card records, peer waits
    bool q_pending = false;       // the pinned query has not been consumed by the peer yet
    bool states_pending = false;  // the peer's states have not been merged yet
  };
  BatchBand band1_;  // the peer's half of the banded attention; allocated with the rest of the batch state
  cudaEvent_t bev_x_[2] = {};         // card 0 records, card 1 waits
  cudaEvent_t bev_x_free_[2] = {};    // card 1 records, card 0 waits
  cudaEvent_t bev_y_[2] = {};         // card 1 records, card 0 waits
  cudaEvent_t bev_y_free_[2] = {};    // card 0 records, card 1 waits
  int bexch_ = 0;
  bool batch_ready_ = false;
  bool batch_logits_kept_ = false;      // QWEN35MOE_BATCH_LOGITS=1: read the block's logits back
  std::vector<float> batch_logits_host_;    // [n, VOCAB] when kept
  std::vector<float> batch_residual_host_;  // [n, HIDDEN] when kept

  float* gdn_ckpt_ = nullptr;   // [NUM_LINEAR_LAYERS, heads, key, value] FP32
  float* conv_ckpt_ = nullptr;  // [NUM_LINEAR_LAYERS, CONV_K - 1, LIN_CONV_DIM]
  bool gdn_ckpt_ready_ = false;
};

}  // namespace ingine::arch::qwen35moe
