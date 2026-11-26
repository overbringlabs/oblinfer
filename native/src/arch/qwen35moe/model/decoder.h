// Decoder of the Qwen3.6-35B-A3B pack: one token per step, greedy sampling.
//
// Weights come from a loaded TextArena (uploader.h): the `placed` table gives every tensor's device span and
// offset, so the driver only resolves names once at init. Activations are FP32; KV caches are BF16 in a
// position-major layout per full-attention layer; the GDN state and conv history live on the device for the
// linear layers.
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

#include "arch/qwen35moe/kernels/moe/fused_experts.h"
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

  // Host copy of the last step's logits.
  const std::vector<float>& logits() const { return logits_host_; }

  // ---- the per-token graph of decode_graph.h: one captured step per attention-split bucket ----

  // Allocates what a capture needs (the two step-input ints, the card-0 capture stream) and captures the
  // card-1 halves of every MoE layer once. Idempotent; a graph bucket is built from it.
  bool prepare_graph(std::string* error);
  int max_pos() const { return max_pos_; }
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
  void* kv_k_[NUM_FULL_LAYERS] = {};
  void* kv_v_[NUM_FULL_LAYERS] = {};
  kernels::KvType kv_type_k_ = kernels::KvType::Bf16;  // -ctk
  kernels::KvType kv_type_v_ = kernels::KvType::Bf16;  // -ctv
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
};

}  // namespace ingine::arch::qwen35moe
