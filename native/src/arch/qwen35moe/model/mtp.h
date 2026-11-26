// MTP draft head of Qwen3.6-35B-A3B as a standalone module: the checkpoint's single nextn block, which the
// speculative decode path (plans/20261126_qwen36-spec-decode.md, K4) uses to propose the next K tokens.
//
// The block is one full-attention layer plus one MoE layer, unquantised (19 BF16 tensors, 1.689 GB), and it
// borrows two trunk artifacts instead of copying them: the token embedding table and the NVFP4 lm_head. Its K/V
// cache is its own (one layer, 2 KV heads x 256 dims per position per side) and its type is selectable, Q4_0 by
// default.
//
// The forward, in order (guards from ~/git/llama.cpp src/models/qwen3next.cpp graph_mtp and the HF block):
//   e   = embed_tokens[draft_token]
//   e_n = rms(e, mtp.pre_fc_norm_embedding)          x/rms * (1 + w), the trunk's convention
//   h_n = rms(h, mtp.pre_fc_norm_hidden)             `h` is the trunk hidden state, pre-final-norm
//   x   = mtp.fc . concat([e_n, h_n])                embedding half first, 4096 columns
//   sa  = x
//   x   = rms(x, mtp.layers.0.input_layernorm)
//   q   = q_proj(x), 8192 rows = 16 heads of [query | gate]; k = k_proj(x), v = v_proj(x)
//   q,k = per-head rms (q_norm/k_norm), partial rotary on 64 of 256 dims, mrope text axis (1-D)
//   a   = causal attention over the drafter cache at `pos`; a *= sigmoid(gate); x = sa + o_proj(a)
//   sp  = x
//   x   = rms(x, mtp.layers.0.post_attention_layernorm)
//   x   = sp + sum_k w_k experts_k(x) + sigmoid(shared_expert_gate . x) * shared_expert(x)
//   h'  = rms(x, mtp.norm)
//   draft = argmax(lm_head . h')
//
// Everything is FP32 on the device; the weights stay in their stored BF16. A draft call runs on the module's own
// stream and synchronises once, because the greedy token is returned as a host int.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/mtp/mtp_attn.h"
#include "arch/qwen35moe/model/model_data.h"

namespace ingine::arch::qwen35moe {

// The cache element format of the drafter's layer; see kernels/mtp/mtp_attn.h for the Q4_0 layout.
using MtpKvType = kernels::MtpKvType;

// Q4_0 is the default: 0.15 GB at 262144 positions instead of 0.51 GB as BF16, at the cost measured in the
// test's cache_quality_case(). A drafter only has to propose tokens.
inline constexpr MtpKvType kMtpKvDefault = MtpKvType::Q4_0;

// The trunk artifacts the drafter reads but never owns.
struct MtpTrunkHead {
  const __nv_bfloat16* embed_tokens = nullptr;  // [VOCAB, HIDDEN] BF16
  const std::uint8_t* head_w = nullptr;         // lm_head NVFP4 weight, [VOCAB, HIDDEN / 2] e2m1 pairs
  const std::uint8_t* head_scales = nullptr;    // lm_head NVFP4 block scales, [VOCAB, HIDDEN / 16] e4m3
  float head_scale = 1.0f;                      // lm_head weight_scale_2
};

// Device time of one draft, split the way the VRAM and bandwidth budget needs it: the dense projections, the
// attention over the drafter cache, the routed experts, the shared expert (with the two residual adds that close
// the MoE block) and the lm_head with its argmax. Filled only when a caller passes the struct to draft().
struct MtpTiming {
  float dense_ms = 0.0f;
  float attn_ms = 0.0f;
  float routed_ms = 0.0f;
  float shared_ms = 0.0f;
  float head_ms = 0.0f;
};

// One tensor of the head as loaded: enough for a test to check every shape and sample any byte of the upload.
struct MtpTensors {
  std::string name;
  std::vector<std::int64_t> shape;  // outermost dimension first, as the safetensors header stores it
  std::uint64_t bytes = 0;
  std::uint64_t offset = 0;  // from the module's weight block
};

class MtpDrafter {
 public:
  MtpDrafter();
  ~MtpDrafter();
  MtpDrafter(const MtpDrafter&) = delete;
  MtpDrafter& operator=(const MtpDrafter&) = delete;

  // Loads the 19 BF16 tensors from `snapshot_dir` (a directory holding model.safetensors.index.json or
  // model.safetensors), allocates the weight block and a `max_pos`-deep K/V cache of `kv_type` on `cuda_device`,
  // and converts the seven norm vectors to FP32 with the (1 + w) convention. Returns false and fills `error`
  // when a tensor is missing, is not BF16, or has a shape other than the one this architecture fixes: there is
  // no fallback, a head that does not match is not a drafter.
  bool init(const std::filesystem::path& snapshot_dir, const MtpTrunkHead& trunk, int max_pos, MtpKvType kv_type,
            int cuda_device, std::string* error);

  // One draft: appends the cache row of `token` seen from the trunk hidden state `hidden` (a device row of
  // HIDDEN BF16 values, pre-final-norm) at `pos`, runs the block, and writes the greedy token. `pos` must be in
  // [0, max_pos) and the cache rows below it belong to the caller's position history, as in the trunk: replaying
  // a position overwrites its row and leaves the rest alone. All launches go to the module's stream; the call
  // returns once the token is known.
  bool draft(int token, const __nv_bfloat16* hidden, int pos, int* next_token, std::string* error,
             MtpTiming* timing = nullptr);

  // The logits row the last draft's argmax came from, on the device [VOCAB].
  const float* logits() const { return logits_; }
  // h' of the last draft, after mtp.norm, on the device [HIDDEN].
  const float* last_hidden() const { return out_; }
  // Router ids of the last draft, on the device [TOP_K], highest logit first.
  const int* router_ids() const { return ids_; }

  // The K/V cache of the last draft: one side each, in `kv_type()`'s layout. Null before init.
  const void* k_cache() const { return k_cache_; }
  const void* v_cache() const { return v_cache_; }
  MtpKvType kv_type() const { return kv_type_; }
  int max_pos() const { return max_pos_; }
  // Base of the contiguous weight block and the inventory inside it, in load order.
  const void* weights() const { return weights_; }
  const std::vector<MtpTensors>& tensors() const { return tensors_; }

  // What init allocated: the 19 tensors, both cache sides, the FP32 activations and the split states.
  std::uint64_t weight_bytes() const { return weight_bytes_; }
  std::uint64_t cache_bytes() const { return 2 * kernels::mtp_kv_bytes(kv_type_, max_pos_); }
  std::uint64_t total_bytes() const { return weight_bytes_ + cache_bytes() + scratch_bytes_; }

 private:
  // The twelve projections of the head, resolved once at init.
  struct Weights {
    const __nv_bfloat16* fc = nullptr;                 // [HIDDEN, 2 * HIDDEN]
    const __nv_bfloat16* q_proj = nullptr;             // [Q_PROJ_ROWS, HIDDEN]
    const __nv_bfloat16* k_proj = nullptr;             // [KV_DIM, HIDDEN]
    const __nv_bfloat16* v_proj = nullptr;             // [KV_DIM, HIDDEN]
    const __nv_bfloat16* o_proj = nullptr;             // [HIDDEN, Q_DIM]
    const __nv_bfloat16* router = nullptr;             // [EXPERTS, HIDDEN]
    const __nv_bfloat16* gate_up = nullptr;            // [EXPERTS, 2 * MOE_INTER, HIDDEN]
    const __nv_bfloat16* down = nullptr;               // [EXPERTS, HIDDEN, MOE_INTER]
    const __nv_bfloat16* shared_gate = nullptr;        // [SHARED_INTER, HIDDEN]
    const __nv_bfloat16* shared_up = nullptr;          // [SHARED_INTER, HIDDEN]
    const __nv_bfloat16* shared_down = nullptr;        // [HIDDEN, SHARED_INTER]
    const __nv_bfloat16* shared_gate_scalar = nullptr; // [1, HIDDEN]
  };
  // The seven norm vectors, FP32 with the (1 + w) convention already applied.
  struct Norms {
    const float* embed = nullptr;   // mtp.pre_fc_norm_embedding
    const float* hidden = nullptr;  // mtp.pre_fc_norm_hidden
    const float* input = nullptr;   // mtp.layers.0.input_layernorm
    const float* post = nullptr;    // mtp.layers.0.post_attention_layernorm
    const float* q = nullptr;       // mtp.layers.0.self_attn.q_norm
    const float* k = nullptr;       // mtp.layers.0.self_attn.k_norm
    const float* final = nullptr;   // mtp.norm
  };

  bool load_weights(const std::filesystem::path& snapshot_dir, std::string* error);
  bool load_norms(const std::filesystem::path& snapshot_dir, std::string* error);
  bool allocate(int max_pos, MtpKvType kv_type, std::string* error);
  void release();

  // The trunk artifacts, borrowed for the module's lifetime.
  const __nv_bfloat16* embed_tokens_ = nullptr;
  const std::uint8_t* head_w_ = nullptr;
  const std::uint8_t* head_scales_ = nullptr;
  float head_scale_ = 1.0f;

  int device_ = -1;
  int max_pos_ = 0;
  MtpKvType kv_type_ = MtpKvType::Q4_0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t events_[7] = {};

  void* weights_ = nullptr;
  std::uint64_t weight_bytes_ = 0;
  std::uint64_t scratch_bytes_ = 0;
  std::vector<MtpTensors> tensors_;
  Weights w_;
  Norms n_;
  std::vector<float*> norm_storage_;  // the FP32 copies `n_` points into

  void* k_cache_ = nullptr;
  void* v_cache_ = nullptr;

  float* e_ = nullptr;        // embedding row of the draft token
  float* hf_ = nullptr;       // trunk hidden row, FP32
  float* cat_ = nullptr;      // [2 * HIDDEN] embedding half first, then hidden half
  float* x_ = nullptr;        // running residual stream
  float* h_ = nullptr;        // attention input, normed
  float* qgate_ = nullptr;    // [Q_PROJ_ROWS] per head [query | gate]
  float* q_ = nullptr;        // [Q_DIM] contiguous query half
  float* gate_ = nullptr;     // [Q_DIM] contiguous gate half
  float* k_ = nullptr;        // [KV_DIM]
  float* v_ = nullptr;        // [KV_DIM]
  float* attn_ = nullptr;     // [Q_DIM]
  float* y_ = nullptr;        // projection output before the residual add
  float* h2_ = nullptr;       // MoE input, normed
  float* moe_ = nullptr;      // routed-expert output
  float* shared_ = nullptr;   // shared-expert output
  float* sh_ = nullptr;       // [SHARED_INTER] shared gate projection
  float* su_ = nullptr;       // [SHARED_INTER] shared up projection
  float* expert_h_ = nullptr; // [TOP_K * MOE_INTER]
  float* out_ = nullptr;      // h' after mtp.norm
  float* logits_ = nullptr;   // [VOCAB]
  float* router_logits_ = nullptr;  // [EXPERTS]
  int* ids_ = nullptr;        // [TOP_K]
  float* weights_r_ = nullptr;     // [TOP_K] router weights
  int* argmax_ = nullptr;
  kernels::SplitState* partial_ = nullptr;  // [ATTN_HEADS * kMaxSplits]
};

}  // namespace ingine::arch::qwen35moe
