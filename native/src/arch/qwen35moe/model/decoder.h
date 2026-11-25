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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

#include "arch/qwen35moe/kernels/moe/fused_experts.h"
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

  __nv_bfloat16* kv_k_[NUM_FULL_LAYERS] = {};
  __nv_bfloat16* kv_v_[NUM_FULL_LAYERS] = {};
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

  int* ids_dev_ = nullptr;
  float* weights_dev_ = nullptr;
  int* argmax_dev_ = nullptr;
  std::vector<float> logits_host_;
};

}  // namespace ingine::arch::qwen35moe
