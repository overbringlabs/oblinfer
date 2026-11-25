// Native prompt processing: model.Model.prefill for one chunk, without PyTorch.
#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "kernels/prefill/prefill.h"
#include "model/model_data.h"

#include <array>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// Processes prompt tokens [T] (device int32) at positions pos0..pos0+T-1 on top of `state` (exactly
// model.Model.prefill for one chunk): writes caches/conv state, sets state.length = pos0 + T, returns the last
// token's logits (device fp32 [UNPADDED_VOCAB], softcapped like the head) and, if taps != nullptr, every row's
// DSpark taps (bf16 [T, 4, H], model.TAP_VISITS + final normed hidden).
//
// The visit sequence is fast.FastDecoder.decode's (the same block-state wiring and mix / message-gate kernels)
// with T rows; projections are cuBLASLt BF16 GEMMs over the dequantized weight rows (weights.dequant), the
// attention a flash-attention prefill kernel, the MoE grouped GEMMs over the experts decoded on the fly.
// Stream-ordered on `stream`; run() only synchronises for its first use of a block-pointer table.
// Embedding rows computed elsewhere (image tokens from the vision tower) for rows [row, row + count) of a
// chunk: src bf16 [count, HIDDEN] on the device replaces the token embeddings of those rows (which the model
// feeds in as they are: no embedding norm).
//
// Two devices (RuntimeOptions::second_gpu): a layer whose experts live on the second device runs its grouped
// expert GEMMs there (expert_hop), a 64/64 split runs half of them on each device (expert_split_hop), and with
// the KV-head split (--kv-split) the full-attention visits write and attend this device's 5 heads and the second
// device's 3 (full_attn_split). One device, or every layer's experts here: no copy and no event beyond the
// one-device launch sequence.
struct EmbedRows {
  int row = 0, count = 0;
  const __nv_bfloat16* src = nullptr;
};

class Prefiller {
 public:
  // `second`: the device opened for expert parallelism (RuntimeOptions::second_gpu), or null. `kv_split`:
  // the runtime's State has kv_heads_hi > 0 (RuntimeOptions::kv_split), i.e. the second device holds 3 of the
  // 8 full-attention KV heads, which this Prefiller writes and attends there.
  Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream,
            const DeviceContext* second = nullptr, bool kv_split = false);
  ~Prefiller();
  Prefiller(const Prefiller&) = delete;
  Prefiller& operator=(const Prefiller&) = delete;

  // Tokens with a negative id must be covered by `rows` (their embedding is the image placeholder's otherwise).
  void run(State& state, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
           std::span<const EmbedRows> rows = {});

  int max_chunk() const { return max_chunk_; }
  std::size_t device_bytes() const;  // scratch held by this object on this device

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);
  // EP: one visit's routed experts, which live on the second device. This device's router output (the tile and
  // pair lists) and xm hop over, the second device runs the grouped gate/up, down and combine GEMMs, and the
  // combined BF16 rows come back for the post-MLP conv. No host wait.
  void expert_hop(const Experts& ex, int T, cudaStream_t stream);
  // EP with a 64/64 split (ExpertResidency split 64): each device runs the grouped gate/up and down GEMMs of
  // its contiguous half of the layer's experts; the second device's per-pair rows come back, this device
  // merges them and combines the slots in order. No host wait.
  void expert_split_hop(const Experts& ex, int T, cudaStream_t stream);
  // KV-head split (--kv-split): one full-attention visit of a chunk. `ap` / `at` describe this device's KV
  // heads [0, kv_heads); the second device writes its cache shard's rows and attends its own heads against
  // them, and its query-head rows come back into attn_out.
  void full_attn_split(State& st, const Layer& layer, kernels::prefill::AttnPrepParams& ap,
                       kernels::prefill::AttnParams& at, int T, int pos0, int v, cudaStream_t stream);
  void layer_weights_second();  // the split's per-layer slices, uploaded once at construction

  const Weights& w_;
  const Ablation& abl_;
  int max_chunk_;
  cudaStream_t stream_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;

  // activations (rows = max_chunk), see prefill.cpp
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  __nv_bfloat16 *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  __nv_bfloat16 *proj_, *q_, *rel16_, *rel_, *krows_, *vrows_, *attn_out_, *wo_out_;
  int* kpos_;
  int4* bounds_;
  __nv_bfloat16 *gin_, *g16_;
  float *fpre_, *gpre_;
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  __nv_bfloat16* wscratch_;  // dequantized FP8 rows of wproj / wo (dense_fp8)
  std::int64_t* ids_;
  int kpos_rows_;
  int tiles_capacity_;

  // second device (nothing here is allocated without one, and the expert hop only with a remote layer)
  const DeviceContext* second_ = nullptr;
  bool kv_split_ = false;
  cudaStream_t hop_stream_ = nullptr;  // the second device's compute stream
  cudaEvent_t hop_ready_ = nullptr;    // recorded on `stream` (device 0), waited on hop_stream_
  cudaEvent_t hop_back_ = nullptr;     // recorded on hop_stream_ (device 1), waited on `stream`
  MappedHostBuffer hop_x_, hop_work_, hop_sum_;  // pinned staging: xm and the work lists out; the rows back
  MappedHostBuffer hop_rows_;  // the 64/64 split's per-slot rows back (sized for the whole chunk)
  DeviceBuffer hop_rows0_;     // ... as they land on this device
  SecondStorage hop_hi_;  // x, tiles_n, tiles, pair_token, pair_dst, wts, act, down, sum on the second device
  // KV-head split (allocated only with kv_split_): the packet in (proj slices + conv state) and the rows back
  cudaEvent_t kv_ready_ = nullptr;     // recorded here, waited on the second device
  cudaEvent_t kv_loaded_ = nullptr;    // recorded there, waited here (the packet may be rewritten)
  cudaEvent_t kv_consumed_ = nullptr;  // recorded here, waited there (the result staging may be rewritten)
  cudaEvent_t kv_back_ = nullptr;      // recorded there, waited here
  MappedHostBuffer kv_in_, kv_out_;
  SecondStorage kv_hi_;  // proj, q, rel16, rel, out, conv, the per-layer slices on the second device
  struct KvLayer {
    const float* q_scale = nullptr;  // [24, 128] by absolute head
    const float* r_scale = nullptr;  // [24, 16]
    const float* rel_proj = nullptr;  // [16, 1024]
    const float* k_w = nullptr;      // k_static rows [4, kv_heads_hi * 128], compact heads
    const float* v_w = nullptr;
  };
  std::array<KvLayer, LAYERS> kv_layer_{};
};

}  // namespace ingine::model
