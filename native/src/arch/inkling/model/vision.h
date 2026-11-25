// Inkling-2's vision tower on the GPU (transformers Inkling2VisionModel): RGB pixels of one image -> one
// 2304-wide embedding row per image token, the rows that replace the embeddings of the image's placeholder
// tokens in the language model's input.
//
// The tower's weights (model.visual.*, 636 MB of BF16 in the checkpoint) stay resident. Matrix products
// take BF16 operands with FP32 accumulation (cuBLASLt, tensor cores); the residual stream, LayerNorm,
// softmax and GELU are FP32, so the result is closer to FP32 arithmetic than transformers' BF16 model.
// Attention is one GEMM pair per head over chunks of queries, the scores of a chunk held in FP32.
#pragma once

#include "core/arena.h"
#include "kernels/prefill/prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

namespace ingine::model {

inline constexpr int VISION_LAYERS = 23;
inline constexpr int VISION_OUT = 2304;  // the language model's hidden size

struct VisionLayer {
  const float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;  // [1024]
  const float *ls1 = nullptr, *ls2 = nullptr;                                          // layer scale [1024]
  const __nv_bfloat16* wqkv = nullptr;  // [3072, 1024]: q, k and v projections stacked
  const float* bqkv = nullptr;          // [3072]
  const __nv_bfloat16* wo = nullptr;    // [1024, 1024]
  const float* bo = nullptr;
  const __nv_bfloat16* fc = nullptr;    // [4096, 1024]
  const float* bfc = nullptr;           // [4096]
  const __nv_bfloat16* proj = nullptr;  // [1024, 4096]
  const float* bproj = nullptr;
};

struct VisionWeights {
  std::vector<DeviceBuffer> storage;
  const __nv_bfloat16* patch_w = nullptr;   // [1024, 1200]
  const float* pos = nullptr;               // [32 * 32, 1024]
  const float* cls = nullptr;               // [1024]: class embedding + class position embedding
  const float *ln_pre_w = nullptr, *ln_pre_b = nullptr;
  std::array<VisionLayer, VISION_LAYERS> layers{};
  const __nv_bfloat16* proj_in = nullptr;   // [4096, 4096]
  const __nv_bfloat16* proj_out = nullptr;  // [2304, 4096]
  const float* inv_freq = nullptr;          // [16] rotary frequencies
  std::size_t bytes = 0;                    // device memory held
};

// Reads model.visual.* from the checkpoint in model_dir; returns once the weights are on the device.
VisionWeights load_vision_weights(const std::filesystem::path& model_dir, cudaStream_t stream);

class VisionEncoder {
 public:
  VisionEncoder(const std::filesystem::path& model_dir, cudaStream_t stream);

  // Embeddings of the image rgb u8 [height, width, 3] (host): bf16 [padded_tokens(N), 2304] on the device,
  // rows [0, N) for its N = ceil(height / 40) * ceil(width / 40) tokens in raster order (the rest is
  // padding). Returns once they are computed. Working memory (workspace_bytes) is held only during the call.
  DeviceBuffer encode(const std::uint8_t* rgb, int height, int width);

  [[nodiscard]] static int padded_tokens(int tokens) { return (tokens + 15) / 16 * 16; }
  [[nodiscard]] std::size_t workspace_bytes(int height, int width) const;
  [[nodiscard]] std::size_t weight_bytes() const { return weights_.bytes; }

  // Device memory for the attention scores and probabilities of one chunk of queries (default 256 MiB):
  // the queries per GEMM are what fits, at least 64.
  void set_score_bytes(std::size_t bytes) { score_bytes_ = bytes; }

  // Tests: called with -1 after the embeddings (before ln_pre) and with i after encoder layer i; h is the
  // residual stream f32 [tokens, 1024] on the device, complete once the stream is synchronised.
  std::function<void(int stage, const float* h, int tokens)> observer;

 private:
  cudaStream_t stream_;
  VisionWeights weights_;
  kernels::prefill::Gemm gemm_;
  std::size_t score_bytes_ = std::size_t(256) << 20;
};

}  // namespace ingine::model
