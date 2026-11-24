// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {

// Expert residency across two devices (EP step 2: load-time placement): split[l] is the number of layer l's
// LAYER_EXPERTS experts kept on the primary device, the rest live on the second one. Only 0 (the whole layer's
// experts there) and LAYER_EXPERTS (all here) are loadable: moe_up/moe_down take one base pointer per expert
// tensor, so a split inside a layer needs per-expert bases (step 3). Default: every expert on the primary.
inline constexpr int EXPERT_RESIDENCY_LAYERS = 24;  // = model.LAYERS
inline constexpr int LAYER_EXPERTS = 128;           // = model.EXPERTS

struct ExpertResidency {
  std::array<std::int16_t, EXPERT_RESIDENCY_LAYERS> split;
  // Not `= {}`: zero-initialising the array would put every layer's experts on the second device.
  ExpertResidency() { split.fill(LAYER_EXPERTS); }
  bool all_primary() const {
    for (std::int16_t s : split)
      if (s != LAYER_EXPERTS) return false;
    return true;
  }
  int device_of(int layer, int expert) const { return expert < split[layer] ? 0 : 1; }
};

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
};

// The token whose embedding an image token replaces (<|unused_200054|>, the chat template's image placeholder).
inline constexpr std::int32_t IMAGE_PLACEHOLDER_ID = 200054;

// One image of a prompt. The chat template marks an image with one placeholder token; the prompt holds
// tokens() ids in its place, one per 40x40-pixel tile in raster order, whose embeddings the vision tower
// computes from the pixels. These ids are negative (never a vocabulary id) and a function of the image
// content and the tile index (image::token_id), so prompts compare equal exactly when their images do.
struct PromptImage {
  int pos = 0;                                           // prompt position of the image's first token
  int height = 0, width = 0;                             // pixels
  std::shared_ptr<const std::vector<std::uint8_t>> rgb;  // [height, width, 3]
  [[nodiscard]] int rows() const { return (height + 39) / 40; }
  [[nodiscard]] int cols() const { return (width + 39) / 40; }
  [[nodiscard]] int tokens() const { return rows() * cols(); }
};

// A prompt: token ids and its images, in the order of their positions (without overlap).
struct Prompt {
  std::span<const std::int32_t> ids;
  std::span<const PromptImage> images;
};

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,        // empty state, the whole prompt prefilled
  Extend,      // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,  // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
  Parked,      // restored a parked conversation's checkpoint (a prefix of this prompt; RuntimeOptions::park_mb)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    case Reuse::Parked: return "parked";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  Reuse reuse = Reuse::None;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
  std::vector<SpecRoundRecord> spec_trace;  // forced replay only
  // Images: tokens of the prompt's images, how many images the vision tower encoded for this request (those
  // not inside the reused prefix) and its time (part of prefill_seconds).
  int image_tokens = 0, images_encoded = 0;
  double vision_seconds = 0;
};

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  // Tokens per prefill chunk. 3072 (the largest the sliding attention takes) reads a long prompt 6-10% faster
  // than 2048 for 0.38 GB more scratch; results differ between chunk sizes only by rounding.
  int prefill_chunk = 3072;
  bool dense_fp8 = true, kv_fp8 = false, spec = false;
  float spec_threshold = 0.8f;
  // Conversation parking: device memory (MiB) for the checkpoints of conversations displaced by another
  // one, least recently used evicted, all released when a device allocation fails; 0 disables; negative:
  // min(3072, free memory after start-up - 4096), at least 0.
  int park_mb = -1;
  // Image input: load the vision tower (636 MB of the checkpoint, resident) and accept prompts with images.
  bool vision = false;
  // Expert parallelism: the CUDA device opened as the second device; expert weights can be loaded onto it
  // (expert_residency) and their visits run there. Decode's other kernels stay on the first device. Negative:
  // one device.
  int second_gpu = -1;
  // Where each layer's experts live (default: all on this device). Only 0 (the whole layer's experts there) and
  // LAYER_EXPERTS load: the moe kernels take one base pointer per expert tensor, so a split inside a layer needs
  // per-expert bases (step 3). With any layer moved, a step spans two devices: it runs eagerly (no CUDA graph),
  // DSpark verifies eagerly at one extra host synchronisation per round, and a prompt is processed one token per
  // decode step.
  ExpertResidency expert_residency;
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
  // The same for a prompt with images. Generators without image input take prompts without images only.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) {
    if (!prompt.images.empty()) throw std::invalid_argument("this generator does not take images");
    return generate(prompt.ids, max_new_tokens, sampling, stop_ids, on_tokens);
  }
  virtual bool supports_images() const { return false; }
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model
