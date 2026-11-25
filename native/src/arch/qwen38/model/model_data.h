// Constants of the Qwen3.8-27B pack: architectures ["Qwen3_5ForConditionalGeneration"], text model_type qwen3_5_text.
// Every value below is copied from the checkpoint config at docs/arch/qwen3.8-27b.config.json unless its comment
// marks it derived. Qwen3.6-27B publishes the same text_config and is the same pack.
//
// Header-only on purpose: no CUDA and no core/ headers, so the pack's model and kernel translation units can both
// include it. Constants only; README.md lists what the pack does and does not implement yet.
#pragma once

#include <array>
#include <cstdint>

namespace ingine::arch::qwen38 {

// ----------------------------------------------------------------------------------------------- text_config

inline constexpr int HIDDEN = 5120;
inline constexpr int LAYERS = 64;
inline constexpr int VOCAB = 248320;
inline constexpr int MAX_POS = 262144;
inline constexpr float EPS = 1e-6f;  // rms_norm_eps

// Dense SwiGLU MLP on every layer (hidden_act silu, no experts). tie_word_embeddings is false.
inline constexpr int INTERMEDIATE = 17408;

inline constexpr std::int32_t EOS_TOKEN_ID = 248044;  // also bos_token_id
inline constexpr std::int32_t VISION_START_TOKEN_ID = 248053, VISION_END_TOKEN_ID = 248054;
inline constexpr std::int32_t IMAGE_TOKEN_ID = 248056, VIDEO_TOKEN_ID = 248057;

// ---------------------------------------------------------------------------------------- full attention

// full_attention_interval: layer_types is a fixed 3 linear / 1 full cycle, so layers 3, 7, ..., 63 are full.
inline constexpr int FULL_ATTENTION_INTERVAL = 4;

inline constexpr int HEADS = 24, KV_HEADS = 4, HEAD_DIM = 256;
inline constexpr int Q_DIM = HEADS * HEAD_DIM;      // 6144
inline constexpr int KV_DIM = KV_HEADS * HEAD_DIM;  // 1024
// attn_output_gate: q_proj emits two Q_DIM halves, query then gate (torch.chunk of the projection, derived)
inline constexpr int Q_PROJ_ROWS = 2 * Q_DIM;  // 12288

// rope_parameters: partial_rotary_factor 0.25 on a 256-dim head, theta 1e7, interleaved mrope with
// mrope_section [11, 11, 10].
inline constexpr int ROTARY_DIM = HEAD_DIM / 4;  // 64 rotated of the 256 head dims
inline constexpr float ROPE_THETA = 1e7f;
inline constexpr std::array<int, 3> MROPE_SECTION = {11, 11, 10};
inline constexpr bool MROPE_INTERLEAVED = true;

// ------------------------------------------------------------ linear attention (GDN-style, 48 layers)

inline constexpr int CONV_K = 4;  // linear_conv_kernel_dim
inline constexpr int LIN_KEY_HEADS = 16, LIN_KEY_DIM = 128;
// 48 value heads to 16 key heads (3:1), unlike the 35B-A3B pack's 32; the recurrence keeps this ratio.
inline constexpr int LIN_VALUE_HEADS = 48, LIN_VALUE_DIM = 128;
inline constexpr int LIN_KEY_DIM_TOT = LIN_KEY_HEADS * LIN_KEY_DIM;        // 2048
inline constexpr int LIN_VALUE_DIM_TOT = LIN_VALUE_HEADS * LIN_VALUE_DIM;  // 6144
// Conv width derived from the Qwen3-Next in_proj layout (q and k channels, then v); not a config field, so it is
// confirmed against the checkpoint's in_proj rows when the GDN weights are loaded.
inline constexpr int LIN_CONV_DIM = 2 * LIN_KEY_DIM_TOT + LIN_VALUE_DIM_TOT;  // 10240
// mamba_ssm_dtype: float32 - the recurrence state and its gates stay FP32.

// ---------------------------------------------------------------------------------------------- MTP head

// mtp_num_hidden_layers: one extra decoder layer after LAYERS that takes the embedding of the next token
// (mtp_use_dedicated_embeddings false).
inline constexpr int MTP_LAYERS = 1;
inline constexpr int MTP_LAYER_BEGIN = LAYERS;

// ----------------------------------------------------------------------------------- layer type table

enum class LayerKind : std::uint8_t { Linear, Full };

inline constexpr bool is_full_layer(int l) {
  return l % FULL_ATTENTION_INTERVAL == FULL_ATTENTION_INTERVAL - 1;
}
inline constexpr LayerKind layer_kind(int l) { return is_full_layer(l) ? LayerKind::Full : LayerKind::Linear; }

inline constexpr int NUM_FULL_LAYERS = LAYERS / FULL_ATTENTION_INTERVAL;  // 16
inline constexpr int NUM_LINEAR_LAYERS = LAYERS - NUM_FULL_LAYERS;        // 48
static_assert(LAYERS % FULL_ATTENTION_INTERVAL == 0, "layer_types must be a whole number of cycles");

inline constexpr std::array<LayerKind, LAYERS> LAYER_KIND = [] {
  std::array<LayerKind, LAYERS> k{};
  for (int l = 0; l < LAYERS; ++l) k[l] = layer_kind(l);
  return k;
}();

// Per-layer slot in the full-attention KV cache (-1 on linear layers) and in the linear-attention state (-1 on full
// layers), the way the Inkling pack's RING_INDEX / FULL_INDEX map visits.
inline constexpr std::array<int, LAYERS> FULL_INDEX = [] {
  std::array<int, LAYERS> r{};
  int n = 0;
  for (int l = 0; l < LAYERS; ++l) r[l] = is_full_layer(l) ? n++ : -1;
  return r;
}();
inline constexpr std::array<int, LAYERS> LINEAR_INDEX = [] {
  std::array<int, LAYERS> r{};
  int n = 0;
  for (int l = 0; l < LAYERS; ++l) r[l] = is_full_layer(l) ? -1 : n++;
  return r;
}();

static_assert(LAYER_KIND[0] == LayerKind::Linear && LAYER_KIND[1] == LayerKind::Linear &&
                  LAYER_KIND[2] == LayerKind::Linear && LAYER_KIND[3] == LayerKind::Full,
              "text_config.layer_types starts linear, linear, linear, full");
static_assert(LAYER_KIND[LAYERS - 1] == LayerKind::Full, "text_config.layer_types ends on a full-attention layer");
static_assert(FULL_INDEX[LAYERS - 1] == NUM_FULL_LAYERS - 1 && LINEAR_INDEX[LAYERS - 1] == -1);

static_assert(LIN_VALUE_HEADS % LIN_KEY_HEADS == 0, "value heads are a whole multiple of key heads");

}  // namespace ingine::arch::qwen38
