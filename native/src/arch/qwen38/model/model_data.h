// Constants of the Qwen3.5/3.8 dense-hybrid pack: architectures ["Qwen3_5ForConditionalGeneration"], text
// model_type qwen3_5_text. One pack holds three shapes: 27B (Qwen3.8-27B = Qwen3.6-27B), 9B and 4B. Every value
// is copied from a checkpoint config at docs/arch/qwen3.8-27b.config.json, docs/arch/qwen3.5-9b.config.json or
// docs/arch/qwen3.5-4b.config.json unless its comment marks it derived.
//
// The recipe (3 linear / 1 full attention, head_dim 256, GDN conv_k 4, partial rotary, mrope [11, 11, 10]) is
// shared by all three shapes. Hidden size, layer count, attention heads, MLP width and linear value heads differ;
// Shape<...> derives the rest per shape. Kernels are templated on `class S` and read `S::HIDDEN`, `S::LAYERS`, so
// one set of kernels serves all three. Pick the binary's shape with Qwen38Shape (see the end of this header).
//
// Header-only on purpose: no CUDA and no core/ headers, so the pack's model and kernel translation units can both
// include it. Constants only; README.md lists what the pack does and does not implement yet.
#pragma once

#include <array>
#include <cstdint>

namespace ingine::arch::qwen38 {

// ------------------------------------------------------------------------------- shared text_config

inline constexpr int VOCAB = 248320;  // vocab_size, all three shapes
inline constexpr int MAX_POS = 262144;
inline constexpr float EPS = 1e-6f;  // rms_norm_eps

inline constexpr std::int32_t EOS_TOKEN_ID = 248044;  // also bos_token_id
inline constexpr std::int32_t VISION_START_TOKEN_ID = 248053, VISION_END_TOKEN_ID = 248054;
inline constexpr std::int32_t IMAGE_TOKEN_ID = 248056, VIDEO_TOKEN_ID = 248057;

// ---------------------------------------------------------------------------------- full attention

// full_attention_interval: layer_types is a fixed 3 linear / 1 full cycle in every shape, so layers 3, 7, ... are
// full and the cycle count is LAYERS / 4.
inline constexpr int FULL_ATTENTION_INTERVAL = 4;

inline constexpr int HEAD_DIM = 256;
// rope_parameters: partial_rotary_factor 0.25 on a 256-dim head, theta 1e7, interleaved mrope with
// mrope_section [11, 11, 10].
inline constexpr int ROTARY_DIM = HEAD_DIM / 4;  // 64 rotated of the 256 head dims
inline constexpr float ROPE_THETA = 1e7f;
inline constexpr std::array<int, 3> MROPE_SECTION = {11, 11, 10};
inline constexpr bool MROPE_INTERLEAVED = true;

// ------------------------------------------------- linear attention (GDN-style, shared head geometry)

inline constexpr int CONV_K = 4;  // linear_conv_kernel_dim
inline constexpr int LIN_KEY_HEADS = 16, LIN_KEY_DIM = 128;
inline constexpr int LIN_VALUE_DIM = 128;
// mamba_ssm_dtype: float32 - the recurrence state and its gates stay FP32.

// -------------------------------------------------------------------------------------- MTP head

// mtp_num_hidden_layers: one extra decoder layer after LAYERS that takes the embedding of the next token
// (mtp_use_dedicated_embeddings false).
inline constexpr int MTP_LAYERS = 1;

// ------------------------------------------------------------------------------ layer type table

enum class LayerKind : std::uint8_t { Linear, Full };

inline constexpr bool is_full_layer(int l) {
  return l % FULL_ATTENTION_INTERVAL == FULL_ATTENTION_INTERVAL - 1;
}
inline constexpr LayerKind layer_kind(int l) { return is_full_layer(l) ? LayerKind::Full : LayerKind::Linear; }

template <int Layers>
inline constexpr std::array<LayerKind, Layers> layer_kind_table() {
  std::array<LayerKind, Layers> k{};
  for (int l = 0; l < Layers; ++l) k[l] = layer_kind(l);
  return k;
}

// Per-layer slot in the full-attention KV cache (-1 on linear layers) and in the linear-attention state (-1 on full
// layers), the way the Inkling pack's RING_INDEX / FULL_INDEX map visits.
template <int Layers>
inline constexpr std::array<int, Layers> full_index_table() {
  std::array<int, Layers> r{};
  int n = 0;
  for (int l = 0; l < Layers; ++l) r[l] = is_full_layer(l) ? n++ : -1;
  return r;
}
template <int Layers>
inline constexpr std::array<int, Layers> linear_index_table() {
  std::array<int, Layers> r{};
  int n = 0;
  for (int l = 0; l < Layers; ++l) r[l] = is_full_layer(l) ? -1 : n++;
  return r;
}

// ----------------------------------------------------------------------------------------- shapes

// One dense-hybrid shape. Only the fields that differ between checkpoints are parameters; the recipe constants
// above are shared. TiedEmbeddings is the checkpoint's tie_word_embeddings (true on the 4B only).
template <int HiddenSize, int LayerCount, int AttentionHeads, int KvHeads, int IntermediateSize,
          int LinearValueHeads, bool TiedEmbeddings>
struct Shape {
  static constexpr int HIDDEN = HiddenSize;
  static constexpr int LAYERS = LayerCount;
  static constexpr int HEADS = AttentionHeads;
  static constexpr int KV_HEADS = KvHeads;
  // Dense SwiGLU MLP on every layer (hidden_act silu, no experts).
  static constexpr int INTERMEDIATE = IntermediateSize;
  static constexpr int LIN_VALUE_HEADS = LinearValueHeads;
  static constexpr bool TIED_EMBEDDINGS = TiedEmbeddings;

  static constexpr int Q_DIM = HEADS * HEAD_DIM;      // 6144 / 4096 / 4096
  static constexpr int KV_DIM = KV_HEADS * HEAD_DIM;  // 1024
  // attn_output_gate: q_proj emits two Q_DIM halves, query then gate (torch.chunk of the projection, derived)
  static constexpr int Q_PROJ_ROWS = 2 * Q_DIM;

  static constexpr int LIN_KEY_DIM_TOT = LIN_KEY_HEADS * LIN_KEY_DIM;        // 2048
  static constexpr int LIN_VALUE_DIM_TOT = LIN_VALUE_HEADS * LIN_VALUE_DIM;  // 6144 / 4096 / 4096
  // Conv width derived from the Qwen3-Next in_proj layout (q and k channels, then v); not a config field, so it is
  // confirmed against the checkpoint's in_proj rows when the GDN weights are loaded.
  static constexpr int LIN_CONV_DIM = 2 * LIN_KEY_DIM_TOT + LIN_VALUE_DIM_TOT;  // 10240 / 8192 / 8192

  static constexpr int NUM_FULL_LAYERS = LAYERS / FULL_ATTENTION_INTERVAL;  // 16 / 8 / 8
  static constexpr int NUM_LINEAR_LAYERS = LAYERS - NUM_FULL_LAYERS;        // 48 / 24 / 24
  static constexpr int MTP_LAYER_BEGIN = LAYERS;

  static constexpr std::array<LayerKind, LAYERS> LAYER_KIND = layer_kind_table<LAYERS>();
  static constexpr std::array<int, LAYERS> FULL_INDEX = full_index_table<LAYERS>();
  static constexpr std::array<int, LAYERS> LINEAR_INDEX = linear_index_table<LAYERS>();

  static_assert(LAYERS > 0 && LAYERS % FULL_ATTENTION_INTERVAL == 0,
                "layer_types must be a whole number of cycles");
  static_assert(INTERMEDIATE > 0 && INTERMEDIATE != HIDDEN,
                "dense MLP width is the config's intermediate_size");
  static_assert(LIN_VALUE_HEADS % LIN_KEY_HEADS == 0, "value heads are a whole multiple of key heads");
  static_assert(!is_full_layer(0) && !is_full_layer(1) && !is_full_layer(2) && is_full_layer(3),
                "text_config.layer_types starts linear, linear, linear, full");
  static_assert(is_full_layer(LAYERS - 1), "text_config.layer_types ends on a full-attention layer");
  static_assert(FULL_INDEX[LAYERS - 1] == NUM_FULL_LAYERS - 1 && LINEAR_INDEX[LAYERS - 1] == -1);
};

using Shape27 = Shape<5120, 64, 24, 4, 17408, 48, false>;  // Qwen3.8-27B = Qwen3.6-27B
using Shape9 = Shape<4096, 32, 16, 4, 12288, 32, false>;   // Qwen3.5-9B
using Shape4 = Shape<2560, 32, 16, 4, 9216, 32, true>;     // Qwen3.5-4B, tied embeddings

using Defaults = Shape27;

// -------------------------------------------------------------------- compile-time shape selection

// Kernels and model code use `Qwen38Shape` (a typedef, so `template <class S>` takes it directly).
// INGINE_QWEN38_SHAPE is a CMake cache variable on qwen38_data and also honors a plain -D override.
#ifndef INGINE_QWEN38_SHAPE
#define INGINE_QWEN38_SHAPE 27
#endif

#if INGINE_QWEN38_SHAPE == 27
using Qwen38Shape = Shape27;
#elif INGINE_QWEN38_SHAPE == 9
using Qwen38Shape = Shape9;
#elif INGINE_QWEN38_SHAPE == 4
using Qwen38Shape = Shape4;
#else
#error "INGINE_QWEN38_SHAPE must be 27, 9, or 4"
#endif

// ---------------------------------------------------------- default-shape aliases (back-compat)

// Bare names resolve to Shape27, the shape the pack shipped with. New code should read Qwen38Shape.
inline constexpr int HIDDEN = Defaults::HIDDEN;
inline constexpr int LAYERS = Defaults::LAYERS;
inline constexpr int INTERMEDIATE = Defaults::INTERMEDIATE;
inline constexpr int HEADS = Defaults::HEADS;
inline constexpr int KV_HEADS = Defaults::KV_HEADS;
inline constexpr int Q_DIM = Defaults::Q_DIM;
inline constexpr int KV_DIM = Defaults::KV_DIM;
inline constexpr int Q_PROJ_ROWS = Defaults::Q_PROJ_ROWS;
inline constexpr int LIN_VALUE_HEADS = Defaults::LIN_VALUE_HEADS;
inline constexpr int LIN_KEY_DIM_TOT = Defaults::LIN_KEY_DIM_TOT;
inline constexpr int LIN_VALUE_DIM_TOT = Defaults::LIN_VALUE_DIM_TOT;
inline constexpr int LIN_CONV_DIM = Defaults::LIN_CONV_DIM;
inline constexpr int NUM_FULL_LAYERS = Defaults::NUM_FULL_LAYERS;
inline constexpr int NUM_LINEAR_LAYERS = Defaults::NUM_LINEAR_LAYERS;
inline constexpr int MTP_LAYER_BEGIN = Defaults::MTP_LAYER_BEGIN;
inline constexpr std::array<LayerKind, LAYERS> LAYER_KIND = Defaults::LAYER_KIND;
inline constexpr std::array<int, LAYERS> FULL_INDEX = Defaults::FULL_INDEX;
inline constexpr std::array<int, LAYERS> LINEAR_INDEX = Defaults::LINEAR_INDEX;

}  // namespace ingine::arch::qwen38
