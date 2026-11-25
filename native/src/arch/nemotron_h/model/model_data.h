// Constants of the Nemotron 3.5 Lightning pack: architectures ["NemotronHForCausalLM"], model_type
// nemotron_h. Every value is copied from the NVFP4 checkpoint config at
// docs/arch/nemotron-lightning-30b-a3b.config.json unless its comment marks it derived. The BF16 card of the
// same model publishes max_position_embeddings 262144; this pack takes the NVFP4 card's 1048576.
//
// The recipe is one stack of three block kinds: Mamba-2, sparse MoE (relu2 experts plus a shared expert) and full
// attention. The 30B-A3B shape is 52 layers, 23 Mamba + 23 MoE + 6 attention. The 74B-A9B Bolt core
// (NemotronHPuzzleForCausalLM) shares the recipe at other sizes; Shape<...> keeps the per-shape fields and Shape30
// is the default. Bolt E4B is Gemma4ForConditionalGeneration and is not this pack.
//
// Header-only on purpose: no CUDA and no core/ headers, so the pack's model and kernel translation units can both
// include it. Constants only; README.md lists what the pack does and does not implement yet.
#pragma once

#include <array>
#include <cstdint>

namespace ingine::arch::nemotron_h {

// ------------------------------------------------------------------------------------ shared text_config

inline constexpr int VOCAB = 131072;      // vocab_size
inline constexpr int MAX_POS = 1048576;   // max_position_embeddings, NVFP4 card
inline constexpr float EPS = 1e-5f;       // norm_eps and layer_norm_epsilon

inline constexpr std::int32_t PAD_TOKEN_ID = 0;
inline constexpr std::int32_t BOS_TOKEN_ID = 1;
inline constexpr std::int32_t EOS_TOKEN_ID = 2;

// ------------------------------------------------------------------------------------- full attention

// Only six layers are attention, so the KV cache is 2 heads * 128 dims on those layers and nothing on the other
// 46. sliding_window is null: every attention layer is full-causal.
inline constexpr int KV_HEADS = 2;  // num_key_value_heads
inline constexpr int HEAD_DIM = 128;
inline constexpr int ROTARY_DIM = HEAD_DIM;  // partial_rotary_factor 1.0, the whole head rotates
inline constexpr float ROPE_THETA = 1e4f;
inline constexpr bool ATTENTION_BIAS = false;  // attention_bias and use_bias

// ---------------------------------------------------------------------------------------------- Mamba-2

// Mamba-2 mixer geometry shared by the shapes. expand is the config's field; the reference mixer width is the
// head count * head dim, and Shape<...> derives the in_proj and conv rows from that.
inline constexpr int SSM_STATE_SIZE = 128;
inline constexpr int CONV_KERNEL = 4;         // conv_kernel, conv1d taps
inline constexpr int MAMBA_EXPAND = 2;        // expand; the mixer width is MAMBA_HEADS * MAMBA_HEAD_DIM below
inline constexpr int MAMBA_CHUNK_SIZE = 128;  // chunk_size
inline constexpr bool MAMBA_CONV_BIAS = true;   // use_conv_bias
inline constexpr bool MAMBA_PROJ_BIAS = false;  // mamba_proj_bias
// time_step_*: the config bounds used to initialize dt; the reference scan clamps dt to [TIME_STEP_MIN, inf).
inline constexpr float TIME_STEP_MIN = 1e-3f;
inline constexpr float TIME_STEP_MAX = 1e-1f;
inline constexpr float TIME_STEP_FLOOR = 1e-4f;
// mamba_ssm_cache_dtype: float32 - the recurrence state and its gates stay FP32.

// -------------------------------------------------------------------------------------------------- MoE

inline constexpr bool SHARED_EXPERT_OVERLAP = true;       // moe_shared_expert_overlap
inline constexpr bool RESCALE_PRE_NORM_RESIDUAL = true;   // rescale_prenorm_residual
inline constexpr bool RESIDUAL_IN_FP32 = false;           // residual_in_fp32
inline constexpr float ROUTED_SCALING_FACTOR = 2.5f;      // routed_scaling_factor

// ------------------------------------------------------------------------------------------ layer kinds

enum class LayerKind : std::uint8_t { Mamba, Moe, Attention };

// layers_block_type verbatim: 23 Mamba + 23 MoE + 6 attention. Attention sits on layers 5, 12, 19, 26, 33 and 42;
// the Mamba/MoE blocks between them alternate starting with Mamba (5 blocks before the first attention, then runs
// of 6, 6, 6, 6, 8 and 9).
inline constexpr std::array<LayerKind, 52> LIGHTNING_BLOCK_TYPE = {
    LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe,
    LayerKind::Mamba, LayerKind::Attention, LayerKind::Moe, LayerKind::Mamba,
    LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba,
    LayerKind::Attention, LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe,
    LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba, LayerKind::Attention,
    LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba,
    LayerKind::Moe, LayerKind::Mamba, LayerKind::Attention, LayerKind::Moe,
    LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe,
    LayerKind::Mamba, LayerKind::Attention, LayerKind::Moe, LayerKind::Mamba,
    LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba,
    LayerKind::Moe, LayerKind::Mamba, LayerKind::Attention, LayerKind::Moe,
    LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe,
    LayerKind::Mamba, LayerKind::Moe, LayerKind::Mamba, LayerKind::Moe};

template <std::size_t Layers>
inline constexpr std::array<int, Layers> kind_index_table(const std::array<LayerKind, Layers>& kinds,
                                                          LayerKind kind) {
  std::array<int, Layers> r{};
  int n = 0;
  for (int l = 0; l < static_cast<int>(Layers); ++l) r[l] = kinds[l] == kind ? n++ : -1;
  return r;
}

template <std::size_t Layers>
inline constexpr int count_kind(const std::array<LayerKind, Layers>& kinds, LayerKind kind) {
  int n = 0;
  for (int l = 0; l < static_cast<int>(Layers); ++l) n += kinds[l] == kind;
  return n;
}

// ---------------------------------------------------------------------------------------------- MTP head

// num_nextn_predict_layers: one MTP prediction step. Its mtp_layers_block_type is two physical layers, named
// mtp.layers.0 (attention) and mtp.layers.1 (MoE) in the checkpoint, not decoder layers of the stack above.
inline constexpr int NEXTN_PREDICT_LAYERS = 1;
inline constexpr std::array<LayerKind, 2> MTP_BLOCK_TYPES = {LayerKind::Attention, LayerKind::Moe};
inline constexpr int MTP_LAYERS = static_cast<int>(MTP_BLOCK_TYPES.size());  // 2

// ----------------------------------------------------------------------------------------------- shapes

// One Nemotron-H shape. Only the fields that differ between checkpoints are parameters; the attention and Mamba-2
// geometry above is shared by Lightning and the 74B-A9B Bolt core. BlockTypes is the checkpoint's
// layers_block_type, so the irregular block order is data, not a formula.
template <int HiddenSize, int LayerCount, int AttentionHeads, int MambaHeads, int MambaHeadDim,
          int MambaGroups, int RoutedExperts, int TopK, int SharedExperts, int MoeIntermediate,
          int SharedIntermediate, int MoeGroups, int MoeTopKGroups, bool NormTopkProb, bool TiedEmbeddings,
          std::array<LayerKind, LayerCount> BlockTypes>
struct Shape {
  static constexpr int HIDDEN = HiddenSize;
  static constexpr int LAYERS = LayerCount;
  static constexpr int HEADS = AttentionHeads;  // num_attention_heads
  static constexpr int Q_DIM = HEADS * HEAD_DIM;
  static constexpr int KV_DIM = KV_HEADS * HEAD_DIM;

  static constexpr int MAMBA_HEADS = MambaHeads;        // mamba_num_heads
  static constexpr int MAMBA_HEAD_DIM = MambaHeadDim;   // mamba_head_dim
  static constexpr int MAMBA_GROUPS = MambaGroups;      // n_groups, evolution-matrix groups
  // Mamba-2 widths from the reference mixer: x + z is MAMBA_INNER, conv carries x + B + C, and in_proj emits
  // gate, conv input and dt.
  static constexpr int MAMBA_INNER = MAMBA_HEADS * MAMBA_HEAD_DIM;  // 4096
  static constexpr int MAMBA_CONV_DIM = MAMBA_INNER + 2 * MAMBA_GROUPS * SSM_STATE_SIZE;  // 6144
  static constexpr int MAMBA_IN_PROJ_ROWS = MAMBA_INNER + MAMBA_CONV_DIM + MAMBA_HEADS;   // 10304
  static constexpr int MAMBA_NORM_GROUP = MAMBA_INNER / MAMBA_GROUPS;                     // 512

  static constexpr int EXPERTS = RoutedExperts;            // n_routed_experts
  static constexpr int TOP_K = TopK;                       // num_experts_per_tok
  static constexpr int SHARED_EXPERTS = SharedExperts;     // n_shared_experts
  static constexpr int MOE_INTER = MoeIntermediate;        // moe_intermediate_size, per routed expert
  static constexpr int SHARED_INTER = SharedIntermediate;  // moe_shared_expert_intermediate_size
  static constexpr int MOE_GROUPS = MoeGroups;             // n_group, routing groups
  static constexpr int MOE_TOPK_GROUPS = MoeTopKGroups;    // topk_group
  static constexpr bool NORM_TOPK_PROB = NormTopkProb;
  static constexpr bool TIED_EMBEDDINGS = TiedEmbeddings;

  static constexpr std::array<LayerKind, LAYERS> LAYER_KIND = BlockTypes;
  static constexpr std::array<int, LAYERS> MAMBA_INDEX = kind_index_table(BlockTypes, LayerKind::Mamba);
  static constexpr std::array<int, LAYERS> MOE_INDEX = kind_index_table(BlockTypes, LayerKind::Moe);
  static constexpr std::array<int, LAYERS> ATTENTION_INDEX = kind_index_table(BlockTypes, LayerKind::Attention);

  static constexpr int NUM_MAMBA_LAYERS = count_kind(BlockTypes, LayerKind::Mamba);
  static constexpr int NUM_MOE_LAYERS = count_kind(BlockTypes, LayerKind::Moe);
  static constexpr int NUM_ATTENTION_LAYERS = count_kind(BlockTypes, LayerKind::Attention);
  static constexpr int MTP_LAYER_BEGIN = LAYERS;

  static_assert(LAYERS == static_cast<int>(BlockTypes.size()));
  static_assert(HEADS > 0 && HEADS % KV_HEADS == 0, "attention heads are a whole multiple of KV heads");
  static_assert(MAMBA_HEADS > 0 && MAMBA_HEAD_DIM > 0);
  static_assert(MAMBA_GROUPS > 0 && MAMBA_INNER % MAMBA_GROUPS == 0,
                "mamba state groups divide the mixer width");
  static_assert(EXPERTS > 0 && TOP_K > 0 && TOP_K <= EXPERTS, "top-k routed experts fit the expert count");
  static_assert(SHARED_EXPERTS >= 0 && MOE_INTER > 0 && SHARED_INTER >= 0);
  static_assert(MOE_GROUPS > 0 && EXPERTS % MOE_GROUPS == 0, "routed experts split into whole routing groups");
  static_assert(MOE_TOPK_GROUPS > 0 && MOE_TOPK_GROUPS <= MOE_GROUPS);
  static_assert(NUM_MAMBA_LAYERS + NUM_MOE_LAYERS + NUM_ATTENTION_LAYERS == LAYERS);
};

using Shape30 =
    Shape<2688, 52, 32, 64, 64, 8, 128, 6, 1, 1856, 3712, 1, 1, true, false, LIGHTNING_BLOCK_TYPE>;

using Defaults = Shape30;
using NemotronHShape = Defaults;

// -------------------------------------------------------------------------- default-shape aliases

// Bare names resolve to Shape30, the Lightning shape. New code should read NemotronHShape. A future Shape74 (Bolt
// 74B-A9B, NemotronHPuzzleForCausalLM) would add its own block table and instantiation; nothing here hardcodes it.
inline constexpr int HIDDEN = Defaults::HIDDEN;
inline constexpr int LAYERS = Defaults::LAYERS;
inline constexpr int HEADS = Defaults::HEADS;
inline constexpr int Q_DIM = Defaults::Q_DIM;
inline constexpr int KV_DIM = Defaults::KV_DIM;
inline constexpr int MAMBA_HEADS = Defaults::MAMBA_HEADS;
inline constexpr int MAMBA_HEAD_DIM = Defaults::MAMBA_HEAD_DIM;
inline constexpr int MAMBA_GROUPS = Defaults::MAMBA_GROUPS;
inline constexpr int MAMBA_INNER = Defaults::MAMBA_INNER;
inline constexpr int MAMBA_CONV_DIM = Defaults::MAMBA_CONV_DIM;
inline constexpr int MAMBA_IN_PROJ_ROWS = Defaults::MAMBA_IN_PROJ_ROWS;
inline constexpr int MAMBA_NORM_GROUP = Defaults::MAMBA_NORM_GROUP;
inline constexpr int EXPERTS = Defaults::EXPERTS;
inline constexpr int TOP_K = Defaults::TOP_K;
inline constexpr int SHARED_EXPERTS = Defaults::SHARED_EXPERTS;
inline constexpr int MOE_INTER = Defaults::MOE_INTER;
inline constexpr int SHARED_INTER = Defaults::SHARED_INTER;
inline constexpr int MOE_GROUPS = Defaults::MOE_GROUPS;
inline constexpr int MOE_TOPK_GROUPS = Defaults::MOE_TOPK_GROUPS;
inline constexpr bool NORM_TOPK_PROB = Defaults::NORM_TOPK_PROB;
inline constexpr bool TIED_EMBEDDINGS = Defaults::TIED_EMBEDDINGS;
inline constexpr int NUM_MAMBA_LAYERS = Defaults::NUM_MAMBA_LAYERS;
inline constexpr int NUM_MOE_LAYERS = Defaults::NUM_MOE_LAYERS;
inline constexpr int NUM_ATTENTION_LAYERS = Defaults::NUM_ATTENTION_LAYERS;
inline constexpr int MTP_LAYER_BEGIN = Defaults::MTP_LAYER_BEGIN;
inline constexpr std::array<LayerKind, LAYERS> LAYER_KIND = Defaults::LAYER_KIND;
inline constexpr std::array<int, LAYERS> MAMBA_INDEX = Defaults::MAMBA_INDEX;
inline constexpr std::array<int, LAYERS> MOE_INDEX = Defaults::MOE_INDEX;
inline constexpr std::array<int, LAYERS> ATTENTION_INDEX = Defaults::ATTENTION_INDEX;

}  // namespace ingine::arch::nemotron_h
