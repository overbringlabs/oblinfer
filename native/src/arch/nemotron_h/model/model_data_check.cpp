// The pack is header-only at this stage, so this translation unit exists to make the default build compile
// model/model_data.h: a broken constant or layer table fails `ingine-serve`'s build without linking anything from
// the pack into it.
#include "model/model_data.h"

#include <type_traits>

namespace {

using namespace ingine::arch::nemotron_h;

// Walks a shape's three slot tables against LAYER_KIND and checks that each kind's slots are dense and exclusive.
template <class S>
constexpr bool kind_maps_cover_every_layer() {
  int mamba = 0, moe = 0, attention = 0;
  for (int l = 0; l < S::LAYERS; ++l) {
    switch (S::LAYER_KIND[l]) {
      case LayerKind::Mamba:
        if (S::MAMBA_INDEX[l] != mamba || S::MOE_INDEX[l] != -1 || S::ATTENTION_INDEX[l] != -1) return false;
        ++mamba;
        break;
      case LayerKind::Moe:
        if (S::MAMBA_INDEX[l] != -1 || S::MOE_INDEX[l] != moe || S::ATTENTION_INDEX[l] != -1) return false;
        ++moe;
        break;
      case LayerKind::Attention:
        if (S::MAMBA_INDEX[l] != -1 || S::MOE_INDEX[l] != -1 || S::ATTENTION_INDEX[l] != attention) return false;
        ++attention;
        break;
    }
  }
  return mamba == S::NUM_MAMBA_LAYERS && moe == S::NUM_MOE_LAYERS && attention == S::NUM_ATTENTION_LAYERS &&
         mamba + moe + attention == S::LAYERS;
}

static_assert(kind_maps_cover_every_layer<Shape30>(), "Shape30 layer tables cover every decoder layer");

static_assert(Shape30::HIDDEN == 2688 && Shape30::LAYERS == 52 && VOCAB == 131072 && MAX_POS == 1048576,
              "Lightning 30B-A3B text_config");
static_assert(Shape30::NUM_MAMBA_LAYERS == 23 && Shape30::NUM_MOE_LAYERS == 23 &&
                  Shape30::NUM_ATTENTION_LAYERS == 6,
              "layers_block_type is 23 Mamba + 23 MoE + 6 attention");
static_assert(Shape30::LAYER_KIND[0] == LayerKind::Mamba && Shape30::LAYER_KIND[51] == LayerKind::Moe,
              "layers_block_type starts on Mamba and ends on MoE");
static_assert(Shape30::LAYER_KIND[5] == LayerKind::Attention && Shape30::LAYER_KIND[12] == LayerKind::Attention &&
                  Shape30::LAYER_KIND[19] == LayerKind::Attention &&
                  Shape30::LAYER_KIND[26] == LayerKind::Attention &&
                  Shape30::LAYER_KIND[33] == LayerKind::Attention &&
                  Shape30::LAYER_KIND[42] == LayerKind::Attention,
              "attention sits on layers 5, 12, 19, 26, 33 and 42");
static_assert(Shape30::ATTENTION_INDEX[5] == 0 && Shape30::ATTENTION_INDEX[12] == 1 &&
                  Shape30::ATTENTION_INDEX[19] == 2 && Shape30::ATTENTION_INDEX[26] == 3 &&
                  Shape30::ATTENTION_INDEX[33] == 4 && Shape30::ATTENTION_INDEX[42] == 5,
              "attention cache slots follow block order");

static_assert(Shape30::HEADS == 32 && KV_HEADS == 2 && HEAD_DIM == 128 && Shape30::Q_DIM == 4096 &&
                  Shape30::KV_DIM == 256,
              "32 query heads, 2 KV heads, head_dim 128");
static_assert(ROTARY_DIM == HEAD_DIM && ROPE_THETA == 1e4f, "full rotary, rope_theta 10000");
static_assert(Shape30::MAMBA_HEADS == 64 && Shape30::MAMBA_HEAD_DIM == 64 && SSM_STATE_SIZE == 128 &&
                  CONV_KERNEL == 4 && MAMBA_EXPAND == 2 && MAMBA_CHUNK_SIZE == 128 && MAMBA_GROUPS == 8,
              "Mamba-2 geometry");
static_assert(Shape30::MAMBA_INNER == 4096 && Shape30::MAMBA_CONV_DIM == 6144 &&
                  Shape30::MAMBA_IN_PROJ_ROWS == 10304 && Shape30::MAMBA_NORM_GROUP == 512,
              "Mamba-2 widths derived from head count, groups and state size");
static_assert(Shape30::EXPERTS == 128 && Shape30::TOP_K == 6 && Shape30::SHARED_EXPERTS == 1 &&
                  Shape30::MOE_INTER == 1856 && Shape30::SHARED_INTER == 3712 &&
                  Shape30::MOE_GROUPS == 1 && Shape30::MOE_TOPK_GROUPS == 1,
              "MoE routing geometry");
static_assert(Shape30::NORM_TOPK_PROB && !Shape30::TIED_EMBEDDINGS);

static_assert(Shape30::MTP_LAYER_BEGIN == Shape30::LAYERS && MTP_LAYERS == 2 && NEXTN_PREDICT_LAYERS == 1,
              "one MTP prediction step with an attention layer and a MoE layer");
static_assert(MTP_BLOCK_TYPES.size() == 2 && MTP_BLOCK_TYPES[0] == LayerKind::Attention &&
                  MTP_BLOCK_TYPES[1] == LayerKind::Moe,
              "mtp_layers_block_type is attention then moe");

static_assert(std::is_same_v<NemotronHShape, Shape30>, "the default shape is Lightning 30B-A3B");

// The bare names stay pinned to Shape30 until the pack has a second shape.
static_assert(HIDDEN == Shape30::HIDDEN && LAYERS == Shape30::LAYERS && HEADS == Shape30::HEADS &&
                  MAMBA_HEADS == Shape30::MAMBA_HEADS && EXPERTS == Shape30::EXPERTS &&
                  MOE_INTER == Shape30::MOE_INTER && MOE_GROUPS == Shape30::MOE_GROUPS &&
                  NUM_ATTENTION_LAYERS == 6,
              "default-shape aliases match Shape30");

}  // namespace
