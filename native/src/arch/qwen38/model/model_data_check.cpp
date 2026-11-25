// The pack is header-only at this stage, so this translation unit exists to make the default build compile
// model/model_data.h: a broken constant or layer table fails `ingine-serve`'s build without linking anything from
// the pack into it. It instantiates all three shapes, not just the selected Qwen38Shape.
#include "model/model_data.h"

#include <type_traits>

namespace {

using namespace ingine::arch::qwen38;

// Walks a shape's three tables against is_full_layer and checks that the slot counters are dense and exclusive.
template <class S>
constexpr bool layer_maps_cover_every_layer() {
  int full = 0;
  int linear = 0;
  for (int l = 0; l < S::LAYERS; ++l) {
    if (is_full_layer(l)) {
      if (S::LAYER_KIND[l] != LayerKind::Full || S::FULL_INDEX[l] != full || S::LINEAR_INDEX[l] != -1) {
        return false;
      }
      ++full;
    } else {
      if (S::LAYER_KIND[l] != LayerKind::Linear || S::LINEAR_INDEX[l] != linear || S::FULL_INDEX[l] != -1) {
        return false;
      }
      ++linear;
    }
  }
  return full == S::NUM_FULL_LAYERS && linear == S::NUM_LINEAR_LAYERS && full + linear == S::LAYERS;
}

static_assert(layer_maps_cover_every_layer<Shape27>(), "Shape27 layer tables cover every decoder layer");
static_assert(layer_maps_cover_every_layer<Shape9>(), "Shape9 layer tables cover every decoder layer");
static_assert(layer_maps_cover_every_layer<Shape4>(), "Shape4 layer tables cover every decoder layer");

static_assert(Shape27::MTP_LAYER_BEGIN == Shape27::LAYERS && Shape9::MTP_LAYER_BEGIN == Shape9::LAYERS &&
                  Shape4::MTP_LAYER_BEGIN == Shape4::LAYERS && MTP_LAYERS == 1,
              "one MTP layer after the decoder in every shape");

// 9B and 4B share layer counts and GDN head counts; only hidden size and MLP width differ.
static_assert(Shape9::LAYERS == Shape4::LAYERS && Shape9::HEADS == Shape4::HEADS &&
                  Shape9::KV_HEADS == Shape4::KV_HEADS &&
                  Shape9::LIN_VALUE_HEADS == Shape4::LIN_VALUE_HEADS);
static_assert(Shape9::HIDDEN != Shape4::HIDDEN && Shape9::INTERMEDIATE != Shape4::INTERMEDIATE);
static_assert(!Shape27::TIED_EMBEDDINGS && !Shape9::TIED_EMBEDDINGS && Shape4::TIED_EMBEDDINGS,
              "tie_word_embeddings is true on the 4B only");

// The bare constants stay pinned to Shape27 until the pack's own sources move to Qwen38Shape.
static_assert(HIDDEN == Shape27::HIDDEN && LAYERS == Shape27::LAYERS && INTERMEDIATE == Shape27::INTERMEDIATE &&
                  HEADS == Shape27::HEADS && KV_HEADS == Shape27::KV_HEADS &&
                  LIN_VALUE_HEADS == Shape27::LIN_VALUE_HEADS && LIN_CONV_DIM == Shape27::LIN_CONV_DIM,
              "default-shape aliases match Shape27");

#if INGINE_QWEN38_SHAPE == 27
static_assert(std::is_same_v<Qwen38Shape, Shape27>, "INGINE_QWEN38_SHAPE=27 selects Shape27");
#elif INGINE_QWEN38_SHAPE == 9
static_assert(std::is_same_v<Qwen38Shape, Shape9>, "INGINE_QWEN38_SHAPE=9 selects Shape9");
#elif INGINE_QWEN38_SHAPE == 4
static_assert(std::is_same_v<Qwen38Shape, Shape4>, "INGINE_QWEN38_SHAPE=4 selects Shape4");
#else
#error "INGINE_QWEN38_SHAPE must be 27, 9, or 4"
#endif

}  // namespace
