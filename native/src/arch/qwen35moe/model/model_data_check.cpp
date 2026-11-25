// The pack is header-only at this stage, so this translation unit exists to make the default build compile
// model/model_data.h: a broken constant or layer table fails `ingine-serve`'s build without linking anything from
// the pack into it.
#include "model/model_data.h"

namespace {

using namespace ingine::arch::qwen35moe;

static_assert(LAYER_KIND.size() == LAYERS && FULL_INDEX.size() == LAYERS && LINEAR_INDEX.size() == LAYERS,
              "layer tables cover every decoder layer");
static_assert(NUM_FULL_LAYERS + NUM_LINEAR_LAYERS == LAYERS);
static_assert(MTP_LAYER_BEGIN == LAYERS && MTP_LAYERS == 1);

}  // namespace
