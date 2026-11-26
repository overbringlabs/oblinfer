// Placement of a resolved checkpoint's tensors: dense slots in the order they are loaded, and for each MoE
// layer one contiguous expert block whose stride covers every artifact of one expert. The loader streams
// source tensors into these offsets; the MoE kernel then sees one base pointer per packed block.
//
// The planner takes the actual byte sizes of the checkpoint's tensors (read from the safetensors headers) and
// never assumes a scale dtype, so the same code serves the NVIDIA and Ornith exports and any future one.
// Host-only, no CUDA.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "arch/qwen35moe/model/tensor_names.h"

namespace ingine::arch::qwen35moe {

struct TensorSize {
  std::string name;
  std::uint64_t bytes = 0;
};

struct TensorSlot {
  std::string source;
  Projection projection = Projection::None;
  Artifact artifact = Artifact::Weight;
  std::uint64_t offset = 0;
};

// One routed expert's artifacts, laid out gate, up, down; inside a projection in artifact order
// weight, weight_scale, weight_scale_2, input_scale. Only the artifacts the export ships are included.
struct ExpertBlockPlan {
  std::uint64_t stride = 0;        // bytes of one expert
  std::uint64_t bytes = 0;         // EXPERTS * stride, or the packed total
  std::uint64_t arena_offset = 0;  // where this block starts in the model arena
  std::vector<TensorSlot> slots;
};

struct LayerPlan {
  int layer = 0;  // LAYERS is the MTP block
  std::vector<TensorSlot> dense;
  std::uint64_t dense_bytes = 0;
  std::uint64_t dense_arena_offset = 0;  // where this layer's dense slots start
  bool has_experts = false;
  ExpertBlockPlan experts;
};

struct PlacementPlan {
  std::vector<LayerPlan> layers;  // LAYERS + MTP_LAYERS entries
  std::vector<TensorSlot> model;  // embeddings, head, MTP glue, vision; offsets from 0
  std::uint64_t total_bytes = 0;  // the full arena
  Dialect dialect = Dialect::Unknown;
};

// Nullopt on a malformed checkpoint; `error` then names the offending tensor.
// Checks: every name is known, no duplicates, every routed-expert layer carries the same slot set with
// identical sizes, and every dense layer present in the checkpoint is planned.
std::optional<PlacementPlan> plan_placement(std::span<const TensorSize> tensors, std::string* error);

// --------------------------------------------------------------- routed-expert split between the two spans

// What the split decision needs to know about the machine. `bandwidth` is bytes/s as the driver reports it
// (memory clock times bus width, doubled for the DDR transfer); `room` is the bytes each span can give the
// expert blocks once the weights it must hold, its share of the cache and its workspace are subtracted.
struct ExpertFit {
  std::uint64_t per_expert_bytes = 0;  // bytes one expert adds to a span, summed over the per-expert layers
  std::uint64_t packed_bytes = 0;      // expert bytes no split can move: a packed block stays on span 0
  std::uint64_t room[2] = {0, 0};
  std::uint64_t bandwidth[2] = {0, 0};
};

struct ExpertSplit {
  int experts_on_first = 0;
  int ideal = 0;                     // the bandwidth (or, on a tie, room) share, before the clamp
  bool fits = true;                  // false: no split of the experts fits both spans at once
  bool bandwidth_balanced = false;   // the fit allowed `ideal`, so both cards finish together
  bool bandwidth_symmetric = false;  // the devices read alike; `ideal` is then proportional to room
  int fit_low = 0, fit_high = 0;     // the range the room allows
};

// The two cards read their expert blocks in parallel and the step waits for the slower one, so the split that
// finishes both together gives each span the share of the experts its bandwidth can read in the same time. When
// a span's room cannot hold that share the clamp wins, and the caller is told which end bound it.
ExpertSplit plan_expert_split(const ExpertFit& fit);

}  // namespace ingine::arch::qwen35moe
