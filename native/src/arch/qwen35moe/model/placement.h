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
  std::uint64_t stride = 0;  // bytes of one expert
  std::uint64_t bytes = 0;   // EXPERTS * stride
  std::vector<TensorSlot> slots;
};

struct LayerPlan {
  int layer = 0;  // LAYERS is the MTP block
  std::vector<TensorSlot> dense;
  std::uint64_t dense_bytes = 0;
  bool has_experts = false;
  ExpertBlockPlan experts;
};

struct PlacementPlan {
  std::vector<LayerPlan> layers;  // LAYERS + MTP_LAYERS entries
  std::vector<TensorSlot> model;  // embeddings, head, MTP glue, vision
  std::uint64_t total_bytes = 0;
  Dialect dialect = Dialect::Unknown;
};

// Nullopt on a malformed checkpoint; `error` then names the offending tensor.
// Checks: every name is known, no duplicates, every routed-expert layer carries the same slot set with
// identical sizes, and every dense layer present in the checkpoint is planned.
std::optional<PlacementPlan> plan_placement(std::span<const TensorSize> tensors, std::string* error);

}  // namespace ingine::arch::qwen35moe
