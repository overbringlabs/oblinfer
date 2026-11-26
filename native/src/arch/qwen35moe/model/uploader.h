// Loads the text half of a Qwen3.6-35B-A3B checkpoint into one device arena laid out by
// plan_placement: the model area (embeddings, head, MTP glue), then each layer's dense slots, then its
// packed expert block. The streaming, chunking and pinned staging come from core/safetensors.
//
// Vision tensors are not uploaded: the text path never reads them, and they cost 0.9 GB.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "arch/qwen35moe/model/placement.h"

namespace ingine::arch::qwen35moe {

// One device's slice of the model: span 0 holds the model area (embeddings, head, MTP glue), every layer's
// dense slots and the first `experts_on_first` experts of each layer; span 1 holds the remaining experts,
// layer by layer. `cuda_device` is the device the span was allocated on.
struct DeviceSpan {
  int cuda_device = 0;
  void* device = nullptr;
  std::uint64_t bytes = 0;
};

// Where one source tensor landed: `span` indexes TextArena::spans.
struct PlacedTensor {
  std::string name;
  int span = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct TextArena {
  std::vector<DeviceSpan> spans;
  std::vector<PlacedTensor> placed;  // one entry per source tensor, unmatched names excluded
  PlacementPlan plan;
  int experts_on_first = 0;
  ExpertSplit split_policy;  // the numbers behind the split; `ideal` is 0 for an explicit override
  int tensors = 0;          // source tensors enqueued
  std::uint64_t uploaded_bytes = 0;
  double seconds = 0;
};

// `experts_on_first`: routed experts of each layer packed onto span 0. Negative picks the split from the read
// bandwidth of the two devices, clamped to what their free memory holds after the dense weights, the cache and
// the workspace; the free-memory proportion stands in when the devices read alike. An explicit value is used
// as given, so a split that does not fit fails at the allocation. Returns false and fills `error` when the
// directory is not a safetensors checkpoint, a tensor name is unknown, the placement is malformed, or an
// allocation fails.
bool upload_text_model(const std::filesystem::path& snapshot_dir, bool include_mtp, int experts_on_first,
                       TextArena* out, std::string* error);

void free_text_arena(TextArena* arena);

// Source tensor name of expert `expert` for a per-expert slot whose stored name is expert 0's.
std::string with_expert(const std::string& name, int expert);

}  // namespace ingine::arch::qwen35moe
