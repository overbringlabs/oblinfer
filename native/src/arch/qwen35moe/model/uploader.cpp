#include "arch/qwen35moe/model/uploader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "core/safetensors.h"

namespace ingine::arch::qwen35moe {
namespace {

// End of the numeric expert id in ".experts.<id>.", or npos for a packed name.
std::size_t expert_digits_end(const std::string& name, std::size_t* digits_begin) {
  const std::size_t at = name.find(".experts.");
  if (at == std::string::npos) return std::string::npos;
  const std::size_t begin = at + std::strlen(".experts.");
  std::size_t end = begin;
  while (end < name.size() && name[end] >= '0' && name[end] <= '9') ++end;
  if (end == begin || end >= name.size() || name[end] != '.') return std::string::npos;
  *digits_begin = begin;
  return end;
}

bool is_per_expert(const std::string& name) {
  std::size_t begin = 0;
  return expert_digits_end(name, &begin) != std::string::npos;
}

}  // namespace

std::string with_expert(const std::string& name, int expert) {
  std::size_t begin = 0;
  const std::size_t end = expert_digits_end(name, &begin);
  if (end == std::string::npos) return name;
  return name.substr(0, begin) + std::to_string(expert) + name.substr(end);
}

bool upload_text_model(const std::filesystem::path& snapshot_dir, bool include_mtp, int experts_on_first,
                       TextArena* out, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };

  std::unique_ptr<SafetensorsModel> model;
  try {
    model = std::make_unique<SafetensorsModel>(snapshot_dir);
  } catch (const std::exception& e) {
    return fail(std::string("open checkpoint: ") + e.what());
  }

  std::vector<TensorSize> sizes;
  sizes.reserve(model->size());
  for (const std::string& name : model->names()) {
    SafetensorsRef ref = model->find(name);
    if (!ref) continue;
    sizes.push_back({name, ref->bytes});
  }

  std::string plan_error;
  const auto planned = plan_placement(sizes, &plan_error);
  if (!planned) return fail("placement: " + plan_error);
  const PlacementPlan& plan = *planned;

  // Model area size: the planner lays the model slots out from 0, so the area ends at the last slot's end.
  std::uint64_t model_bytes = 0;
  for (const TensorSlot& slot : plan.model) {
    const SafetensorsRef ref = model->find(slot.source);
    if (!ref) return fail("missing tensor: " + slot.source);
    model_bytes = std::max(model_bytes, slot.offset + ref->bytes);
  }

  constexpr int kSpans = 2;
  int split = experts_on_first;
  if (split < 0) {
    // Reserve 1 GiB of slack on each card, put the model and dense areas on device 0, and divide the expert
    // bytes by the remaining free memory.
    std::uint64_t free_bytes[kSpans] = {0, 0};
    std::uint64_t expert_bytes = 0;
    std::uint64_t dense_bytes = 0;
    for (const LayerPlan& lp : plan.layers) {
      dense_bytes += lp.dense_bytes;
      if (lp.has_experts) expert_bytes += lp.experts.bytes;
    }
    for (int d = 0; d < kSpans; ++d) {
      std::size_t free_b = 0, total_b = 0;
      if (cudaSetDevice(d) != cudaSuccess || cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return fail("cannot query the free memory of device " + std::to_string(d));
      }
      free_bytes[d] = free_b > (1ull << 30) ? free_b - (1ull << 30) : 0;
    }
    const std::uint64_t needed0 = model_bytes + dense_bytes;
    const std::uint64_t room0 = free_bytes[0] > needed0 ? free_bytes[0] - needed0 : 0;
    const std::uint64_t room1 = free_bytes[1];
    const std::uint64_t total_room = room0 + room1;
    split = total_room == 0 ? 0 : int(std::min<std::uint64_t>(
        EXPERTS, (std::uint64_t(EXPERTS) * room0 + total_room - 1) / std::max<std::uint64_t>(1, total_room)));
    if (expert_bytes > total_room) return fail("the expert blocks do not fit the free memory of both cards");
  }
  split = std::clamp(split, 0, EXPERTS);

  // Layout per span. Span 0: model area, then every layer's dense slots, then experts [0, split).
  // Span 1: every layer's experts [split, EXPERTS).
  struct LayerSpan {
    std::uint64_t dense = 0;    // span 0 only
    std::uint64_t experts[2] = {0, 0};  // base of the layer's block on each span
    std::uint64_t experts_per_span[2] = {0, 0};
  };
  std::vector<LayerSpan> layout(plan.layers.size());
  std::uint64_t cursor[kSpans] = {model_bytes, 0};
  for (std::size_t i = 0; i < plan.layers.size(); ++i) {
    const LayerPlan& lp = plan.layers[i];
    layout[i].dense = cursor[0];
    cursor[0] += lp.dense_bytes;
    if (!lp.has_experts) continue;
    const bool packed = lp.experts.slots.empty() ? true : !is_per_expert(lp.experts.slots.front().source);
    if (packed) {
      layout[i].experts[0] = cursor[0];
      layout[i].experts_per_span[0] = 1;
      cursor[0] += lp.experts.bytes;
      continue;
    }
    layout[i].experts[0] = cursor[0];
    layout[i].experts_per_span[0] = std::uint64_t(split);
    cursor[0] += std::uint64_t(split) * lp.experts.stride;
    layout[i].experts[1] = cursor[1];
    layout[i].experts_per_span[1] = std::uint64_t(EXPERTS - split);
    cursor[1] += std::uint64_t(EXPERTS - split) * lp.experts.stride;
  }

  TextArena arena;
  arena.plan = plan;
  arena.experts_on_first = split;
  for (int d = 0; d < kSpans; ++d) {
    DeviceSpan span;
    span.cuda_device = d;
    span.bytes = std::max<std::uint64_t>(cursor[d], 1);  // both spans exist so span indices stay stable
    if (cudaSetDevice(d) != cudaSuccess) return fail("cudaSetDevice failed");
    if (cudaMalloc(&span.device, span.bytes) != cudaSuccess) {
      return fail("cudaMalloc of " + std::to_string(span.bytes >> 20) + " MiB on device " + std::to_string(d) +
                  " failed");
    }
    arena.spans.push_back(span);
  }

  // Requests per span, then one uploader per device.
  std::vector<UploadRequest> requests[kSpans];
  auto base = [&](int d) { return static_cast<std::byte*>(arena.spans[std::size_t(d)].device); };

  auto enqueue = [&](int d, const std::string& name, void* dst) -> bool {
    const SafetensorsRef ref = model->find(name);
    if (!ref) return false;
    requests[d].push_back(upload_request(ref, dst));
    arena.placed.push_back({name, d, std::uint64_t(static_cast<std::byte*>(dst) - base(d)), ref->bytes});
    return true;
  };

  for (const TensorSlot& slot : plan.model) {
    if (!enqueue(0, slot.source, base(0) + slot.offset)) return fail("missing tensor: " + slot.source);
  }
  for (std::size_t i = 0; i < plan.layers.size(); ++i) {
    const LayerPlan& lp = plan.layers[i];
    if (lp.layer == LAYERS && !include_mtp) continue;
    for (const TensorSlot& slot : lp.dense) {
      if (!enqueue(0, slot.source, base(0) + layout[i].dense + slot.offset)) {
        return fail("missing tensor: " + slot.source);
      }
    }
    if (!lp.has_experts) continue;
    const bool packed = !is_per_expert(lp.experts.slots.front().source);
    if (packed) {
      for (const TensorSlot& slot : lp.experts.slots) {
        if (!enqueue(0, slot.source, base(0) + layout[i].experts[0] + slot.offset)) {
          return fail("missing tensor: " + slot.source);
        }
      }
      continue;
    }
    for (const TensorSlot& slot : lp.experts.slots) {
      for (int e = 0; e < EXPERTS; ++e) {
        const int d = e < split ? 0 : 1;
        const int local = e - (d == 0 ? 0 : split);
        const std::string name = with_expert(slot.source, e);
        void* dst = base(d) + layout[i].experts[d] + std::uint64_t(local) * lp.experts.stride + slot.offset;
        if (!enqueue(d, name, dst)) return fail("missing tensor: " + name);
      }
    }
  }

  double seconds = 0;
  for (int d = 0; d < kSpans; ++d) {
    if (requests[d].empty()) continue;
    if (cudaSetDevice(d) != cudaSuccess) return fail("cudaSetDevice failed");
    DeviceUploader uploader({}, d);
    const UploadStats stats = uploader.upload(requests[d], nullptr);
    arena.tensors += int(requests[d].size());
    arena.uploaded_bytes += stats.bytes;
    seconds = std::max(seconds, stats.seconds);
  }
  arena.seconds = seconds;
  *out = std::move(arena);
  return true;
}

void free_text_arena(TextArena* arena) {
  if (!arena) return;
  for (DeviceSpan& span : arena->spans) {
    if (!span.device) continue;
    cudaSetDevice(span.cuda_device);
    cudaFree(span.device);
    span.device = nullptr;
  }
  arena->spans.clear();
}

}  // namespace ingine::arch::qwen35moe
