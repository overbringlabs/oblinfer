#include "arch/qwen35moe/model/uploader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "arch/qwen35moe/kernels/attn/kv_bands.h"
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

// A packed expert block holds every expert in one tensor, so it goes to span 0 whole and no split can shrink it.
bool packed_block(const ExpertBlockPlan& block) {
  return block.slots.empty() || !is_per_expert(block.slots.front().source);
}

// What each span must still hold when the upload ends: the decoder's workspace, plus its share of the cache. A
// split chosen only against free memory fills a span to the brim, and the decoder's first cache allocation then
// fails long after the upload looked healthy.
//
// The workspace is the decoder's scratch, state, tables and logits: 85 MiB beyond the arena on span 0 and under
// 10 MiB on span 1 when measured after init on supra, so 512 MiB covers both with room for the graph path.
constexpr std::uint64_t kWorkspaceBytes = 512ull << 20;
// The cache is the serving profile's: K and V of KV_HEADS x HEAD_DIM for each of the ten full-attention layers at
// MAX_POS. q8_0 stores one byte per value plus a half-precision scale per 32 values, so 17/16 of the value
// count; it measures 1,786 MiB on span 0 and 894 MiB on span 1 at 258048 positions.
constexpr std::uint64_t kKvValues = std::uint64_t(NUM_FULL_LAYERS) * KV_HEADS * HEAD_DIM * 2 * MAX_POS;
constexpr std::uint64_t kKvBytesAtMaxPos = kKvValues * 17 / 16;
// Band ratio 2 puts two thirds of the cache on span 0. Below the banding break-even the whole cache stays on
// span 0, but 1.0 GB at 98304 positions is less than that share, so the worst case is the banded one.
constexpr std::uint64_t kKvBytesOnFirst = static_cast<std::uint64_t>(
    static_cast<double>(kKvBytesAtMaxPos) * KV_BAND_RATIO / (1.0 + KV_BAND_RATIO));

// Theoretical read bandwidth of `device` in bytes/s: the driver's peak memory clock (kHz) times the bus width,
// two transfers per clock. Zero when the driver reports neither, which sends the split back to free memory.
std::uint64_t device_bandwidth(int cuda_device) {
  int clock_khz = 0, bus_bits = 0;
  if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, cuda_device) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, cuda_device) != cudaSuccess) return 0;
  if (clock_khz <= 0 || bus_bits <= 0) return 0;
  return std::uint64_t(clock_khz) * 1000ull * 2ull * std::uint64_t(bus_bits) / 8ull;
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
  std::uint64_t expert_stride_sum = 0;  // bytes one expert adds to a span, every per-expert layer summed
  std::uint64_t packed_expert_bytes = 0;
  std::uint64_t dense_bytes = 0;
  for (const LayerPlan& lp : plan.layers) {
    dense_bytes += lp.dense_bytes;
    if (!lp.has_experts) continue;
    if (packed_block(lp.experts)) {
      packed_expert_bytes += lp.experts.bytes;
    } else {
      expert_stride_sum += lp.experts.stride;
    }
  }

  int split = experts_on_first;
  ExpertSplit decision;
  if (split < 0) {
    // The step waits for the slower card, so the split that finishes both together is proportional to their
    // read bandwidth; the free memory decides only how far the fit allows it to go.
    ExpertFit fit;
    fit.per_expert_bytes = expert_stride_sum;
    fit.packed_bytes = packed_expert_bytes;
    for (int d = 0; d < kSpans; ++d) {
      std::size_t free_b = 0, total_b = 0;
      if (cudaSetDevice(d) != cudaSuccess || cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return fail("cannot query the free memory of device " + std::to_string(d));
      }
      const std::uint64_t reserve =
          kWorkspaceBytes + (d == 0 ? kKvBytesOnFirst : kKvBytesAtMaxPos - kKvBytesOnFirst);
      const std::uint64_t needed = (d == 0 ? model_bytes + dense_bytes : 0) + reserve;
      fit.room[d] = std::uint64_t(free_b) > needed ? std::uint64_t(free_b) - needed : 0;
      fit.bandwidth[d] = device_bandwidth(d);
    }
    decision = plan_expert_split(fit);
    if (!decision.fits) {
      return fail("the routed experts do not fit both cards: " + std::to_string(fit.room[0] >> 20) +
                  " MiB free for experts on device 0 and " + std::to_string(fit.room[1] >> 20) +
                  " MiB on device 1, each expert " + std::to_string(fit.per_expert_bytes >> 20) + " MiB");
    }
    split = decision.experts_on_first;
    std::fprintf(stderr,
                 "[qwen35moe] expert split %d/%d: bandwidth %.0f/%.0f GB/s, %.0f/%.0f MiB free for experts, "
                 "bandwidth split %d, fit range %d..%d%s\n",
                 split, EXPERTS - split, double(fit.bandwidth[0]) / 1e9, double(fit.bandwidth[1]) / 1e9,
                 double(fit.room[0]) / (1u << 20), double(fit.room[1]) / (1u << 20), decision.ideal,
                 decision.fit_low, decision.fit_high,
                 decision.bandwidth_balanced ? "" : " (the fit holds it below the bandwidth split)");
    std::fflush(stderr);
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
    if (packed_block(lp.experts)) {
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
  arena.split_policy = decision;
  arena.split_policy.experts_on_first = split;
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
    if (packed_block(lp.experts)) {
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
