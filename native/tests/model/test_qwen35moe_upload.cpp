// End-to-end load of the real checkpoint through plan_placement: one arena, every text tensor streamed in.
// Runs only when QWEN35MOE_SNAPSHOT points at the snapshot directory (the checkpoint is 23 GB, so it is not
// part of a default test run). Verifies the uploaded bytes, then byte-compares a dense tensor and one
// expert's gate weights against the files.
#include "common/test_util.h"

#include "arch/qwen35moe/model/placement.h"
#include "arch/qwen35moe/model/uploader.h"
#include "core/safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::test;

namespace qwen = ingine::arch::qwen35moe;

namespace {

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  return p.major == 12 && p.minor == 0;
}

const qwen::TensorSlot* find_slot(const qwen::LayerPlan& layer, qwen::Projection projection, qwen::Artifact artifact) {
  for (const auto& slot : layer.dense) {
    if (slot.projection == projection && slot.artifact == artifact) return &slot;
  }
  for (const auto& slot : layer.experts.slots) {
    if (slot.projection == projection && slot.artifact == artifact) return &slot;
  }
  return nullptr;
}

bool compare_prefix(const std::string& tag, void* device_ptr, std::uint64_t bytes, const std::vector<std::byte>& file) {
  const std::size_t n = std::size_t(std::min<std::uint64_t>(bytes, 1 << 16));
  std::vector<std::byte> got(n);
  if (cudaMemcpy(got.data(), device_ptr, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL %s: cudaMemcpy\n", tag.c_str());
    return false;
  }
  const bool ok = n <= file.size() && std::memcmp(got.data(), file.data(), n) == 0;
  std::printf("%s %s (%zu bytes)\n", ok ? "PASS" : "FAIL", tag.c_str(), n);
  return ok;
}

}  // namespace

int main() {
  const char* snapshot = std::getenv("QWEN35MOE_SNAPSHOT");
  if (!snapshot) {
    std::printf("skip: set QWEN35MOE_SNAPSHOT to the snapshot directory\n");
    return 0;
  }
  if (!have_matching_gpu()) {
    std::printf("skip: needs sm_120\n");
    return 0;
  }

  qwen::TextArena arena;
  std::string error;
  if (!qwen::upload_text_model(snapshot, true, -1, &arena, &error)) {
    std::printf("FAIL upload: %s\n", error.c_str());
    return 1;
  }
  std::printf("loaded %llu bytes in %.2f s (%d tensors)\n", static_cast<unsigned long long>(arena.uploaded_bytes),
              arena.seconds, arena.tensors);

  bool ok = true;

  // Every text tensor accounted for, and the arena is the plan's size.
  SafetensorsModel model{std::filesystem::path(snapshot)};
  std::uint64_t expected = 0;
  for (const std::string& name : model.names()) {
    const auto parsed = qwen::parse_tensor_name(name);
    if (parsed && parsed->role == qwen::Role::Vision) continue;
    const auto ref = model.find(name);
    expected += ref->bytes;
  }
  const bool bytes_ok = arena.uploaded_bytes == expected;
  std::printf("%s uploaded bytes: got %llu want %llu\n", bytes_ok ? "PASS" : "FAIL",
              static_cast<unsigned long long>(arena.uploaded_bytes), static_cast<unsigned long long>(expected));
  ok &= bytes_ok;

  std::printf("split: %d experts on span 0, %d on span 1; spans:", arena.experts_on_first,
              qwen::EXPERTS - arena.experts_on_first);
  for (const auto& span : arena.spans) {
    std::printf(" dev%d=%llu MiB", span.cuda_device, static_cast<unsigned long long>(span.bytes >> 20));
  }
  std::printf("\n");

  // Layer 0 is a linear-attention layer; a dense tensor must match the file byte for byte.
  const auto* dense = find_slot(arena.plan.layers[0], qwen::Projection::InQkv, qwen::Artifact::Weight);
  if (!dense) {
    std::printf("FAIL: layer 0 has no in_proj_qkv weight\n");
    return 1;
  }
  auto place = [&](const std::string& name) -> const qwen::PlacedTensor* {
    for (const auto& p : arena.placed) {
      if (p.name == name) return &p;
    }
    return nullptr;
  };
  const qwen::PlacedTensor* dp = place(dense->source);
  if (!dp) {
    std::printf("FAIL: dense tensor not placed\n");
    return 1;
  }
  void* dense_ptr = static_cast<std::byte*>(arena.spans[std::size_t(dp->span)].device) + dp->offset;
  ok &= compare_prefix("dense in_proj_qkv weight", dense_ptr, dp->bytes, model.read(dense->source));

  // Expert 5's gate weight inside the packed block.
  const auto& experts = arena.plan.layers[0].experts;
  const qwen::TensorSlot* gate = nullptr;
  for (const auto& slot : experts.slots) {
    if (slot.projection == qwen::Projection::GateProj && slot.artifact == qwen::Artifact::Weight) gate = &slot;
  }
  if (!gate) {
    std::printf("FAIL: layer 0 has no expert gate weight\n");
    return 1;
  }
  for (int expert : {0, arena.experts_on_first, qwen::EXPERTS - 1}) {
    if (expert < 0 || expert >= qwen::EXPERTS) continue;
    const std::string name = qwen::with_expert(gate->source, expert);
    const qwen::PlacedTensor* e = place(name);
    if (!e) {
      std::printf("FAIL: expert %d gate not placed\n", expert);
      ok = false;
      continue;
    }
    void* dst = static_cast<std::byte*>(arena.spans[std::size_t(e->span)].device) + e->offset;
    ok &= compare_prefix("expert " + std::to_string(expert) + " gate weight on dev" + std::to_string(e->span), dst,
                         e->bytes, model.read(name));
  }

  qwen::free_text_arena(&arena);
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
