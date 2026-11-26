// Depth benchmark for the Qwen3.6-35B-A3B decoder: prefill `depth` tokens, then decode `gen`, printing
// prefill and decode tok/s. Timing only — correctness is the parity test's job.
// Env: QWEN35MOE_SNAPSHOT (required), QWEN35MOE_DEPTH (default 4096), QWEN35MOE_GEN (default 32).
#include "common/test_util.h"

#include "arch/qwen35moe/model/decoder.h"
#include "arch/qwen35moe/model/uploader.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ingine;
namespace qwen = ingine::arch::qwen35moe;

namespace {

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  return p.major == 12 && p.minor == 0;
}

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : fallback;
}

}  // namespace

int main() {
  const char* snapshot = std::getenv("QWEN35MOE_SNAPSHOT");
  if (!snapshot) {
    std::printf("skip: set QWEN35MOE_SNAPSHOT\n");
    return 0;
  }
  if (!have_matching_gpu()) {
    std::printf("skip: needs sm_120\n");
    return 0;
  }
  const int depth = env_int("QWEN35MOE_DEPTH", 4096);
  const int gen = env_int("QWEN35MOE_GEN", 32);
  // QWEN35MOE_MAXPOS allocates a full-context cache while only a few tokens run (allocation proof).
  const char* maxpos_env = std::getenv("QWEN35MOE_MAXPOS");
  const char* fake_max = std::getenv("QWEN35MOE_FAKE_DEPTH");
  const int max_pos = fake_max ? std::atoi(fake_max) + gen + 8
                               : (maxpos_env ? std::atoi(maxpos_env) : depth + gen + 8);

  // Expert split: -1 keeps the uploader's free-memory choice; set QWEN35MOE_SPLIT to A/B a bandwidth split.
  const char* split_env = std::getenv("QWEN35MOE_SPLIT");
  const int split = split_env ? std::atoi(split_env) : -1;
  qwen::TextArena arena;
  std::string error;
  if (!qwen::upload_text_model(snapshot, true, split, &arena, &error)) {
    std::printf("FAIL upload: %s\n", error.c_str());
    return 1;
  }

  std::printf("expert split %d/%d (bandwidth split %d, fit range %d..%d); spans:", arena.experts_on_first,
              qwen::EXPERTS - arena.experts_on_first, arena.split_policy.ideal, arena.split_policy.fit_low,
              arena.split_policy.fit_high);
  for (const auto& span : arena.spans) std::printf(" dev%d=%.2f GiB", span.cuda_device, span.bytes / 1073741824.0);
  std::printf("\n");
  qwen::Decoder decoder;
  if (!decoder.init(arena, max_pos, &error)) {
    std::printf("FAIL decoder init: %s\n", error.c_str());
    return 1;
  }
  // Headroom left on each card once the weights, the cache and the workspace are in: what a wider split would have
  // had to fit into.
  for (const auto& span : arena.spans) {
    std::size_t free_b = 0, total_b = 0;
    cudaSetDevice(span.cuda_device);
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      std::printf("  dev%d after init: %.0f MiB free of %.0f\n", span.cuda_device, free_b / 1048576.0,
                  total_b / 1048576.0);
    }
  }
  cudaSetDevice(arena.spans[0].cuda_device);

  // Prompt ids: a fixed rotation of ordinary text tokens, so the timing is representative.
  const std::vector<int> base = {760, 6511, 314, 9338, 369, 11, 264, 3177, 34756, 364, 1141, 8807,
                                 3712, 11, 7431, 11, 321, 25438, 57902, 13};
  int next = 0;
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  // QWEN35MOE_FAKE_DEPTH=D: skip prefill and decode at position D against a zeroed cache. Decode cost depends
  // on how many KV rows are read, not on their values, so this measures depth without an hour of prefill.
  const char* fake_env = std::getenv("QWEN35MOE_FAKE_DEPTH");
  if (fake_env) {
    const int start = std::atoi(fake_env);
    cudaEventRecord(t0);
    for (int j = 0; j < gen; ++j) {
      if (!decoder.step(base[std::size_t(j) % base.size()], start + j, &next, &error)) {
        std::printf("FAIL decode: %s\n", error.c_str());
        return 1;
      }
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    std::printf("depth %d (synthetic) gen %d: decode %.2f tok/s (%.2f ms/token)\n", start, gen,
                gen / (ms / 1000.0), ms / gen);
    qwen::free_text_arena(&arena);
    return 0;
  }

  cudaEventRecord(t0);
  for (int i = 0; i < depth; ++i) {
    if (!decoder.step(base[std::size_t(i) % base.size()], i, &next, &error)) {
      std::printf("FAIL prefill %d: %s\n", i, error.c_str());
      return 1;
    }
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float prefill_ms = 0;
  cudaEventElapsedTime(&prefill_ms, t0, t1);

  cudaEventRecord(t0);
  for (int j = 0; j < gen; ++j) {
    if (!decoder.step(next, depth + j, &next, &error)) {
      std::printf("FAIL decode %d: %s\n", j, error.c_str());
      return 1;
    }
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float decode_ms = 0;
  cudaEventElapsedTime(&decode_ms, t0, t1);

  std::printf("depth %d gen %d: prefill %.1f tok/s (%.1f ms), decode %.2f tok/s (%.2f ms/token)\n", depth,
              gen, depth / (prefill_ms / 1000.0), prefill_ms, gen / (decode_ms / 1000.0), decode_ms / gen);
  qwen::free_text_arena(&arena);
  return 0;
}
