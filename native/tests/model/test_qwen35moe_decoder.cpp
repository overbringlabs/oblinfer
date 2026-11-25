// End-to-end smoke test of the Qwen3.6-35B-A3B decoder on the real checkpoint: load, run a fixed prompt of
// token ids, generate greedily, and check that two independent decoders agree and every logit is finite.
// Skipped unless QWEN35MOE_SNAPSHOT points at the snapshot. Tokenizer parity is a separate step (the ids are
// printed so a tokenizer on the same machine can decode them).
#include "common/test_util.h"

#include "arch/qwen35moe/model/decoder.h"
#include "arch/qwen35moe/model/uploader.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

bool finite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
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
  std::printf("loaded %llu bytes, split %d/%d\n", static_cast<unsigned long long>(arena.uploaded_bytes),
              arena.experts_on_first, qwen::EXPERTS - arena.experts_on_first);

  const int kMaxPos = 2048;
  std::vector<int> prompt = {9707, 11, 1879, 314, 2419, 13, 9656, 11};
  int kGenerate = 8;
  // Pass prompt ids on the command line (`--ids 760,6511,...`) to test a real tokenized prompt.
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ids" && i + 1 < argc) {
      prompt.clear();
      std::string text = argv[++i];
      std::size_t at = 0;
      while (at < text.size()) {
        const std::size_t comma = text.find(',', at);
        prompt.push_back(std::atoi(text.substr(at, comma == std::string::npos ? std::string::npos : comma - at).c_str()));
        if (comma == std::string::npos) break;
        at = comma + 1;
      }
    } else if (arg == "--generate" && i + 1 < argc) {
      kGenerate = std::atoi(argv[++i]);
    } else if (arg == "--quick") {
      quick = true;
    }
  }

  bool ok = true;
  std::vector<int> run_a, run_b;
  double seconds = 0;
  for (int run = 0; run < (quick ? 1 : 2); ++run) {
    qwen::Decoder decoder;
    if (!decoder.init(arena, kMaxPos, &error)) {
      std::printf("FAIL decoder init: %s\n", error.c_str());
      return 1;
    }
    std::printf("decoder init ok (run %d)\n", run);
    std::fflush(stdout);
    std::vector<int>& out = run == 0 ? run_a : run_b;
    int next = 0;
    const cudaEvent_t start = [] {
      cudaEvent_t e;
      cudaEventCreate(&e);
      cudaEventRecord(e);
      return e;
    }();
    // Prefill: the argmax after the last prompt token is the first generated token.
    for (int i = 0; i < int(prompt.size()); ++i) {
      if (!decoder.step(prompt[std::size_t(i)], i, &next, &error)) {
        std::printf("FAIL prefill %d: %s\n", i, error.c_str());
        return 1;
      }
    }
    for (int j = 0; j < kGenerate; ++j) {
      out.push_back(next);
      if (!finite(decoder.logits())) {
        std::printf("FAIL step %d: non-finite logits\n", j);
        ok = false;
        break;
      }
      const int pos = int(prompt.size()) + j;
      if (!decoder.step(next, pos, &next, &error)) {
        std::printf("FAIL decode %d: %s\n", j, error.c_str());
        return 1;
      }
    }
    std::printf("step 0 ok\n");
    std::fflush(stdout);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    if (run == 0) seconds = ms / 1000.0;
    std::printf("run %d: %zu tokens in %.3f s (%.1f tok/s)\n", run, prompt.size() + kGenerate, ms / 1000.0,
                (ms > 0 ? (prompt.size() + kGenerate) / (ms / 1000.0) : 0.0));
  }

  const bool same = run_a == run_b;
  std::printf("%s determinism: two decoders agree\n", same ? "PASS" : "FAIL");
  ok &= same;

  bool in_range = true;
  for (int t : run_a) {
    if (t < 0 || t >= qwen::VOCAB) in_range = false;
  }
  std::printf("%s generated ids in range\n", in_range ? "PASS" : "FAIL");
  ok &= in_range;

  std::printf("generated ids:");
  for (int t : run_a) std::printf(" %d", t);
  std::printf("\nprompt ids:");
  for (int t : prompt) std::printf(" %d", t);
  std::printf("\nprefill+decode %.1f tok/s\n", (prompt.size() + kGenerate) / seconds);

  qwen::free_text_arena(&arena);
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
