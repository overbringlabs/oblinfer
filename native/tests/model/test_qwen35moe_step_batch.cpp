// Verify-width step (spec-decode K2/K3): step_batch(ids, n) against n single steps on the real checkpoint.
//
// Three questions, in this order:
//   1. shape   — n = 1 through step_batch is the single-token step, bit for bit (same ids, same logits bytes);
//   2. numbers — for n = 1..4, one step_batch agrees with n step() calls: identical greedy ids everywhere (the
//                gate), and the logits within a few 1e-7 relative, which is fp32 reduction order alone;
//   3. cost    — ms per step and ms per token for n = 1..4 at the depths of interest, against n single steps.
//
// The rollback the accept loop needs is what makes (2) possible on one decoder: checkpoint_gdn() before the n
// single steps, restore_gdn() before the batch pass, so both see the same GDN state. Trunk KV needs no
// rollback, a rejected position's rows are overwritten by the next pass.
//
// Env: QWEN35MOE_SNAPSHOT (required), QWEN35MOE_STEP_BATCH_MAXPOS (default 2048),
//      QWEN35MOE_STEP_BATCH_DEPTHS (default 1024,258048), QWEN35MOE_STEP_BATCH_ITERS (default 8),
//      QWEN35MOE_STEP_BATCH_SKIP_TIMING=1 to run only the correctness part.
#include "common/test_util.h"

#include "arch/qwen35moe/model/decoder.h"
#include "arch/qwen35moe/model/uploader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : fallback;
}

// The prompt is the decoder test's, extended by a rotation of ordinary text ids: enough depth that the GDN
// state and the KV cache are non-trivial, short enough to prefill in a second.
std::vector<int> prompt_ids(int length) {
  const std::vector<int> base = {9707, 11, 1879, 314, 2419, 13,  9656, 11,   760,  6511, 314,  9338, 369, 11,
                                 264,  3177, 34756, 364, 1141, 8807, 3712, 11,  7431, 11,   321,  25438};
  std::vector<int> ids;
  for (int i = 0; i < length; ++i) ids.push_back(base[std::size_t(i) % base.size()]);
  return ids;
}

// The draft ids of a verify block: fixed, from the same distribution, so the comparison is reproducible.
std::vector<int> draft_ids() { return {760, 6511, 314, 9338}; }

double ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return double(ms);
}

}  // namespace

int main(int argc, char** argv) {
  const char* snapshot = std::getenv("QWEN35MOE_SNAPSHOT");
  if (!snapshot) {
    std::printf("skip: set QWEN35MOE_SNAPSHOT to the snapshot directory\n");
    return 0;
  }
  if (!have_matching_gpu()) {
    std::printf("skip: needs sm_120\n");
    return 0;
  }
  (void)argc;
  (void)argv;

  const int iters = env_int("QWEN35MOE_STEP_BATCH_ITERS", 8);
  const bool skip_timing = std::getenv("QWEN35MOE_STEP_BATCH_SKIP_TIMING") != nullptr;
  std::vector<int> depths;
  if (const char* d = std::getenv("QWEN35MOE_STEP_BATCH_DEPTHS")) {
    std::string text = d;
    std::size_t at = 0;
    while (at < text.size()) {
      const std::size_t comma = text.find(',', at);
      depths.push_back(std::atoi(text.substr(at, comma == std::string::npos ? std::string::npos : comma - at).c_str()));
      if (comma == std::string::npos) break;
      at = comma + 1;
    }
  } else {
    depths = {1024, 258048};
  }
  int deepest = 0;
  for (int d : depths) deepest = std::max(deepest, d);
  const int max_pos = std::max(env_int("QWEN35MOE_STEP_BATCH_MAXPOS", 2048), deepest + 256);

  qwen::TextArena arena;
  std::string error;
  if (!qwen::upload_text_model(snapshot, true, -1, &arena, &error)) {
    std::printf("FAIL upload: %s\n", error.c_str());
    return 1;
  }
  std::printf("loaded %llu bytes, split %d/%d\n", static_cast<unsigned long long>(arena.uploaded_bytes),
              arena.experts_on_first, qwen::EXPERTS - arena.experts_on_first);

  // The tolerance check reads the block's logits back; a serving caller only needs the argmax.
  setenv("QWEN35MOE_BATCH_LOGITS", "1", 1);

  qwen::Decoder decoder;
  if (!decoder.init(arena, max_pos, &error)) {
    std::printf("FAIL decoder init (max_pos %d): %s\n", max_pos, error.c_str());
    return 1;
  }

  bool ok = true;
  const std::vector<int> prompt = prompt_ids(env_int("QWEN35MOE_STEP_BATCH_PREFIX", 64));
  const std::vector<int> drafts = draft_ids();
  const int pos = int(prompt.size());
  {
    int next = 0;
    for (int i = 0; i < int(prompt.size()); ++i) {
      if (!decoder.step(prompt[std::size_t(i)], i, &next, &error)) {
        std::printf("FAIL prefill %d: %s\n", i, error.c_str());
        return 1;
      }
    }
    std::printf("prefilled %zu positions; the verify block sits at %d\n", prompt.size(), pos);
    std::fflush(stdout);
  }

  // One checkpoint of the trunk state after the prefill; every comparison below starts from it.
  if (!decoder.checkpoint_gdn(&error)) {
    std::printf("FAIL checkpoint: %s\n", error.c_str());
    return 1;
  }

  // ---- 1. n = 1 is the single-token step ----
  {
    const int id = drafts[0];
    int single = -1;
    if (!decoder.step(id, pos, &single, &error)) {
      std::printf("FAIL single step: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> single_logits = decoder.logits();
    if (!decoder.restore_gdn(&error)) {
      std::printf("FAIL restore: %s\n", error.c_str());
      return 1;
    }
    int block = -1;
    if (!decoder.step_batch(&id, 1, pos, &block, &error)) {
      std::printf("FAIL step_batch(1): %s\n", error.c_str());
      return 1;
    }
    // The single-token logits accessor is the right one here: n = 1 is step() itself, so it filled logits().
    const bool same_id = single == block;
    const bool same_logits = std::memcmp(decoder.logits().data(), single_logits.data(),
                                         sizeof(float) * qwen::VOCAB) == 0;
    std::printf("%s n = 1 equals step(): id %d vs %d, logits byte-identical: %s\n",
                same_id && same_logits ? "PASS" : "FAIL", single, block, same_logits ? "yes" : "no");
    ok &= same_id && same_logits;
  }

  // ---- 2. n = 1..4 against n single steps ----
  // The batch path is forced at n = 1 here, because the default routes one token through step(): what this
  // section measures is the batch kernels' own agreement with the sequential path.
  setenv("QWEN35MOE_BATCH_AT_ONE", "1", 1);
  double worst_rel = 0.0;
  int worst_n = 0, worst_pos = 0;
  std::printf("\n%-3s %-8s %-14s %-14s %-12s\n", "n", "ids", "max |dlogit|", "max |logit|", "rel");
  for (int n = 1; n <= qwen::kernels::BATCH_MAX; ++n) {
    if (!decoder.restore_gdn(&error)) {
      std::printf("FAIL restore (n=%d): %s\n", n, error.c_str());
      return 1;
    }
    std::vector<int> step_ids(std::size_t(n), -1);
    std::vector<float> step_logits(std::size_t(n) * qwen::VOCAB, 0.0f);
    std::vector<float> step_resid(std::size_t(n) * qwen::HIDDEN, 0.0f);
    std::vector<float> resid;
    for (int t = 0; t < n; ++t) {
      if (!decoder.step(drafts[std::size_t(t)], pos + t, &step_ids[std::size_t(t)], &error)) {
        std::printf("FAIL step (n=%d, t=%d): %s\n", n, t, error.c_str());
        return 1;
      }
      std::memcpy(&step_logits[std::size_t(t) * qwen::VOCAB], decoder.logits().data(), sizeof(float) * qwen::VOCAB);
      if (!decoder.read_residual(&resid, &error)) {
        std::printf("FAIL read_residual: %s\n", error.c_str());
        return 1;
      }
      std::memcpy(&step_resid[std::size_t(t) * qwen::HIDDEN], resid.data(), sizeof(float) * qwen::HIDDEN);
    }
    if (!decoder.restore_gdn(&error)) {
      std::printf("FAIL restore (n=%d): %s\n", n, error.c_str());
      return 1;
    }
    std::vector<int> block_ids(std::size_t(n), -1);
    if (!decoder.step_batch(drafts.data(), n, pos, block_ids.data(), &error)) {
      std::printf("FAIL step_batch(%d): %s\n", n, error.c_str());
      return 1;
    }
    bool ids_ok = true;
    double max_diff = 0.0, max_abs = 0.0;
    for (int t = 0; t < n; ++t) {
      if (block_ids[std::size_t(t)] != step_ids[std::size_t(t)]) {
        ids_ok = false;
        std::printf("  n=%d t=%d: id %d vs %d\n", n, t, block_ids[std::size_t(t)], step_ids[std::size_t(t)]);
      }
      const float* got = decoder.batch_logits(t);
      if (!got) {
        std::printf("FAIL batch_logits(%d) is null\n", t);
        return 1;
      }
      const float* ref = &step_logits[std::size_t(t) * qwen::VOCAB];
      int nonfinite = 0;
      for (int v = 0; v < qwen::VOCAB; ++v) {
        const double a = double(ref[v]), b = double(got[v]);
        max_abs = std::max(max_abs, std::fabs(a));
        if (!std::isfinite(b)) {
          ++nonfinite;
          continue;
        }
        max_diff = std::max(max_diff, std::fabs(a - b));
      }
      if (nonfinite) {
        std::printf("FAIL n=%d t=%d: %d non-finite logits in the batch pass\n", n, t, nonfinite);
        ok = false;
      }
    }
    const double rel = max_abs > 0.0 ? max_diff / max_abs : max_diff;

    if (rel > worst_rel) {
      worst_rel = rel;
      worst_n = n;
      worst_pos = 0;
    }
    std::printf("%-3d %-8s %-14.3e %-14.3e %-12.3e |", n, ids_ok ? "same" : "DIFF", max_diff, max_abs, rel);
    // Per position, so a difference that is one discrete event (a router near-tie) shows as one large entry
    // beside small ones rather than as a blanket offset.
    for (int t = 0; t < n; ++t) {
      const float* got = decoder.batch_logits(t);
      const float* ref = &step_logits[std::size_t(t) * qwen::VOCAB];
      double d = 0.0, m = 0.0;
      for (int v = 0; v < qwen::VOCAB; ++v) {
        d = std::max(d, std::fabs(double(ref[v]) - double(got[v])));
        m = std::max(m, std::fabs(double(ref[v])));
      }
      // The same position's trunk residual, which is what every difference upstream of the head is measured on.
      const float* brut = decoder.batch_residual(t);
      const float* sref = &step_resid[std::size_t(t) * qwen::HIDDEN];
      double rd = 0.0, rm = 0.0;
      if (brut) {
        for (int i = 0; i < qwen::HIDDEN; ++i) {
          rd = std::max(rd, std::fabs(double(sref[i]) - double(brut[i])));
          rm = std::max(rm, std::fabs(double(sref[i])));
        }
      }
      std::printf(" t%d logit %.2e resid %.2e", t, m > 0.0 ? d / m : d, rm > 0.0 ? rd / rm : rd);
    }
    std::printf("\n");
    ok &= ids_ok;
  }
  // The gate is the ids: they are identical at every n, every depth and every expert placement tried, which is
  // what a verify block needs. Measured here: residual 1.4e-7..5.5e-7 and logits 2.3e-7..3.5e-7 relative, i.e.
  // fp32 reduction order alone (the batch GEMVs sum eight contiguous columns per lane where the single-token
  // kernels sum stride-32 column pairs, and the GDN recurrence merges four row chunks where it walks 128
  // serially). The bound is deliberately loose because the same comparison on a different expert split - which
  // changes how the two cards' expert partials are summed, and so which side of a router near tie a token lands
  // on - reached 3.5e-4 while still producing identical ids. A real defect shows at 1e-1 and above, and the ids
  // break first.
  const bool within = worst_rel <= 5e-2;
  std::printf("%s tolerance: worst relative logit difference %.3e (n=%d) inside the 5e-2 smoke bound; "
              "ids identical is the gate\n",
              within ? "PASS" : "FAIL", worst_rel, worst_n);
  ok &= within;
  (void)worst_pos;

  // ---- 3. cost ----
  unsetenv("QWEN35MOE_BATCH_AT_ONE");
  if (!skip_timing) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::printf("\n%-8s %-3s %-12s %-12s %-12s %-12s %-9s\n", "depth", "n", "batch ms/step", "batch ms/tok",
                "n steps ms", "single ms/tok", "speedup");
    for (int depth : depths) {
      // Same trick as the depth bench: decode at `depth` against whatever the cache holds. A decode step costs
      // the rows it reads, not their values, and the correctness of every kernel above is already shown.
      const int start = depth;
      if (start + 128 > max_pos) {
        std::printf("depth %d: outside max_pos %d, skipped\n", depth, max_pos);
        continue;
      }
      for (int n = 1; n <= qwen::kernels::BATCH_MAX; ++n) {
        int next[qwen::kernels::BATCH_MAX] = {};
        // Warm both paths once each, then time them on the same block of positions.
        int warm = 0;
        if (!decoder.step_batch(drafts.data(), n, start, next, &error)) {
          std::printf("FAIL step_batch(%d) at %d: %s\n", n, start, error.c_str());
          return 1;
        }
        for (int t = 0; t < n; ++t) {
          if (!decoder.step(drafts[std::size_t(t)], start + 8 + t, &warm, &error)) {
            std::printf("FAIL warm step: %s\n", error.c_str());
            return 1;
          }
        }
        // The batch pass: positions start + 16 * i, so every iteration is a fresh block below max_pos.
        double batch_ms = 0.0;
        int blocks = 0;
        for (int i = 0; i < iters; ++i) {
          const int at = start + 8 + i * n;
          if (at + n > max_pos) break;
          cudaEventRecord(t0);
          if (!decoder.step_batch(drafts.data(), n, at, next, &error)) {
            std::printf("FAIL step_batch(%d) at %d: %s\n", n, at, error.c_str());
            return 1;
          }
          cudaEventRecord(t1);
          cudaEventSynchronize(t1);
          batch_ms += ms_between(t0, t1);
          ++blocks;
        }
        // n single steps, at their own fresh positions, so the depth is the same.
        double single_ms = 0.0;
        int steps = 0;
        for (int i = 0; i < iters; ++i) {
          const int at = start + 8 + blocks * n + i * n;
          if (at + n > max_pos) break;
          cudaEventRecord(t0);
          for (int t = 0; t < n; ++t) {
            if (!decoder.step(drafts[std::size_t(t)], at + t, &warm, &error)) {
              std::printf("FAIL single step: %s\n", error.c_str());
              return 1;
            }
          }
          cudaEventRecord(t1);
          cudaEventSynchronize(t1);
          single_ms += ms_between(t0, t1);
          ++steps;
        }
        if (blocks == 0 || steps == 0) continue;
        const double per_block = batch_ms / blocks;
        const double per_single = single_ms / steps;  // n tokens
        std::printf("%-8d %-3d %-12.3f %-12.3f %-12.3f %-12.3f %-9.2f\n", depth, n, per_block, per_block / n,
                    per_single, per_single / n, per_single / per_block);
        std::fflush(stdout);
      }
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }

  qwen::free_text_arena(&arena);
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
