// MTP drafter acceptance rate on the real checkpoint: K0 of plans/20261126_qwen36-spec-decode.md.
//
// The trunk (Decoder) runs a fixed prompt greedily and records, per step, the pre-final-norm residual row it
// ended on: decoder.cu's QWEN35MOE_DUMP_STATE hook appends {int32 pos, int32 token, float32 x[2048]}. The
// drafter (MtpDrafter) is then replayed along the same token sequence in the MTP chain llama.cpp uses:
//   draft 1 = MTP(embed(t[p+1]), h[p]) at position p+1, predicting t[p+2];
//   draft 2 = MTP(embed(t[p+2]), h' of draft 1) at position p+2, predicting t[p+3];
//   draft i likewise from draft i-1, up to eight drafts.
// Each position runs one draft_chain(): the forwards are per token, the trunk's lm_head once for the whole chain.
// Drafts 2.. only count where the earlier drafts of that chain were accepted, which is what makes the expected
// tokens per verify step 1 + a1 + a1*a2 + ... + a1*...*a(n-1) for n = k + 1 candidates. The chain is carried to
// eight drafts and the table prints n = 2..9, alongside the measured cost of a chain that long, so the draft
// count is a measured decision and not a guess.
//
// Prompt positions are tallied apart from generated ones: the target of a sample belongs to the generated
// region when its index in the token sequence is at or past the prompt length, and continuations behave
// differently from a teacher-forced prefix. Every prompt runs through both drafter caches, the shipped Q4_0 and
// a BF16 one, because Q4_0's acceptance cost is a K0 question in the plan.
//
// Nothing here changes the engine: trunk and drafter are the shipped modules, the dump hook is the only
// measurement glue, and this file adds no kernel. Skipped unless QWEN35MOE_SNAPSHOT is set; needs sm_120.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/mtp/mtp_attn.h"
#include "arch/qwen35moe/model/decoder.h"
#include "arch/qwen35moe/model/mtp.h"
#include "arch/qwen35moe/model/uploader.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::test;

namespace qwen = ingine::arch::qwen35moe;
namespace mtpk = ingine::arch::qwen35moe::kernels;

namespace {

constexpr int HIDDEN = qwen::HIDDEN;
// The trunk's cache is small on purpose: the trunk plus the drafter must fit in 16 + 16 GB, and the prompts
// below never reach 200 positions.
constexpr int kMaxPos = 4096;
constexpr int kGenerate = 32;

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  return p.major == 12 && p.minor == 0;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// The drafter borrows the trunk's embedding table and lm_head instead of holding copies; both are already in the
// arena the decoder runs from, at the span and offset the placement gave them.
const void* arena_tensor(const qwen::TextArena& arena, const char* name) {
  for (const qwen::PlacedTensor& t : arena.placed) {
    if (t.name != name) continue;
    return static_cast<const std::byte*>(arena.spans[std::size_t(t.span)].device) + t.offset;
  }
  return nullptr;
}

float arena_scalar(const qwen::TextArena& arena, const char* name) {
  const void* at = arena_tensor(arena, name);
  float v = 1.0f;
  if (at) INGINE_TEST_CUDA(cudaMemcpy(&v, at, 4, cudaMemcpyDeviceToHost));
  return v;
}

// -------------------------------------------------------------------------------------------------- the prompts
//
// Tokenised with the snapshot's own tokenizer.json (Qwen tokenizer, `tokenizers` 0.22), no chat template: these
// are raw prefixes of prose and of code, which is what a continuation measures. The text of each prompt is in
// the comment above its ids so the ids can be re-derived on any machine that can run the tokenizer.

struct Prompt {
  const char* name;
  std::vector<int> ids;
};

std::vector<Prompt> make_prompts() {
  return {
      // "The old harbour had been rebuilt three times since the flood, and each time the engineers promised
      //  that this pier would outlast the sea. ..."
      {"prose_city",
       {760, 2235, 72103, 995, 978, 57571, 2250, 2942, 2394, 279, 17198, 11, 321, 1754, 854, 279, 23438, 18378,
        421, 411, 21268, 1000, 680, 4119, 279, 9117, 13, 37762, 3055, 424, 1381, 11, 3162, 279, 1505, 14234,
        321, 279, 8981, 5638, 73342, 11, 488, 1362, 1990, 1436, 279, 1500, 1332, 279, 24733, 13769, 2197, 279,
        8742, 9506, 13, 16120, 57157, 295, 2789, 50692, 303, 279, 79117, 314, 264, 5075, 11, 321, 264, 5054,
        5898, 10482, 494, 264, 41829, 303, 279, 6784, 314, 279, 33619, 3626, 13, 35254, 70398, 13, 561, 30592,
        3571, 303, 948, 781, 3571, 303, 11, 321, 279, 40554, 557, 524, 2414, 5316, 17736, 440, 13, 1848, 279,
        2944, 5883, 279, 491, 9014, 52331, 290, 11, 660, 8720, 321, 4147, 36070, 536, 11, 8318, 364, 38592,
        852, 995, 524, 11136, 3482, 13}},
      // "By the middle of October the mornings had turned sharp enough that the first thing you did was check
      //  the window for frost. ..."
      {"prose_weather",
       {1311, 279, 5956, 314, 6324, 279, 55287, 995, 6316, 16724, 3213, 421, 279, 1118, 3065, 488, 1467, 557,
        1716, 279, 3136, 364, 43247, 13, 561, 13159, 995, 7795, 11012, 303, 279, 1560, 424, 1503, 948, 279,
        39465, 2842, 11, 321, 279, 3000, 3571, 303, 3238, 321, 13358, 3808, 279, 9488, 1898, 13, 1049, 427,
        2540, 364, 2943, 2756, 1973, 21835, 11, 264, 8455, 10849, 421, 9896, 279, 17325, 4883, 321, 1801, 279,
        20619, 92285, 470, 11, 321, 1179, 279, 12515, 22235, 430, 413, 4161, 995, 6716, 13, 561, 33002, 5467,
        264, 2545, 11456, 383, 836, 17631, 11947, 11, 321, 279, 3594, 82738, 314, 13988, 10583, 321, 7339,
        3445, 4587, 13, 20798, 1018, 279, 12050, 1000, 381, 1248, 11, 321, 4858, 1018, 424, 303, 279, 1788,
        15754, 11, 4139, 11833, 321, 4139, 6624, 13}},
      // "The recipe is not complicated, but it is unforgiving about time. You mix the flour and water first
      //  and let them sit for an hour, so the gluten begins to form on its own, ..."
      {"prose_bread",
       {760, 10793, 369, 524, 16468, 11, 694, 424, 369, 9194, 1712, 2177, 883, 854, 13, 1394, 6311, 279, 19223,
        321, 2919, 1118, 321, 1042, 1070, 2364, 364, 449, 6258, 11, 748, 279, 36120, 11690, 310, 1304, 383,
        1141, 1765, 11, 321, 1132, 1179, 635, 488, 884, 279, 11679, 321, 279, 39142, 13, 561, 29363, 1220, 2579,
        8153, 321, 9781, 13471, 88, 11, 264, 2545, 27467, 310, 6601, 3023, 494, 279, 18632, 1056, 488, 1660, 13,
        58435, 424, 1396, 4139, 6258, 364, 279, 1118, 1330, 3979, 11, 28314, 11, 430, 3384, 488, 998, 259, 44630,
        424, 303, 11, 321, 1179, 5106, 424, 7249, 2980, 424, 682, 33472, 13, 21826, 424, 3200, 303, 279, 11129,
        321, 21844, 424, 303, 264, 1546, 3882, 22660, 440, 264, 33540, 314, 2919, 28396, 11, 321, 279, 38300,
        668, 7534, 948, 424, 3905, 680, 13}},
      // "They left the coast in a small ship with a crew of nineteen and a cargo of dried fish, intending to be
      //  gone for a year and to return with charts of the northern passage. ..."
      {"prose_explore",
       {6651, 2047, 279, 13255, 303, 264, 2526, 8026, 440, 264, 13234, 314, 90600, 321, 264, 24834, 314, 30581,
        7397, 11, 81428, 310, 381, 7795, 364, 264, 1007, 321, 310, 460, 440, 25296, 314, 279, 17631, 20438, 13,
        561, 1118, 5381, 998, 3999, 26, 279, 9662, 5467, 23271, 494, 279, 51206, 321, 279, 9117, 557, 4409, 9766,
        13, 4844, 279, 29262, 3571, 1441, 321, 19013, 364, 42744, 2756, 11, 321, 781, 32306, 639, 539, 4950, 321,
        539, 279, 7739, 314, 279, 1500, 11, 864, 279, 28222, 2512, 680, 1396, 4139, 6258, 303, 264, 7497, 421,
        13538, 1274, 1592, 430, 279, 2756, 5642, 13, 3095, 279, 29262, 28761, 1017, 557, 9561, 383, 279, 32946,
        11, 3238, 321, 4022, 321, 1691, 7875, 1056, 424, 995, 6749, 303, 279, 36590, 13}},
      // static size_t ring_push(struct ring *r, const void *item, size_t len) { ... }
      {"code_c",
       {1911, 1331, 516, 9761, 13812, 5970, 9761, 348, 81, 11, 713, 715, 348, 1163, 11, 1331, 516, 2343, 8, 313,
        198, 262, 1331, 516, 1429, 283, 318, 81, 397, 1916, 478, 427, 397, 11018, 471, 427, 397, 14166, 8, 984,
        427, 397, 11018, 26, 198, 262, 413, 318, 81, 397, 11018, 471, 1429, 361, 2343, 8, 313, 198, 285, 460,
        220, 15, 26, 1343, 874, 2956, 11, 19260, 26731, 3315, 310, 5758, 621, 198, 262, 333, 198, 262, 364, 318,
        2073, 516, 585, 283, 220, 15, 26, 585, 361, 2343, 26, 3333, 72, 8, 313, 198, 285, 713, 3646, 1122, 348,
        3431, 283, 318, 989, 3646, 1122, 2501, 1163, 26, 198, 285, 427, 397, 5725, 9409, 81, 397, 1916, 478,
        585, 8, 984, 427, 397, 11018, 60, 283, 2212, 957, 5098, 198, 262, 333, 198, 262, 427, 397, 1916, 283,
        318, 81, 397, 1916, 478, 2343, 8, 984, 427, 397, 11018, 26, 198, 262, 460, 2343, 26, 198, 92, 198}},
      // def merge_intervals(intervals): ...
      {"code_python",
       {727, 10562, 87682, 1494, 40798, 1590, 198, 262, 413, 524, 26126, 25, 198, 285, 460, 2958, 198, 262, 11128,
        283, 10300, 1494, 40798, 11, 1328, 26612, 6505, 25, 6505, 58, 15, 2387, 198, 262, 25170, 283, 498, 1551,
        7, 10231, 58, 15, 69631, 198, 262, 364, 1151, 11, 809, 303, 11128, 58, 16, 25, 5491, 198, 285, 1483, 283,
        25170, 7368, 16, 60, 198, 285, 413, 1151, 2564, 1483, 58, 16, 5491, 198, 309, 1483, 58, 16, 60, 283, 1866,
        22634, 58, 16, 1089, 809, 8, 198, 285, 745, 25, 198, 309, 25170, 1989, 2477, 2388, 11, 809, 2387, 198,
        262, 460, 498, 23814, 37848, 8, 364, 6505, 303, 25170, 60, 198}},
  };
}

// ------------------------------------------------------------------------------------------ the trunk records

// One record of the dump hook: the step's position, the token it was fed, and the residual it ended on.
struct StepRecord {
  int pos = 0;
  int token = 0;
  std::vector<float> residual;
};

constexpr std::size_t kRecordBytes = 2 * sizeof(int) + HIDDEN * sizeof(float);

bool read_dump(const std::string& path, std::vector<StepRecord>* steps, std::string* error) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    *error = "cannot open " + path;
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0 || std::size_t(size) % kRecordBytes != 0) {
    *error = "dump size " + std::to_string(size) + " is not a whole number of records";
    std::fclose(f);
    return false;
  }
  steps->assign(std::size_t(size) / kRecordBytes, StepRecord{});
  for (StepRecord& s : *steps) {
    const int header[2] = {0, 0};
    if (std::fread(const_cast<int*>(header), sizeof(int), 2, f) != 2) {
      *error = "short read of a record header";
      std::fclose(f);
      return false;
    }
    s.pos = header[0];
    s.token = header[1];
    s.residual.resize(std::size_t(HIDDEN));
    if (std::fread(s.residual.data(), sizeof(float), HIDDEN, f) != std::size_t(HIDDEN)) {
      *error = "short read of a residual row";
      std::fclose(f);
      return false;
    }
  }
  std::fclose(f);
  for (std::size_t i = 0; i < steps->size(); ++i) {
    if ((*steps)[i].pos != int(i)) {
      *error = "record " + std::to_string(i) + " is at pos " + std::to_string((*steps)[i].pos);
      return false;
    }
  }
  return true;
}

// A prompt run: the recorded steps, the one token the trunk predicted past the last recorded step, and what the
// trunk cost while producing them.
struct TrunkRun {
  std::vector<StepRecord> steps;
  int tail_token = -1;
  double prefill_ms_per_step = 0.0;
  double decode_ms_per_step = 0.0;
  std::vector<int> generated;  // the new tokens, in order, tail_token included
};

// Prefills the prompt, generates kGenerate tokens greedily, and reads back what the dump hook saw. A fresh
// decoder per prompt: the GDN state and the conv history are recurrent and carry no reset, so a second prompt
// on the same decoder would not start from an empty state.
bool run_trunk(const qwen::TextArena& arena, const std::vector<int>& prompt, const std::string& dump_path,
               TrunkRun* run, std::string* error) {
  if (FILE* f = std::fopen(dump_path.c_str(), "wb")) std::fclose(f);  // the hook appends; start empty
  if (setenv("QWEN35MOE_DUMP_STATE", dump_path.c_str(), 1) != 0) {
    *error = "setenv QWEN35MOE_DUMP_STATE";
    return false;
  }

  qwen::Decoder decoder;
  bool ok = decoder.init(arena, kMaxPos, error);
  int next = 0;
  if (ok) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < int(prompt.size()); ++i) {
      if (!decoder.step(prompt[std::size_t(i)], i, &next, error)) {
        ok = false;
        break;
      }
    }
    run->prefill_ms_per_step = ms_since(t0) / double(prompt.size());
  }
  if (ok) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int j = 0; j < kGenerate; ++j) {
      run->generated.push_back(next);
      if (!decoder.step(next, int(prompt.size()) + j, &next, error)) {
        ok = false;
        break;
      }
    }
    run->decode_ms_per_step = ms_since(t0) / double(kGenerate);
    run->tail_token = next;
  }
  unsetenv("QWEN35MOE_DUMP_STATE");
  if (!ok) return false;

  if (!read_dump(dump_path, &run->steps, error)) return false;
  const std::size_t want = prompt.size() + kGenerate;
  if (run->steps.size() != want) {
    *error = "dump holds " + std::to_string(run->steps.size()) + " steps, the run made " + std::to_string(want);
    return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------- the tallies

// s[i]/a[i] are the samples and hits of draft i + 1: s[0] counts the chains with a target for draft 1, s[1] only
// those whose draft 1 was accepted (and whose target exists), and so on. That is the denominator the multiplier
// uses. The chain is carried to kChainDrafts drafts, so the table reaches alpha8.
constexpr int kChainDrafts = qwen::kMtpChainMax;

struct Tally {
  std::array<long long, kChainDrafts> s{};
  std::array<long long, kChainDrafts> a{};

  void add(const Tally& o) {
    for (int i = 0; i < kChainDrafts; ++i) {
      s[std::size_t(i)] += o.s[std::size_t(i)];
      a[std::size_t(i)] += o.a[std::size_t(i)];
    }
  }
};

struct Rates {
  std::array<double, kChainDrafts> a{};
};

Rates rates_of(const Tally& t) {
  Rates r;
  for (int i = 0; i < kChainDrafts; ++i) {
    r.a[std::size_t(i)] = t.s[std::size_t(i)] ? double(t.a[std::size_t(i)]) / double(t.s[std::size_t(i)]) : 0.0;
  }
  return r;
}

// 1 + a1 + a1*a2 + ... + a1*...*a(n-1): the tokens one verify step of n candidates yields. A rate with no
// samples is zero here, so the terms past the measured depth drop out instead of being invented.
double expected_tokens(const Rates& r, int n) {
  double e = 1.0, run = 1.0;
  for (int i = 0; i < n - 1 && i < kChainDrafts; ++i) {
    run *= r.a[std::size_t(i)];
    e += run;
  }
  return e;
}

// One prompt's chain result: the same tally over every position, and over the positions whose target token is
// a generated one.
struct ChainResult {
  std::string name;
  int prompt_tokens = 0;
  int steps = 0;
  Tally all;
  Tally generated_only;
  double draft_ms = 0.0;      // the draft_chain() calls, summed
  long long drafts = 0;
  // Pairing diagnostics: an off-by-one in the chain (draft 1 pointed at the token it was fed, or at the one
  // before it, instead of the trunk's next token) would show up as hits here rather than as the alpha1 above.
  long long fed_hits = 0;
  long long prev_hits = 0;
};

// ------------------------------------------------------------------------------------------------ the chain

// Replays the MTP chain over `run`'s token sequence with the module's batched head. The drafter's cache row p is
// the K/V of the token fed at position p with the trunk hidden of p-1, which is exactly what draft 1 of the chain
// at p-1 writes; row 0 has no such hidden, so it is seeded with the only row available (t[0] with h[0]) to give
// the first positions a cache to attend over, and that seed is not scored.
//
// Each position runs one draft_chain() of up to kChainDrafts candidates: the n forwards are per token, the trunk's
// lm_head runs once over all n rows. Draft i's input does not depend on whether drafts 1..i-1 were accepted (the
// tokens are the trunk's own), so a full chain computes every row the conditional walk would have reached and the
// tally applies the acceptance condition to the answers, not to the work.
bool run_chain(qwen::MtpDrafter& drafter, const TrunkRun& run, int prompt_len, ChainResult* out,
               std::string* error) {
  const int n = int(run.steps.size());
  if (n < kChainDrafts + 2) {
    *error = "a chain needs at least " + std::to_string(kChainDrafts + 2) + " trunk steps";
    return false;
  }
  std::vector<int> tok(std::size_t(n) + 1);
  for (int i = 0; i < n; ++i) tok[std::size_t(i)] = run.steps[std::size_t(i)].token;
  tok[std::size_t(n)] = run.tail_token;

  DeviceArray<__nv_bfloat16> d_hidden{static_cast<std::size_t>(HIDDEN)};
  std::vector<__nv_bfloat16> row(static_cast<std::size_t>(HIDDEN));

  // What draft() takes is a BF16 row; the trunk's residual and the drafter's h' are both FP32 on the device.
  auto upload_row = [&](const float* src) {
    for (int i = 0; i < HIDDEN; ++i) row[std::size_t(i)] = float_to_bf16(src[i]);
    INGINE_TEST_CUDA(cudaMemcpy(d_hidden.get(), row.data(), HIDDEN * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
  };

  int seed_token = -1;
  upload_row(run.steps[0].residual.data());
  if (!drafter.draft(tok[0], d_hidden.get(), 0, &seed_token, error)) return false;

  out->name = "";
  out->prompt_tokens = prompt_len;
  out->steps = n;

  std::vector<int> drafted(kChainDrafts);
  for (int p = 0; p + 2 <= n; ++p) {
    // Draft i consumes tok[p + i] and predicts tok[p + 1 + i]; the last draft with a target caps the chain.
    const int len = std::min(kChainDrafts, n - p - 1);
    upload_row(run.steps[std::size_t(p)].residual.data());
    const auto t0 = std::chrono::steady_clock::now();
    if (!drafter.draft_chain(tok.data() + p + 1, d_hidden.get(), len, p + 1, drafted.data(), error)) return false;
    out->draft_ms += ms_since(t0);
    out->drafts += len;
    for (int i = 0; i < len; ++i) {
      // s[i] needs drafts 1..i accepted and a target at p + 2 + i; both are decided in this loop.
      const int target = p + 2 + i;
      if (target > n) break;
      ++out->all.s[std::size_t(i)];
      const bool generated = target >= prompt_len;
      if (generated) ++out->generated_only.s[std::size_t(i)];
      if (i == 0) {
        // Pairing diagnostics on draft 1: an off-by-one in the chain (the token it was fed, or the one before it,
        // instead of the trunk's next token) would show up here rather than as the alpha1 above.
        if (drafted[0] == tok[std::size_t(p + 1)]) ++out->fed_hits;
        if (drafted[0] == tok[std::size_t(p)]) ++out->prev_hits;
      }
      if (drafted[std::size_t(i)] != tok[std::size_t(target)]) break;
      ++out->all.a[std::size_t(i)];
      if (generated) ++out->generated_only.a[std::size_t(i)];
    }
  }
  return true;
}

// The drafter's own cost at `pos`: warm, then kIters drafts, averaged, with the five stages the module times.
bool probe_drafter(qwen::MtpDrafter& drafter, int pos, const std::vector<float>& residual, int token,
                   double* wall_ms, qwen::MtpTiming* stage, std::string* error) {
  DeviceArray<__nv_bfloat16> d_hidden{std::size_t(HIDDEN)};
  std::vector<__nv_bfloat16> row(static_cast<std::size_t>(HIDDEN));
  for (int i = 0; i < HIDDEN; ++i) row[std::size_t(i)] = float_to_bf16(residual[std::size_t(i)]);
  d_hidden.upload(row);
  int next = -1;
  for (int warm = 0; warm < 3; ++warm) {
    if (!drafter.draft(token, d_hidden.get(), pos, &next, error)) return false;
  }
  constexpr int kIters = 20;
  qwen::MtpTiming sum{};
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < kIters; ++it) {
    qwen::MtpTiming timing;
    if (!drafter.draft(token, d_hidden.get(), pos, &next, error, &timing)) return false;
    sum.dense_ms += timing.dense_ms;
    sum.attn_ms += timing.attn_ms;
    sum.routed_ms += timing.routed_ms;
    sum.shared_ms += timing.shared_ms;
    sum.head_ms += timing.head_ms;
  }
  *wall_ms = ms_since(t0) / double(kIters);
  sum.dense_ms /= kIters;
  sum.attn_ms /= kIters;
  sum.routed_ms /= kIters;
  sum.shared_ms /= kIters;
  sum.head_ms /= kIters;
  *stage = sum;
  return true;
}

// ------------------------------------------------------------------------------------------------- the table

// One line of rates with their counts, one of the expected tokens per verify step they imply. n = k + 1
// candidates for k drafts, so the n column of a rate is the one that rate first contributes to.
void print_header() {
  std::printf("  %-14s %4s %4s", "prompt", "ctx", "gen");
  for (int i = 1; i <= kChainDrafts; ++i) {
    char cell[32];
    std::snprintf(cell, sizeof cell, "alpha%d", i);
    std::printf(" | %-13s", cell);
  }
  std::printf("\n");
}

void print_row(const std::string& label, int ctx, int gen, const Tally& t) {
  const Rates r = rates_of(t);
  std::printf("  %-14s %4d %4d", label.c_str(), ctx, gen);
  for (int i = 0; i < kChainDrafts; ++i) {
    std::printf(" | %5.3f %3lld/%-3lld", r.a[std::size_t(i)], t.a[std::size_t(i)], t.s[std::size_t(i)]);
  }
  std::printf("\n  %-14s %4s %4s", "  tokens/step", "", "");
  for (int n = 2; n <= kChainDrafts + 1; ++n) std::printf(" | n=%-2d %5.2f  ", n, expected_tokens(r, n));
  std::printf("\n");
}

}  // namespace

int main() {
  const std::string snapshot = env_or_skip("QWEN35MOE_SNAPSHOT");
  if (snapshot.empty()) return 0;
  if (!have_matching_gpu()) {
    std::printf("skip: needs sm_120\n");
    return 0;
  }
  const std::filesystem::path dir(snapshot);

  // The trunk does not need the checkpoint's MTP tensors (the drafter reads them itself), so they stay off the
  // cards: 1.69 GB of headroom the fit does not have to make.
  qwen::TextArena arena;
  std::string error;
  if (!qwen::upload_text_model(dir, false, -1, &arena, &error)) {
    std::printf("FAIL upload: %s\n", error.c_str());
    return 1;
  }
  std::printf("trunk: %.2f GB, expert split %d/%d; drafter cache at %d positions\n",
              double(arena.uploaded_bytes) / 1e9, arena.experts_on_first, qwen::EXPERTS - arena.experts_on_first,
              kMaxPos);

  qwen::MtpTrunkHead head;
  head.embed_tokens =
      static_cast<const __nv_bfloat16*>(arena_tensor(arena, "model.language_model.embed_tokens.weight"));
  head.head_w = static_cast<const std::uint8_t*>(arena_tensor(arena, "lm_head.weight"));
  head.head_scales = static_cast<const std::uint8_t*>(arena_tensor(arena, "lm_head.weight_scale"));
  head.head_scale = arena_scalar(arena, "lm_head.weight_scale_2");
  if (!head.embed_tokens || !head.head_w || !head.head_scales) {
    std::printf("FAIL the arena does not hold the embedding table and lm_head\n");
    qwen::free_text_arena(&arena);
    return 1;
  }

  // ---- the trunk runs: one per prompt, recorded through the dump hook ----
  const std::vector<Prompt> prompts = make_prompts();
  std::vector<TrunkRun> runs(prompts.size());
  const std::string dump_prefix = "/tmp/qwen35moe_mtp_accept_" + std::to_string(getpid());
  std::printf("trunk runs (%d prompts, %d greedy tokens each):\n", int(prompts.size()), kGenerate);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    const std::string path = dump_prefix + "_" + prompts[i].name + ".bin";
    if (!run_trunk(arena, prompts[i].ids, path, &runs[i], &error)) {
      std::printf("FAIL trunk run %s: %s\n", prompts[i].name, error.c_str());
      qwen::free_text_arena(&arena);
      return 1;
    }
    std::printf("  %-14s %3zu prompt tokens, prefill %.2f ms/step, decode %.2f ms/step\n",
                prompts[i].name, prompts[i].ids.size(), runs[i].prefill_ms_per_step, runs[i].decode_ms_per_step);
    // The whole continuation, so the tokenizer that produced the prompt ids can confirm the trunk is running
    // the real model and not a broken one.
    std::printf("  %-14s generated:", prompts[i].name);
    for (int t : runs[i].generated) std::printf(" %d", t);
    std::printf("\n");
  }

  bool ok = true;
  // ---- the chains, on both drafter caches: Q4_0 is what the plan ships, BF16 is the best the drafter can do,
  // and the difference is the cache's cost in acceptance ----
  for (qwen::MtpKvType type : {qwen::MtpKvType::Q4_0, qwen::MtpKvType::Bf16}) {
    qwen::MtpDrafter drafter;
    if (!drafter.init(dir, head, kMaxPos, type, 0, &error)) {
      std::printf("FAIL drafter init (%s cache): %s\n", mtpk::mtp_kv_type_name(type), error.c_str());
      qwen::free_text_arena(&arena);
      return 1;
    }
    std::printf("\ndrafter: %s cache, weights %.4f GB, cache %.4f GB at %d positions\n",
                mtpk::mtp_kv_type_name(type), double(drafter.weight_bytes()) / 1e9,
                double(drafter.cache_bytes()) / 1e9, kMaxPos);

    std::vector<ChainResult> results(prompts.size());
    Tally pooled, pooled_gen;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      ChainResult& r = results[i];
      if (!run_chain(drafter, runs[i], int(prompts[i].ids.size()), &r, &error)) {
        std::printf("FAIL chain %s: %s\n", prompts[i].name, error.c_str());
        qwen::free_text_arena(&arena);
        return 1;
      }
      r.name = prompts[i].name;
      pooled.add(r.all);
      pooled_gen.add(r.generated_only);
    }

    std::printf("\nall positions (target token = the trunk's own next token at that position):\n");
    print_header();
    for (const ChainResult& r : results) {
      print_row(r.name, r.prompt_tokens, kGenerate, r.all);
    }
    print_row("pooled", -1, -1, pooled);

    std::printf("\ngenerated positions only (the target token's index is at or past the prompt length):\n");
    print_header();
    for (const ChainResult& r : results) {
      print_row(r.name, r.prompt_tokens, kGenerate, r.generated_only);
    }
    print_row("pooled", -1, -1, pooled_gen);

    // Stability across prompts: the spread of each rate over the rows that have samples for it.
    auto spread = [](const std::vector<ChainResult>& v, bool generated, int which, double* lo, double* hi,
                     int* rows) {
      *lo = std::numeric_limits<double>::infinity();
      *hi = -std::numeric_limits<double>::infinity();
      *rows = 0;
      for (const ChainResult& r : v) {
        const Tally& t = generated ? r.generated_only : r.all;
        const Rates q = rates_of(t);
        if (t.s[std::size_t(which)] == 0) continue;
        *lo = std::min(*lo, q.a[std::size_t(which)]);
        *hi = std::max(*hi, q.a[std::size_t(which)]);
        ++*rows;
      }
    };
    for (bool generated : {false, true}) {
      std::printf("\nspread over prompts (%s):\n  ", generated ? "generated positions" : "all positions");
      for (int which = 0; which < kChainDrafts; ++which) {
        double lo = 0, hi = 0;
        int rows = 0;
        spread(results, generated, which, &lo, &hi, &rows);
        if (rows) std::printf("alpha%d %.3f..%.3f (%d)  ", which + 1, lo, hi, rows);
      }
      if (generated) std::printf("\n");
    }

    // What each draft is worth in tokens, pooled: the product of the rates up to and including draft i, which is
    // the marginal gain of carrying the chain one draft further.
    const Rates all = rates_of(pooled);
    const Rates gen = rates_of(pooled_gen);
    for (const Rates* r : {&all, &gen}) {
      std::printf("marginal tokens per draft, pooled (%s):\n          ", r == &all ? "all positions" : "generated");
      double run = 1.0;
      for (int i = 0; i < kChainDrafts; ++i) {
        run *= r->a[std::size_t(i)];
        std::printf(" a%-2d %.3f", i + 1, run);
      }
      std::printf("\n");
    }

    long long fed = 0, prev = 0;
    for (const ChainResult& r : results) {
      fed += r.fed_hits;
      prev += r.prev_hits;
    }
    std::printf("pairing check: over %lld draft-1 calls, the draft equalled the token it was fed %lld times and "
                "the token before it %lld times; a shifted comparison cannot produce the alpha1 above\n",
                pooled.s[0], fed, prev);

    // Cost side at the depth the chains ran: the drafter's own latency, and what one trunk token costs there.
    double draft_ms = 0.0;
    long long drafts = 0;
    double prefill_ms = 0.0, decode_ms = 0.0;
    for (std::size_t i = 0; i < results.size(); ++i) {
      draft_ms += results[i].draft_ms;
      drafts += results[i].drafts;
      prefill_ms += runs[i].prefill_ms_per_step;
      decode_ms += runs[i].decode_ms_per_step;
    }
    prefill_ms /= double(results.size());
    decode_ms /= double(results.size());
    const int deepest = int(runs.back().steps.size());
    double probe_ms = 0.0;
    qwen::MtpTiming stage;
    if (!probe_drafter(drafter, deepest, runs.back().steps.back().residual, runs.back().steps.back().token,
                       &probe_ms, &stage, &error)) {
      std::printf("FAIL drafter probe: %s\n", error.c_str());
      qwen::free_text_arena(&arena);
      return 1;
    }
    std::printf("cost: %lld drafts over the chains, %.3f ms/draft, the BF16 row rebuild between drafts on the "
                "device\n",
                drafts, drafts ? draft_ms / double(drafts) : 0.0);
    std::printf("cost: probe at pos %d: %.3f ms/draft = dense %.3f + attn %.3f + routed %.3f + shared %.3f + "
                "head %.3f; one trunk token at that depth is %.2f ms (decode) and %.2f ms (prefill)\n",
                deepest, probe_ms, stage.dense_ms, stage.attn_ms, stage.routed_ms, stage.shared_ms, stage.head_ms,
                decode_ms, prefill_ms);
    std::printf("cost: a 4-candidate verify step would spend 3 drafts = %.2f ms (%.0f%% of one fetched token at "
                "this depth, before the verify step itself)\n",
                3.0 * probe_ms, 100.0 * 3.0 * probe_ms / (decode_ms > 0 ? decode_ms : 1.0));

    // ---- the ms-per-token curve over n ----------------------------------------------------------------
    // chain_ms is one draft_chain(n - 1) at the depth the chains ran, and the verify step is charged one trunk
    // token: decode_ms is the floor for it, because the real step also reads the cache for n queries and runs the
    // hop. Every row of the acceptance table's n column lands in this one.
    {
      constexpr int kIters = 10;
      DeviceArray<__nv_bfloat16> d_hidden{static_cast<std::size_t>(HIDDEN)};
      std::vector<__nv_bfloat16> hidden_row(static_cast<std::size_t>(HIDDEN));
      const std::vector<float>& residual = runs.back().steps.back().residual;
      for (int i = 0; i < HIDDEN; ++i) hidden_row[std::size_t(i)] = float_to_bf16(residual[std::size_t(i)]);
      d_hidden.upload(hidden_row);
      std::vector<int> candidates(kChainDrafts);
      for (int i = 0; i < kChainDrafts; ++i) {
        candidates[std::size_t(i)] = runs.back().steps[std::size_t(deepest - 1 - i)].token;
      }
      std::printf("\ncost curve: a verify step of n candidates at depth %d, %s cache; the trunk side is one token "
                  "(%.2f ms), the drafter side one draft_chain(n - 1)\n",
                  deepest, mtpk::mtp_kv_type_name(type), decode_ms);
      std::printf("   n  drafts   chain ms   ms/draft   forward ms   head ms | E(n) all   ms/token all | E(n) gen   "
                  "ms/token gen\n");
      for (int n = 2; n <= kChainDrafts + 1; ++n) {
        const int k = n - 1;
        std::vector<int> out_tokens(k);
        const int base = deepest + 1 - k;
        auto once = [&](qwen::MtpChainTiming* t) {
          return drafter.draft_chain(candidates.data(), d_hidden.get(), k, base, out_tokens.data(), &error, t);
        };
        bool ran = true;
        for (int warm = 0; warm < 2 && ran; ++warm) ran = once(nullptr);
        if (!ran) {
          std::printf("FAIL chain probe k=%d: %s\n", k, error.c_str());
          qwen::free_text_arena(&arena);
          return 1;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < kIters; ++it) ran = once(nullptr);
        const double chain_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / kIters;
        qwen::MtpChainTiming timing;
        if (!ran || !once(&timing)) {
          std::printf("FAIL chain probe k=%d: %s\n", k, error.c_str());
          qwen::free_text_arena(&arena);
          return 1;
        }
        const double e_all = expected_tokens(all, n), e_gen = expected_tokens(gen, n);
        const double step_ms = decode_ms + chain_ms;
        std::printf("  %2d  %6d   %8.3f   %8.3f   %10.3f   %7.3f | %6.2f   %10.3f     | %6.2f   %10.3f\n", n, k,
                    chain_ms, chain_ms / k, timing.forward_ms, timing.head_ms, e_all,
                    e_all > 0 ? step_ms / e_all : 0.0, e_gen, e_gen > 0 ? step_ms / e_gen : 0.0);
      }
    }

    if (pooled.s[0] == 0 || pooled.s[1] == 0) ok = false;
    (void)ok;
  }

  qwen::free_text_arena(&arena);
  if (!ok) {
    std::printf("FAIL: a prompt produced no chains\n");
    return 1;
  }
  std::printf("\nALL PASS\n");
  return 0;
}
