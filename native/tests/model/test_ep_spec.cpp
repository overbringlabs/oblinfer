// Expert parallelism with DSpark: with experts on the second device a verify step spans two devices, so it is
// no CUDA graph and the speculative loop runs it eagerly (SpecLoop reads the verify width back and steps that
// width). A greedy speculative generation must still walk the plain decode's tokens: the accept rule only takes
// drafts equal to the target's argmax. The prompt is processed one token per decode step
// (EngineGenerator::prefill_steps), so this covers that path too.
// Two GPUs and the whole checkpoint (~16.5 GB across both cards): skips (77) unless INGINE_RESIDENCY_TEST=1.
#include "common/test_util.h"
#include "ingine/generator.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <vector>

using namespace ingine;
using namespace ingine::test;
namespace model = ingine::model;

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

// The start of a captured agent prompt (test_forced), where a greedy continuation is confident enough that the
// verify widths' rounding does not move a tie.
const std::vector<std::int32_t> kPrompt = {200002, 200004, 1735,   149914, 220,    65281,  6,      82,     194368,
                                           163872, 171159, 13,     68014,  2,      154675, 198,    12,     167508,
                                           1255,   11,     150126, 151894, 12,     2409,   152725, 69137,  13,
                                           165179, 154393, 152059, 26,     152293, 220,    67211,  150665, 149882,
                                           150261, 1528,   183763, 13};

model::GenerateStats run(model::Generator& gen, int tokens, std::vector<std::int32_t>& out) {
  out.clear();
  model::Sampling greedy;  // temperature 0: the drafts are checked by equality
  return gen.generate(kPrompt, tokens, greedy, {}, [&](std::span<const std::int32_t> t) {
    out.insert(out.end(), t.begin(), t.end());
    return true;
  });
}

// The residency of the two-GPU runs: layers 19-23 keep their experts (4.55 GiB) on the second device.
model::RuntimeOptions options(const std::filesystem::path& dir) {
  model::RuntimeOptions o;
  o.model_dir = dir;
  o.max_ctx = 4096;
  o.dense_fp8 = true;
  o.kv_fp8 = true;
  o.second_gpu = 1;
  for (int l = 19; l < model::EXPERT_RESIDENCY_LAYERS; ++l) o.expert_residency.split[l] = 0;
  return o;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  int devices = 0;
  INGINE_TEST_CUDA(cudaGetDeviceCount(&devices));
  if (devices < 2) {
    std::printf("skip: needs two CUDA devices, found %d\n", devices);
    return kSkip;
  }
  if (std::getenv("INGINE_RESIDENCY_TEST") == nullptr) {
    std::printf("skip: set INGINE_RESIDENCY_TEST=1 (loads the whole model, ~16.5 GB across both cards)\n");
    return kSkip;
  }
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return kSkip;
  }

  const int tokens = 16;
  std::vector<std::int32_t> spec_out, plain_out;
  model::GenerateStats spec;
  {
    model::RuntimeOptions o = options(dir);
    o.spec = true;
    auto gen = model::make_generator(o);
    spec = run(*gen, tokens, spec_out);
  }
  {
    model::RuntimeOptions o = options(dir);
    o.spec = false;
    auto gen = model::make_generator(o);
    run(*gen, tokens, plain_out);
  }
  auto show = [](const char* name, const std::vector<std::int32_t>& v) {
    std::printf("%s:", name);
    for (std::int32_t t : v) std::printf(" %d", t);
    std::printf("\n");
  };
  show("spec", spec_out);
  show("plain", plain_out);
  std::printf("spec: %d rounds, %d accepted\n", spec.spec_rounds, spec.spec_accepted);

  expect(spec_out.size() == std::size_t(tokens) && plain_out.size() == std::size_t(tokens),
         "both runs emit the token budget");
  // a round emits the drafts it accepted plus one token
  expect(spec.spec_rounds > 0 && spec.spec_rounds + spec.spec_accepted >= tokens,
         "the speculative loop ran its rounds on the eager verify step");
  int same = 0;
  for (std::size_t i = 0; i < spec_out.size() && i < plain_out.size(); ++i) same += spec_out[i] == plain_out[i];
  std::printf("positions equal to plain decode: %d of %d\n", same, tokens);
  // a width-2/4/8 verify rounds differently from the width-1 step, so one near tie may move (and everything
  // after it); a broken step moves most of the continuation
  expect(same >= tokens - 1, "greedy speculative decoding walks the plain decode's tokens");
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
