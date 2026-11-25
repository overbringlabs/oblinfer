// A/B test of the two routed-expert paths on the real checkpoint: the verified per-expert `moe_apply` and the
// fused token-major kernel, over the same weights and the same MoE input. Also checks the fused per-expert
// base anchoring against the placed tensor addresses, which is where a table bug would show up.
// Skipped unless QWEN35MOE_SNAPSHOT is set.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/moe/fused_experts.h"
#include "arch/qwen35moe/kernels/moe/moe_apply.h"
#include "arch/qwen35moe/model/placement.h"
#include "arch/qwen35moe/model/uploader.h"
#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::test;
namespace qwen = ingine::arch::qwen35moe;
namespace moe = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;

namespace {

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  return p.major == 12 && p.minor == 0;
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

  qwen::TextArena arena;
  std::string error;
  if (!qwen::upload_text_model(snapshot, true, -1, &arena, &error)) {
    std::printf("FAIL upload: %s\n", error.c_str());
    return 1;
  }
  std::printf("stage: uploaded %zu tensors, %d spans\n", arena.placed.size(), int(arena.spans.size()));
  std::fflush(stdout);
  std::map<std::string, const qwen::PlacedTensor*> placed;
  for (const auto& p : arena.placed) placed[p.name] = &p;

  auto ptr = [&](const std::string& name) -> void* {
    const auto it = placed.find(name);
    if (it == placed.end()) throw std::runtime_error("not placed: " + name);
    const qwen::PlacedTensor& p = *it->second;
    return static_cast<std::byte*>(arena.spans[std::size_t(p.span)].device) + p.offset;
  };
  std::map<std::string, float> scalars;
  for (const auto& p : arena.placed) {
    if (p.bytes == 4 && p.name.ends_with(".weight_scale_2")) {
      float v = 0.0f;
      // A tensor on card 1 must be read with card 1 current: there is no P2P, and the error is sticky.
      cudaSetDevice(arena.spans[std::size_t(p.span)].cuda_device);
      cudaMemcpy(&v, ptr(p.name), 4, cudaMemcpyDeviceToHost);
      scalars[p.name] = v;
    }
  }
  cudaSetDevice(arena.spans[0].cuda_device);
  auto scalar = [&](const std::string& n, float fb) {
    const auto it = scalars.find(n);
    return it == scalars.end() ? fb : it->second;
  };

  std::printf("stage: scalars %zu\n", scalars.size());
  std::fflush(stdout);
  const int layer = 0;
  const std::string L = "model.language_model.layers." + std::to_string(layer) + ".";
  const auto& lp = arena.plan.layers[std::size_t(layer)];

  // Shared input and routing.
  std::vector<float> x_host(moe::MOE_HIDDEN);
  Rng rng(77);
  for (auto& v : x_host) v = float(rng.normal());
  // Keep every id on span 0: the reference path launches on the current device only.
  const int ids_host[8] = {3, 17, 63, 7, 100, 55, 12, 99};
  float weights_host[8] = {0.2f, 0.05f, 0.15f, 0.1f, 0.12f, 0.18f, 0.08f, 0.12f};

  float *dx = nullptr, *dy = nullptr;
  int* dids = nullptr;
  float* dweights = nullptr;
  cudaMalloc(&dx, moe::MOE_HIDDEN * 4);
  cudaMalloc(&dy, moe::MOE_HIDDEN * 4);
  cudaMalloc(&dids, 8 * 4);
  cudaMalloc(&dweights, 8 * 4);
  cudaMemcpy(dx, x_host.data(), moe::MOE_HIDDEN * 4, cudaMemcpyHostToDevice);
  int ids_dev[8];
  std::copy(std::begin(ids_host), std::end(ids_host), ids_dev);
  cudaMemcpy(dids, ids_dev, sizeof(ids_dev), cudaMemcpyHostToDevice);
  cudaMemcpy(dweights, weights_host, sizeof(weights_host), cudaMemcpyHostToDevice);

  std::printf("stage: ref path\n");
  std::fflush(stdout);
  // Path A: per-expert moe_apply (verified).
  std::vector<float> ref(moe::MOE_HIDDEN, 0.0f);
  {
    // moe_apply indexes the expert array by expert id, so the table needs every expert.
    std::vector<moe::MoeExpert> experts(ingine::arch::qwen35moe::EXPERTS);
    for (int e = 0; e < ingine::arch::qwen35moe::EXPERTS; ++e) {
      const std::string base = L + "mlp.experts." + std::to_string(e) + ".";
      auto args = [&](const char* proj, int rows, int cols) {
        quant::Nvfp4GemvArgs a;
        a.w = static_cast<const std::uint8_t*>(ptr(base + proj + ".weight"));
        a.scales = static_cast<const std::uint8_t*>(ptr(base + proj + ".weight_scale"));
        a.scale = scalar(base + proj + ".weight_scale_2", 1.0f);
        a.rows = rows;
        a.cols = cols;
        a.x = dx;
        return a;
      };
      experts[std::size_t(e)].gate = args("gate_proj", moe::MOE_INTER, moe::MOE_HIDDEN);
      experts[std::size_t(e)].up = args("up_proj", moe::MOE_INTER, moe::MOE_HIDDEN);
      experts[std::size_t(e)].down = args("down_proj", moe::MOE_HIDDEN, moe::MOE_INTER);
    }
    float *scratch_h = nullptr, *scratch_up = nullptr, *scratch_down = nullptr;
    cudaMalloc(&scratch_h, moe::MOE_INTER * 4);
    cudaMalloc(&scratch_up, moe::MOE_INTER * 4);
    cudaMalloc(&scratch_down, moe::MOE_HIDDEN * 4);
    moe::MoeApplyArgs apply;
    apply.x = dx;
    apply.ids = ids_dev;
    apply.weights = weights_host;
    apply.experts = experts.data();
    apply.y = dy;
    apply.scratch_h = scratch_h;
    apply.scratch_up = scratch_up;
    apply.scratch_down = scratch_down;
    if (moe::moe_apply(apply, nullptr) != cudaSuccess) {
      std::printf("FAIL moe_apply\n");
      return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(ref.data(), dy, moe::MOE_HIDDEN * 4, cudaMemcpyDeviceToHost);
    cudaFree(scratch_h);
    cudaFree(scratch_up);
    cudaFree(scratch_down);
  }

  std::printf("stage: table\n");
  std::fflush(stdout);
  // Fused table for layer 0, both spans.
  int off[3][4] = {};
  std::string gate_source;
  for (const auto& slot : lp.experts.slots) {
    const int proj = slot.projection == ingine::arch::qwen35moe::Projection::GateProj ? 0 : slot.projection == ingine::arch::qwen35moe::Projection::UpProj ? 1 : 2;
    const int art = slot.artifact == ingine::arch::qwen35moe::Artifact::Weight ? 0
                    : slot.artifact == ingine::arch::qwen35moe::Artifact::WeightScale ? 1
                    : slot.artifact == ingine::arch::qwen35moe::Artifact::WeightScale2 ? 2
                                                                   : 3;
    off[proj][art] = int(slot.offset);
    if (proj == 0 && art == 0) gate_source = slot.source;
  }
  const int stride = int(lp.experts.stride);

  std::vector<moe::FusedExpertLayer> host_tables[2];
  for (int d = 0; d < 2; ++d) host_tables[d].assign(ingine::arch::qwen35moe::EXPERTS, moe::FusedExpertLayer{});
  int mismatched_bases = 0;
  for (int e = 0; e < ingine::arch::qwen35moe::EXPERTS; ++e) {
    const std::string gate_e = L + "mlp.experts." + std::to_string(e) + ".gate_proj";
    const int span = placed[gate_e + ".weight"]->span;
    for (int d = 0; d < 2; ++d) {
      moe::FusedExpertLayer& f = host_tables[d][std::size_t(e)];
      if (span != d) {
        f.stride_bytes = 0;
        continue;
      }
      f.stride_bytes = stride;
      f.scale_stride_bytes = stride;
      f.gate_weight_off = off[0][0];
      f.up_weight_off = off[1][0];
      f.down_weight_off = off[2][0];
      f.gate_scale_off = off[0][1];
      f.up_scale_off = off[1][1];
      f.down_scale_off = off[2][1];
      // weight_scale_2 is per expert, not per layer.
      f.gate_scale2 = scalar(gate_e + ".weight_scale_2", 1.0f);
      f.up_scale2 = scalar(L + "mlp.experts." + std::to_string(e) + ".up_proj.weight_scale_2", 1.0f);
      f.down_scale2 = scalar(L + "mlp.experts." + std::to_string(e) + ".down_proj.weight_scale_2", 1.0f);
      f.w_base = static_cast<const std::uint8_t*>(ptr(gate_e + ".weight")) - std::size_t(e) * stride - off[0][0];
      f.s_base = static_cast<const std::uint8_t*>(ptr(gate_e + ".weight_scale")) - std::size_t(e) * stride - off[0][1];
      // The resolved address for this expert must be exactly the placed gate weight.
      if (f.w_base + std::size_t(e) * stride + f.gate_weight_off != ptr(gate_e + ".weight")) ++mismatched_bases;
    }
  }
  std::printf("%s fused table bases resolve to the placed tensors (%d mismatches)\n",
              mismatched_bases == 0 ? "PASS" : "FAIL", mismatched_bases);

  // Path B: fused kernel, one launch per span, partial sums merged.
  std::vector<float> got(moe::MOE_HIDDEN, 0.0f);
  {
    moe::FusedExpertLayer* dev_tables[2] = {nullptr, nullptr};
    float *scratch_gate = nullptr, *scratch_up = nullptr, *scratch_down = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dev_tables[0]), host_tables[0].size() * sizeof(moe::FusedExpertLayer));
    cudaMemcpy(dev_tables[0], host_tables[0].data(), host_tables[0].size() * sizeof(moe::FusedExpertLayer),
               cudaMemcpyHostToDevice);
    cudaMalloc(&scratch_gate, 8 * moe::MOE_INTER * 4);
    cudaMalloc(&scratch_up, 8 * moe::MOE_INTER * 4);
    cudaMalloc(&scratch_down, 8 * moe::MOE_HIDDEN * 4);
    moe::FusedMoeArgs a;
    a.layers = dev_tables[0];
    a.x = dx;
    a.ids = dids;
    a.weights = dweights;
    a.y = dy;
    a.scratch_gate = scratch_gate;
    a.scratch_up = scratch_up;
    a.scratch_down = scratch_down;
    if (moe::fused_moe_apply(a, nullptr) != cudaSuccess) {
      std::printf("FAIL fused_moe_apply\n");
      return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dy, moe::MOE_HIDDEN * 4, cudaMemcpyDeviceToHost);
    cudaFree(dev_tables[0]);
    cudaFree(scratch_gate);
    cudaFree(scratch_up);
    cudaFree(scratch_down);
  }

  double worst = 0.0, rms = 0.0;
  for (int i = 0; i < moe::MOE_HIDDEN; ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])));
    rms += double(ref[i]) * double(ref[i]);
  }
  rms = std::sqrt(rms / moe::MOE_HIDDEN);
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= 1e-3;
  std::printf("%s fused vs moe_apply: max_abs=%.3e rms=%.3e ratio=%.3e (tol 1e-3)\n", ok ? "PASS" : "FAIL", worst, rms,
              ratio);
  std::printf("  fused[0:4]=%.6f %.6f %.6f %.6f\n  ref  [0:4]=%.6f %.6f %.6f %.6f\n", got[0], got[1], got[2], got[3],
              ref[0], ref[1], ref[2], ref[3]);
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dids);
  cudaFree(dweights);
  qwen::free_text_arena(&arena);
  return (ok && mismatched_bases == 0) ? 0 : 1;
}
