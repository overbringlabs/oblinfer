// Verify-width (n-token) batch kernels against FP64 oracles, plus the n = 1 agreement with the single-token
// paths they are meant to replace:
//
//   batch_gemv   fp8 / nvfp4 / bf16 GEMVs over n rows   vs their FP64 host mirror, n = 1 vs kernels/quant
//   batch_moe    dedup + per-unique-expert gate/up/down vs the FP64 host mirror, n = 1 vs fused_moe_apply
//   batch_attn   n queries over one KV cache            vs the FP64 host mirror, n = 1 vs decode_attention
//   batch_gdn    n staged delta-rule updates            vs the FP64 host mirror, n = 1 vs gdn_recurrent_step
//
// Everything here runs on synthetic bytes at the checkpoint's shapes, so the numerics are checked without a
// 23 GB load; test_qwen35moe_batch_chain.cpp repeats the MoE check on the real weights. The device runs need
// sm_120 (the pack builds for sm_120a); the binary skips elsewhere.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/batch/batch_attn.h"
#include "arch/qwen35moe/kernels/batch/batch_gdn.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"
#include "arch/qwen35moe/kernels/batch/batch_moe.h"
#include "arch/qwen35moe/kernels/moe/fused_experts.h"
#include "kernels/gdn/gdn.h"
#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::test;
namespace bk = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;
namespace gdn = ingine::kernels::gdn;

namespace {

bool ok_all = true;
long checks_run = 0;

void check(bool cond, const std::string& what) {
  ++checks_run;
  std::printf("%s %s\n", cond ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  ok_all = ok_all && cond;
}

bool have_sm120() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  return p.major == 12 && p.minor == 0;
}

// Relative-to-rms comparison of two device/host float buffers over the first `count` elements.
double worst_ratio(const float* got, const float* want, std::size_t count, double* rms_out) {
  double worst = 0.0, sq = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - double(want[i])));
    sq += double(want[i]) * double(want[i]);
  }
  const double rms = std::sqrt(sq / double(count ? count : 1));
  if (rms_out) *rms_out = rms;
  return rms > 0.0 ? worst / rms : worst;
}

double worst_ratio(const std::vector<float>& got, const std::vector<float>& want, std::size_t count,
                   double* rms_out) {
  return worst_ratio(got.data(), want.data(), count, rms_out);
}

// ---------------------------------------------------------------------- gemvs

void test_gemvs() {
  Rng rng(2026);
  const int nmax = bk::BATCH_MAX;

  // fp8 e4m3 [512, 2048]: an in_proj_z-sized projection.
  {
    const int rows = 512, cols = bk::MOE_HIDDEN;
    std::vector<std::uint8_t> hw(std::size_t(rows) * cols);
    for (auto& b : hw) b = std::uint8_t(rng.next() & 0x7Eu);
    std::vector<float> hx(std::size_t(nmax) * cols);
    for (auto& v : hx) v = float(rng.normal());
    std::uint8_t* dw = nullptr;
    float *dx = nullptr, *dy = nullptr, *dy1 = nullptr;
    INGINE_TEST_CUDA(cudaMalloc(&dw, hw.size()));
    INGINE_TEST_CUDA(cudaMalloc(&dx, hx.size() * 4));
    INGINE_TEST_CUDA(cudaMalloc(&dy, std::size_t(nmax) * rows * 4));
    INGINE_TEST_CUDA(cudaMalloc(&dy1, std::size_t(rows) * 4));
    INGINE_TEST_CUDA(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));

    // n = 1 first, through the shipped single-token kernel, so the batch rows can be compared against it.
    quant::Fp8GemvArgs s;
    s.w = dw;
    s.scale = 0.037f;
    s.x = dx;
    s.y = dy1;
    s.rows = rows;
    s.cols = cols;
    check(quant::fp8_gemv(s, nullptr) == cudaSuccess, "single-token fp8_gemv launch");
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> single(rows);
    INGINE_TEST_CUDA(cudaMemcpy(single.data(), dy1, single.size() * 4, cudaMemcpyDeviceToHost));

    for (int nn = 1; nn <= nmax; ++nn) {
      bk::Fp8GemvBatchArgs a;
      a.w = dw;
      a.scale = 0.037f;
      a.x = dx;
      a.y = dy;
      a.rows = rows;
      a.cols = cols;
      a.n = nn;
      check(bk::fp8_gemv_batch(a, nullptr) == cudaSuccess, "fp8_gemv_batch launch n=" + std::to_string(nn));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      std::vector<float> got(std::size_t(nn) * rows);
      INGINE_TEST_CUDA(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<float> want(std::size_t(nn) * rows);
      bk::Fp8GemvBatchArgs r = a;
      r.w = hw.data();
      r.x = hx.data();
      r.y = want.data();
      bk::fp8_gemv_batch_cpu(r);
      double rms = 0.0;
      const double ratio = worst_ratio(got, want, want.size(), &rms);
      check(ratio <= 2e-5, "fp8_gemv_batch n=" + std::to_string(nn) + " vs FP64: ratio=" + std::to_string(ratio) +
                               " rms=" + std::to_string(rms));
      if (nn == 1) {
        const double one = worst_ratio(got, single, single.size(), nullptr);
        check(one <= 1e-6, "fp8_gemv_batch n=1 matches the single-token kernel: ratio=" + std::to_string(one));
      }
    }
    {
      bk::Fp8GemvBatchArgs bad;
      bad.w = dw;
      bad.x = dx;
      bad.y = dy;
      bad.rows = rows;
      bad.cols = cols;
      bad.n = 0;
      check(bk::fp8_gemv_batch(bad, nullptr) == cudaErrorInvalidValue, "fp8_gemv_batch rejects n=0");
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dy1);
  }

  // nvfp4 [2048, 512]: a down_proj-sized projection.
  {
    const int rows = bk::MOE_HIDDEN, cols = bk::MOE_INTER;
    std::vector<std::uint8_t> hw(std::size_t(rows) * (cols / 2)), hs(std::size_t(rows) * (cols / 16));
    for (auto& b : hw) b = std::uint8_t(rng.next() & 0xFFu);
    for (auto& b : hs) b = std::uint8_t(rng.next() & 0x7Eu);
    std::vector<float> hx(std::size_t(nmax) * cols);
    for (auto& v : hx) v = float(rng.normal());
    std::uint8_t *dw = nullptr, *ds = nullptr;
    float *dx = nullptr, *dy = nullptr;
    INGINE_TEST_CUDA(cudaMalloc(&dw, hw.size()));
    INGINE_TEST_CUDA(cudaMalloc(&ds, hs.size()));
    INGINE_TEST_CUDA(cudaMalloc(&dx, hx.size() * 4));
    INGINE_TEST_CUDA(cudaMalloc(&dy, std::size_t(nmax) * rows * 4));
    INGINE_TEST_CUDA(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    for (int nn = 1; nn <= nmax; ++nn) {
      bk::Nvfp4GemvBatchArgs a;
      a.w = dw;
      a.scales = ds;
      a.scale = 0.11f;
      a.x = dx;
      a.y = dy;
      a.rows = rows;
      a.cols = cols;
      a.n = nn;
      check(bk::nvfp4_gemv_batch(a, nullptr) == cudaSuccess, "nvfp4_gemv_batch launch n=" + std::to_string(nn));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      std::vector<float> got(std::size_t(nn) * rows);
      INGINE_TEST_CUDA(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<float> want(std::size_t(nn) * rows);
      bk::Nvfp4GemvBatchArgs r = a;
      r.w = hw.data();
      r.scales = hs.data();
      r.x = hx.data();
      r.y = want.data();
      bk::nvfp4_gemv_batch_cpu(r);
      double rms = 0.0;
      const double ratio = worst_ratio(got, want, want.size(), &rms);
      check(ratio <= 2e-5, "nvfp4_gemv_batch n=" + std::to_string(nn) + " vs FP64: ratio=" +
                               std::to_string(ratio) + " rms=" + std::to_string(rms));
    }
    cudaFree(dw);
    cudaFree(ds);
    cudaFree(dx);
    cudaFree(dy);
  }

  // bf16 [256, 2048]: a router-sized projection.
  {
    const int rows = 256, cols = bk::MOE_HIDDEN;
    std::vector<__nv_bfloat16> hw(std::size_t(rows) * cols);
    for (auto& b : hw) b = float_to_bf16(float(rng.normal()) * 0.05f);
    std::vector<float> hx(std::size_t(nmax) * cols);
    for (auto& v : hx) v = float(rng.normal());
    __nv_bfloat16* dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    INGINE_TEST_CUDA(cudaMalloc(&dw, hw.size() * 2));
    INGINE_TEST_CUDA(cudaMalloc(&dx, hx.size() * 4));
    INGINE_TEST_CUDA(cudaMalloc(&dy, std::size_t(nmax) * rows * 4));
    INGINE_TEST_CUDA(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    for (int nn = 1; nn <= nmax; ++nn) {
      bk::Bf16GemvBatchArgs a;
      a.w = dw;
      a.x = dx;
      a.y = dy;
      a.rows = rows;
      a.cols = cols;
      a.n = nn;
      check(bk::bf16_gemv_batch(a, nullptr) == cudaSuccess, "bf16_gemv_batch launch n=" + std::to_string(nn));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      std::vector<float> got(std::size_t(nn) * rows);
      INGINE_TEST_CUDA(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<float> want(std::size_t(nn) * rows);
      bk::Bf16GemvBatchArgs r = a;
      r.w = hw.data();
      r.x = hx.data();
      r.y = want.data();
      bk::bf16_gemv_batch_cpu(r);
      double rms = 0.0;
      const double ratio = worst_ratio(got, want, want.size(), &rms);
      check(ratio <= 2e-5, "bf16_gemv_batch n=" + std::to_string(nn) + " vs FP64: ratio=" +
                               std::to_string(ratio) + " rms=" + std::to_string(rms));
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
  }
}

// ----------------------------------------------------------------------- moe

// One expert's packed block, laid out the way placement packs the checkpoint: gate, up and down weights
// followed by their e4m3 block scales.
struct ExpertBytes {
  static constexpr std::size_t kW = std::size_t(bk::MOE_INTER) * (bk::MOE_HIDDEN / 2);  // gate or up weight
  static constexpr std::size_t kS = std::size_t(bk::MOE_INTER) * (bk::MOE_HIDDEN / 16);  // its block scales
  static constexpr std::size_t kDw = std::size_t(bk::MOE_HIDDEN) * (bk::MOE_INTER / 2);
  static constexpr std::size_t kDs = std::size_t(bk::MOE_HIDDEN) * (bk::MOE_INTER / 16);
  static constexpr std::size_t kStride = kW + kS + kW + kS + kDw + kDs;

  std::vector<std::uint8_t> bytes;
  std::vector<float> scale2;  // gate, up, down per expert
};

struct Offsets {
  std::int64_t gate_w, up_w, down_w, gate_s, up_s, down_s;
};

Offsets expert_offsets() {
  Offsets o;
  o.gate_w = 0;
  o.up_w = std::int64_t(ExpertBytes::kW + ExpertBytes::kS);
  o.down_w = std::int64_t(2 * (ExpertBytes::kW + ExpertBytes::kS));
  o.gate_s = std::int64_t(ExpertBytes::kW);
  o.up_s = std::int64_t(ExpertBytes::kW + ExpertBytes::kS + ExpertBytes::kW);
  o.down_s = std::int64_t(2 * (ExpertBytes::kW + ExpertBytes::kS) + ExpertBytes::kDw);
  return o;
}

void fill_expert_bytes(ExpertBytes& eb, int experts, Rng& rng) {
  eb.bytes.assign(std::size_t(experts) * ExpertBytes::kStride, 0);
  eb.scale2.assign(std::size_t(experts) * 3, 0.0f);
  // Codes avoid the e4m3 NaN patterns so the hardware decode and the reference agree byte for byte.
  for (auto& b : eb.bytes) b = std::uint8_t(rng.next() & 0x7Eu);
  for (auto& v : eb.scale2) v = float(rng.uniform(0.02, 0.2));
}

void test_batch_moe() {
  Rng rng(7717);
  constexpr int kExperts = 8;
  ExpertBytes eb;
  fill_expert_bytes(eb, kExperts, rng);
  const Offsets off = expert_offsets();
  const int nmax = bk::BATCH_MAX;

  std::uint8_t* dbytes = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&dbytes, eb.bytes.size()));
  INGINE_TEST_CUDA(cudaMemcpy(dbytes, eb.bytes.data(), eb.bytes.size(), cudaMemcpyHostToDevice));

  auto make_table = [&](const std::uint8_t* base, int remote) {
    std::vector<bk::BatchExpertBlock> t(kExperts);
    for (int e = 0; e < kExperts; ++e) {
      bk::BatchExpertBlock& b = t[std::size_t(e)];
      if (e == remote) continue;  // stride 0: the kernel sees an expert hosted on the other card
      b.w_base = base;
      b.s_base = base;
      b.stride_bytes = std::int64_t(ExpertBytes::kStride);
      b.scale_stride_bytes = std::int64_t(ExpertBytes::kStride);
      b.gate_w = off.gate_w;
      b.up_w = off.up_w;
      b.down_w = off.down_w;
      b.gate_s = off.gate_s;
      b.up_s = off.up_s;
      b.down_s = off.down_s;
      b.gate_scale2 = eb.scale2[std::size_t(e) * 3 + 0];
      b.up_scale2 = eb.scale2[std::size_t(e) * 3 + 1];
      b.down_scale2 = eb.scale2[std::size_t(e) * 3 + 2];
    }
    return t;
  };

  float *dx = nullptr, *dy = nullptr;
  int* dids = nullptr;
  float *dwts = nullptr, *dg = nullptr, *du = nullptr, *dd = nullptr;
  int* dplan = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&dx, std::size_t(nmax) * bk::MOE_HIDDEN * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dy, std::size_t(nmax) * bk::MOE_HIDDEN * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dids, bk::BATCH_PAIRS * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dwts, bk::BATCH_PAIRS * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dplan, (3 * bk::BATCH_PAIRS + 1 + bk::BATCH_PAIRS * bk::BATCH_MAX) * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dg, std::size_t(bk::BATCH_PAIRS) * bk::MOE_INTER * 4));
  INGINE_TEST_CUDA(cudaMalloc(&du, std::size_t(bk::BATCH_PAIRS) * bk::MOE_INTER * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dd, std::size_t(bk::BATCH_PAIRS) * bk::MOE_HIDDEN * 4));

  bk::BatchMoePlan plan;
  plan.unique_ids = dplan;
  plan.unique_count = dplan + bk::BATCH_PAIRS;
  plan.pair_unique = dplan + bk::BATCH_PAIRS + 1;
  plan.rows = dplan + 2 * bk::BATCH_PAIRS + 1;
  plan.pair_of = dplan + 3 * bk::BATCH_PAIRS + 1;

  std::vector<float> host_x(std::size_t(nmax) * bk::MOE_HIDDEN);
  for (auto& v : host_x) v = float(rng.normal());
  INGINE_TEST_CUDA(cudaMemcpy(dx, host_x.data(), host_x.size() * 4, cudaMemcpyHostToDevice));

  // Three routing patterns: every token picks the same eight experts (maximum sharing), a rotation (sharing
  // by four), and a rotation whose expert 3 is marked remote, which is how a cross-device expert arrives.
  const int patterns[3][bk::BATCH_PAIRS] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
      {0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 7, 0, 1, 3, 4, 5, 6, 7, 0, 1, 2},
      {0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 7, 0, 1, 3, 4, 5, 6, 7, 0, 1, 2},
  };

  for (int pt = 0; pt < 3; ++pt) {
    const int remote = pt == 2 ? 3 : -1;
    auto host_table = make_table(eb.bytes.data(), remote);
    auto dev_table = make_table(dbytes, remote);
    bk::BatchExpertBlock* dtab = nullptr;
    INGINE_TEST_CUDA(cudaMalloc(&dtab, sizeof(bk::BatchExpertBlock) * kExperts));
    INGINE_TEST_CUDA(cudaMemcpy(dtab, dev_table.data(), sizeof(bk::BatchExpertBlock) * kExperts,
                                cudaMemcpyHostToDevice));

    std::vector<int> ids(bk::BATCH_PAIRS);
    std::vector<float> wts(bk::BATCH_PAIRS);
    Rng wr(4242 + pt);
    for (int i = 0; i < bk::BATCH_PAIRS; ++i) {
      ids[std::size_t(i)] = patterns[pt][std::size_t(i)];
      wts[std::size_t(i)] = float(wr.uniform(0.02, 0.3));
    }
    INGINE_TEST_CUDA(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(dwts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));

    for (int nn = 1; nn <= nmax; ++nn) {
      bk::BatchMoeArgs a;
      a.layers = dtab;
      a.x = dx;
      a.ids = dids;
      a.weights = dwts;
      a.y = dy;
      a.scratch_gate = dg;
      a.scratch_up = du;
      a.scratch_down = dd;
      a.plan = plan;
      a.n = nn;
      check(bk::batch_moe_apply(a, nullptr) == cudaSuccess,
            "batch_moe_apply launch n=" + std::to_string(nn) + " pattern=" + std::to_string(pt));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());

      bk::BatchMoePlanHost hp;
      bk::batch_moe_plan_cpu(ids.data(), nn, &hp);
      int dev_count = -1;
      std::vector<int> got_u(bk::BATCH_PAIRS), got_pu(bk::BATCH_PAIRS), got_rows(bk::BATCH_PAIRS);
      INGINE_TEST_CUDA(cudaMemcpy(&dev_count, plan.unique_count, 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(got_u.data(), plan.unique_ids, got_u.size() * 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(got_pu.data(), plan.pair_unique, got_pu.size() * 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(got_rows.data(), plan.rows, got_rows.size() * 4, cudaMemcpyDeviceToHost));
      bool plan_ok = dev_count == hp.unique_count;
      for (int u = 0; u < dev_count && u < hp.unique_count; ++u) {
        plan_ok = plan_ok && got_u[std::size_t(u)] == hp.unique_ids[std::size_t(u)];
        plan_ok = plan_ok && got_rows[std::size_t(u)] == hp.rows[std::size_t(u)];
      }
      for (int d = 0; d < nn * bk::MOE_TOP_K; ++d) plan_ok = plan_ok && got_pu[std::size_t(d)] == hp.pair_unique[std::size_t(d)];
      check(plan_ok, "dedup plan n=" + std::to_string(nn) + " pattern=" + std::to_string(pt) +
                         " unique device=" + std::to_string(dev_count) + " host=" + std::to_string(hp.unique_count));

      std::vector<float> got(std::size_t(nn) * bk::MOE_HIDDEN);
      INGINE_TEST_CUDA(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<float> want(std::size_t(nn) * bk::MOE_HIDDEN, 0.0f);
      bk::BatchMoeArgs r = a;
      r.layers = host_table.data();
      r.x = host_x.data();
      r.ids = ids.data();
      r.weights = wts.data();
      r.y = want.data();
      bk::batch_moe_apply_cpu(r, hp);
      double rms = 0.0;
      const double ratio = worst_ratio(got, want, want.size(), &rms);
      check(ratio <= 1e-5, "batch_moe n=" + std::to_string(nn) + " pattern=" + std::to_string(pt) +
                               " vs FP64: ratio=" + std::to_string(ratio) + " rms=" + std::to_string(rms));

      if (nn == 1) {  // the shipped single-token fused kernel over the same bytes
        std::vector<bk::FusedExpertLayer> ft(kExperts);
        for (int e = 0; e < kExperts; ++e) {
          bk::FusedExpertLayer& f = ft[std::size_t(e)];
          if (e == remote) continue;
          f.stride_bytes = int(ExpertBytes::kStride);
          f.scale_stride_bytes = int(ExpertBytes::kStride);
          f.gate_weight_off = int(off.gate_w);
          f.up_weight_off = int(off.up_w);
          f.down_weight_off = int(off.down_w);
          f.gate_scale_off = int(off.gate_s);
          f.up_scale_off = int(off.up_s);
          f.down_scale_off = int(off.down_s);
          f.gate_scale2 = eb.scale2[std::size_t(e) * 3 + 0];
          f.up_scale2 = eb.scale2[std::size_t(e) * 3 + 1];
          f.down_scale2 = eb.scale2[std::size_t(e) * 3 + 2];
          f.w_base = dbytes;
          f.s_base = dbytes;
        }
        bk::FusedExpertLayer* dft = nullptr;
        float *fy = nullptr, *fg = nullptr, *fu = nullptr, *fd = nullptr;
        INGINE_TEST_CUDA(cudaMalloc(&dft, sizeof(bk::FusedExpertLayer) * kExperts));
        INGINE_TEST_CUDA(cudaMemcpy(dft, ft.data(), sizeof(bk::FusedExpertLayer) * kExperts,
                                    cudaMemcpyHostToDevice));
        INGINE_TEST_CUDA(cudaMalloc(&fy, std::size_t(bk::MOE_HIDDEN) * 4));
        INGINE_TEST_CUDA(cudaMalloc(&fg, std::size_t(bk::MOE_TOP_K) * bk::MOE_INTER * 4));
        INGINE_TEST_CUDA(cudaMalloc(&fu, std::size_t(bk::MOE_TOP_K) * bk::MOE_INTER * 4));
        INGINE_TEST_CUDA(cudaMalloc(&fd, std::size_t(bk::MOE_TOP_K) * bk::MOE_HIDDEN * 4));
        bk::FusedMoeArgs fa;
        fa.layers = dft;
        fa.x = dx;
        fa.ids = dids;
        fa.weights = dwts;
        fa.y = fy;
        fa.scratch_gate = fg;
        fa.scratch_up = fu;
        fa.scratch_down = fd;
        check(bk::fused_moe_apply(fa, nullptr) == cudaSuccess, "fused_moe_apply launch (single-token ref)");
        INGINE_TEST_CUDA(cudaDeviceSynchronize());
        std::vector<float> one(bk::MOE_HIDDEN);
        INGINE_TEST_CUDA(cudaMemcpy(one.data(), fy, one.size() * 4, cudaMemcpyDeviceToHost));
        const double ratio1 = worst_ratio(got, one, one.size(), nullptr);
        check(ratio1 <= 1e-5, "batch_moe n=1 equals fused single-token path: ratio=" + std::to_string(ratio1));
        cudaFree(dft);
        cudaFree(fy);
        cudaFree(fg);
        cudaFree(fu);
        cudaFree(fd);
      }
    }
    cudaFree(dtab);
  }
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dids);
  cudaFree(dwts);
  cudaFree(dplan);
  cudaFree(dg);
  cudaFree(du);
  cudaFree(dd);
  cudaFree(dbytes);
}

// ------------------------------------------------------------------- attention

void fill_cache(void* base, bk::KvType type, int max_pos, Rng& rng) {
  const std::size_t values = std::size_t(max_pos) * bk::ATTN_KV_HEADS * bk::ATTN_HEAD_DIM;
  if (type == bk::KvType::Bf16) {
    std::vector<__nv_bfloat16> h(values);
    for (auto& v : h) v = float_to_bf16(float(rng.normal()) * 0.6f);
    INGINE_TEST_CUDA(cudaMemcpy(base, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    return;
  }
  std::vector<std::int8_t> v(values);
  for (auto& b : v) b = std::int8_t((rng.next() % 250) - 125);
  std::vector<__half> scales(values / bk::KV_BLOCK);
  for (auto& s : scales) s = __float2half(float(rng.uniform(0.01, 0.4)));
  INGINE_TEST_CUDA(cudaMemcpy(base, v.data(), v.size(), cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(static_cast<std::int8_t*>(base) + values, scales.data(), scales.size() * 2,
                              cudaMemcpyHostToDevice));
}

void test_batch_attn() {
  Rng rng(31337);
  const int nmax = bk::BATCH_MAX;
  const std::size_t qn = std::size_t(nmax) * bk::ATTN_HEADS * bk::ATTN_HEAD_DIM;
  const std::size_t kn = std::size_t(nmax) * bk::ATTN_KV_HEADS * bk::ATTN_HEAD_DIM;
  float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *dref = nullptr, *dother = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&dq, qn * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dk, kn * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dv, kn * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dout, qn * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dref, qn * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dother, qn * 4));
  std::vector<float> hq(qn), hk(kn), hv(kn);
  for (auto& v : hq) v = float(rng.normal());
  for (auto& v : hk) v = float(rng.normal()) * 0.7f;
  for (auto& v : hv) v = float(rng.normal()) * 0.7f;
  INGINE_TEST_CUDA(cudaMemcpy(dq, hq.data(), qn * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(dk, hk.data(), kn * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(dv, hv.data(), kn * 4, cudaMemcpyHostToDevice));

  const int depths[] = {1, 9, 64, 133, 300};
  for (int dt = 0; dt < 5; ++dt) {
    const int pos0 = depths[dt] - 1;
    const int max_pos = 384;
    const int splits = pos0 + 1 < 64 ? 1 : 7;
    for (int ty = 0; ty < 2; ++ty) {
      const bk::KvType type = ty == 0 ? bk::KvType::Q8_0 : bk::KvType::Bf16;
      void *kc = nullptr, *vc = nullptr, *k2 = nullptr, *v2 = nullptr;
      const std::size_t cb = bk::kv_cache_bytes(type, max_pos);
      INGINE_TEST_CUDA(cudaMalloc(&kc, cb));
      INGINE_TEST_CUDA(cudaMalloc(&vc, cb));
      INGINE_TEST_CUDA(cudaMalloc(&k2, cb));
      INGINE_TEST_CUDA(cudaMalloc(&v2, cb));
      fill_cache(kc, type, max_pos, rng);
      fill_cache(vc, type, max_pos, rng);
      INGINE_TEST_CUDA(cudaMemcpy(k2, kc, cb, cudaMemcpyDeviceToDevice));
      INGINE_TEST_CUDA(cudaMemcpy(v2, vc, cb, cudaMemcpyDeviceToDevice));

      bk::BatchSplitState* partial = nullptr;
      INGINE_TEST_CUDA(cudaMalloc(&partial, sizeof(bk::BatchSplitState) * bk::ATTN_HEADS * splits));
      // Pristine host copies of the cache bytes: the FP64 reference runs entirely on host memory and writes
      // the same n rows the device store does, so both sides see identical bytes.
      std::vector<std::uint8_t> hc_host(cb), vc_host(cb);
      INGINE_TEST_CUDA(cudaMemcpy(hc_host.data(), kc, cb, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(vc_host.data(), vc, cb, cudaMemcpyDeviceToHost));
      std::vector<float> ref_out(qn);
      std::vector<float> got(qn);
      std::vector<std::uint8_t> hkc, hvc;

      for (int nn = 1; nn <= nmax; ++nn) {
        bk::BatchAttnArgs a;
        a.q = dq;
        a.k_new = dk;
        a.v_new = dv;
        a.k_cache = kc;
        a.v_cache = vc;
        a.kv_type_k = type;
        a.kv_type_v = type;
        a.out = dout;
        a.pos0 = pos0;
        a.max_pos = max_pos;
        a.partial = partial;
        a.splits = splits;
        a.n = nn;
        check(bk::batch_attention(a, nullptr) == cudaSuccess,
              "batch_attention launch n=" + std::to_string(nn) + " depth=" + std::to_string(pos0 + 1));
        INGINE_TEST_CUDA(cudaDeviceSynchronize());
        INGINE_TEST_CUDA(cudaMemcpy(got.data(), dout, qn * 4, cudaMemcpyDeviceToHost));

        hkc = hc_host;
        hvc = vc_host;
        std::fill(ref_out.begin(), ref_out.end(), 0.0f);
        bk::BatchAttnArgs r;
        r.q = hq.data();
        r.k_new = hk.data();
        r.v_new = hv.data();
        r.k_cache = hkc.data();
        r.v_cache = hvc.data();
        r.kv_type_k = type;
        r.kv_type_v = type;
        r.out = ref_out.data();
        r.pos0 = pos0;
        r.max_pos = max_pos;
        r.head_count = bk::ATTN_HEADS;
        r.n = nn;
        bk::batch_attention_cpu(r);
        double rms = 0.0;
        const double ratio = worst_ratio(got, ref_out, std::size_t(nn) * bk::ATTN_HEADS * bk::ATTN_HEAD_DIM, &rms);
        check(ratio <= 1e-5, "batch_attention n=" + std::to_string(nn) + " depth=" + std::to_string(pos0 + 1) +
                                 " kv=" + (ty == 0 ? "q8_0" : "bf16") + " vs FP64: ratio=" + std::to_string(ratio) +
                                 " rms=" + std::to_string(rms));
      }

      // n = 1 against the shipped single-token flash decode over the same cache bytes.
      {
        bk::BatchAttnArgs a;
        a.q = dq;
        a.k_new = dk;
        a.v_new = dv;
        a.k_cache = kc;
        a.v_cache = vc;
        a.kv_type_k = type;
        a.kv_type_v = type;
        a.out = dout;
        a.pos0 = pos0;
        a.max_pos = max_pos;
        a.partial = partial;
        a.splits = splits;
        a.n = 1;
        INGINE_TEST_CUDA(cudaDeviceSynchronize());
        bk::batch_attention(a, nullptr);
        INGINE_TEST_CUDA(cudaDeviceSynchronize());
        std::vector<float> mine(std::size_t(bk::ATTN_HEADS) * bk::ATTN_HEAD_DIM);
        INGINE_TEST_CUDA(cudaMemcpy(mine.data(), dout, mine.size() * 4, cudaMemcpyDeviceToHost));

        bk::DecodeAttnArgs d;
        d.q = dq;
        d.k_new = dk;
        d.v_new = dv;
        d.k_cache = k2;
        d.v_cache = v2;
        d.kv_type_k = type;
        d.kv_type_v = type;
        d.out = dother;
        d.pos = pos0;
        d.max_pos = max_pos;
        d.splits = splits;
        d.partial = reinterpret_cast<bk::SplitState*>(partial);
        check(bk::decode_attention(d, nullptr) == cudaSuccess, "decode_attention launch (single-token ref)");
        INGINE_TEST_CUDA(cudaDeviceSynchronize());
        std::vector<float> theirs(std::size_t(bk::ATTN_HEADS) * bk::ATTN_HEAD_DIM);
        INGINE_TEST_CUDA(cudaMemcpy(theirs.data(), dother, theirs.size() * 4, cudaMemcpyDeviceToHost));
        double rms = 0.0;
        const double ratio = worst_ratio(mine, theirs, theirs.size(), &rms);
        check(ratio <= 1e-5, "batch_attention n=1 equals decode_attention: ratio=" + std::to_string(ratio) +
                                 " rms=" + std::to_string(rms));
      }
      cudaFree(partial);
      cudaFree(kc);
      cudaFree(vc);
      cudaFree(k2);
      cudaFree(v2);
    }
  }
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dout);
  cudaFree(dref);
  cudaFree(dother);
}

// ----------------------------------------------------------------- delta rule

void test_batch_gdn() {
  Rng rng(999);
  const int heads = 32;
  const int nmax = bk::BATCH_MAX;
  const std::size_t qk = std::size_t(nmax) * gdn::GDN_K_HEADS * gdn::GDN_K_DIM;
  const std::size_t vv = std::size_t(nmax) * heads * gdn::GDN_V_DIM;
  const std::size_t st = std::size_t(heads) * gdn::GDN_K_DIM * gdn::GDN_V_DIM;
  std::vector<float> hq(qk), hk(qk), hv(vv), hb(std::size_t(nmax) * heads), hg(std::size_t(nmax) * heads);
  for (auto& v : hq) v = float(rng.normal());
  for (auto& v : hk) v = float(rng.normal());
  for (auto& v : hv) v = float(rng.normal());
  for (auto& v : hb) v = float(rng.uniform(0.0, 1.0));
  for (auto& v : hg) v = float(rng.uniform(-4.0, -0.2));
  std::vector<float> hstate(st);
  for (auto& v : hstate) v = float(rng.normal());

  float *dq = nullptr, *dk = nullptr, *dv = nullptr, *db = nullptr, *dg = nullptr, *dst = nullptr, *dout = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&dq, qk * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dk, qk * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dv, vv * 4));
  INGINE_TEST_CUDA(cudaMalloc(&db, hb.size() * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dg, hg.size() * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dst, st * 4));
  INGINE_TEST_CUDA(cudaMalloc(&dout, vv * 4));
  INGINE_TEST_CUDA(cudaMemcpy(dq, hq.data(), qk * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(dk, hk.data(), qk * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(dv, hv.data(), vv * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(dg, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice));

  for (int nn = 1; nn <= nmax; ++nn) {
    INGINE_TEST_CUDA(cudaMemcpy(dst, hstate.data(), st * 4, cudaMemcpyHostToDevice));
    bk::GdnStepsArgs a;
    a.q = dq;
    a.k = dk;
    a.v = dv;
    a.beta = db;
    a.g = dg;
    a.state = dst;
    a.out = dout;
    a.v_heads = heads;
    a.n = nn;
    check(bk::gdn_recurrent_steps(a, nullptr) == cudaSuccess,
          "gdn_recurrent_steps launch n=" + std::to_string(nn));
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> got_state(st), got_out(std::size_t(nn) * heads * gdn::GDN_V_DIM);
    INGINE_TEST_CUDA(cudaMemcpy(got_state.data(), dst, st * 4, cudaMemcpyDeviceToHost));
    INGINE_TEST_CUDA(cudaMemcpy(got_out.data(), dout, got_out.size() * 4, cudaMemcpyDeviceToHost));

    std::vector<double> ref_state(st), ref_out(got_out.size());
    for (std::size_t i = 0; i < st; ++i) ref_state[i] = double(hstate[i]);
    bk::GdnStepsArgs r;
    r.q = hq.data();
    r.k = hk.data();
    r.v = hv.data();
    r.beta = hb.data();
    r.g = hg.data();
    r.v_heads = heads;
    r.n = nn;
    bk::gdn_recurrent_steps_cpu(r, ref_state.data(), ref_out.data());
    double worst = 0.0, sq = 0.0;
    for (std::size_t i = 0; i < got_out.size(); ++i) {
      worst = std::max(worst, std::fabs(double(got_out[i]) - ref_out[i]));
      sq += ref_out[i] * ref_out[i];
    }
    const double rms = std::sqrt(sq / double(got_out.size()));
    double sworst = 0.0, smax = 0.0;
    for (std::size_t i = 0; i < st; ++i) {
      sworst = std::max(sworst, std::fabs(double(got_state[i]) - ref_state[i]));
      smax = std::max(smax, std::fabs(ref_state[i]));
    }
    check(worst <= 1e-4 * rms && sworst <= 1e-4 * smax,
          "gdn_recurrent_steps n=" + std::to_string(nn) + " vs FP64: out ratio=" + std::to_string(worst / rms) +
              " state ratio=" + std::to_string(sworst / smax));
  }

  // n = 1 against the shipped single-step kernel, outputs and the updated state.
  {
    float* dst2 = nullptr;
    INGINE_TEST_CUDA(cudaMalloc(&dst2, st * 4));
    INGINE_TEST_CUDA(cudaMemcpy(dst, hstate.data(), st * 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(dst2, hstate.data(), st * 4, cudaMemcpyHostToDevice));
    gdn::GdnStepArgs s;
    s.q = dq;
    s.k = dk;
    s.v = dv;
    s.beta = db;
    s.g = dg;
    s.state = dst;
    s.out = dout;
    s.v_heads = heads;
    check(gdn::gdn_recurrent_step(s, nullptr) == cudaSuccess, "gdn_recurrent_step launch (single-token ref)");
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> one(std::size_t(heads) * gdn::GDN_V_DIM), one_state(st);
    INGINE_TEST_CUDA(cudaMemcpy(one.data(), dout, one.size() * 4, cudaMemcpyDeviceToHost));
    INGINE_TEST_CUDA(cudaMemcpy(one_state.data(), dst, st * 4, cudaMemcpyDeviceToHost));

    bk::GdnStepsArgs a;
    a.q = dq;
    a.k = dk;
    a.v = dv;
    a.beta = db;
    a.g = dg;
    a.state = dst2;
    a.out = dout;
    a.v_heads = heads;
    a.n = 1;
    check(bk::gdn_recurrent_steps(a, nullptr) == cudaSuccess, "gdn_recurrent_steps launch n=1 (comparison)");
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> got(one.size()), got_state(st);
    INGINE_TEST_CUDA(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    INGINE_TEST_CUDA(cudaMemcpy(got_state.data(), dst2, st * 4, cudaMemcpyDeviceToHost));
    double rms = 0.0;
    const double ratio = worst_ratio(got, one, one.size(), &rms);
    double srms = 0.0;
    const double sratio = worst_ratio(got_state, one_state, one_state.size(), &srms);
    check(ratio <= 1e-5 && sratio <= 1e-5, "gdn_recurrent_steps n=1 equals the single-step kernel: out ratio=" +
                                              std::to_string(ratio) + " state ratio=" + std::to_string(sratio));
    cudaFree(dst2);
  }
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(db);
  cudaFree(dg);
  cudaFree(dst);
  cudaFree(dout);
}

}  // namespace

int main() {
  if (!have_sm120()) {
    std::printf("skip: needs sm_120 (the pack builds for sm_120a)\n");
    return 0;
  }
  test_gemvs();
  test_batch_moe();
  test_batch_attn();
  test_batch_gdn();
  std::printf("%s (%ld checks)\n", ok_all ? "ALL PASS" : "FAILURES PRESENT", checks_run);
  return ok_all ? 0 : 1;
}
