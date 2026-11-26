// Real-weights chain harness for the verify path. It runs the shipped single-token kernels over a real
// prompt on the real checkpoint, records the router's 8 expert ids and the post-attention norm per layer per
// position, and prints the number the business case rests on: how many *unique* experts a verify block of n
// consecutive positions touches. It then checks the batch kernels on those real activations against the
// single-token path and an FP64 oracle, and times the pieces the cost model needs.
//
// Skipped unless QWEN35MOE_SNAPSHOT points at nvidia/Qwen3.6-35B-A3B-NVFP4; needs sm_120. One model load
// carries every measurement, and every GPU process runs under /tmp/gpu-run.sh.
//
// The chain is width-1 by construction — the same per-token kernels the decoder uses, with the cross-card
// expert hop done through blocking copies (slow, and irrelevant here: this measures routing and rates, not
// the shipped step). Layer parity against the decoder's own QWEN35MOE_DUMP files (/tmp/qwen_layer0_h2.bin,
// /tmp/qwen_x_layer_NN.bin) is checked when those exist, which pins the harness to the verified network.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/attn/rope_norm.h"
#include "arch/qwen35moe/kernels/batch/batch_attn.h"
#include "arch/qwen35moe/kernels/batch/batch_gdn.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"
#include "arch/qwen35moe/kernels/batch/batch_moe.h"
#include "arch/qwen35moe/kernels/moe/fused_experts.h"
#include "arch/qwen35moe/kernels/moe/moe_apply.h"
#include "arch/qwen35moe/kernels/moe/router.h"
#include "arch/qwen35moe/kernels/moe/shared_expert.h"
#include "arch/qwen35moe/model/model_data.h"
#include "arch/qwen35moe/model/placement.h"
#include "arch/qwen35moe/model/uploader.h"
#include "kernels/dense/dense_ops.h"
#include "kernels/gdn/conv.h"
#include "kernels/gdn/gates.h"
#include "kernels/gdn/gdn.h"
#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::test;
namespace qwen = ingine::arch::qwen35moe;
namespace bk = ingine::arch::qwen35moe::kernels;
namespace dense = ingine::kernels::dense;
namespace gdn = ingine::kernels::gdn;
namespace quant = ingine::kernels::quant;

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

std::string layer_name(int l, const char* suffix) {
  return "model.language_model.layers." + std::to_string(l) + "." + suffix;
}

#define CUDA_OK(expr)                                                                     \
  do {                                                                                    \
    cudaError_t e_ = (expr);                                                              \
    if (e_ != cudaSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,                  \
                   cudaGetErrorString(e_));                                               \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

float ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.0f;
  CUDA_OK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

bool read_floats(const char* path, std::vector<float>& out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f) / 4;
  std::fseek(f, 0, SEEK_SET);
  out.resize(std::size_t(n));
  const bool good = n > 0 && std::fread(out.data(), 4, std::size_t(n), f) == std::size_t(n);
  std::fclose(f);
  return good;
}

struct Driver {
  qwen::TextArena* arena = nullptr;
  std::map<std::string, const qwen::PlacedTensor*> placed;
  std::map<std::string, float> scalars;
  std::map<std::string, float*> f32cache;
  int dev0 = 0, dev1 = 1;
  int max_pos = 0;
  bk::KvType kv_type = bk::KvType::Bf16;  // CHAIN_KV=q8_0 switches the chain cache; bf16 matches the decoder

  float *x = nullptr, *h = nullptr, *h2 = nullptr, *y = nullptr;
  float *qkv = nullptr, *qkv2 = nullptr, *z = nullptr, *a = nullptr, *b = nullptr, *beta = nullptr,
        *g = nullptr, *gout = nullptr;
  float *qgate = nullptr, *q_attn = nullptr, *gate_attn = nullptr, *k_ = nullptr, *v_ = nullptr,
        *attn_out = nullptr;
  float *moe_out = nullptr, *shared_out = nullptr, *shared_h = nullptr, *shared_up = nullptr;
  float* logits = nullptr;
  float* gdn_state = nullptr;
  float* conv_state = nullptr;
  std::vector<void*> kv_k, kv_v;
  bk::SplitState* kv_partial = nullptr;

  int* ids_dev = nullptr;
  float* wts_dev = nullptr;
  float* router_logits = nullptr;

  float *peer_x = nullptr, *peer_out = nullptr, *peer_g = nullptr, *peer_u = nullptr, *peer_d = nullptr;
  int* peer_ids = nullptr;
  float* peer_wts = nullptr;

  std::vector<bk::FusedExpertLayer> fused_host[2];
  std::vector<bk::BatchExpertBlock> batch_host[2];
  bk::FusedExpertLayer* fused_dev[2] = {nullptr, nullptr};
  bk::BatchExpertBlock* batch_dev[2] = {nullptr, nullptr};
  float *fscratch_g[2] = {nullptr, nullptr}, *fscratch_u[2] = {nullptr, nullptr},
        *fscratch_d[2] = {nullptr, nullptr};

  float *bx = nullptr, *by = nullptr, *bg = nullptr, *bu = nullptr, *bd = nullptr;
  int* bids = nullptr;
  float* bwts = nullptr;
  int* bplan = nullptr;
  int* bplan1 = nullptr;
  bk::BatchMoePlan plan;
  bk::BatchMoePlan plan1;

  // recorded per position: router ids (positions * LAYERS * 8) and post-attention norms (positions * LAYERS *
  // HIDDEN), which is 10 MB at 32 positions and lets the batch tests run on real activations.
  std::vector<int> ids_log;
  std::vector<float> h2_log;
  std::vector<float> x_layer0;  // residual after each layer at pos 0, for the decoder-dump parity
  int positions = 0;

  void* ptr(const std::string& name) {
    const auto it = placed.find(name);
    if (it == placed.end()) throw std::runtime_error("not placed: " + name);
    const qwen::PlacedTensor& p = *it->second;
    return static_cast<std::byte*>(arena->spans[std::size_t(p.span)].device) + p.offset;
  }
  int span(const std::string& name) {
    const auto it = placed.find(name);
    if (it == placed.end()) throw std::runtime_error("not placed: " + name);
    return it->second->span;
  }
  float scalar(const std::string& name, float fb) {
    const auto it = scalars.find(name);
    return it == scalars.end() ? fb : it->second;
  }
  const float* f32(const std::string& name, bool plus_one) {
    const std::string key = plus_one ? name + "#1" : name;
    const auto hit = f32cache.find(key);
    if (hit != f32cache.end()) return hit->second;
    const std::size_t count = placed.at(name)->bytes / 2;
    std::vector<__nv_bfloat16> staging(count);
    CUDA_OK(cudaSetDevice(arena->spans[std::size_t(placed.at(name)->span)].cuda_device));
    CUDA_OK(cudaMemcpy(staging.data(), ptr(name), count * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = __bfloat162float(staging[i]) + (plus_one ? 1.0f : 0.0f);
    CUDA_OK(cudaSetDevice(dev0));
    float* dev = nullptr;
    CUDA_OK(cudaMalloc(&dev, count * 4));
    CUDA_OK(cudaMemcpy(dev, out.data(), count * 4, cudaMemcpyHostToDevice));
    f32cache[key] = dev;
    return dev;
  }

  void* alloc_on(int dev, std::size_t bytes) {
    CUDA_OK(cudaSetDevice(dev));
    void* p = nullptr;
    CUDA_OK(cudaMalloc(&p, bytes));
    return p;
  }
  float* A0(std::size_t count) { return static_cast<float*>(alloc_on(dev0, count * 4)); }

  // First four values of a device vector, in the format the decoder's QWEN35MOE_DUMP lines use.
  std::string peek(const float* dev) {
    float v[4] = {0, 0, 0, 0};
    CUDA_OK(cudaSetDevice(dev0));
    CUDA_OK(cudaMemcpy(v, dev, 16, cudaMemcpyDeviceToHost));
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%.9g %.9g %.9g %.9g", v[0], v[1], v[2], v[3]);
    return buf;
  }
  std::string banner;

  void init(qwen::TextArena& ar, int maxpos) {
    arena = &ar;
    max_pos = maxpos;
    dev0 = arena->spans[0].cuda_device;
    dev1 = arena->spans.size() > 1 ? arena->spans[1].cuda_device : dev0;
    for (const auto& p : arena->placed) placed[p.name] = &p;
    CUDA_OK(cudaSetDevice(dev0));
    for (const auto& p : arena->placed) {
      // Both per-tensor scales: weight_scale_2 for the NVFP4 experts, weight_scale for every FP8 projection.
      const bool s2 = p.name.ends_with(".weight_scale_2");
      const bool s1 = !s2 && p.name.ends_with(".weight_scale");
      if (p.bytes == 4 && (s1 || s2)) {
        CUDA_OK(cudaSetDevice(arena->spans[std::size_t(p.span)].cuda_device));
        float v = 0.0f;
        CUDA_OK(cudaMemcpy(&v, ptr(p.name), 4, cudaMemcpyDeviceToHost));
        scalars[p.name] = v;
      }
    }
    CUDA_OK(cudaSetDevice(dev0));

    x = A0(qwen::HIDDEN);
    h = A0(qwen::HIDDEN);
    h2 = A0(qwen::HIDDEN);
    y = A0(qwen::HIDDEN);
    qkv = A0(qwen::LIN_CONV_DIM);
    qkv2 = A0(qwen::LIN_CONV_DIM);
    z = A0(qwen::LIN_VALUE_DIM_TOT);
    a = A0(qwen::LIN_VALUE_HEADS);
    b = A0(qwen::LIN_VALUE_HEADS);
    beta = A0(qwen::LIN_VALUE_HEADS);
    g = A0(qwen::LIN_VALUE_HEADS);
    gout = A0(qwen::LIN_VALUE_DIM_TOT);
    qgate = A0(qwen::Q_PROJ_ROWS);
    q_attn = A0(qwen::Q_DIM);
    gate_attn = A0(qwen::Q_DIM);
    k_ = A0(qwen::KV_DIM);
    v_ = A0(qwen::KV_DIM);
    attn_out = A0(qwen::Q_DIM);
    moe_out = A0(qwen::HIDDEN);
    shared_out = A0(qwen::HIDDEN);
    shared_h = A0(qwen::SHARED_INTER);
    shared_up = A0(qwen::SHARED_INTER);
    logits = A0(qwen::VOCAB);
    CUDA_OK(cudaMalloc(&ids_dev, bk::BATCH_PAIRS * 4));
    CUDA_OK(cudaMalloc(&wts_dev, bk::BATCH_PAIRS * 4));
    CUDA_OK(cudaMalloc(&router_logits, bk::ROUTER_EXPERTS * 4));

    const std::size_t kbytes = bk::kv_cache_bytes(kv_type, max_pos);
    for (int f = 0; f < qwen::NUM_FULL_LAYERS; ++f) {
      void* kk = alloc_on(dev0, kbytes);
      void* vv = alloc_on(dev0, kbytes);
      CUDA_OK(cudaMemset(kk, 0, kbytes));
      CUDA_OK(cudaMemset(vv, 0, kbytes));
      kv_k.push_back(kk);
      kv_v.push_back(vv);
    }
    CUDA_OK(cudaMalloc(&kv_partial, sizeof(bk::SplitState) * qwen::HEADS * bk::kMaxSplits));
    gdn_state = A0(std::size_t(qwen::NUM_LINEAR_LAYERS) * qwen::LIN_VALUE_HEADS * qwen::LIN_KEY_DIM *
                   qwen::LIN_VALUE_DIM);
    CUDA_OK(cudaMemset(gdn_state, 0, std::size_t(qwen::NUM_LINEAR_LAYERS) * qwen::LIN_VALUE_HEADS *
                                        qwen::LIN_KEY_DIM * qwen::LIN_VALUE_DIM * 4));
    conv_state = A0(std::size_t(qwen::NUM_LINEAR_LAYERS) * (qwen::CONV_K - 1) * qwen::LIN_CONV_DIM);
    CUDA_OK(cudaMemset(conv_state, 0, std::size_t(qwen::NUM_LINEAR_LAYERS) * (qwen::CONV_K - 1) *
                                          qwen::LIN_CONV_DIM * 4));

    if (dev1 != dev0) {
      peer_x = static_cast<float*>(alloc_on(dev1, std::size_t(bk::BATCH_MAX) * qwen::HIDDEN * 4));
      peer_out = static_cast<float*>(alloc_on(dev1, std::size_t(bk::BATCH_MAX) * qwen::HIDDEN * 4));
      peer_g = static_cast<float*>(alloc_on(dev1, std::size_t(bk::BATCH_PAIRS) * bk::MOE_INTER * 4));
      peer_u = static_cast<float*>(alloc_on(dev1, std::size_t(bk::BATCH_PAIRS) * bk::MOE_INTER * 4));
      peer_d = static_cast<float*>(alloc_on(dev1, std::size_t(bk::BATCH_PAIRS) * bk::MOE_HIDDEN * 4));
      CUDA_OK(cudaSetDevice(dev1));
      CUDA_OK(cudaMalloc(&peer_ids, bk::BATCH_PAIRS * 4));
      CUDA_OK(cudaMalloc(&peer_wts, bk::BATCH_PAIRS * 4));
      CUDA_OK(cudaMalloc(&bplan1, (3 * bk::BATCH_PAIRS + 1 + bk::BATCH_PAIRS * bk::BATCH_MAX) * 4));
    }
    CUDA_OK(cudaSetDevice(dev0));

    for (int d = 0; d < 2; ++d) {
      CUDA_OK(cudaSetDevice(d == 0 ? dev0 : dev1));
      CUDA_OK(cudaMalloc(&fscratch_g[d], std::size_t(bk::MOE_TOP_K) * bk::MOE_INTER * 4));
      CUDA_OK(cudaMalloc(&fscratch_u[d], std::size_t(bk::MOE_TOP_K) * bk::MOE_INTER * 4));
      CUDA_OK(cudaMalloc(&fscratch_d[d], std::size_t(bk::MOE_TOP_K) * bk::MOE_HIDDEN * 4));
    }
    CUDA_OK(cudaSetDevice(dev0));
    build_tables();

    bx = A0(std::size_t(bk::BATCH_MAX) * qwen::HIDDEN);
    by = A0(std::size_t(bk::BATCH_MAX) * qwen::HIDDEN);
    bg = A0(std::size_t(bk::BATCH_PAIRS) * bk::MOE_INTER);
    bu = A0(std::size_t(bk::BATCH_PAIRS) * bk::MOE_INTER);
    bd = A0(std::size_t(bk::BATCH_PAIRS) * bk::MOE_HIDDEN);
    CUDA_OK(cudaMalloc(&bids, bk::BATCH_PAIRS * 4));
    CUDA_OK(cudaMalloc(&bwts, bk::BATCH_PAIRS * 4));
    CUDA_OK(cudaMalloc(&bplan, (3 * bk::BATCH_PAIRS + 1 + bk::BATCH_PAIRS * bk::BATCH_MAX) * 4));
    plan.unique_ids = bplan;
    plan.unique_count = bplan + bk::BATCH_PAIRS;
    plan.pair_unique = bplan + bk::BATCH_PAIRS + 1;
    plan.rows = bplan + 2 * bk::BATCH_PAIRS + 1;
    plan.pair_of = bplan + 3 * bk::BATCH_PAIRS + 1;
    if (dev1 != dev0) {
      plan1.unique_ids = bplan1;
      plan1.unique_count = bplan1 + bk::BATCH_PAIRS;
      plan1.pair_unique = bplan1 + bk::BATCH_PAIRS + 1;
      plan1.rows = bplan1 + 2 * bk::BATCH_PAIRS + 1;
      plan1.pair_of = bplan1 + 3 * bk::BATCH_PAIRS + 1;
    }
  }

  void build_tables() {
    for (int d = 0; d < 2; ++d) {
      fused_host[d].assign(std::size_t(qwen::LAYERS) * qwen::EXPERTS, bk::FusedExpertLayer{});
      batch_host[d].assign(std::size_t(qwen::LAYERS) * qwen::EXPERTS, bk::BatchExpertBlock{});
    }
    for (int l = 0; l < qwen::LAYERS; ++l) {
      const qwen::LayerPlan& lp = arena->plan.layers[std::size_t(l)];
      std::int64_t off[3][4] = {};
      for (const qwen::TensorSlot& slot : lp.experts.slots) {
        const int proj = slot.projection == qwen::Projection::GateProj   ? 0
                         : slot.projection == qwen::Projection::UpProj   ? 1
                         : slot.projection == qwen::Projection::DownProj ? 2
                                                                         : 3;
        const int art = slot.artifact == qwen::Artifact::Weight          ? 0
                        : slot.artifact == qwen::Artifact::WeightScale   ? 1
                        : slot.artifact == qwen::Artifact::WeightScale2  ? 2
                                                                         : 3;
        if (proj < 3 && art < 2) off[proj][art] = std::int64_t(slot.offset);
      }
      const std::int64_t stride = std::int64_t(lp.experts.stride);
      for (int e = 0; e < qwen::EXPERTS; ++e) {
        const std::string base = layer_name(l, "mlp.experts.") + std::to_string(e) + ".";
        const int sp = span(base + "gate_proj.weight");
        const float g2 = scalar(base + "gate_proj.weight_scale_2", 1.0f);
        const float u2 = scalar(base + "up_proj.weight_scale_2", 1.0f);
        const float d2 = scalar(base + "down_proj.weight_scale_2", 1.0f);
        if (sp < 0 || sp > 1) continue;
        const std::uint8_t* wbase = static_cast<const std::uint8_t*>(ptr(base + "gate_proj.weight")) -
                                    std::size_t(e) * std::size_t(stride) - std::size_t(off[0][0]);
        const std::uint8_t* sbase = static_cast<const std::uint8_t*>(ptr(base + "gate_proj.weight_scale")) -
                                    std::size_t(e) * std::size_t(stride) - std::size_t(off[0][1]);
        bk::FusedExpertLayer& f = fused_host[sp][std::size_t(l) * qwen::EXPERTS + e];
        f.w_base = wbase;
        f.s_base = sbase;
        f.stride_bytes = int(stride);
        f.scale_stride_bytes = int(stride);
        f.gate_weight_off = int(off[0][0]);
        f.up_weight_off = int(off[1][0]);
        f.down_weight_off = int(off[2][0]);
        f.gate_scale_off = int(off[0][1]);
        f.up_scale_off = int(off[1][1]);
        f.down_scale_off = int(off[2][1]);
        f.gate_scale2 = g2;
        f.up_scale2 = u2;
        f.down_scale2 = d2;
        bk::BatchExpertBlock& bb = batch_host[sp][std::size_t(l) * qwen::EXPERTS + e];
        bb.w_base = wbase;
        bb.s_base = sbase;
        bb.stride_bytes = stride;
        bb.scale_stride_bytes = stride;
        bb.gate_w = off[0][0];
        bb.up_w = off[1][0];
        bb.down_w = off[2][0];
        bb.gate_s = off[0][1];
        bb.up_s = off[1][1];
        bb.down_s = off[2][1];
        bb.gate_scale2 = g2;
        bb.up_scale2 = u2;
        bb.down_scale2 = d2;
      }
    }
    for (int d = 0; d < 2; ++d) {
      CUDA_OK(cudaSetDevice(d == 0 ? dev0 : dev1));
      const std::size_t fb = fused_host[d].size() * sizeof(bk::FusedExpertLayer);
      CUDA_OK(cudaMalloc(&fused_dev[d], fb));
      CUDA_OK(cudaMemcpy(fused_dev[d], fused_host[d].data(), fb, cudaMemcpyHostToDevice));
      const std::size_t bb = batch_host[d].size() * sizeof(bk::BatchExpertBlock);
      CUDA_OK(cudaMalloc(&batch_dev[d], bb));
      CUDA_OK(cudaMemcpy(batch_dev[d], batch_host[d].data(), bb, cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaSetDevice(dev0));
  }

  // `base` is the projection's dotted name without the artifact suffix, as in decoder.cu.
  void fp8_gemv(const std::string& base, const float* xv, float* yv, int rows, int cols) {
    quant::Fp8GemvArgs a;
    a.w = static_cast<const std::uint8_t*>(ptr(base + ".weight"));
    a.scale = scalar(base + ".weight_scale", 1.0f);
    a.x = xv;
    a.y = yv;
    a.rows = rows;
    a.cols = cols;
    CUDA_OK(quant::fp8_gemv(a, nullptr));
  }

  void route(int l, int host_ids[bk::MOE_TOP_K], float host_w[bk::MOE_TOP_K]) {
    bk::RouterArgs r;
    r.weight = static_cast<const __nv_bfloat16*>(ptr(layer_name(l, "mlp.gate.weight")));
    r.x = h2;
    r.ids = ids_dev;
    r.weights = wts_dev;
    r.logits = router_logits;
    CUDA_OK(bk::route(r, nullptr));
    CUDA_OK(cudaMemcpy(host_ids, ids_dev, bk::MOE_TOP_K * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(host_w, wts_dev, bk::MOE_TOP_K * 4, cudaMemcpyDeviceToHost));
  }

  // One token's routed experts across both cards, the shipped fused kernel twice plus a host-side merge.
  void routed_one(int l, const int* host_ids, const float* host_w, float* out_host) {
    CUDA_OK(cudaMemcpy(ids_dev, host_ids, bk::MOE_TOP_K * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(wts_dev, host_w, bk::MOE_TOP_K * 4, cudaMemcpyHostToDevice));
    bk::FusedMoeArgs a0;
    a0.layers = fused_dev[0] + std::size_t(l) * qwen::EXPERTS;
    a0.x = h2;
    a0.ids = ids_dev;
    a0.weights = wts_dev;
    a0.y = moe_out;
    a0.scratch_gate = fscratch_g[0];
    a0.scratch_up = fscratch_u[0];
    a0.scratch_down = fscratch_d[0];
    CUDA_OK(cudaSetDevice(dev0));
    CUDA_OK(bk::fused_moe_apply(a0, nullptr));
    std::vector<float> acc(qwen::HIDDEN);
    CUDA_OK(cudaMemcpy(acc.data(), moe_out, qwen::HIDDEN * 4, cudaMemcpyDeviceToHost));
    if (dev1 == dev0) {
      std::copy(acc.begin(), acc.end(), out_host);
      return;
    }
    std::vector<float> host_h2(qwen::HIDDEN);
    CUDA_OK(cudaMemcpy(host_h2.data(), h2, qwen::HIDDEN * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaSetDevice(dev1));
    CUDA_OK(cudaMemcpy(peer_x, host_h2.data(), qwen::HIDDEN * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(peer_ids, host_ids, bk::MOE_TOP_K * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(peer_wts, host_w, bk::MOE_TOP_K * 4, cudaMemcpyHostToDevice));
    bk::FusedMoeArgs a1;
    a1.layers = fused_dev[1] + std::size_t(l) * qwen::EXPERTS;
    a1.x = peer_x;
    a1.ids = peer_ids;
    a1.weights = peer_wts;
    a1.y = peer_out;
    a1.scratch_gate = peer_g;
    a1.scratch_up = peer_u;
    a1.scratch_down = peer_d;
    CUDA_OK(bk::fused_moe_apply(a1, nullptr));
    std::vector<float> part(qwen::HIDDEN);
    CUDA_OK(cudaMemcpy(part.data(), peer_out, qwen::HIDDEN * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaSetDevice(dev0));
    for (int i = 0; i < qwen::HIDDEN; ++i) out_host[i] = acc[std::size_t(i)] + part[std::size_t(i)];
  }

  void linear_layer(int l) {
    const std::string p = layer_name(l, "linear_attn.");
    fp8_gemv(p + "in_proj_qkv", h, qkv, qwen::LIN_CONV_DIM, qwen::HIDDEN);
    fp8_gemv(p + "in_proj_z", h, z, qwen::LIN_VALUE_DIM_TOT, qwen::HIDDEN);
    CUDA_OK(dense::gemv_bf16(static_cast<const __nv_bfloat16*>(ptr(p + "in_proj_a.weight")), h, a,
                             qwen::LIN_VALUE_HEADS, qwen::HIDDEN, nullptr));
    CUDA_OK(dense::gemv_bf16(static_cast<const __nv_bfloat16*>(ptr(p + "in_proj_b.weight")), h, b,
                             qwen::LIN_VALUE_HEADS, qwen::HIDDEN, nullptr));
    int slot = 0;
    for (int i = 0; i < l; ++i) slot += qwen::is_full_layer(i) ? 0 : 1;

    gdn::ConvStepArgs conv;
    conv.x = qkv;
    conv.weight = f32(p + "conv1d.weight", false);
    conv.history = conv_state + std::size_t(slot) * (qwen::CONV_K - 1) * qwen::LIN_CONV_DIM;
    conv.y = qkv2;
    conv.conv_dim = qwen::LIN_CONV_DIM;
    CUDA_OK(gdn::conv_silu_step(conv, nullptr));

    gdn::GdnGateArgs gates;
    gates.a = a;
    gates.b = b;
    gates.dt_bias = f32(p + "dt_bias", false);
    gates.a_log = f32(p + "A_log", false);
    gates.beta = beta;
    gates.g = g;
    gates.heads = qwen::LIN_VALUE_HEADS;
    CUDA_OK(gdn::gdn_gates(gates, nullptr));

    gdn::GdnStepArgs step;
    step.q = qkv2;
    step.k = qkv2 + qwen::LIN_KEY_DIM_TOT;
    step.v = qkv2 + 2 * qwen::LIN_KEY_DIM_TOT;
    step.beta = beta;
    step.g = g;
    step.state = gdn_state + std::size_t(slot) * qwen::LIN_VALUE_HEADS * qwen::LIN_KEY_DIM * qwen::LIN_VALUE_DIM;
    step.out = gout;
    step.v_heads = qwen::LIN_VALUE_HEADS;
    CUDA_OK(gdn::gdn_recurrent_step(step, nullptr));

    gdn::GatedNormArgs norm;
    norm.x = gout;
    norm.z = z;
    norm.weight = f32(p + "norm.weight", false);
    norm.v_heads = qwen::LIN_VALUE_HEADS;
    norm.v_dim = qwen::LIN_VALUE_DIM;
    norm.eps = qwen::EPS;
    CUDA_OK(gdn::gated_norm(norm, nullptr));

    fp8_gemv(p + "out_proj", gout, y, qwen::HIDDEN, qwen::LIN_VALUE_DIM_TOT);
  }

  void full_layer(int l, int pos) {
    const std::string p = layer_name(l, "self_attn.");
    const int slot = l / qwen::FULL_ATTENTION_INTERVAL;
    fp8_gemv(p + "q_proj", h, qgate, qwen::Q_PROJ_ROWS, qwen::HIDDEN);
    fp8_gemv(p + "k_proj", h, k_, qwen::KV_DIM, qwen::HIDDEN);
    fp8_gemv(p + "v_proj", h, v_, qwen::KV_DIM, qwen::HIDDEN);

    bk::SplitQGateArgs split;
    split.qgate = qgate;
    split.q = q_attn;
    split.gate = gate_attn;
    split.heads = qwen::HEADS;
    split.head_dim = qwen::HEAD_DIM;
    CUDA_OK(bk::split_q_gate(split, nullptr));

    bk::NormRopeArgs rope;
    rope.q = q_attn;
    rope.k = k_;
    rope.q_weight = f32(p + "q_norm.weight", true);
    rope.k_weight = f32(p + "k_norm.weight", true);
    rope.pos = pos;
    rope.eps = qwen::EPS;
    CUDA_OK(bk::qk_norm_rope(rope, nullptr));

    bk::DecodeAttnArgs dec;
    dec.q = q_attn;
    dec.k_new = k_;
    dec.v_new = v_;
    dec.k_cache = kv_k[std::size_t(slot)];
    dec.v_cache = kv_v[std::size_t(slot)];
    dec.kv_type_k = kv_type;
    dec.kv_type_v = kv_type;
    dec.out = attn_out;
    dec.pos = pos;
    dec.max_pos = max_pos;
    dec.partial = kv_partial;
    dec.splits = std::max(1, std::min(bk::kMaxSplits, (pos + 2048) / 2048));
    CUDA_OK(bk::decode_attention(dec, nullptr));

    bk::OutputGateArgs gate;
    gate.x = attn_out;
    gate.gate = gate_attn;
    gate.count = qwen::Q_DIM;
    CUDA_OK(bk::output_gate(gate, nullptr));

    fp8_gemv(p + "o_proj", attn_out, y, qwen::HIDDEN, qwen::Q_DIM);
  }

  void shared_one(int l) {
    bk::SharedExpertArgs shared;
    const std::string sb = layer_name(l, "mlp.shared_expert.");
    shared.gate = {static_cast<const std::uint8_t*>(ptr(sb + "gate_proj.weight")),
                   static_cast<const std::uint8_t*>(ptr(sb + "gate_proj.weight_scale")),
                   scalar(sb + "gate_proj.weight_scale_2", 1.0f), h2, nullptr, bk::MOE_INTER, bk::MOE_HIDDEN};
    shared.up = {static_cast<const std::uint8_t*>(ptr(sb + "up_proj.weight")),
                 static_cast<const std::uint8_t*>(ptr(sb + "up_proj.weight_scale")),
                 scalar(sb + "up_proj.weight_scale_2", 1.0f), h2, nullptr, bk::MOE_INTER, bk::MOE_HIDDEN};
    shared.down = {static_cast<const std::uint8_t*>(ptr(sb + "down_proj.weight")),
                   static_cast<const std::uint8_t*>(ptr(sb + "down_proj.weight_scale")),
                   scalar(sb + "down_proj.weight_scale_2", 1.0f), nullptr, nullptr, bk::MOE_HIDDEN,
                   bk::MOE_INTER};
    shared.gate_weight =
        static_cast<const __nv_bfloat16*>(ptr(layer_name(l, "mlp.shared_expert_gate.weight")));
    shared.x = h2;
    shared.y = shared_out;
    shared.scratch_h = shared_h;
    shared.scratch_up = shared_up;
    CUDA_OK(bk::shared_expert(shared, nullptr));
  }

  int step(int token, int pos) {
    CUDA_OK(cudaSetDevice(dev0));
    const auto* embed = static_cast<const __nv_bfloat16*>(ptr("model.language_model.embed_tokens.weight"));
    CUDA_OK(dense::embedding_row(embed, token, x, qwen::HIDDEN, nullptr));
    if (pos == 0) banner += "embed: " + peek(x) + "\n";
    for (int l = 0; l < qwen::LAYERS; ++l) {
      CUDA_OK(dense::rms_norm(x, f32(layer_name(l, "input_layernorm.weight"), true), h, qwen::HIDDEN,
                              qwen::EPS, nullptr));
      if (pos == 0 && l == 0) banner += "h0: " + peek(h) + "\n";
      if (qwen::is_full_layer(l)) {
        full_layer(l, pos);
      } else {
        linear_layer(l);
      }
      if (pos == 0 && l == 0) banner += "qkv0(y): " + peek(y) + "\n";
      CUDA_OK(bk::axpy(x, y, 1.0f, qwen::HIDDEN, nullptr));
      CUDA_OK(dense::rms_norm(x, f32(layer_name(l, "post_attention_layernorm.weight"), true), h2,
                              qwen::HIDDEN, qwen::EPS, nullptr));
      if (pos == 0 && l == 0) banner += "h2: " + peek(h2) + "\n";
      int host_ids[bk::MOE_TOP_K];
      float host_w[bk::MOE_TOP_K];
      route(l, host_ids, host_w);
      for (int k = 0; k < bk::MOE_TOP_K; ++k) ids_log.push_back(host_ids[k]);
      std::vector<float> h2_host(qwen::HIDDEN);
      CUDA_OK(cudaMemcpy(h2_host.data(), h2, qwen::HIDDEN * 4, cudaMemcpyDeviceToHost));
      h2_log.insert(h2_log.end(), h2_host.begin(), h2_host.end());
      if (pos == 0) x_layer0.resize(std::size_t(l + 1) * qwen::HIDDEN);
      std::vector<float> moe_host(qwen::HIDDEN);
      routed_one(l, host_ids, host_w, moe_host.data());
      CUDA_OK(cudaMemcpy(moe_out, moe_host.data(), qwen::HIDDEN * 4, cudaMemcpyHostToDevice));
      shared_one(l);
      CUDA_OK(bk::axpy(x, moe_out, 1.0f, qwen::HIDDEN, nullptr));
      CUDA_OK(bk::axpy(x, shared_out, 1.0f, qwen::HIDDEN, nullptr));
      if (pos == 0) {
        CUDA_OK(cudaMemcpy(x_layer0.data() + std::size_t(l) * qwen::HIDDEN, x, qwen::HIDDEN * 4,
                           cudaMemcpyDeviceToHost));
      }
    }
    CUDA_OK(dense::rms_norm(x, f32("model.language_model.norm.weight", true), h, qwen::HIDDEN, qwen::EPS,
                            nullptr));
    quant::Nvfp4GemvArgs head;
    head.w = static_cast<const std::uint8_t*>(ptr("lm_head.weight"));
    head.scales = static_cast<const std::uint8_t*>(ptr("lm_head.weight_scale"));
    head.scale = scalar("lm_head.weight_scale_2", 1.0f);
    head.x = h;
    head.y = logits;
    head.rows = qwen::VOCAB;
    head.cols = qwen::HIDDEN;
    CUDA_OK(quant::nvfp4_gemv(head, nullptr));
    std::vector<float> host_logits(std::size_t(qwen::VOCAB));
    CUDA_OK(cudaMemcpy(host_logits.data(), logits, std::size_t(qwen::VOCAB) * 4, cudaMemcpyDeviceToHost));
    int best = 0;
    float bv = -1e30f;
    for (int v = 0; v < qwen::VOCAB; ++v) {
      if (host_logits[std::size_t(v)] > bv) {
        bv = host_logits[std::size_t(v)];
        best = v;
      }
    }
    last_logits = std::move(host_logits);
    return best;
  }

  std::vector<float> last_logits;
};

// ------------------------------------------------------------- histogram

void print_histogram(const Driver& d, int w, int from, int to) {
  const int layers = qwen::LAYERS;
  if (to - from + 1 < w) return;
  std::vector<int> hist(w * bk::MOE_TOP_K + 1, 0);
  double sum = 0.0;
  long samples = 0;
  int mn = w * bk::MOE_TOP_K, mx = 0;
  std::vector<int> per_layer(layers, 0);
  for (int l = 0; l < layers; ++l) {
    double lsum = 0.0;
    for (int p0 = from; p0 + w <= to; ++p0) {
      bool seen[qwen::EXPERTS] = {};
      int uniq = 0;
      for (int t = 0; t < w; ++t) {
        for (int k = 0; k < bk::MOE_TOP_K; ++k) {
          const int e = int(d.ids_log[(std::size_t(p0 + t) * layers + l) * bk::MOE_TOP_K + k]);
          if (!seen[e]) {
            seen[e] = true;
            ++uniq;
          }
        }
      }
      ++hist[std::size_t(uniq)];
      sum += uniq;
      lsum += uniq;
      mn = std::min(mn, uniq);
      mx = std::max(mx, uniq);
      ++samples;
    }
    per_layer[std::size_t(l)] = int(lsum / double(std::max(1, to - from + 1 - w + 1)));
  }
  const double mean = sum / double(samples);
  std::printf("HIST n=%d samples=%ld mean=%.2f min=%d max=%d of %d -> expert bytes saved %.1f%%\n", w, samples,
              mean, mn, mx, w * bk::MOE_TOP_K, 100.0 * (1.0 - mean / double(w * bk::MOE_TOP_K)));
  for (int u = w; u <= w * bk::MOE_TOP_K; ++u) {
    if (hist[std::size_t(u)]) std::printf("HIST n=%d U=%d count=%ld\n", w, u, hist[std::size_t(u)]);
  }
  std::printf("HISTL n=%d per-layer mean:", w);
  for (int l = 0; l < layers; ++l) std::printf(" %d", per_layer[std::size_t(l)]);
  std::printf("\n");
  std::fflush(stdout);
}

// ------------------------------------------- batch path checks on real bytes

// Download the experts a window touches, so the FP64 oracle can read the same bytes the device did.
std::int64_t card_stride(Driver& d, int card, int l) {
  for (int e = 0; e < qwen::EXPERTS; ++e) {
    const std::int64_t s = d.batch_host[card][std::size_t(l) * qwen::EXPERTS + e].stride_bytes;
    if (s != 0) return s;
  }
  return 0;
}

std::map<int, std::vector<std::uint8_t>> fetch_experts(Driver& d, int l, const int* eids, int count,
                                                       int card) {
  std::map<int, std::vector<std::uint8_t>> out;
  const std::int64_t stride = card_stride(d, card, l);
  if (stride == 0) return out;
  CUDA_OK(cudaSetDevice(card == 0 ? d.dev0 : d.dev1));
  for (int i = 0; i < count; ++i) {
    const int e = eids[i];
    if (out.count(e)) continue;
    const bk::BatchExpertBlock& src = d.batch_host[card][std::size_t(l) * qwen::EXPERTS + e];
    if (src.stride_bytes == 0) continue;  // this expert lives on the other card
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(stride));
    CUDA_OK(cudaMemcpy(buf.data(), src.w_base + std::size_t(e) * std::size_t(stride), buf.size(),
                       cudaMemcpyDeviceToHost));
    out[e] = std::move(buf);
  }
  CUDA_OK(cudaSetDevice(d.dev0));
  return out;
}

// A host table over downloaded bytes: the offsets, scales and stride come from the device table, the base is
// shifted so the kernel's `base + e * stride + offset` lands on the start of the buffer that holds expert e.
std::vector<bk::BatchExpertBlock> host_oracle_table(Driver& d, int l, int card, const int* eids, int count,
                                                    std::map<int, std::vector<std::uint8_t>>* keep) {
  *keep = fetch_experts(d, l, eids, count, card);
  std::vector<bk::BatchExpertBlock> table(qwen::EXPERTS);
  for (const auto& kv : *keep) {
    const bk::BatchExpertBlock& src = d.batch_host[card][std::size_t(l) * qwen::EXPERTS + kv.first];
    bk::BatchExpertBlock& b = table[std::size_t(kv.first)];
    b = src;
    b.w_base = kv.second.data() - std::size_t(kv.first) * std::size_t(src.stride_bytes);
    b.s_base = b.w_base + (src.s_base - src.w_base);
  }
  return table;
}

}  // namespace

int main() {
  const char* snapshot = std::getenv("QWEN35MOE_SNAPSHOT");
  if (!snapshot) {
    std::printf("skip: set QWEN35MOE_SNAPSHOT\n");
    return 0;
  }
  if (!have_sm120()) {
    std::printf("skip: needs sm_120\n");
    return 0;
  }
  const int kGen = std::atoi(std::getenv("CHAIN_GEN") ? std::getenv("CHAIN_GEN") : "24");
  const bool chain_q8 = std::getenv("CHAIN_KV") && std::string(std::getenv("CHAIN_KV")) == "q8_0";
  const std::vector<int> prompt = {9707, 11, 1879, 314, 2419, 13, 9656, 11, 625, 429, 316, 3645, 702, 562,
                                   310, 279};
  const int plen = int(prompt.size());
  const int max_pos = plen + kGen + 8;

  qwen::TextArena arena;
  std::string error;
  if (!qwen::upload_text_model(snapshot, false, -1, &arena, &error)) {
    std::printf("FAIL upload: %s\n", error.c_str());
    return 1;
  }
  std::printf("stage: uploaded %llu bytes, split %d/%d in %.2f s\n",
              static_cast<unsigned long long>(arena.uploaded_bytes), arena.experts_on_first,
              qwen::EXPERTS - arena.experts_on_first, arena.seconds);
  std::fflush(stdout);

  Driver d;
  d.kv_type = chain_q8 ? bk::KvType::Q8_0 : bk::KvType::Bf16;
  std::printf("stage: chain kv = %s\n", chain_q8 ? "q8_0" : "bf16");
  try {
    d.init(arena, max_pos);
  } catch (const std::exception& e) {
    std::printf("FAIL init: %s\n", e.what());
    return 1;
  }
  std::printf("stage: driver ready\n");
  std::fflush(stdout);

  // ---- Part A: the real chain, width 1, recording ids and h2 per layer per position
  std::vector<int> tokens(prompt);
  int next = 0;
  cudaEvent_t ev_a, ev_b;
  cudaEventCreate(&ev_a);
  cudaEventCreate(&ev_b);
  CUDA_OK(cudaEventRecord(ev_a));
  for (int i = 0; i < plen; ++i) next = d.step(tokens[std::size_t(i)], i);
  for (int j = 0; j < kGen; ++j) {
    tokens.push_back(next);
    next = d.step(next, plen + j);
  }
  CUDA_OK(cudaEventRecord(ev_b));
  CUDA_OK(cudaEventSynchronize(ev_b));
  d.positions = int(tokens.size());
  const double chain_ms = ms_between(ev_a, ev_b);
  std::printf("chain: %d positions in %.1f ms (%.1f tok/s width-1 with a blocking hop)\n", d.positions,
              chain_ms, d.positions / (chain_ms / 1000.0));
  {
    auto head = [&](const char* what, const float* dev, int n) {
      std::vector<float> v(n);
      CUDA_OK(cudaSetDevice(d.dev0));
      CUDA_OK(cudaMemcpy(v.data(), dev, n * 4, cudaMemcpyDeviceToHost));
      std::printf("%s:", what);
      for (int i = 0; i < n; ++i) std::printf(" %.9g", v[std::size_t(i)]);
      std::printf("\n");
    };
    std::printf("harness values (compare with the decoder's QWEN35MOE_DUMP lines):\n%s", d.banner.c_str());
    std::fflush(stdout);
  }
  std::printf("generated ids:");
  for (int i = plen; i < int(tokens.size()); ++i) std::printf(" %d", tokens[std::size_t(i)]);
  std::printf("\n");
  std::fflush(stdout);

  const int layers = qwen::LAYERS;
  // Parity of the harness against the decoder's own dumps, when a decoder run wrote them.
  {
    std::vector<float> theirs;
    if (read_floats("/tmp/qwen_layer0_h2.bin", theirs) && theirs.size() == std::size_t(qwen::HIDDEN)) {
      const float* mine = d.h2_log.data();  // position 0, layer 0
      double worst = 0.0, scale = 0.0;
      for (std::size_t i = 0; i < theirs.size(); ++i) {
        worst = std::max(worst, std::fabs(double(mine[i]) - double(theirs[i])));
        scale = std::max(scale, std::fabs(double(theirs[i])));
      }
      check(worst <= 1e-3 * scale, "harness h2(l=0,pos=0) matches the decoder dump: " + std::to_string(worst) +
                                       " / " + std::to_string(scale));
    } else {
      std::printf("note: no /tmp/qwen_layer0_h2.bin, harness-vs-decoder parity not checked this run\n");
    }
    for (int l = 0; l < layers; ++l) {
      char path[64];
      std::snprintf(path, sizeof(path), "/tmp/qwen_x_layer_%02d.bin", l);
      std::vector<float> dump;
      if (!read_floats(path, dump) || dump.size() != std::size_t(qwen::HIDDEN)) continue;
      const float* mine = d.x_layer0.data() + std::size_t(l) * qwen::HIDDEN;
      double worst = 0.0, scale = 0.0;
      for (std::size_t i = 0; i < dump.size(); ++i) {
        worst = std::max(worst, std::fabs(double(mine[i]) - double(dump[i])));
        scale = std::max(scale, std::fabs(double(dump[i])));
      }
      std::printf("parity pos0 layer%02d: rel=%.3e %s\n", l, worst / scale, worst <= 1e-3 * scale ? "ok" : "off");
    }
  }

  for (int w = 2; w <= bk::BATCH_MAX; ++w) {
    print_histogram(d, w, 0, plen - 1);    // prompt windows (prefill chunks)
    print_histogram(d, w, plen, d.positions - 1);  // generated chain windows (the draft chain)
  }
  print_histogram(d, 1, 0, d.positions - 1);

  // ---- Part B: the batch expert path on real weights, against the single-token path and FP64
  for (int l : {3, 17, 39}) {
    const int p0 = d.positions - bk::BATCH_MAX;
    std::vector<float> xs(std::size_t(bk::BATCH_MAX) * bk::MOE_HIDDEN);
    std::vector<int> ids(bk::BATCH_PAIRS);
    std::vector<float> wts(bk::BATCH_PAIRS);
    for (int t = 0; t < bk::BATCH_MAX; ++t) {
      const float* src = d.h2_log.data() + (std::size_t(p0 + t) * layers + l) * qwen::HIDDEN;
      std::copy_n(src, bk::MOE_HIDDEN, xs.begin() + std::ptrdiff_t(t) * bk::MOE_HIDDEN);
      for (int k = 0; k < bk::MOE_TOP_K; ++k) {
        const std::size_t at = (std::size_t(p0 + t) * layers + l) * bk::MOE_TOP_K + k;
        ids[std::size_t(t * bk::MOE_TOP_K + k)] = d.ids_log[at];
        wts[std::size_t(t * bk::MOE_TOP_K + k)] = 1.0f / bk::MOE_TOP_K;
      }
    }
    bk::BatchMoePlanHost hp;
    bk::batch_moe_plan_cpu(ids.data(), bk::BATCH_MAX, &hp);

    // device: both cards, summed on the host
    CUDA_OK(cudaSetDevice(d.dev0));
    CUDA_OK(cudaMemcpy(d.bx, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d.bids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d.bwts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> dev_y(std::size_t(bk::BATCH_MAX) * bk::MOE_HIDDEN, 0.0f);
    std::vector<float> part0(std::size_t(bk::BATCH_MAX) * bk::MOE_HIDDEN, 0.0f);
    float batch_ms = 0.0f, fused_ms = 0.0f;
    for (int card = 0; card < 2; ++card) {
      bk::BatchMoeArgs a;
      a.layers = d.batch_dev[card] + std::size_t(l) * qwen::EXPERTS;
      a.scratch_gate = card == 0 ? d.bg : d.peer_g;
      a.scratch_up = card == 0 ? d.bu : d.peer_u;
      a.scratch_down = card == 0 ? d.bd : d.peer_d;
      a.n = bk::BATCH_MAX;
      if (card == 0) {
        CUDA_OK(cudaSetDevice(d.dev0));
        a.x = d.bx;
        a.ids = d.bids;
        a.weights = d.bwts;
        a.y = d.by;
        a.plan = d.plan;
      } else {
        CUDA_OK(cudaSetDevice(d.dev1));
        CUDA_OK(cudaMemcpy(d.peer_x, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d.peer_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d.peer_wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
        a.x = d.peer_x;
        a.ids = d.peer_ids;
        a.weights = d.peer_wts;
        a.y = d.peer_out;
        a.plan = d.plan1;
      }
      CUDA_OK(bk::batch_moe_apply(a, nullptr));
      CUDA_OK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      CUDA_OK(cudaEventRecord(e0));
      for (int it = 0; it < 20; ++it) CUDA_OK(bk::batch_moe_apply(a, nullptr));
      CUDA_OK(cudaEventRecord(e1));
      CUDA_OK(cudaEventSynchronize(e1));
      batch_ms += ms_between(e0, e1) / 20.0f;
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
      std::vector<float> part(std::size_t(bk::BATCH_MAX) * bk::MOE_HIDDEN);
      CUDA_OK(cudaMemcpy(part.data(), a.y, part.size() * 4, cudaMemcpyDeviceToHost));
      if (card == 0) part0 = part;
      for (std::size_t i = 0; i < part.size(); ++i) dev_y[i] += part[i];
    }

    // single-token reference: the shipped fused kernel, once per position, both cards, merged on the host
    std::vector<float> ref_y(std::size_t(bk::BATCH_MAX) * bk::MOE_HIDDEN, 0.0f);
    for (int t = 0; t < bk::BATCH_MAX; ++t) {
      std::vector<float> h2_backup(qwen::HIDDEN);
      CUDA_OK(cudaSetDevice(d.dev0));
      CUDA_OK(cudaMemcpy(h2_backup.data(), d.h2, qwen::HIDDEN * 4, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(d.h2, &xs[std::size_t(t) * bk::MOE_HIDDEN], qwen::HIDDEN * 4, cudaMemcpyHostToDevice));
      std::vector<float> one(qwen::HIDDEN);
      int host_ids[bk::MOE_TOP_K];
      float host_w[bk::MOE_TOP_K];
      for (int k = 0; k < bk::MOE_TOP_K; ++k) {
        host_ids[k] = ids[std::size_t(t * bk::MOE_TOP_K + k)];
        host_w[k] = wts[std::size_t(t * bk::MOE_TOP_K + k)];
      }
      d.routed_one(l, host_ids, host_w, one.data());
      std::copy_n(one.data(), qwen::HIDDEN, ref_y.begin() + std::ptrdiff_t(t) * bk::MOE_HIDDEN);
      CUDA_OK(cudaMemcpy(d.h2, h2_backup.data(), qwen::HIDDEN * 4, cudaMemcpyHostToDevice));
    }
    double worst = 0.0, sq = 0.0;
    for (std::size_t i = 0; i < dev_y.size(); ++i) {
      worst = std::max(worst, std::fabs(double(dev_y[i]) - double(ref_y[i])));
      sq += double(ref_y[i]) * double(ref_y[i]);
    }
    const double rms = std::sqrt(sq / double(dev_y.size()));
    check(worst <= 1e-4 * rms, "layer " + std::to_string(l) +
                                   " batch n=4 on real weights equals 4 single-token runs: ratio=" +
                                   std::to_string(worst / rms) + " rms=" + std::to_string(rms));

    // Fair rate: the fused kernel once per position per card, device-resident inputs, no staging inside the
    // timing window, so both sides of the A/B measure kernel time only.
    {
      float fused_only = 0.0f;
      for (int card = 0; card < 2; ++card) {
        CUDA_OK(cudaSetDevice(card == 0 ? d.dev0 : d.dev1));
        bk::FusedMoeArgs fa;
        fa.layers = d.fused_dev[card] + std::size_t(l) * qwen::EXPERTS;
        fa.y = card == 0 ? d.by : d.peer_out;
        fa.scratch_gate = card == 0 ? d.bg : d.peer_g;
        fa.scratch_up = card == 0 ? d.bu : d.peer_u;
        fa.scratch_down = card == 0 ? d.bd : d.peer_d;
        float* xb = card == 0 ? d.bx : d.peer_x;
        int* ib = card == 0 ? d.bids : d.peer_ids;
        float* wb = card == 0 ? d.bwts : d.peer_wts;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        CUDA_OK(cudaEventRecord(e0));
        for (int r = 0; r < 20; ++r) {
          for (int t = 0; t < bk::BATCH_MAX; ++t) {
            fa.x = xb + std::size_t(t) * bk::MOE_HIDDEN;
            fa.ids = ib + t * bk::MOE_TOP_K;
            fa.weights = wb + t * bk::MOE_TOP_K;
            cudaError_t le = bk::fused_moe_apply(fa, nullptr);
            if (std::getenv("CHAIN_DEBUG")) {
              if (le == cudaSuccess) le = cudaDeviceSynchronize();
              if (le != cudaSuccess) {
                std::printf("FAIL fused layer=%d card=%d t=%d: %s (x=%p ids=%p w=%p y=%p sg=%p su=%p sd=%p "
                            "layers=%p)\n",
                            l, card, t, cudaGetErrorString(le), (void*)fa.x, (void*)fa.ids, (void*)fa.weights,
                            (void*)fa.y, (void*)fa.scratch_gate, (void*)fa.scratch_up, (void*)fa.scratch_down,
                            (void*)fa.layers);
                std::fflush(stdout);
                break;
              }
            } else {
              CUDA_OK(le);
            }
          }
        }
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        fused_only += ms_between(e0, e1) / 20.0f;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
      }
      CUDA_OK(cudaSetDevice(d.dev0));
      std::printf("RATE layer=%d unique=%d/32 batch_n4=%.3f ms  fused_x4=%.3f ms  speedup=%.2fx  "
                  "fused_per_token=%.3f ms\n",
                  l, hp.unique_count, batch_ms, fused_only, fused_only / std::max(1e-6f, batch_ms),
                  fused_only / bk::BATCH_MAX);
    }

    // FP64 oracle over the same real bytes, card 0's experts only, for one layer.
    if (l == 3) {
      std::map<int, std::vector<std::uint8_t>> keep;
      auto host_tab = host_oracle_table(d, l, 0, hp.unique_ids, hp.unique_count, &keep);
      std::vector<float> want(part0.size(), 0.0f);
      bk::BatchMoeArgs oa;
      oa.layers = host_tab.data();
      oa.x = xs.data();
      oa.ids = ids.data();
      oa.weights = wts.data();
      oa.y = want.data();
      oa.n = bk::BATCH_MAX;
      std::printf("stage: FP64 oracle over %zu real experts\n", keep.size());
      std::fflush(stdout);
      bk::batch_moe_apply_cpu(oa, hp);
      double w2 = 0.0, sq2 = 0.0;
      for (std::size_t i = 0; i < part0.size(); ++i) {
        w2 = std::max(w2, std::fabs(double(part0[i]) - double(want[i])));
        sq2 += double(want[i]) * double(want[i]);
      }
      const double rms2 = std::sqrt(sq2 / double(part0.size()));
      check(w2 <= 1e-5 * rms2, "layer 3 batch n=4 card-0 experts vs FP64 oracle on real bytes: ratio=" +
                                   std::to_string(w2 / rms2) + " rms=" + std::to_string(rms2));
    }
  }

  // ---- Part C: dense projections at verify width, real weights
  float* gx = nullptr;  // [BATCH_MAX, max cols]
  float* gy = nullptr;  // [BATCH_MAX, max rows]
  CUDA_OK(cudaMalloc(&gx, std::size_t(bk::BATCH_MAX) * 8192 * 4));
  CUDA_OK(cudaMalloc(&gy, std::size_t(bk::BATCH_MAX) * 8192 * 4));
  CUDA_OK(cudaMemset(gx, 0, std::size_t(bk::BATCH_MAX) * 8192 * 4));
  if (!std::getenv("CHAIN_PARTS") || std::string(std::getenv("CHAIN_PARTS")).find('c') != std::string::npos) {
  {
    struct Item {
      const char* what;
      std::string base;
      int rows;
      int cols;
    };
    const std::vector<Item> items = {
        {"gdn in_proj_qkv", layer_name(0, "linear_attn.in_proj_qkv"), qwen::LIN_CONV_DIM, qwen::HIDDEN},
        {"gdn in_proj_z", layer_name(0, "linear_attn.in_proj_z"), qwen::LIN_VALUE_DIM_TOT, qwen::HIDDEN},
        {"gdn out_proj", layer_name(0, "linear_attn.out_proj"), qwen::HIDDEN, qwen::LIN_VALUE_DIM_TOT},
        {"attn q_proj", layer_name(3, "self_attn.q_proj"), qwen::Q_PROJ_ROWS, qwen::HIDDEN},
        {"attn o_proj", layer_name(3, "self_attn.o_proj"), qwen::HIDDEN, qwen::Q_DIM},
    };
    for (const auto& it : items) {
      if (!d.placed.count(it.base + ".weight")) {
        std::printf("note: %s absent, skipped\n", it.what);
        continue;
      }
      const std::uint8_t* w = static_cast<const std::uint8_t*>(d.ptr(it.base + ".weight"));
      const float scale = d.scalar(it.base + ".weight_scale", 1.0f);
      for (int nn = 1; nn <= bk::BATCH_MAX; ++nn) {
        bk::Fp8GemvBatchArgs a;
        a.w = w;
        a.scale = scale;
        a.x = gx;
        a.y = gy;
        a.rows = it.rows;
        a.cols = it.cols;
        a.n = nn;
        CUDA_OK(cudaSetDevice(d.dev0));
        if (bk::fp8_gemv_batch(a, nullptr) != cudaSuccess) {
          std::printf("note: %s shape (%d,%d) outside the batch kernel's splits\n", it.what, it.rows, it.cols);
          break;
        }
        CUDA_OK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        CUDA_OK(cudaEventRecord(e0));
        for (int r = 0; r < 20; ++r) CUDA_OK(bk::fp8_gemv_batch(a, nullptr));
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        const float ms = ms_between(e0, e1) / 20.0f;
        const double gb = double(it.rows) * it.cols * (nn == 1 ? 1.0 : 1.0);
        if (nn == 1) {
        std::printf("ALIGN %-16s weight @%p align=%zu\n", it.what, (void*)w,
                    std::size_t(reinterpret_cast<std::uintptr_t>(w) & 15u));
      }
      std::printf("RATE fp8_gemv_batch %s n=%d %.3f ms %.0f GB/s(weight)  per-token %.3f ms\n", it.what, nn,
                    ms, gb / ms / 1e6, ms / nn);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
      }
    }
    // lm_head: NVFP4 [248320, 2048], 0.286 GB, one read for n positions.
    float* hlogits = nullptr;
    CUDA_OK(cudaMalloc(&hlogits, std::size_t(bk::BATCH_MAX) * qwen::VOCAB * 4));
    for (int nn = 1; nn <= bk::BATCH_MAX; ++nn) {
      bk::Nvfp4GemvBatchArgs a;
      a.w = static_cast<const std::uint8_t*>(d.ptr("lm_head.weight"));
      a.scales = static_cast<const std::uint8_t*>(d.ptr("lm_head.weight_scale"));
      a.scale = d.scalar("lm_head.weight_scale_2", 1.0f);
      a.x = gx;
      a.y = hlogits;
      a.rows = qwen::VOCAB;
      a.cols = qwen::HIDDEN;
      a.n = nn;
      CUDA_OK(cudaSetDevice(d.dev0));
      if (bk::nvfp4_gemv_batch(a, nullptr) != cudaSuccess) {
        std::printf("note: lm_head rejected rows=%d\n", a.rows);
        break;
      }
      CUDA_OK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      CUDA_OK(cudaEventRecord(e0));
      for (int r = 0; r < 5; ++r) CUDA_OK(bk::nvfp4_gemv_batch(a, nullptr));
      CUDA_OK(cudaEventRecord(e1));
      CUDA_OK(cudaEventSynchronize(e1));
      const float ms = ms_between(e0, e1) / 5.0f;
      if (nn == 1) {
        std::printf("ALIGN lm_head weight @%p align=%zu scale @%p align=%zu\n", (void*)a.w,
                    std::size_t(reinterpret_cast<std::uintptr_t>(a.w) & 15u), (void*)a.scales,
                    std::size_t(reinterpret_cast<std::uintptr_t>(a.scales) & 15u));
      }
      std::printf("RATE nvfp4_gemv_batch lm_head n=%d %.3f ms %.0f GB/s(weight) per-token %.3f ms\n", nn, ms,
                  286.0 / ms, ms / nn);
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
    }
    cudaFree(hlogits);
  }
  cudaFree(gx);
  cudaFree(gy);

  }
  // ---- Part D: attention depth scaling, batched against n single-token calls
  if (!std::getenv("CHAIN_PARTS") || std::string(std::getenv("CHAIN_PARTS")).find('d') != std::string::npos) {
  {
    const int depths[] = {1024, 32768, 131072, 262144};
    const int cap = 262208;  // past the deepest block so pos0 + n stays inside the cache
    const std::size_t cbytes = bk::kv_cache_bytes(bk::KvType::Q8_0, cap);
    void *kc = nullptr, *vc = nullptr;
    CUDA_OK(cudaMalloc(&kc, cbytes));
    CUDA_OK(cudaMalloc(&vc, cbytes));
    CUDA_OK(cudaMemset(kc, 0, cbytes));
    CUDA_OK(cudaMemset(vc, 0, cbytes));
    float* aout = nullptr;
    CUDA_OK(cudaMalloc(&aout, std::size_t(bk::BATCH_MAX) * qwen::HEADS * qwen::HEAD_DIM * 4));
    float* aq = nullptr;
    float* akv = nullptr;
    CUDA_OK(cudaMalloc(&aq, std::size_t(bk::BATCH_MAX) * qwen::HEADS * qwen::HEAD_DIM * 4));
    CUDA_OK(cudaMalloc(&akv, std::size_t(bk::BATCH_MAX) * qwen::KV_HEADS * qwen::HEAD_DIM * 4));
    bk::BatchSplitState* bpartial = nullptr;
    CUDA_OK(cudaMalloc(&bpartial, sizeof(bk::BatchSplitState) * qwen::HEADS * bk::kMaxSplits));
    bk::SplitState* spartial = nullptr;
    CUDA_OK(cudaMalloc(&spartial, sizeof(bk::SplitState) * qwen::HEADS * bk::kMaxSplits));
    for (int depth : depths) {
      const int pos0 = depth - 1;
      for (int nn = 1; nn <= bk::BATCH_MAX; ++nn) {
        const int splits = std::max(1, std::min(bk::kMaxSplits, (pos0 + nn + 511) / 512));
        bk::BatchAttnArgs a;
        a.q = aq;
        a.k_new = akv;
        a.v_new = akv;
        a.k_cache = kc;
        a.v_cache = vc;
        a.kv_type_k = bk::KvType::Q8_0;
        a.kv_type_v = bk::KvType::Q8_0;
        a.out = aout;
        a.pos0 = pos0;
        a.max_pos = cap;
        a.partial = bpartial;
        a.splits = splits;
        a.n = nn;
        CUDA_OK(cudaSetDevice(d.dev0));
        CUDA_OK(bk::batch_attention(a, nullptr));
        CUDA_OK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        CUDA_OK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) CUDA_OK(bk::batch_attention(a, nullptr));
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        const float ms = ms_between(e0, e1) / 10.0f;
        const double gb = 2.0 * double(depth) * qwen::KV_HEADS * qwen::HEAD_DIM;
        std::printf("RATE batch_attention depth=%d n=%d splits=%d %.3f ms %.0f GB/s(cache) per-token %.3f ms\n",
                    depth, nn, splits, ms, gb / ms / 1e6, ms / nn);

        // the shipped path: n independent single-token decodes over the same cache
        bk::DecodeAttnArgs s;
        s.q = aq;
        s.k_new = akv;
        s.v_new = akv;
        s.k_cache = kc;
        s.v_cache = vc;
        s.kv_type_k = bk::KvType::Q8_0;
        s.kv_type_v = bk::KvType::Q8_0;
        s.out = aout;
        s.max_pos = cap;
        s.partial = spartial;
        s.splits = splits;
        CUDA_OK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) {
          for (int t = 0; t < nn; ++t) {
            s.pos = pos0 + t;
            CUDA_OK(bk::decode_attention(s, nullptr));
          }
        }
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        const float ms1 = ms_between(e0, e1) / 10.0f;
        std::printf("RATE single_attention depth=%d n=%d %.3f ms (per-token %.3f) batched speedup %.2fx\n", depth,
                    nn, ms1, ms1 / nn, ms1 / std::max(1e-6f, ms));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
      }
    }
    cudaFree(kc);
    cudaFree(vc);
    cudaFree(aout);
    cudaFree(aq);
    cudaFree(akv);
    cudaFree(bpartial);
    cudaFree(spartial);
  }

  }
  // ---- Part E: the delta-rule recurrence, staged (registers) against n single-token steps
  if (!std::getenv("CHAIN_PARTS") || std::string(std::getenv("CHAIN_PARTS")).find('e') != std::string::npos) {
  {
    float *sq = nullptr, *skv = nullptr, *sv = nullptr, *sb = nullptr, *sg = nullptr, *sout = nullptr,
          *ss2 = nullptr;
    const std::size_t qk = std::size_t(bk::BATCH_MAX) * qwen::LIN_KEY_HEADS * qwen::LIN_KEY_DIM;
    const std::size_t vv = std::size_t(bk::BATCH_MAX) * qwen::LIN_VALUE_HEADS * qwen::LIN_VALUE_DIM;
    CUDA_OK(cudaMalloc(&sq, qk * 4));
    CUDA_OK(cudaMalloc(&skv, qk * 4));
    CUDA_OK(cudaMalloc(&sv, vv * 4));
    CUDA_OK(cudaMalloc(&sb, std::size_t(bk::BATCH_MAX) * qwen::LIN_VALUE_HEADS * 4));
    CUDA_OK(cudaMalloc(&sg, std::size_t(bk::BATCH_MAX) * qwen::LIN_VALUE_HEADS * 4));
    CUDA_OK(cudaMalloc(&sout, vv * 4));
    const std::size_t st = std::size_t(qwen::LIN_VALUE_HEADS) * qwen::LIN_KEY_DIM * qwen::LIN_VALUE_DIM;
    CUDA_OK(cudaMalloc(&ss2, st * 4));
    CUDA_OK(cudaMemset(sq, 0, qk * 4));
    CUDA_OK(cudaMemset(skv, 0, qk * 4));
    CUDA_OK(cudaMemset(sv, 0, vv * 4));
    CUDA_OK(cudaMemset(sb, 0, std::size_t(bk::BATCH_MAX) * qwen::LIN_VALUE_HEADS * 4));
    CUDA_OK(cudaMemset(sg, 0, std::size_t(bk::BATCH_MAX) * qwen::LIN_VALUE_HEADS * 4));
    CUDA_OK(cudaMemcpy(ss2, d.gdn_state, st * 4, cudaMemcpyDeviceToDevice));
    float* sstate1 = nullptr;
    CUDA_OK(cudaMalloc(&sstate1, st * 4));
    CUDA_OK(cudaMemcpy(sstate1, d.gdn_state, st * 4, cudaMemcpyDeviceToDevice));
    for (int nn = 1; nn <= bk::BATCH_MAX; ++nn) {
      bk::GdnStepsArgs a;
      a.q = sq;
      a.k = skv;
      a.v = sv;
      a.beta = sb;
      a.g = sg;
      a.state = ss2;
      a.out = sout;
      a.v_heads = qwen::LIN_VALUE_HEADS;
      a.n = nn;
      CUDA_OK(cudaSetDevice(d.dev0));
      CUDA_OK(bk::gdn_recurrent_steps(a, nullptr));
      CUDA_OK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      CUDA_OK(cudaEventRecord(e0));
      for (int r = 0; r < 50; ++r) CUDA_OK(bk::gdn_recurrent_steps(a, nullptr));
      CUDA_OK(cudaEventRecord(e1));
      CUDA_OK(cudaEventSynchronize(e1));
      const float ms = ms_between(e0, e1) / 50.0f;

      gdn::GdnStepArgs s;
      s.q = sq;
      s.k = skv;
      s.v = sv;
      s.beta = sb;
      s.g = sg;
      s.state = sstate1;
      s.out = sout;
      s.v_heads = qwen::LIN_VALUE_HEADS;
      CUDA_OK(cudaEventRecord(e0));
      for (int r = 0; r < 50; ++r) {
        for (int t = 0; t < nn; ++t) CUDA_OK(gdn::gdn_recurrent_step(s, nullptr));
      }
      CUDA_OK(cudaEventRecord(e1));
      CUDA_OK(cudaEventSynchronize(e1));
      const float ms1 = ms_between(e0, e1) / 50.0f;
      std::printf("RATE gdn layer n=%d staged=%.3f ms loop=%.3f ms speedup=%.2fx (2 MB state, 4 MB traffic)\n",
                  nn, ms, ms1, ms1 / std::max(1e-6f, ms));
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
    }
    cudaFree(sq);
    cudaFree(skv);
    cudaFree(sv);
    cudaFree(sb);
    cudaFree(sg);
    cudaFree(sout);
    cudaFree(ss2);
    cudaFree(sstate1);
  }

  }
  // ---- Part F: the expert A/B across all 40 layers in sequence, so neither side runs on a warm L2 the way
  // a repeated single-layer loop does. This is the number the cost model should use.
  if (!std::getenv("CHAIN_PARTS") || std::string(std::getenv("CHAIN_PARTS")).find('f') != std::string::npos) {
    const int p0 = d.positions - bk::BATCH_MAX;
    double uniq_sum = 0.0;
    int uniq_n = 0;
    float batch_all = 0.0f, fused_all = 0.0f;
    const int reps = 3;
    for (int rep = 0; rep < reps; ++rep) {
      for (int l = 0; l < qwen::LAYERS; ++l) {
        std::vector<float> xs(std::size_t(bk::BATCH_MAX) * bk::MOE_HIDDEN);
        std::vector<int> ids(bk::BATCH_PAIRS);
        std::vector<float> wts(bk::BATCH_PAIRS, 1.0f / bk::MOE_TOP_K);
        for (int t = 0; t < bk::BATCH_MAX; ++t) {
          const float* src = d.h2_log.data() + (std::size_t(p0 + t) * layers + l) * qwen::HIDDEN;
          std::copy_n(src, bk::MOE_HIDDEN, xs.begin() + std::ptrdiff_t(t) * bk::MOE_HIDDEN);
          for (int k = 0; k < bk::MOE_TOP_K; ++k) {
            ids[std::size_t(t * bk::MOE_TOP_K + k)] =
                d.ids_log[(std::size_t(p0 + t) * layers + l) * bk::MOE_TOP_K + k];
          }
        }
        bk::BatchMoePlanHost hp;
        bk::batch_moe_plan_cpu(ids.data(), bk::BATCH_MAX, &hp);
        if (rep == reps - 1) {
          uniq_sum += hp.unique_count;
          ++uniq_n;
        }
        CUDA_OK(cudaSetDevice(d.dev0));
        CUDA_OK(cudaMemcpy(d.bx, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d.bids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d.bwts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaSetDevice(d.dev1));
        CUDA_OK(cudaMemcpy(d.peer_x, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d.peer_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d.peer_wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));

        auto time_batch = [&](int card) {
          CUDA_OK(cudaSetDevice(card == 0 ? d.dev0 : d.dev1));
          bk::BatchMoeArgs a;
          a.layers = d.batch_dev[card] + std::size_t(l) * qwen::EXPERTS;
          a.x = card == 0 ? d.bx : d.peer_x;
          a.ids = card == 0 ? d.bids : d.peer_ids;
          a.weights = card == 0 ? d.bwts : d.peer_wts;
          a.y = card == 0 ? d.by : d.peer_out;
          a.scratch_gate = card == 0 ? d.bg : d.peer_g;
          a.scratch_up = card == 0 ? d.bu : d.peer_u;
          a.scratch_down = card == 0 ? d.bd : d.peer_d;
          a.plan = card == 0 ? d.plan : d.plan1;
          a.n = bk::BATCH_MAX;
          cudaEvent_t e0, e1;
          cudaEventCreate(&e0);
          cudaEventCreate(&e1);
          CUDA_OK(cudaEventRecord(e0));
          CUDA_OK(bk::batch_moe_apply(a, nullptr));
          CUDA_OK(cudaEventRecord(e1));
          CUDA_OK(cudaEventSynchronize(e1));
          batch_all += ms_between(e0, e1);
          cudaEventDestroy(e0);
          cudaEventDestroy(e1);
        };
        auto time_fused = [&](int card) {
          CUDA_OK(cudaSetDevice(card == 0 ? d.dev0 : d.dev1));
          bk::FusedMoeArgs fa;
          fa.layers = d.fused_dev[card] + std::size_t(l) * qwen::EXPERTS;
          fa.y = card == 0 ? d.by : d.peer_out;
          fa.scratch_gate = card == 0 ? d.bg : d.peer_g;
          fa.scratch_up = card == 0 ? d.bu : d.peer_u;
          fa.scratch_down = card == 0 ? d.bd : d.peer_d;
          float* xb = card == 0 ? d.bx : d.peer_x;
          int* ib = card == 0 ? d.bids : d.peer_ids;
          float* wb = card == 0 ? d.bwts : d.peer_wts;
          cudaEvent_t e0, e1;
          cudaEventCreate(&e0);
          cudaEventCreate(&e1);
          CUDA_OK(cudaEventRecord(e0));
          for (int t = 0; t < bk::BATCH_MAX; ++t) {
            fa.x = xb + std::size_t(t) * bk::MOE_HIDDEN;
            fa.ids = ib + t * bk::MOE_TOP_K;
            fa.weights = wb + t * bk::MOE_TOP_K;
            CUDA_OK(bk::fused_moe_apply(fa, nullptr));
          }
          CUDA_OK(cudaEventRecord(e1));
          CUDA_OK(cudaEventSynchronize(e1));
          fused_all += ms_between(e0, e1);
          cudaEventDestroy(e0);
          cudaEventDestroy(e1);
        };
        // One card at a time, its own events, its device current. The shipped step overlaps the two cards,
        // so summing their windows is a conservative stand-in for the wall time on both sides of the A/B.
        time_batch(0);
        time_batch(1);
        time_fused(0);
        time_fused(1);
      }
    }
    CUDA_OK(cudaSetDevice(d.dev0));
    std::printf("COLD experts over %d layers: batch_n4=%.3f ms/step (%.3f ms/token)  fused_x4=%.3f ms/step "
                "(%.3f ms/token)  speedup=%.2fx  mean unique=%.1f/32\n",
                qwen::LAYERS, batch_all / reps, batch_all / reps / bk::BATCH_MAX, fused_all / reps,
                fused_all / reps / bk::BATCH_MAX, fused_all / std::max(1e-6f, batch_all),
                uniq_sum / std::max(1, uniq_n));
  }

  qwen::free_text_arena(&arena);
  std::printf("%s (%ld checks)\n", ok_all ? "ALL PASS" : "FAILURES PRESENT", checks_run);
  return ok_all ? 0 : 1;
}
