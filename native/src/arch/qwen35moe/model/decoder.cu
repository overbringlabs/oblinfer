#include "arch/qwen35moe/model/decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/attn/rope_norm.h"
#include "arch/qwen35moe/kernels/moe/moe_apply.h"
#include "arch/qwen35moe/kernels/moe/router.h"
#include "arch/qwen35moe/kernels/moe/shared_expert.h"
#include "arch/qwen35moe/model/model_data.h"
#include "kernels/dense/dense_ops.h"
#include "kernels/gdn/conv.h"
#include "kernels/gdn/gates.h"
#include "kernels/gdn/gdn.h"
#include "kernels/quant/gemv.h"

namespace ingine::arch::qwen35moe {
namespace {

namespace attn = ingine::arch::qwen35moe::kernels;
namespace gdn = ingine::kernels::gdn;
namespace dense = ingine::kernels::dense;
namespace quant = ingine::kernels::quant;

constexpr const char* kLayer = "model.language_model.layers.";
std::string layer_name(int l, const char* suffix) {
  return std::string(kLayer) + std::to_string(l) + "." + suffix;
}

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void* alloc(std::size_t bytes) {
  void* p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

const char* kv_name(attn::KvType type) {
  return type == attn::KvType::Q8_0 ? "q8_0" : type == attn::KvType::Q4_0 ? "q4_0" : "bf16";
}

}  // namespace

Decoder::Decoder() = default;

Decoder::~Decoder() {
  // The step graphs before anything they reference: their child nodes clone the per-layer hop graphs, which go
  // with the rest of device 1 further down.
  cudaSetDevice(dev0_);
  for (auto& [splits, graph] : graphs_) {
    (void)splits;
    decode_graph_free(&graph);
  }
  graphs_.clear();
  if (cap0_) cudaStreamDestroy(cap0_);
  if (token_dev_) cudaFree(token_dev_);
  if (pos_dev_) cudaFree(pos_dev_);
  for (float* p : {x_, h_, h2_, y_, qkv_, qkv2_, z_, a_, b_, beta_, g_, gout_, q_lin_, k_lin_, v_lin_, qgate_,
                   k_, v_, attn_, q_attn_, gate_attn_, moe_out_, shared_out_, shared_h_, shared_up_, logits_, gdn_state_,
                   conv_state_}) {
    if (p) cudaFree(p);
  }
  // A banded step leaves work on the peer stream; the bands and the staging it writes into go next.
  if (s1_ && dev1_ != dev0_) {
    cudaSetDevice(dev1_);
    cudaStreamSynchronize(s1_);
    cudaSetDevice(dev0_);
  }
  for (auto& bands : kv_bands_) {
    kv_bands_free(&bands);
  }
  kv_band_exchange_free(&kv_exchange_);
  if (ids_dev_) cudaFree(ids_dev_);
  if (weights_dev_) cudaFree(weights_dev_);
  for (int d = 0; d < 2; ++d) {
    cudaSetDevice(d == 0 ? dev0_ : dev1_);
    for (float* p : {fscratch_gate_[d], fscratch_up_[d], fscratch_down_[d]}) {
      if (p) cudaFree(p);
    }
    if (fused_[d].device) cudaFree(fused_[d].device);
  }
  if (dev1_ != dev0_) {
    if (pinned_) cudaFreeHost(pinned_);
    for (int i = 0; i < 2; ++i) {
      if (ev_x_[i]) cudaEventDestroy(ev_x_[i]);
      if (ev_x_free_[i]) cudaEventDestroy(ev_x_free_[i]);
      if (ev_y_[i]) cudaEventDestroy(ev_y_[i]);
      if (ev_y_free_[i]) cudaEventDestroy(ev_y_free_[i]);
    }
    if (s1_) cudaStreamDestroy(s1_);
  }
  if (peer_ids_) cudaFree(peer_ids_);
  if (peer_weights_) cudaFree(peer_weights_);
  cudaSetDevice(dev0_);
  for (auto& [name, buffer] : f32_) {
    (void)name;
    if (buffer) cudaFree(buffer);
  }
  if (router_logits_) cudaFree(router_logits_);
  if (kv_partial_) cudaFree(kv_partial_);
  if (argmax_dev_) cudaFree(argmax_dev_);
  if (dev1_ != dev0_) {
    cudaSetDevice(dev1_);
    // The hop graphs were captured with device 1 current.
    for (cudaGraph_t hop : hop_graphs_) {
      if (hop) cudaGraphDestroy(hop);
    }
    hop_graphs_.clear();
    for (float* p : {peer_x_, peer_h_, peer_up_, peer_out_}) {
      if (p) cudaFree(p);
    }
    cudaSetDevice(dev0_);
    if (pinned_in_) cudaFreeHost(pinned_in_);
    if (pinned_out_) cudaFreeHost(pinned_out_);
  }
}

void* Decoder::ptr(const std::string& name) const {
  const auto it = placed_.find(name);
  if (it == placed_.end()) throw std::runtime_error("tensor not placed: " + name);
  const PlacedTensor& p = *it->second;
  return static_cast<std::byte*>(arena_->spans[std::size_t(p.span)].device) + p.offset;
}

const float* Decoder::f32(const std::string& name, bool plus_one) {
  const std::string key = plus_one ? name + "#1" : name;
  const auto hit = f32_.find(key);
  if (hit != f32_.end()) return hit->second;
  const auto it = placed_.find(name);
  if (it == placed_.end()) throw std::runtime_error("tensor not placed: " + name);
  const PlacedTensor& p = *it->second;
  const std::size_t count = p.bytes / 2;
  std::vector<__nv_bfloat16> staging(count);
  check(cudaMemcpy(staging.data(), ptr(name), p.bytes, cudaMemcpyDeviceToHost), "bf16 read");
  std::vector<float> fp32(count);
  for (std::size_t i = 0; i < count; ++i) fp32[i] = __bfloat162float(staging[i]) + (plus_one ? 1.0f : 0.0f);
  float* device = static_cast<float*>(alloc(count * 4));
  check(cudaMemcpy(device, fp32.data(), count * 4, cudaMemcpyHostToDevice), "fp32 write");
  f32_[key] = device;
  return device;
}

float Decoder::scalar(const std::string& name, float fallback) const {
  const auto it = scalars_.find(name);
  return it == scalars_.end() ? fallback : it->second;
}

bool Decoder::init(const TextArena& arena, int max_pos, std::string* error) {
  try {
    arena_ = &arena;
    max_pos_ = max_pos;
    dev0_ = arena.spans.empty() ? 0 : arena.spans[0].cuda_device;
    dev1_ = arena.spans.size() > 1 ? arena.spans[1].cuda_device : dev0_;
    check(cudaSetDevice(dev0_), "set device 0");
    for (const PlacedTensor& p : arena.placed) placed_[p.name] = &p;

    // Every 4-byte F32 tensor ending in a scale name is a per-tensor scalar.
    for (const PlacedTensor& p : arena.placed) {
      if (p.bytes != 4) continue;
      if (p.name.ends_with(".weight_scale") || p.name.ends_with(".weight_scale_2") ||
          p.name.ends_with(".input_scale")) {
        float value = 0.0f;
        check(cudaSetDevice(arena.spans[std::size_t(p.span)].cuda_device), "scale device");
        check(cudaMemcpy(&value, ptr(p.name), 4, cudaMemcpyDeviceToHost), "scale read");
        scalars_[p.name] = value;
      }
    }
    check(cudaSetDevice(dev0_), "set device 0");

    x_ = static_cast<float*>(alloc(HIDDEN * 4));
    h_ = static_cast<float*>(alloc(HIDDEN * 4));
    h2_ = static_cast<float*>(alloc(HIDDEN * 4));
    y_ = static_cast<float*>(alloc(HIDDEN * 4));
    qkv_ = static_cast<float*>(alloc(LIN_CONV_DIM * 4));
    qkv2_ = static_cast<float*>(alloc(LIN_CONV_DIM * 4));
    z_ = static_cast<float*>(alloc(LIN_VALUE_DIM_TOT * 4));
    a_ = static_cast<float*>(alloc(LIN_VALUE_HEADS * 4));
    b_ = static_cast<float*>(alloc(LIN_VALUE_HEADS * 4));
    beta_ = static_cast<float*>(alloc(LIN_VALUE_HEADS * 4));
    g_ = static_cast<float*>(alloc(LIN_VALUE_HEADS * 4));
    gout_ = static_cast<float*>(alloc(LIN_VALUE_DIM_TOT * 4));
    q_lin_ = static_cast<float*>(alloc(LIN_KEY_DIM_TOT * 4));
    k_lin_ = static_cast<float*>(alloc(LIN_KEY_DIM_TOT * 4));
    v_lin_ = static_cast<float*>(alloc(LIN_VALUE_DIM_TOT * 4));
    grouped_qkv_ = std::getenv("QWEN35MOE_GROUPED_QKV") != nullptr;
    qgate_ = static_cast<float*>(alloc(Q_PROJ_ROWS * 4));
    k_ = static_cast<float*>(alloc(KV_DIM * 4));
    v_ = static_cast<float*>(alloc(KV_DIM * 4));
    attn_ = static_cast<float*>(alloc(Q_DIM * 4));
    q_attn_ = static_cast<float*>(alloc(Q_DIM * 4));
    gate_attn_ = static_cast<float*>(alloc(Q_DIM * 4));
    moe_out_ = static_cast<float*>(alloc(HIDDEN * 4));
    shared_out_ = static_cast<float*>(alloc(HIDDEN * 4));
    shared_h_ = static_cast<float*>(alloc(MOE_INTER * 4));
    shared_up_ = static_cast<float*>(alloc(MOE_INTER * 4));
    logits_ = static_cast<float*>(alloc(std::size_t(VOCAB) * 4));
    logits_host_.resize(VOCAB);

    // KV element formats, selected per cache like llama-server's -ctk/-ctv/-ctkd/-ctvd. q8_0 is a quarter of
    // BF16 and is what makes 262144 fit on two 16 GB cards; q4_0 is half of q8_0 again and stays available per
    // side, but it is not the trunk default (K8V8 loses nothing against what supra runs today).
    auto parse_kv = [](const char* name, attn::KvType fallback) {
      const char* v = std::getenv(name);
      if (!v) return fallback;
      if (std::string(v) == "q8_0") return attn::KvType::Q8_0;
      if (std::string(v) == "q4_0") return attn::KvType::Q4_0;
      if (std::string(v) == "bf16" || std::string(v) == "f16") return attn::KvType::Bf16;
      throw std::runtime_error(std::string(name) + " must be bf16, q8_0 or q4_0, got " + v);
    };
    kv_type_k_ = parse_kv("QWEN35MOE_CTK", attn::KvType::Bf16);
    kv_type_v_ = parse_kv("QWEN35MOE_CT_V", attn::KvType::Bf16);
    // The drafter's own cache is the -ctkd/-ctvd of the speculation plan. It only has to propose tokens, so a
    // coarse cache costs acceptance rather than correctness and K4V4 is its default. No drafter module exists
    // yet: the pair carries the setting and its default, and allocates nothing.
    kv_type_k_drafter_ = parse_kv("QWEN35MOE_CTKD", attn::KvType::Q4_0);
    kv_type_v_drafter_ = parse_kv("QWEN35MOE_CT_VD", attn::KvType::Q4_0);
    std::fprintf(stderr, "[qwen35moe] KV cache: trunk K=%s V=%s, drafter K=%s V=%s (drafter cache not built)\n",
                 kv_name(kv_type_k_), kv_name(kv_type_v_), kv_name(kv_type_k_drafter_),
                 kv_name(kv_type_v_drafter_));
    std::fflush(stderr);
    // Position bands: on by default, because the capacity gate below keeps them dormant under 98304 positions
    // and from there up they are a measured 9% (258048: 20.99 -> 19.28 ms/token). QWEN35MOE_KV_BANDS=0 pins the
    // single cache; with bands off the plan is one band covering the whole capacity on card 0, same allocation
    // and same launch as before.
    const char* bands_env = std::getenv("QWEN35MOE_KV_BANDS");
    kv_bands_on_ = !bands_env || std::string(bands_env) != "0";
    if (const char* r = std::getenv("QWEN35MOE_KV_BAND_RATIO")) {
      kv_band_ratio_ = float(std::atof(r));
      if (!(kv_band_ratio_ > 0.0f)) throw std::runtime_error(std::string("QWEN35MOE_KV_BAND_RATIO must be > 0, got ") + r);
    }
    if (kv_bands_on_ && dev1_ == dev0_) {
      std::fprintf(stderr, "[qwen35moe] KV bands idle: one device holds both bands\n");
      std::fflush(stderr);
      kv_bands_on_ = false;
    }
    // Measured break-even (supra, q8_0, gen 8, eager, ms/token): 65536 14.30 -> 14.85 loss, 98304 15.52 ->
    // 15.39 gain, 114688 16.29 -> 16.01, 131072 16.51 -> 16.22, 258048 20.99 -> 19.28. A banded step costs
    // roughly 0.5 ms/token of pinned staging whatever the depth and saves a share of the cache read that grows
    // with it, so the plan only splits from 98304 positions up. QWEN35MOE_KV_BAND_MIN_POS=0 forces the split at
    // any capacity, which is how the shallow parity and sanitizer runs exercise the banded path.
    kv_band_min_pos_ = 98304;
    if (const char* m = std::getenv("QWEN35MOE_KV_BAND_MIN_POS")) kv_band_min_pos_ = std::atoi(m);
    const bool engage = kv_bands_on_ && max_pos >= kv_band_min_pos_;
    if (kv_bands_on_ && !engage) {
      std::fprintf(stderr, "[qwen35moe] QWEN35MOE_KV_BANDS=1 idle: capacity %d below the %d break-even\n", max_pos,
                   kv_band_min_pos_);
      std::fflush(stderr);
    }
    // One plan per full-attention layer. The decision is all-or-nothing: a mix of two-band and one-band
    // layers would leave the plain path pointing at a band 0 that does not cover the capacity.
    auto plan_all = [&](bool split, std::string* why) -> bool {
      for (int i = 0; i < NUM_FULL_LAYERS; ++i) {
        if (!kv_bands_init(&kv_bands_[i], max_pos, dev0_, split ? dev1_ : dev0_, kv_type_k_, kv_type_v_, why,
                           kv_band_ratio_))
          return false;
        if (split && kv_bands_[i].count != 2) {
          *why = "layer " + std::to_string(i) + " collapsed to one band";
          return false;
        }
      }
      return true;
    };
    auto free_all_bands = [&]() {
      for (int i = 0; i < NUM_FULL_LAYERS; ++i) kv_bands_free(&kv_bands_[i]);
    };
    std::string band_error;
    if (engage && plan_all(true, &band_error)) {
      kv_bands_on_ = true;
    } else {
      if (engage) {
        std::fprintf(stderr, "[qwen35moe] bands off: %s\n", band_error.c_str());
        std::fflush(stderr);
      }
      free_all_bands();
      band_error.clear();
      if (!plan_all(false, &band_error)) {
        if (error) *error = band_error;
        return false;
      }
      kv_bands_on_ = false;
    }
    // Flash-decoding scratch: one SplitState per (query head, position split).
    kv_partial_ = static_cast<kernels::SplitState*>(
        alloc(std::size_t(HEADS) * kernels::kMaxSplits * sizeof(kernels::SplitState)));
    gdn_state_ = static_cast<float*>(
        alloc(std::size_t(NUM_LINEAR_LAYERS) * LIN_VALUE_HEADS * LIN_KEY_DIM * LIN_VALUE_DIM * 4));
    conv_state_ = static_cast<float*>(alloc(std::size_t(NUM_LINEAR_LAYERS) * (CONV_K - 1) * LIN_CONV_DIM * 4));
    check(cudaMemset(gdn_state_, 0, std::size_t(NUM_LINEAR_LAYERS) * LIN_VALUE_HEADS * LIN_KEY_DIM * LIN_VALUE_DIM * 4),
          "state zero");
    check(cudaMemset(conv_state_, 0, std::size_t(NUM_LINEAR_LAYERS) * (CONV_K - 1) * LIN_CONV_DIM * 4), "conv zero");

    if (dev1_ != dev0_) {
      check(cudaSetDevice(dev1_), "set device 1");
      check(cudaStreamCreateWithFlags(&s1_, cudaStreamNonBlocking), "stream 1");
      for (int i = 0; i < 2; ++i) {
        check(cudaEventCreateWithFlags(&ev_x_free_[i], cudaEventDisableTiming), "event x free");
        check(cudaEventCreateWithFlags(&ev_y_[i], cudaEventDisableTiming), "event y");
      }
      peer_x_ = static_cast<float*>(alloc(HIDDEN * 4));
      peer_h_ = static_cast<float*>(alloc(MOE_INTER * 4));
      peer_up_ = static_cast<float*>(alloc(MOE_INTER * 4));
      peer_out_ = static_cast<float*>(alloc(HIDDEN * 4));
      check(cudaSetDevice(dev0_), "set device 0");
      check(cudaHostAlloc(reinterpret_cast<void**>(&pinned_), 2 * sizeof(Exchange), cudaHostAllocDefault),
            "host alloc");
      for (int i = 0; i < 2; ++i) {
        check(cudaEventCreateWithFlags(&ev_x_[i], cudaEventDisableTiming), "event x");
        check(cudaEventCreateWithFlags(&ev_y_free_[i], cudaEventDisableTiming), "event y free");
      }
    }

    router_logits_ = static_cast<float*>(alloc(attn::ROUTER_EXPERTS * 4));
    ids_dev_ = static_cast<int*>(alloc(attn::ROUTER_TOP_K * 4));
    weights_dev_ = static_cast<float*>(alloc(attn::ROUTER_TOP_K * 4));
    argmax_dev_ = static_cast<int*>(alloc(4));

    // Per-expert pointers, one row per expert per layer.
    experts_.resize(LAYERS);
    for (int l = 0; l < LAYERS; ++l) {
      experts_[l].resize(EXPERTS);
      for (int e = 0; e < EXPERTS; ++e) {
        const std::string base = layer_name(l, "mlp.experts.") + std::to_string(e) + ".";
        auto fill = [&](ExpertRow::Args& args, const char* proj) {
          const std::string name = base + proj + "_proj";
          args.w = static_cast<const std::uint8_t*>(ptr(name + ".weight"));
          args.scales = static_cast<const std::uint8_t*>(ptr(name + ".weight_scale"));
          args.scale = scalar(name + ".weight_scale_2", 1.0f);
        };
        fill(experts_[l][e].gate, "gate");
        fill(experts_[l][e].up, "up");
        fill(experts_[l][e].down, "down");
        experts_[l][e].span = placed_.at(base + "gate_proj.weight")->span;
      }
    }

    // Fused MoE tables, one per device: entry [l * EXPERTS + e]. A foreign expert carries stride 0 and the
    // kernel skips it, so both cards run the same router ids and each contributes its own experts.
    for (int d = 0; d < 2; ++d) {
      fused_[d].host.assign(std::size_t(LAYERS) * EXPERTS, kernels::FusedExpertLayer{});
      check(cudaSetDevice(d == 0 ? dev0_ : dev1_), "set scratch device");
      fscratch_gate_[d] = static_cast<float*>(alloc(std::size_t(attn::ROUTER_TOP_K) * MOE_INTER * 4));
      fscratch_up_[d] = static_cast<float*>(alloc(std::size_t(attn::ROUTER_TOP_K) * MOE_INTER * 4));
      fscratch_down_[d] = static_cast<float*>(alloc(std::size_t(attn::ROUTER_TOP_K) * HIDDEN * 4));
    }
    check(cudaSetDevice(dev0_), "set device 0");
    for (int l = 0; l < LAYERS; ++l) {
      const LayerPlan& lp = arena.plan.layers[std::size_t(l)];
      if (!lp.has_experts) throw std::runtime_error("layer " + std::to_string(l) + " has no expert block");
      int off[3][4] = {};
      for (const TensorSlot& slot : lp.experts.slots) {
        const int proj = slot.projection == Projection::GateProj ? 0 : slot.projection == Projection::UpProj ? 1 : 2;
        const int art = slot.artifact == Artifact::Weight          ? 0
                        : slot.artifact == Artifact::WeightScale   ? 1
                        : slot.artifact == Artifact::WeightScale2  ? 2
                                                                   : 3;
        off[proj][art] = int(slot.offset);
      }
      const int stride = int(lp.experts.stride);
      for (int e = 0; e < EXPERTS; ++e) {
        const std::string ebase = layer_name(l, "mlp.experts.") + std::to_string(e) + ".";
        const int span = placed_.at(ebase + "gate_proj.weight")->span;
        // weight_scale_2 is per expert, not per layer.
        const float gate2 = scalar(ebase + "gate_proj.weight_scale_2", 1.0f);
        const float up2 = scalar(ebase + "up_proj.weight_scale_2", 1.0f);
        const float down2 = scalar(ebase + "down_proj.weight_scale_2", 1.0f);
        for (int d = 0; d < 2; ++d) {
          kernels::FusedExpertLayer& f = fused_[d].host[std::size_t(l) * EXPERTS + e];
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
          f.gate_scale2 = gate2;
          f.up_scale2 = up2;
          f.down_scale2 = down2;
          f.w_base = static_cast<const std::uint8_t*>(ptr(ebase + "gate_proj.weight")) -
                     std::size_t(e) * std::size_t(stride) - std::size_t(off[0][0]);
          f.s_base = static_cast<const std::uint8_t*>(ptr(ebase + "gate_proj.weight_scale")) -
                     std::size_t(e) * std::size_t(stride) - std::size_t(off[0][1]);
        }
      }
    }
    for (int d = 0; d < 2; ++d) {
      const std::size_t bytes = fused_[d].host.size() * sizeof(kernels::FusedExpertLayer);
      check(cudaSetDevice(d == 0 ? dev0_ : dev1_), "set table device");
      fused_[d].device = static_cast<kernels::FusedExpertLayer*>(alloc(bytes));
      check(cudaMemcpy(fused_[d].device, fused_[d].host.data(), bytes, cudaMemcpyHostToDevice), "fused table");
    }
    check(cudaSetDevice(dev1_), "set device 1");
    peer_ids_ = static_cast<int*>(alloc(attn::ROUTER_TOP_K * 4));
    peer_weights_ = static_cast<float*>(alloc(attn::ROUTER_TOP_K * 4));
    check(cudaSetDevice(dev0_), "set device 0");

    // The peer staging for a banded cache, last: kv_band_exchange_init leaves the host card current, and the
    // allocations above must stay on device 1.
    if (kv_bands_on_) {
      std::string exchange_error;
      if (!kv_band_exchange_init(&kv_exchange_, dev0_, dev1_, &exchange_error)) {
        if (error) *error = "kv_band_exchange_init: " + exchange_error;
        return false;
      }
      const KvBands& b = kv_bands_[0];
      const int top = splits_at(max_pos - 1);
      const int c1 = kv_band_columns(b, max_pos - 1, top);
      // Bytes one token reads per position in the cache's own formats (both sides, all full layers).
      const double per_pos = double(attn::kv_cache_bytes(kv_type_k_, 32) + attn::kv_cache_bytes(kv_type_v_, 32)) /
                             32.0 * NUM_FULL_LAYERS / 1e6;
      std::fprintf(stderr,
                   "[qwen35moe] banded KV on: %d layers, boundary at %d/%d positions (%.4f), mirror %d rows; "
                   "%.0f MB on card %d plus %.0f MB on card %d; at the top of the cache card 1 runs %d of %d "
                   "columns and %.0f MB per token, card 0 the rest\n",
                   NUM_FULL_LAYERS, b.cut, b.max_pos, double(b.cut) / double(b.max_pos), b.overlap,
                   (kv_band_bytes(b.bands[0].pos_end - b.bands[0].pos_begin, kv_type_k_) +
                    kv_band_bytes(b.bands[0].pos_end - b.bands[0].pos_begin, kv_type_v_)) * NUM_FULL_LAYERS / 1e6,
                   b.bands[0].device,
                   (kv_band_bytes(b.bands[1].pos_end - b.bands[1].pos_begin, kv_type_k_) +
                    kv_band_bytes(b.bands[1].pos_end - b.bands[1].pos_begin, kv_type_v_)) * NUM_FULL_LAYERS / 1e6,
                   b.bands[1].device, c1 >= 0 ? top - c1 : 0, top,
                   double(b.max_pos - b.cut) * per_pos);
      std::fflush(stderr);
    }

    use_graph_ = std::getenv("QWEN35MOE_GRAPH") && std::string(std::getenv("QWEN35MOE_GRAPH")) == "1";
    // A banded step switches the current device inside the layer loop, which a capture may not contain, so the
    // step stays eager while bands are on. The per-device child graphs the MoE hop uses are the way back.
    if (use_graph_ && kv_bands_on_) {
      std::fprintf(stderr, "[qwen35moe] QWEN35MOE_GRAPH=1 ignored: banded KV runs eager on two cards\n");
      std::fflush(stderr);
      use_graph_ = false;
    }
    // Depth-scaled split count: on by default. The fixed rule asked for one split below 2048 positions, which
    // falls out of the flash path into the old single-pass kernel; measured 68.7 -> 83.6 tok/s at 1k and
    // 57.3 -> 84.1 at 2k, +3% at 4k, and neutral from 32k up. QWEN35MOE_ATTN_SPLITS_DEPTH=0 restores the fixed
    // bucket.
    const char* splits_env = std::getenv("QWEN35MOE_ATTN_SPLITS_DEPTH");
    attn_splits_depth_ = !splits_env || std::string(splits_env) != "0";
    // Capturing a step cannot coexist with these: SWAP_QG reassigns the query/gate buffers on every call, so a
    // fixed graph would not follow the layer loop, and the dump and debug paths read activations back to the
    // host inside the loop.
    for (const char* forbid : {"QWEN35MOE_SWAP_QG", "QWEN35MOE_DUMP", "QWEN35MOE_DEBUG"}) {
      if (use_graph_ && std::getenv(forbid)) {
        std::fprintf(stderr, "[qwen35moe] QWEN35MOE_GRAPH=1 ignored: %s is set\n", forbid);
        std::fflush(stderr);
        use_graph_ = false;
      }
    }
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

void Decoder::fp8_gemv(const std::string& base, const float* x, float* y, int rows, int cols) {
  quant::Fp8GemvArgs args;
  args.w = static_cast<const std::uint8_t*>(ptr(base + ".weight"));
  args.scale = scalar(base + ".weight_scale", 1.0f);
  args.x = x;
  args.y = y;
  args.rows = rows;
  args.cols = cols;
  check(quant::fp8_gemv(args, s_), "fp8 gemv");
}

void Decoder::nvfp4_gemv(const std::string& base, const float* x, float* y, int rows, int cols) {
  quant::Nvfp4GemvArgs args;
  args.w = static_cast<const std::uint8_t*>(ptr(base + ".weight"));
  args.scales = static_cast<const std::uint8_t*>(ptr(base + ".weight_scale"));
  args.scale = scalar(base + ".weight_scale_2", 1.0f);
  args.x = x;
  args.y = y;
  args.rows = rows;
  args.cols = cols;
  check(quant::nvfp4_gemv(args, s_), "nvfp4 gemv");
}

void Decoder::linear_layer(int l) {
  const std::string p = layer_name(l, "linear_attn.");
  fp8_gemv(p + "in_proj_qkv", h_, qkv_, LIN_CONV_DIM, HIDDEN);
  if (std::getenv("QWEN35MOE_DUMP") && l == 0) {
    std::vector<float> q(LIN_CONV_DIM);
    check(cudaMemcpy(q.data(), qkv_, LIN_CONV_DIM * 4, cudaMemcpyDeviceToHost), "dump qkv");
    std::printf("in_proj_qkv:");
    for (int i = 0; i < 4; ++i) std::printf(" %.9g", q[std::size_t(i)]);
    std::printf("\n");
  }
  fp8_gemv(p + "in_proj_z", h_, z_, LIN_VALUE_DIM_TOT, HIDDEN);
  check(dense::gemv_bf16(static_cast<const __nv_bfloat16*>(ptr(p + "in_proj_a.weight")), h_, a_, LIN_VALUE_HEADS,
                         HIDDEN, s_),
        "in_proj_a");
  check(dense::gemv_bf16(static_cast<const __nv_bfloat16*>(ptr(p + "in_proj_b.weight")), h_, b_, LIN_VALUE_HEADS,
                         HIDDEN, s_),
        "in_proj_b");

  // Conv state per linear layer, in checkpoint order (no active layer index compression).
  const int linear_slot = l * NUM_LINEAR_LAYERS / LAYERS;  // only used as a base below
  (void)linear_slot;
  int slot = 0;
  for (int i = 0; i < l; ++i) slot += is_full_layer(i) ? 0 : 1;
  float* conv_hist = conv_state_ + std::size_t(slot) * (CONV_K - 1) * LIN_CONV_DIM;
  gdn::ConvStepArgs conv;
  conv.x = qkv_;
  conv.weight = f32(p + "conv1d.weight");
  conv.history = conv_hist;
  conv.y = qkv2_;
  conv.conv_dim = LIN_CONV_DIM;
  check(gdn::conv_silu_step(conv, s_), "conv");

  gdn::GdnGateArgs gates;
  gates.a = a_;
  gates.b = b_;
  gates.dt_bias = f32(p + "dt_bias");
  gates.a_log = f32(p + "A_log");
  gates.beta = beta_;
  gates.g = g_;
  gates.heads = LIN_VALUE_HEADS;
  check(gdn::gdn_gates(gates, s_), "gdn gates");

  const float* q_ptr = qkv2_;
  const float* k_ptr = qkv2_ + LIN_KEY_DIM_TOT;
  const float* v_ptr = qkv2_ + 2 * LIN_KEY_DIM_TOT;
  if (grouped_qkv_) {
    gdn::GatherQkvArgs g;
    g.qkv = qkv2_;
    g.q = q_lin_;
    g.k = k_lin_;
    g.v = v_lin_;
    g.key_heads = LIN_KEY_HEADS;
    g.key_dim = LIN_KEY_DIM;
    g.value_heads = LIN_VALUE_HEADS;
    g.value_dim = LIN_VALUE_DIM;
    check(gdn::gather_qkv_grouped(g, s_), "gather qkv");
    q_ptr = q_lin_;
    k_ptr = k_lin_;
    v_ptr = v_lin_;
  }

  gdn::GdnStepArgs step;
  step.q = q_ptr;
  step.k = k_ptr;
  step.v = v_ptr;
  step.beta = beta_;
  step.g = g_;
  step.state = gdn_state_ + std::size_t(slot) * LIN_VALUE_HEADS * LIN_KEY_DIM * LIN_VALUE_DIM;
  step.out = gout_;
  step.v_heads = LIN_VALUE_HEADS;
  check(gdn::gdn_recurrent_step(step, s_), "gdn step");

  gdn::GatedNormArgs norm;
  norm.x = gout_;
  norm.z = z_;
  norm.weight = f32(p + "norm.weight");
  norm.v_heads = LIN_VALUE_HEADS;
  norm.v_dim = LIN_VALUE_DIM;
  norm.eps = EPS;
  check(gdn::gated_norm(norm, s_), "gdn norm");

  fp8_gemv(p + "out_proj", gout_, y_, HIDDEN, LIN_VALUE_DIM_TOT);
}

// A by-value kernel argument is frozen when the capture ends, so a launch whose argument carries the position
// has to be found in the graph and rewritten before every replay. Outside a capture this is the plain launch.
template <typename Args, typename Launch>
void Decoder::at(const Args& args, std::size_t pos_offset, Launch&& launch) {
  if (!cap_) {
    launch();
    return;
  }
  cap_->begin(s_);
  launch();
  cap_->end(s_, &args, sizeof(args), pos_offset);
}

void Decoder::clear(void* p, std::size_t bytes) {
  // cudaMemset enqueues no node inside a capture and reports no error: the work would simply be missing.
  check(cudaMemsetAsync(p, 0, bytes, s_), "clear");
}

void Decoder::copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
  check(cudaMemcpyAsync(dst, src, bytes, kind, s_), "copy");
}

void Decoder::read_back(void* dst, const void* src, std::size_t bytes) {
  // A synchronous cudaMemcpy orders the legacy stream only, so a replay on the capture stream needs the copy
  // enqueued there first.
  if (s_) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s_), "read back");
    check(cudaStreamSynchronize(s_), "read back sync");
  } else {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "read back");
  }
}

void Decoder::full_layer(int l, int pos) {
  const std::string p = layer_name(l, "self_attn.");
  const int slot = l / FULL_ATTENTION_INTERVAL;

  fp8_gemv(p + "q_proj", h_, qgate_, Q_PROJ_ROWS, HIDDEN);
  fp8_gemv(p + "k_proj", h_, k_, KV_DIM, HIDDEN);
  fp8_gemv(p + "v_proj", h_, v_, KV_DIM, HIDDEN);

  auto dump = [&](const char* file, const float* buffer, int count) {
    std::vector<float> host(count);
    check(cudaMemcpy(host.data(), buffer, count * 4, cudaMemcpyDeviceToHost), "dump");
    if (FILE* f = std::fopen(file, "wb")) {
      std::fwrite(host.data(), 4, count, f);
      std::fclose(f);
    }
  };
  const bool dump_here = std::getenv("QWEN35MOE_DUMP") && l == 3 && pos == std::atoi(std::getenv("QWEN35MOE_DUMP_POS") ? std::getenv("QWEN35MOE_DUMP_POS") : "0");
  if (dump_here) {
    dump("/tmp/qwen_attn_h.bin", h_, HIDDEN);
    dump("/tmp/qwen_attn_qgate.bin", qgate_, Q_PROJ_ROWS);
    dump("/tmp/qwen_attn_k.bin", k_, KV_DIM);
    dump("/tmp/qwen_attn_v.bin", v_, KV_DIM);
  }

  if (std::getenv("QWEN35MOE_SWAP_QG")) {
    // per head [gate | query] instead of [query | gate]
    attn::SplitQGateArgs split;
    split.qgate = qgate_;
    split.q = gate_attn_;  // swapped: second half is the query
    split.gate = q_attn_;  // first half is the gate
    split.heads = HEADS;
    split.head_dim = HEAD_DIM;
    check(attn::split_q_gate(split, s_), "split q gate");
    float* tmp = q_attn_;
    q_attn_ = gate_attn_;
    gate_attn_ = tmp;
  } else if (std::getenv("QWEN35MOE_FLAT_QG")) {
    // Exporter fused q and gate as two contiguous halves instead of per head.
    copy(q_attn_, qgate_, Q_DIM * 4, cudaMemcpyDeviceToDevice);
    copy(gate_attn_, qgate_ + Q_DIM, Q_DIM * 4, cudaMemcpyDeviceToDevice);
  } else {
    attn::SplitQGateArgs split;
    split.qgate = qgate_;
    split.q = q_attn_;
    split.gate = gate_attn_;
    split.heads = HEADS;
    split.head_dim = HEAD_DIM;
    check(attn::split_q_gate(split, s_), "split q gate");
  }

  attn::NormRopeArgs rope;
  rope.q = q_attn_;
  rope.k = k_;
  rope.q_weight = f32(p + "q_norm.weight", true);
  rope.k_weight = f32(p + "k_norm.weight", true);
  rope.pos = pos;
  rope.eps = EPS;
  at(rope, offsetof(attn::NormRopeArgs, pos), [&] { check(attn::qk_norm_rope(rope, s_), "qk norm rope"); });

  attn::DecodeAttnArgs dec;
  dec.q = q_attn_;
  dec.k_new = k_;
  dec.v_new = v_;
  const KvBands& bands = kv_bands_[slot];
  dec.k_cache = bands.bands[0].k;
  dec.v_cache = bands.bands[0].v;
  dec.kv_type_k = kv_type_k_;
  dec.kv_type_v = kv_type_v_;
  dec.out = attn_;
  dec.pos = pos;
  dec.max_pos = max_pos_;
  dec.partial = kv_partial_;
  // One split per 2048 positions keeps the SMs busy; the count is also the graph bucket.
  dec.splits = splits_at(pos);
  if (kv_bands_on_ && bands.count == 2) {
    // Both cards stream the positions they hold. The call switches the current device, so it stays outside
    // the graph (see the note on QWEN35MOE_GRAPH in init) and the position needs no node patch.
    std::string band_error;
    if (!decode_attention_bands(dec, bands, kv_partial_, dec.splits, s_, s1_, &band_error, &kv_exchange_)) {
      throw std::runtime_error("banded attention: " + band_error);
    }
  } else {
    at(dec, offsetof(attn::DecodeAttnArgs, pos), [&] { check(attn::decode_attention(dec, s_), "attention"); });
  }

  attn::OutputGateArgs gate;
  gate.x = attn_;
  gate.gate = gate_attn_;
  gate.count = Q_DIM;
  check(attn::output_gate(gate, s_), "output gate");
  if (dump_here) dump("/tmp/qwen_attn_out.bin", attn_, Q_DIM);

  fp8_gemv(p + "o_proj", attn_, y_, HIDDEN, Q_DIM);
  if (dump_here) dump("/tmp/qwen_attn_y.bin", y_, HIDDEN);
}

void Decoder::moe(int l) {
  attn::RouterArgs router;
  router.weight = static_cast<const __nv_bfloat16*>(ptr(layer_name(l, "mlp.gate.weight")));
  router.x = h2_;
  router.ids = ids_dev_;
  router.weights = weights_dev_;
  router.logits = router_logits_;
  check(cudaSetDevice(dev0_), "set device 0");
  check(attn::route(router, s_), "router");

  exch_ = l & 1;
  Exchange* slot = pinned_ + exch_;
  const bool debug = std::getenv("QWEN35MOE_DEBUG") != nullptr;
  const bool hop = dev1_ != dev0_;

  if (hop) {
    // The routing reaches card 1 through this pinned slot. Eagerly the four events order it; in a captured step
    // the edges of the spliced child node do, and they also cover the slot reuse that ev_x_free_ and ev_y_free_
    // guard between the layer two steps back.
    if (!cap_ && l >= 2) check(cudaStreamWaitEvent(s_, ev_x_free_[exch_], 0), "x free");
    check(cudaMemcpyAsync(slot->x, h2_, HIDDEN * 4, cudaMemcpyDeviceToHost, s_), "x out");
    check(cudaMemcpyAsync(slot->ids, ids_dev_, sizeof(slot->ids), cudaMemcpyDeviceToHost, s_), "ids out");
    check(cudaMemcpyAsync(slot->weights, weights_dev_, sizeof(slot->weights), cudaMemcpyDeviceToHost, s_),
          "weights out");
    if (cap_) {
      cap_->splice(s_, hop_graph(l));  // device 1 stays unbound here: the hop is a node, not a launch
    } else {
      check(cudaEventRecord(ev_x_[exch_], s_), "x ready");
      check(cudaSetDevice(dev1_), "set device 1");
      check(cudaStreamWaitEvent(s1_, ev_x_[exch_], 0), "wait x");
      check(cudaMemcpyAsync(peer_x_, slot->x, HIDDEN * 4, cudaMemcpyHostToDevice, s1_), "x in");
      check(cudaMemcpyAsync(peer_ids_, slot->ids, sizeof(slot->ids), cudaMemcpyHostToDevice, s1_), "ids in");
      check(cudaMemcpyAsync(peer_weights_, slot->weights, sizeof(slot->weights), cudaMemcpyHostToDevice, s1_),
            "weights in");
      kernels::FusedMoeArgs a1;
      a1.layers = fused_[1].device + std::size_t(l) * EXPERTS;
      a1.x = peer_x_;
      a1.ids = peer_ids_;
      a1.weights = peer_weights_;
      a1.y = peer_out_;
      a1.scratch_gate = fscratch_gate_[1];
      a1.scratch_up = fscratch_up_[1];
      a1.scratch_down = fscratch_down_[1];
      check(kernels::fused_moe_apply(a1, s1_), "fused moe dev1");
      check(cudaEventRecord(ev_x_free_[exch_], s1_), "x consumed");
      if (l >= 2) check(cudaStreamWaitEvent(s1_, ev_y_free_[exch_], 0), "y free");
      check(cudaMemcpyAsync(slot->y, peer_out_, HIDDEN * 4, cudaMemcpyDeviceToHost, s1_), "y back");
      check(cudaEventRecord(ev_y_[exch_], s1_), "y ready");
      check(cudaSetDevice(dev0_), "set device 0");
    }
  } else {
    clear(y_, HIDDEN * 4);
  }

  kernels::FusedMoeArgs a0;
  a0.layers = fused_[0].device + std::size_t(l) * EXPERTS;
  a0.x = h2_;
  a0.ids = ids_dev_;
  a0.weights = weights_dev_;
  a0.y = moe_out_;
  a0.scratch_gate = fscratch_gate_[0];
  a0.scratch_up = fscratch_up_[0];
  a0.scratch_down = fscratch_down_[0];
  check(kernels::fused_moe_apply(a0, s_), "fused moe dev0");

  if (hop) {
    // The join leaves the device-0 experts above running beside the hop, as the eager event pair does.
    if (cap_) cap_->join(s_);
    else check(cudaStreamWaitEvent(s_, ev_y_[exch_], 0), "wait y");
    check(cudaMemcpyAsync(y_, slot->y, HIDDEN * 4, cudaMemcpyHostToDevice, s_), "y in");
    check(attn::axpy(moe_out_, y_, 1.0f, HIDDEN, s_), "merge expert partials");
    if (!cap_) check(cudaEventRecord(ev_y_free_[exch_], s_), "y consumed");
  }

  if (debug && l == 0) {
    int ids[attn::ROUTER_TOP_K] = {};
    int peer[attn::ROUTER_TOP_K] = {};
    std::vector<float> dev0_out(HIDDEN), merged(HIDDEN);
    check(cudaMemcpyAsync(ids, ids_dev_, sizeof(ids), cudaMemcpyDeviceToHost, nullptr), "dbg ids");
    check(cudaMemcpyAsync(merged.data(), moe_out_, HIDDEN * 4, cudaMemcpyDeviceToHost, nullptr), "dbg merged");
    check(cudaStreamSynchronize(nullptr), "dbg sync");
    check(cudaSetDevice(dev1_), "dbg dev1");
    check(cudaMemcpy(peer, peer_ids_, sizeof(peer), cudaMemcpyDeviceToHost), "dbg peer ids");
    check(cudaSetDevice(dev0_), "dbg dev0");
    float amax0 = 0, amaxm = 0;
    for (float f : dev0_out) amax0 = std::max(amax0, std::fabs(f));
    for (float f : merged) amaxm = std::max(amaxm, std::fabs(f));
    std::printf("[moe dbg] id0=%d peer0=%d merged|max|=%.5g\n", ids[0], peer[0], amaxm);
    std::fflush(stdout);
  }
  attn::SharedExpertArgs shared;
  shared.gate = {static_cast<const std::uint8_t*>(ptr(layer_name(l, "mlp.shared_expert.gate_proj.weight"))),
                 static_cast<const std::uint8_t*>(ptr(layer_name(l, "mlp.shared_expert.gate_proj.weight_scale"))),
                 scalar(layer_name(l, "mlp.shared_expert.gate_proj.weight_scale_2"), 1.0f),
                 h2_, nullptr, SHARED_INTER, HIDDEN};
  shared.up = {static_cast<const std::uint8_t*>(ptr(layer_name(l, "mlp.shared_expert.up_proj.weight"))),
               static_cast<const std::uint8_t*>(ptr(layer_name(l, "mlp.shared_expert.up_proj.weight_scale"))),
               scalar(layer_name(l, "mlp.shared_expert.up_proj.weight_scale_2"), 1.0f),
               h2_, nullptr, SHARED_INTER, HIDDEN};
  shared.down = {static_cast<const std::uint8_t*>(ptr(layer_name(l, "mlp.shared_expert.down_proj.weight"))),
                 static_cast<const std::uint8_t*>(ptr(layer_name(l, "mlp.shared_expert.down_proj.weight_scale"))),
                 scalar(layer_name(l, "mlp.shared_expert.down_proj.weight_scale_2"), 1.0f),
                 nullptr, nullptr, HIDDEN, SHARED_INTER};
  shared.gate_weight = static_cast<const __nv_bfloat16*>(ptr(layer_name(l, "mlp.shared_expert_gate.weight")));
  shared.x = h2_;
  shared.y = shared_out_;
  shared.scratch_h = shared_h_;
  shared.scratch_up = shared_up_;
  check(attn::shared_expert(shared, s_), "shared expert");
}

int Decoder::splits_at(int pos) const {
  // One flash-decode split per kSplitPositions positions, capped: the bucket the graph is keyed by.
  if (!attn_splits_depth_) return std::max(1, std::min(kernels::kMaxSplits, (pos + kSplitPositions) / kSplitPositions));
  // Depth-scaled: one column per kSplitPositionsFine positions, so a shallow cache still spreads over waves
  // of blocks instead of one block per query head. Monotone in pos, and it reaches the cap at
  // kMaxSplits * kSplitPositionsFine positions, where the fixed rule is still asking for a tenth of that —
  // so 32k and deeper run the same capped grid and only the shallow buckets change.
  const int columns = (pos + 1 + kSplitPositionsFine - 1) / kSplitPositionsFine;
  return std::max(1, std::min(kernels::kMaxSplits, columns));
}

int Decoder::capture_pos(int splits) const {
  // The first position of the bucket. Buckets are [width * (splits - 1), ... ) up to the cap, which absorbs
  // everything beyond it. Negative when the bucket is out of reach for this context length.
  const int width = attn_splits_depth_ ? kSplitPositionsFine : kSplitPositions;
  const int first = (splits - 1) * width;
  return first < max_pos_ ? first : -1;
}

cudaGraph_t Decoder::hop_graph(int l) const {
  return l >= 0 && l < int(hop_graphs_.size()) ? hop_graphs_[l] : nullptr;
}

bool Decoder::prepare_graph(std::string* error) {
  try {
    if (cap0_ && token_dev_ && pos_dev_ && (dev1_ == dev0_ || !hop_graphs_.empty())) return true;
    check(cudaSetDevice(dev0_), "set device 0");
    // The legacy default stream cannot be captured, so the step gets a named non-blocking stream of its own.
    if (!cap0_) check(cudaStreamCreateWithFlags(&cap0_, cudaStreamNonBlocking), "step stream");
    if (!token_dev_) token_dev_ = static_cast<int*>(alloc(sizeof(int)));
    if (!pos_dev_) pos_dev_ = static_cast<int*>(alloc(sizeof(int)));
    check(cudaMemset(token_dev_, 0, sizeof(int)), "token zero");
    check(cudaMemset(pos_dev_, 0, sizeof(int)), "pos zero");
    if (dev1_ != dev0_ && hop_graphs_.empty()) {
      // Card 1's half of every MoE layer, one graph per layer, captured with device 1 bound outside the capture
      // (a graph may not switch device inside itself) and with no event in it: the step graph's child node
      // edges are the hop's fork and join, so the same graphs serve every bucket.
      check(cudaSetDevice(dev1_), "set device 1");
      hop_graphs_.assign(LAYERS, nullptr);
      for (int l = 0; l < LAYERS; ++l) {
        Exchange* slot = pinned_ + (l & 1);
        kernels::FusedMoeArgs a1;
        a1.layers = fused_[1].device + std::size_t(l) * EXPERTS;
        a1.x = peer_x_;
        a1.ids = peer_ids_;
        a1.weights = peer_weights_;
        a1.y = peer_out_;
        a1.scratch_gate = fscratch_gate_[1];
        a1.scratch_up = fscratch_up_[1];
        a1.scratch_down = fscratch_down_[1];
        check(cudaStreamBeginCapture(s1_, cudaStreamCaptureModeThreadLocal), "begin hop capture");
        cudaGraph_t hop = nullptr;
        try {
          check(cudaMemcpyAsync(peer_x_, slot->x, HIDDEN * 4, cudaMemcpyHostToDevice, s1_), "x in");
          check(cudaMemcpyAsync(peer_ids_, slot->ids, sizeof(slot->ids), cudaMemcpyHostToDevice, s1_), "ids in");
          check(cudaMemcpyAsync(peer_weights_, slot->weights, sizeof(slot->weights), cudaMemcpyHostToDevice, s1_),
                "weights in");
          check(kernels::fused_moe_apply(a1, s1_), "fused moe dev1 capture");
          check(cudaMemcpyAsync(slot->y, peer_out_, HIDDEN * 4, cudaMemcpyDeviceToHost, s1_), "y back");
          check(cudaStreamEndCapture(s1_, &hop), "end hop capture");
        } catch (...) {
          cudaGraph_t discard = nullptr;
          if (cudaStreamEndCapture(s1_, &discard) == cudaSuccess && discard) cudaGraphDestroy(discard);
          throw;
        }
        hop_graphs_[l] = hop;
      }
    }
    check(cudaSetDevice(dev0_), "set device 0");
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

void Decoder::enqueue_graph(int pos, DecodeGraphCapture* cap) {
  const cudaStream_t saved_stream = s_;
  DecodeGraphCapture* const saved_cap = cap_;
  s_ = cap0_;
  cap_ = cap;
  try {
    // The token id is irrelevant here: the captured embedding node reads token_dev_, and capture runs nothing.
    body(0, pos, true);
  } catch (...) {
    s_ = saved_stream;
    cap_ = saved_cap;
    throw;
  }
  s_ = saved_stream;
  cap_ = saved_cap;
}

DecodeGraph* Decoder::graph_for(int pos) {
  if (!use_graph_) return nullptr;
  const int splits = splits_at(pos);
  const auto hit = graphs_.find(splits);
  if (hit != graphs_.end()) return &hit->second;
  if (!primed_) return nullptr;  // the first step of the run goes eager: see the note on body()
  DecodeGraph& graph = graphs_[splits];
  std::string why;
  if (decode_graph_build(&graph, *this, splits, &why)) {
    std::fprintf(stderr,
                 "[qwen35moe] captured one token: splits=%d at pos=%d, %d nodes (%d kernels, %d copies, "
                 "%d memsets, %d second-device hops), %d position nodes patched per replay\n",
                 splits, capture_pos(splits), graph.stats.nodes, graph.stats.kernels, graph.stats.copies,
                 graph.stats.memsets, graph.stats.hops, graph.stats.pos_nodes);
    std::fflush(stderr);
    return &graph;
  }
  graphs_.erase(splits);
  use_graph_ = false;  // one failed capture is enough: the step stays eager for the rest of the run
  if (!graph_reported_) {
    graph_reported_ = true;
    std::fprintf(stderr, "[qwen35moe] CUDA graph off, stepping eagerly: %s\n", why.c_str());
    std::fflush(stderr);
  }
  return nullptr;
}

// One token's kernels: the embedding, LAYERS layers, the final norm and the lm_head GEMV, all on `s_`. The
// host readback stays with step(), outside the graph. `captured` marks the graph form, which differs in two
// ways: the embedding reads the token id out of token_dev_ because a captured argument cannot vary, and the
// position-bearing launches are registered with the capture so a replay can rewrite their arguments.
//
// Nothing in here may block or allocate. The first step of a run always takes the eager path, which is what
// loads every module, materialises every cached f32() norm weight and resolves every ptr() name that a later
// capture would otherwise try to build inside itself.
void Decoder::body(int token, int pos, bool captured) {
  const __nv_bfloat16* embed =
      static_cast<const __nv_bfloat16*>(ptr("model.language_model.embed_tokens.weight"));
  if (captured) {
    check(embedding_row_dev(embed, token_dev_, x_, HIDDEN, s_), "embedding");
  } else {
    check(dense::embedding_row(embed, token, x_, HIDDEN, s_), "embedding");
  }
  if (std::getenv("QWEN35MOE_DUMP") && pos == 0) {
    std::vector<float> e(HIDDEN);
    check(cudaMemcpy(e.data(), x_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump embed");
    std::printf("embed:");
    for (int i = 0; i < 4; ++i) std::printf(" %.9g", e[std::size_t(i)]);
    std::printf("\n");
  }
  for (int l = 0; l < LAYERS; ++l) {
    if (std::getenv("QWEN35MOE_DUMP") && pos == 0 && l == 0) {
      std::vector<float> host(HIDDEN);
      check(cudaMemcpy(host.data(), x_, HIDDEN * 4, cudaMemcpyDeviceToHost), "save x_in");
      if (FILE* f = std::fopen("/tmp/qwen_layer0_x.bin", "wb")) {
        std::fwrite(host.data(), 4, HIDDEN, f);
        std::fclose(f);
      }
    }
    if (std::getenv("QWEN35MOE_DEBUG")) {
      std::printf("[step] pos=%d layer=%d\n", pos, l);
      std::fflush(stdout);
    }
    check(dense::rms_norm(x_, f32(layer_name(l, "input_layernorm.weight"), true), h_, HIDDEN, EPS, s_),
          "input norm");
    const bool skip_attn = std::getenv("QWEN35MOE_SKIP_ATTN") != nullptr;
    if (is_full_layer(l) && skip_attn) {
      clear(y_, HIDDEN * 4);
    } else if (is_full_layer(l)) {
      full_layer(l, pos);
    } else {
      linear_layer(l);
    }
    if (std::getenv("QWEN35MOE_DUMP") && pos == 0 && l == 0) {
      std::vector<float> h(HIDDEN), out(HIDDEN);
      check(cudaMemcpy(h.data(), h_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump h");
      check(cudaMemcpy(out.data(), y_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump y");
      std::printf("h0:");
      for (int i = 0; i < 4; ++i) std::printf(" %.9g", h[std::size_t(i)]);
      std::printf("\n");
      if (FILE* fh = std::fopen("/tmp/qwen_h0.bin", "wb")) {
        std::fwrite(h.data(), 4, HIDDEN, fh);
        std::fclose(fh);
      }
      std::printf("qkv0:");
      for (int i = 0; i < 4; ++i) std::printf(" %.9g", out[std::size_t(i)]);
      std::printf("\n");
    }
    if (std::getenv("QWEN35MOE_DUMP") && pos == 0 && l == 0) {
      std::vector<float> host(HIDDEN);
      check(cudaMemcpy(host.data(), y_, HIDDEN * 4, cudaMemcpyDeviceToHost), "save mixer");
      if (FILE* f = std::fopen("/tmp/qwen_layer0_mixer.bin", "wb")) {
        std::fwrite(host.data(), 4, HIDDEN, f);
        std::fclose(f);
      }
    }
    check(attn::axpy(x_, y_, 1.0f, HIDDEN, s_), "residual");

    check(dense::rms_norm(x_, f32(layer_name(l, "post_attention_layernorm.weight"), true), h2_, HIDDEN, EPS, s_),
          "post norm");
    if (!std::getenv("QWEN35MOE_SKIP_MOE")) {
      moe(l);
    } else {
      clear(moe_out_, HIDDEN * 4);
      clear(shared_out_, HIDDEN * 4);
    }
    if (std::getenv("QWEN35MOE_DUMP") && pos == 0 && (l < 4 || l == 39)) {
      std::vector<float> mo(HIDDEN), sh(HIDDEN), hx(HIDDEN);
      check(cudaMemcpy(mo.data(), moe_out_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump moe");
      check(cudaMemcpy(sh.data(), shared_out_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump shared");
      check(cudaMemcpy(hx.data(), x_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump x");
      auto amax = [](const std::vector<float>& v) {
        float m = 0;
        for (float a : v) m = std::max(m, std::fabs(a));
        return m;
      };
      std::printf("[mag] l=%d |moe|=%.4g |shared|=%.4g |x|=%.4g\n", l, amax(mo), amax(sh), amax(hx));
      if (l == 0) {
        auto save = [&](const char* file, const float* data, int n) {
          std::vector<float> host(n);
          check(cudaMemcpy(host.data(), data, n * 4, cudaMemcpyDeviceToHost), "save");
          if (FILE* f = std::fopen(file, "wb")) {
            std::fwrite(host.data(), 4, n, f);
            std::fclose(f);
          }
        };
        save("/tmp/qwen_layer0_x.bin", x_, HIDDEN);
        save("/tmp/qwen_layer0_h2.bin", h2_, HIDDEN);
        save("/tmp/qwen_layer0_moe.bin", moe_out_, HIDDEN);
        save("/tmp/qwen_layer0_shared.bin", shared_out_, HIDDEN);
        save("/tmp/qwen_layer0_x_after.bin", x_, HIDDEN);  // x before the MoE residuals; fixed below
      }
    }
    check(attn::axpy(x_, moe_out_, 1.0f, HIDDEN, s_), "moe residual");
    check(attn::axpy(x_, shared_out_, 1.0f, HIDDEN, s_), "shared residual");
    if (std::getenv("QWEN35MOE_DUMP") && pos == 0) {
      std::vector<float> host(HIDDEN);
      check(cudaMemcpy(host.data(), x_, HIDDEN * 4, cudaMemcpyDeviceToHost), "save x_after");
      char path[64];
      std::snprintf(path, sizeof(path), "/tmp/qwen_x_layer_%02d.bin", l);
      if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(host.data(), 4, HIDDEN, f);
        std::fclose(f);
      }
      if (FILE* f = std::fopen("/tmp/qwen_layer0_x_after.bin", "wb")) {
        std::fwrite(host.data(), 4, HIDDEN, f);
        std::fclose(f);
      }
    }
  }

  check(dense::rms_norm(x_, f32("model.language_model.norm.weight", true), h_, HIDDEN, EPS, s_), "final norm");
  nvfp4_gemv("lm_head", h_, logits_, VOCAB, HIDDEN);
}

bool Decoder::step(int token, int pos, int* next_token, std::string* error) {
  try {
    last_pos_ = pos;
    check(cudaSetDevice(dev0_), "set device 0");
    if (DecodeGraph* graph = graph_for(pos)) {
      std::string why;
      // The replay goes to the capture stream, so the readback below has to follow it there rather than on the
      // legacy stream, which shares no ordering with it.
      s_ = graph->stream;
      cap_ = nullptr;
      if (!decode_graph_run(graph, token, pos, &why)) throw std::runtime_error("step graph replay: " + why);
    } else {
      s_ = nullptr;
      cap_ = nullptr;
      body(token, pos, false);
      primed_ = true;  // a capture may now assume every module is loaded and every cache warm
    }
    // Outside the graph: the argmax reads the logits and the host reads both back.
    check(dense::argmax(logits_, VOCAB, argmax_dev_, nullptr, s_), "argmax");
    int index = 0;
    read_back(&index, argmax_dev_, 4);
    read_back(logits_host_.data(), logits_, std::size_t(VOCAB) * 4);
    if (std::getenv("QWEN35MOE_DUMP") && pos == 4) {
      auto dump = [&](const char* file, const float* buffer, int count) {
        std::vector<float> host(count);
        check(cudaMemcpy(host.data(), buffer, count * 4, cudaMemcpyDeviceToHost), "dump");
        if (FILE* f = std::fopen(file, "wb")) {
          std::fwrite(host.data(), 4, count, f);
          std::fclose(f);
        }
      };
      dump("/tmp/qwen_final_x.bin", x_, HIDDEN);
      if (FILE* fl = std::fopen("/tmp/qwen_logits.bin", "wb")) {
        std::fwrite(logits_host_.data(), 4, VOCAB, fl);
        std::fclose(fl);
      }
      dump("/tmp/qwen_final_h.bin", h_, HIDDEN);
      // top 5
      int top[5] = {-1, -1, -1, -1, -1};
      for (int v = 0; v < VOCAB; ++v) {
        for (int k = 0; k < 5; ++k) {
          if (top[k] < 0 || logits_host_[std::size_t(v)] > logits_host_[std::size_t(top[k])]) {
            for (int m = 4; m > k; --m) top[m] = top[m - 1];
            top[k] = v;
            break;
          }
        }
      }
      std::printf("top5:");
      for (int k = 0; k < 5; ++k) std::printf(" %d(%.6g)", top[k], logits_host_[std::size_t(top[k])]);
      std::printf("\n");
    }
    *next_token = index;
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

}  // namespace ingine::arch::qwen35moe
