#include "arch/qwen35moe/model/decoder.h"

#include <cmath>
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

}  // namespace

Decoder::Decoder() = default;

Decoder::~Decoder() {
  for (float* p : {x_, h_, h2_, y_, qkv_, qkv2_, z_, a_, b_, beta_, g_, gout_, q_lin_, k_lin_, v_lin_, qgate_,
                   k_, v_, attn_, q_attn_, gate_attn_, moe_out_, shared_out_, shared_h_, shared_up_, logits_, gdn_state_,
                   conv_state_}) {
    if (p) cudaFree(p);
  }
  for (auto* p : kv_k_) {
    if (p) cudaFree(p);
  }
  for (auto* p : kv_v_) {
    if (p) cudaFree(p);
  }
  if (ids_dev_) cudaFree(ids_dev_);
  if (weights_dev_) cudaFree(weights_dev_);
  for (auto& [name, buffer] : f32_) {
    (void)name;
    if (buffer) cudaFree(buffer);
  }
  if (argmax_dev_) cudaFree(argmax_dev_);
  if (dev1_ != dev0_) {
    cudaSetDevice(dev1_);
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
        check(cudaMemcpy(&value, ptr(p.name), 4, cudaMemcpyDeviceToHost), "scale read");
        scalars_[p.name] = value;
      }
    }

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

    const std::size_t per_layer_kv = std::size_t(max_pos) * KV_HEADS * HEAD_DIM * sizeof(__nv_bfloat16);
    for (int i = 0; i < NUM_FULL_LAYERS; ++i) {
      kv_k_[i] = static_cast<__nv_bfloat16*>(alloc(per_layer_kv));
      kv_v_[i] = static_cast<__nv_bfloat16*>(alloc(per_layer_kv));
    }
    gdn_state_ = static_cast<float*>(
        alloc(std::size_t(NUM_LINEAR_LAYERS) * LIN_VALUE_HEADS * LIN_KEY_DIM * LIN_VALUE_DIM * 4));
    conv_state_ = static_cast<float*>(alloc(std::size_t(NUM_LINEAR_LAYERS) * (CONV_K - 1) * LIN_CONV_DIM * 4));
    check(cudaMemset(gdn_state_, 0, std::size_t(NUM_LINEAR_LAYERS) * LIN_VALUE_HEADS * LIN_KEY_DIM * LIN_VALUE_DIM * 4),
          "state zero");
    check(cudaMemset(conv_state_, 0, std::size_t(NUM_LINEAR_LAYERS) * (CONV_K - 1) * LIN_CONV_DIM * 4), "conv zero");

    if (dev1_ != dev0_) {
      check(cudaSetDevice(dev1_), "set device 1");
      peer_x_ = static_cast<float*>(alloc(HIDDEN * 4));
      peer_h_ = static_cast<float*>(alloc(MOE_INTER * 4));
      peer_up_ = static_cast<float*>(alloc(MOE_INTER * 4));
      peer_out_ = static_cast<float*>(alloc(HIDDEN * 4));
      check(cudaSetDevice(dev0_), "set device 0");
      check(cudaHostAlloc(reinterpret_cast<void**>(&pinned_in_), HIDDEN * 4, cudaHostAllocDefault), "host alloc");
      check(cudaHostAlloc(reinterpret_cast<void**>(&pinned_out_), HIDDEN * 4, cudaHostAllocDefault), "host alloc");
    }

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
  check(quant::fp8_gemv(args, nullptr), "fp8 gemv");
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
  check(quant::nvfp4_gemv(args, nullptr), "nvfp4 gemv");
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
                         HIDDEN, nullptr),
        "in_proj_a");
  check(dense::gemv_bf16(static_cast<const __nv_bfloat16*>(ptr(p + "in_proj_b.weight")), h_, b_, LIN_VALUE_HEADS,
                         HIDDEN, nullptr),
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
  check(gdn::conv_silu_step(conv, nullptr), "conv");

  gdn::GdnGateArgs gates;
  gates.a = a_;
  gates.b = b_;
  gates.dt_bias = f32(p + "dt_bias");
  gates.a_log = f32(p + "A_log");
  gates.beta = beta_;
  gates.g = g_;
  gates.heads = LIN_VALUE_HEADS;
  check(gdn::gdn_gates(gates, nullptr), "gdn gates");

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
    check(gdn::gather_qkv_grouped(g, nullptr), "gather qkv");
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
  check(gdn::gdn_recurrent_step(step, nullptr), "gdn step");

  gdn::GatedNormArgs norm;
  norm.x = gout_;
  norm.z = z_;
  norm.weight = f32(p + "norm.weight");
  norm.v_heads = LIN_VALUE_HEADS;
  norm.v_dim = LIN_VALUE_DIM;
  norm.eps = EPS;
  check(gdn::gated_norm(norm, nullptr), "gdn norm");

  fp8_gemv(p + "out_proj", gout_, y_, HIDDEN, LIN_VALUE_DIM_TOT);
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
    check(attn::split_q_gate(split, nullptr), "split q gate");
    float* tmp = q_attn_;
    q_attn_ = gate_attn_;
    gate_attn_ = tmp;
  } else if (std::getenv("QWEN35MOE_FLAT_QG")) {
    // Exporter fused q and gate as two contiguous halves instead of per head.
    check(cudaMemcpy(q_attn_, qgate_, Q_DIM * 4, cudaMemcpyDeviceToDevice), "q half");
    check(cudaMemcpy(gate_attn_, qgate_ + Q_DIM, Q_DIM * 4, cudaMemcpyDeviceToDevice), "gate half");
  } else {
    attn::SplitQGateArgs split;
    split.qgate = qgate_;
    split.q = q_attn_;
    split.gate = gate_attn_;
    split.heads = HEADS;
    split.head_dim = HEAD_DIM;
    check(attn::split_q_gate(split, nullptr), "split q gate");
  }

  attn::NormRopeArgs rope;
  rope.q = q_attn_;
  rope.k = k_;
  rope.q_weight = f32(p + "q_norm.weight", true);
  rope.k_weight = f32(p + "k_norm.weight", true);
  rope.pos = pos;
  rope.eps = EPS;
  rope.interleaved = std::getenv("QWEN35MOE_INTERLEAVED_ROPE") != nullptr;
  check(attn::qk_norm_rope(rope, nullptr), "qk norm rope");

  attn::DecodeAttnArgs dec;
  dec.q = q_attn_;
  dec.k_new = k_;
  dec.v_new = v_;
  dec.k_cache = kv_k_[slot];
  dec.v_cache = kv_v_[slot];
  dec.out = attn_;
  dec.pos = pos;
  dec.max_pos = max_pos_;
  check(attn::decode_attention(dec, nullptr), "attention");

  attn::OutputGateArgs gate;
  gate.x = attn_;
  gate.gate = gate_attn_;
  gate.count = Q_DIM;
  check(attn::output_gate(gate, nullptr), "output gate");
  if (dump_here) dump("/tmp/qwen_attn_out.bin", attn_, Q_DIM);

  fp8_gemv(p + "o_proj", attn_, y_, HIDDEN, Q_DIM);
  if (dump_here) dump("/tmp/qwen_attn_y.bin", y_, HIDDEN);
}

void Decoder::run_expert(int l, int expert, int device, const float* x, float* out, float* scratch_h,
                         float* scratch_up) {
  const ExpertRow& row = experts_[l][std::size_t(expert)];
  quant::Nvfp4GemvArgs gate{row.gate.w, row.gate.scales, row.gate.scale, x, scratch_h, MOE_INTER, HIDDEN};
  quant::Nvfp4GemvArgs up{row.up.w, row.up.scales, row.up.scale, x, scratch_up, MOE_INTER, HIDDEN};
  quant::Nvfp4GemvArgs down{row.down.w, row.down.scales, row.down.scale, scratch_h, out, HIDDEN, MOE_INTER};
  check(cudaSetDevice(device), "set expert device");
  if (std::getenv("QWEN35MOE_DEBUG")) {
    std::printf("[expert] l=%d e=%d dev=%d span=%d w=%p scales=%p x=%p y=%p\n", l, expert, device,
                row.span, static_cast<const void*>(gate.w), static_cast<const void*>(gate.scales),
                static_cast<const void*>(gate.x), static_cast<void*>(gate.y));
  }
  check(quant::nvfp4_gemv(gate, nullptr), "expert gate");
  check(quant::nvfp4_gemv(up, nullptr), "expert up");
  check(attn::silu_mul(scratch_h, scratch_h, scratch_up, MOE_INTER, nullptr), "expert silu");
  check(quant::nvfp4_gemv(down, nullptr), "expert down");
}

void Decoder::moe(int l) {
  attn::RouterArgs router;
  router.weight = static_cast<const __nv_bfloat16*>(ptr(layer_name(l, "mlp.gate.weight")));
  router.x = h2_;
  router.ids = ids_dev_;
  router.weights = weights_dev_;
  check(cudaSetDevice(dev0_), "set device 0");
  check(attn::route(router, nullptr), "router");

  int ids[attn::ROUTER_TOP_K] = {};
  float weights[attn::ROUTER_TOP_K] = {};
  check(cudaMemcpy(ids, ids_dev_, sizeof(ids), cudaMemcpyDeviceToHost), "ids copy");
  check(cudaMemcpy(weights, weights_dev_, sizeof(weights), cudaMemcpyDeviceToHost), "weights copy");

  check(cudaMemset(moe_out_, 0, HIDDEN * 4), "moe zero");
  for (int k = 0; k < attn::ROUTER_TOP_K; ++k) {
    const int expert = ids[k];
    const int span = experts_[l][std::size_t(expert)].span;
    float* out = y_;
    if (span == dev0_ || dev1_ == dev0_) {
      run_expert(l, expert, dev0_, h2_, out, shared_h_, shared_up_);
      check(cudaSetDevice(dev0_), "set device 0");
    } else {
      // No P2P between the cards: stage the 2048-wide input and output through pinned host memory.
      check(cudaMemcpy(pinned_in_, h2_, HIDDEN * 4, cudaMemcpyDeviceToHost), "peer in");
      check(cudaSetDevice(dev1_), "set device 1");
      check(cudaMemcpy(peer_x_, pinned_in_, HIDDEN * 4, cudaMemcpyHostToDevice), "peer in h2d");
      run_expert(l, expert, dev1_, peer_x_, peer_out_, peer_h_, peer_up_);
      check(cudaMemcpy(pinned_out_, peer_out_, HIDDEN * 4, cudaMemcpyDeviceToHost), "peer out");
      check(cudaSetDevice(dev0_), "set device 0");
      check(cudaMemcpy(out, pinned_out_, HIDDEN * 4, cudaMemcpyHostToDevice), "peer out h2d");
    }
    if (std::getenv("QWEN35MOE_DUMP") && last_pos_ == 0 && l == 0 && k == 0) {
      std::vector<float> xv(HIDDEN), ov(HIDDEN);
      check(cudaMemcpy(xv.data(), h2_, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump moe x");
      check(cudaMemcpy(ov.data(), out, HIDDEN * 4, cudaMemcpyDeviceToHost), "dump moe out");
      std::printf("moe_e0 expert=%d x0:", expert);
      for (int i = 0; i < 4; ++i) std::printf(" %.9g", xv[std::size_t(i)]);
      std::printf("\n");
      std::printf("moe_e0 out0:");
      for (int i = 0; i < 4; ++i) std::printf(" %.9g", ov[std::size_t(i)]);
      std::printf("\n");
      if (FILE* fx = std::fopen("/tmp/qwen_moe_x.bin", "wb")) {
        std::fwrite(xv.data(), 4, HIDDEN, fx);
        std::fclose(fx);
      }
      if (FILE* fo = std::fopen("/tmp/qwen_moe_out.bin", "wb")) {
        std::fwrite(ov.data(), 4, HIDDEN, fo);
        std::fclose(fo);
      }
      if (FILE* fw = std::fopen("/tmp/qwen_moe_w.bin", "wb")) {
        std::fwrite(weights, 4, attn::ROUTER_TOP_K, fw);
        std::fclose(fw);
      }
    }
    check(attn::axpy(moe_out_, out, weights[k], HIDDEN, nullptr), "moe accumulate");
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
  check(attn::shared_expert(shared, nullptr), "shared expert");
}

bool Decoder::step(int token, int pos, int* next_token, std::string* error) {
  try {
    last_pos_ = pos;
    check(cudaSetDevice(dev0_), "set device 0");
    check(dense::embedding_row(static_cast<const __nv_bfloat16*>(ptr("model.language_model.embed_tokens.weight")),
                               token, x_, HIDDEN, nullptr),
          "embedding");
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
      check(dense::rms_norm(x_, f32(layer_name(l, "input_layernorm.weight"), true), h_, HIDDEN, EPS, nullptr),
            "input norm");
      if (is_full_layer(l)) {
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
      check(attn::axpy(x_, y_, 1.0f, HIDDEN, nullptr), "residual");

      check(dense::rms_norm(x_, f32(layer_name(l, "post_attention_layernorm.weight"), true), h2_, HIDDEN, EPS, nullptr),
            "post norm");
      moe(l);
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
      check(attn::axpy(x_, moe_out_, 1.0f, HIDDEN, nullptr), "moe residual");
      check(attn::axpy(x_, shared_out_, 1.0f, HIDDEN, nullptr), "shared residual");
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

    check(dense::rms_norm(x_, f32("model.language_model.norm.weight", true), h_, HIDDEN, EPS, nullptr), "final norm");
    nvfp4_gemv("lm_head", h_, logits_, VOCAB, HIDDEN);
    check(dense::argmax(logits_, VOCAB, argmax_dev_, nullptr, nullptr), "argmax");
    int index = 0;
    check(cudaMemcpy(&index, argmax_dev_, 4, cudaMemcpyDeviceToHost), "argmax copy");
    check(cudaMemcpy(logits_host_.data(), logits_, std::size_t(VOCAB) * 4, cudaMemcpyDeviceToHost), "logits copy");
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
