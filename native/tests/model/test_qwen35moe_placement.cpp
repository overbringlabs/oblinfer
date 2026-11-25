// Placement planner for the Qwen3.6-35B-A3B packs: dense slots and one contiguous expert block per MoE
// layer, from the two published export dialects. Host only; the byte sizes are synthetic because the
// planner takes them from the caller (the safetensors headers) and must stay dtype-agnostic.
#include "arch/qwen35moe/model/placement.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qwen = ingine::arch::qwen35moe;

#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);   \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

namespace {

std::uint64_t bytes_for(qwen::Projection projection, qwen::Artifact artifact) {
  if (artifact == qwen::Artifact::Weight) {
    if (projection == qwen::Projection::GateUpProj) return 2000;
    if (projection == qwen::Projection::DownProj) return 1000;
    return 1000;
  }
  if (artifact == qwen::Artifact::WeightScale) return 64;
  if (artifact == qwen::Artifact::WeightScale2) return 4;
  return 4;  // input_scale
}

void add(std::vector<qwen::TensorSize>& out, const std::string& name, qwen::Projection projection,
         qwen::Artifact artifact) {
  out.push_back({name, bytes_for(projection, artifact)});
}

void add_quantized(std::vector<qwen::TensorSize>& out, const std::string& base, qwen::Projection projection,
                   bool with_input_scale) {
  add(out, base + ".weight", projection, qwen::Artifact::Weight);
  add(out, base + ".weight_scale", projection, qwen::Artifact::WeightScale);
  add(out, base + ".weight_scale_2", projection, qwen::Artifact::WeightScale2);
  if (with_input_scale) add(out, base + ".input_scale", projection, qwen::Artifact::InputScale);
}

std::vector<qwen::TensorSize> checkpoint(bool ornith, bool with_mtp_experts = true) {
  std::vector<qwen::TensorSize> out;
  const bool q = !ornith;
  for (int l = 0; l < qwen::LAYERS; ++l) {
    const std::string p = "model.language_model.layers." + std::to_string(l);
    add(out, p + ".input_layernorm.weight", qwen::Projection::None, qwen::Artifact::Weight);
    add(out, p + ".post_attention_layernorm.weight", qwen::Projection::None, qwen::Artifact::Weight);
    add(out, p + ".mlp.gate.weight", qwen::Projection::Router, qwen::Artifact::Weight);
    add(out, p + ".mlp.shared_expert_gate.weight", qwen::Projection::SharedGate, qwen::Artifact::Weight);
    if (l % qwen::FULL_ATTENTION_INTERVAL == qwen::FULL_ATTENTION_INTERVAL - 1) {
      add_quantized(out, p + ".self_attn.q_proj", qwen::Projection::QProj, q);
      add_quantized(out, p + ".self_attn.k_proj", qwen::Projection::KProj, q);
      add_quantized(out, p + ".self_attn.v_proj", qwen::Projection::VProj, q);
      add_quantized(out, p + ".self_attn.o_proj", qwen::Projection::OProj, q);
      add(out, p + ".self_attn.q_norm.weight", qwen::Projection::QNorm, qwen::Artifact::Weight);
      add(out, p + ".self_attn.k_norm.weight", qwen::Projection::KNorm, qwen::Artifact::Weight);
    } else {
      add_quantized(out, p + ".linear_attn.in_proj_qkv", qwen::Projection::InQkv, q);
      add_quantized(out, p + ".linear_attn.in_proj_z", qwen::Projection::InZ, q);
      add_quantized(out, p + ".linear_attn.out_proj", qwen::Projection::OutProj, q);
      add(out, p + ".linear_attn.in_proj_a.weight", qwen::Projection::InA, qwen::Artifact::Weight);
      add(out, p + ".linear_attn.in_proj_b.weight", qwen::Projection::InB, qwen::Artifact::Weight);
      add(out, p + ".linear_attn.conv1d.weight", qwen::Projection::Conv1d, qwen::Artifact::Weight);
      add(out, p + ".linear_attn.A_log", qwen::Projection::ALog, qwen::Artifact::Weight);
      add(out, p + ".linear_attn.dt_bias", qwen::Projection::DtBias, qwen::Artifact::Weight);
      add(out, p + ".linear_attn.norm.weight", qwen::Projection::None, qwen::Artifact::Weight);
    }
  }
  add(out, "model.language_model.embed_tokens.weight", qwen::Projection::None, qwen::Artifact::Weight);
  add_quantized(out, "lm_head", qwen::Projection::LmHead, q);

  // Experts on two layers only; the planner checks the other layers have dense tensors.
  for (int l : {0, 3}) {
    const std::string p = "model.language_model.layers." + std::to_string(l);
    for (int e = 0; e < qwen::EXPERTS; ++e) {
      const std::string b = p + ".mlp.experts." + std::to_string(e);
      add_quantized(out, b + ".gate_proj", qwen::Projection::GateProj, q);
      add_quantized(out, b + ".up_proj", qwen::Projection::UpProj, q);
      add_quantized(out, b + ".down_proj", qwen::Projection::DownProj, q);
    }
  }

  add(out, "mtp.fc.weight", qwen::Projection::None, qwen::Artifact::Weight);
  add(out, "mtp.pre_fc_norm_embedding.weight", qwen::Projection::None, qwen::Artifact::Weight);
  add(out, "mtp.pre_fc_norm_hidden.weight", qwen::Projection::None, qwen::Artifact::Weight);
  add(out, "mtp.norm.weight", qwen::Projection::None, qwen::Artifact::Weight);
  add(out, "mtp.layers.0.input_layernorm.weight", qwen::Projection::None, qwen::Artifact::Weight);
  add(out, "mtp.layers.0.self_attn.q_proj.weight", qwen::Projection::QProj, qwen::Artifact::Weight);
  if (with_mtp_experts) {
    if (ornith) {
      for (int e = 0; e < qwen::EXPERTS; ++e) {
        const std::string b = "mtp.layers.0.mlp.experts." + std::to_string(e);
        add_quantized(out, b + ".gate_proj", qwen::Projection::GateProj, false);
        add_quantized(out, b + ".up_proj", qwen::Projection::UpProj, false);
        add_quantized(out, b + ".down_proj", qwen::Projection::DownProj, false);
      }
    } else {
      add(out, "mtp.layers.0.mlp.experts.gate_up_proj", qwen::Projection::GateUpProj, qwen::Artifact::Weight);
      add(out, "mtp.layers.0.mlp.experts.down_proj", qwen::Projection::DownProj, qwen::Artifact::Weight);
    }
  }
  return out;
}

const qwen::LayerPlan& layer(const qwen::PlacementPlan& plan, int l) {
  CHECK(l >= 0 && l < int(plan.layers.size()));
  return plan.layers[std::size_t(l)];
}

int nvidia_case() {
  std::vector<qwen::TensorSize> tensors = checkpoint(false);
  std::string error;
  const auto plan = qwen::plan_placement(tensors, &error);
  if (!plan) std::fprintf(stderr, "nvidia plan failed: %s\n", error.c_str());
  CHECK(plan.has_value());
  CHECK(plan->dialect == qwen::Dialect::Nvidia);
  CHECK(plan->layers.size() == std::size_t(qwen::LAYERS + qwen::MTP_LAYERS));

  const qwen::LayerPlan& l0 = layer(*plan, 0);
  CHECK(l0.has_experts);
  // Four artifacts per projection, each 1000/64/4/4 bytes: stride 3216.
  CHECK(l0.experts.stride == 3216);
  CHECK(l0.experts.bytes == 3216 * std::uint64_t(qwen::EXPERTS));
  CHECK(l0.experts.slots.size() == 12);
  CHECK(l0.experts.slots[0].projection == qwen::Projection::GateProj);
  CHECK(l0.experts.slots[0].artifact == qwen::Artifact::Weight);
  CHECK(l0.experts.slots[0].offset == 0);
  CHECK(l0.experts.slots[1].artifact == qwen::Artifact::WeightScale);
  CHECK(l0.experts.slots[1].offset == 1000);
  CHECK(l0.experts.slots[3].artifact == qwen::Artifact::InputScale);
  CHECK(l0.experts.slots[3].offset == 1068);
  CHECK(l0.experts.slots[4].projection == qwen::Projection::UpProj);
  CHECK(l0.experts.slots[4].offset == 1072);
  CHECK(l0.experts.slots[11].offset == 3212);

  // Dense layers carry no expert block.
  CHECK(!layer(*plan, 1).has_experts);
  CHECK(layer(*plan, 1).dense_bytes > 0);
  CHECK(!layer(*plan, 3).dense.empty());

  // Packed MTP experts: one block, no per-expert multiplication.
  const qwen::LayerPlan& mtp = layer(*plan, qwen::LAYERS);
  CHECK(mtp.has_experts);
  CHECK(mtp.experts.slots.size() == 2);
  CHECK(mtp.experts.stride == 3000);
  CHECK(mtp.experts.bytes == 3000);

  CHECK(!plan->model.empty());
  CHECK(plan->total_bytes > 0);
  return 0;
}

int ornith_case() {
  std::vector<qwen::TensorSize> tensors = checkpoint(true);
  std::string error;
  const auto plan = qwen::plan_placement(tensors, &error);
  if (!plan) std::fprintf(stderr, "ornith plan failed: %s\n", error.c_str());
  CHECK(plan.has_value());
  CHECK(plan->dialect == qwen::Dialect::Ornith);
  const qwen::LayerPlan& l0 = layer(*plan, 0);
  // Three artifacts per projection: 1000 + 64 + 4.
  CHECK(l0.experts.stride == 3204);
  CHECK(l0.experts.slots.size() == 9);
  CHECK(layer(*plan, qwen::LAYERS).experts.bytes == 3204 * std::uint64_t(qwen::EXPERTS));
  return 0;
}

int error_cases() {
  // A missing artifact on one expert must be caught, not silently mixed into the stride.
  {
    std::vector<qwen::TensorSize> tensors = checkpoint(false);
    for (auto it = tensors.begin(); it != tensors.end(); ++it) {
      if (it->name == "model.language_model.layers.0.mlp.experts.17.down_proj.input_scale") {
        tensors.erase(it);
        break;
      }
    }
    std::string error;
    const auto plan = qwen::plan_placement(tensors, &error);
    CHECK(!plan.has_value());
    CHECK(error.find("expert 17") != std::string::npos);
  }
  // A size that differs between experts must be caught.
  {
    std::vector<qwen::TensorSize> tensors = checkpoint(false);
    for (auto& t : tensors) {
      if (t.name == "model.language_model.layers.0.mlp.experts.5.gate_proj.weight") t.bytes += 1;
    }
    std::string error;
    const auto plan = qwen::plan_placement(tensors, &error);
    CHECK(!plan.has_value());
    CHECK(error.find("size differs") != std::string::npos);
  }
  // Duplicates and unknown names are rejected.
  {
    std::vector<qwen::TensorSize> tensors = checkpoint(false);
    tensors.push_back(tensors.front());
    std::string error;
    CHECK(!qwen::plan_placement(tensors, &error).has_value());
    CHECK(error.find("duplicate") != std::string::npos);
  }
  {
    std::vector<qwen::TensorSize> tensors = checkpoint(false);
    tensors.push_back({"model.language_model.layers.0.mlp.experts.0.bogus.weight", 4});
    std::string error;
    CHECK(!qwen::plan_placement(tensors, &error).has_value());
    CHECK(error.find("unknown") != std::string::npos);
  }
  return 0;
}

}  // namespace

int main() {
  nvidia_case();
  ornith_case();
  error_cases();
  std::printf("qwen35moe placement: ok\n");
  return 0;
}
