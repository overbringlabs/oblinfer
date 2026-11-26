#include "arch/qwen35moe/model/tensor_names.h"

#include <array>

namespace ingine::arch::qwen35moe {
namespace {

constexpr std::string_view kModelPrefix = "model.language_model.";
constexpr std::string_view kMtpPrefix = "mtp.";
constexpr std::string_view kLayersPrefix = "layers.";

bool parse_int(std::string_view text, int* out) {
  if (text.empty()) return false;
  int value = 0;
  for (char c : text) {
    if (c < '0' || c > '9') return false;
    value = value * 10 + (c - '0');
    if (value > 1000000) return false;
  }
  *out = value;
  return true;
}

std::optional<Artifact> artifact_of(std::string_view last) {
  if (last == "weight") return Artifact::Weight;
  if (last == "weight_scale") return Artifact::WeightScale;
  if (last == "weight_scale_2") return Artifact::WeightScale2;
  if (last == "input_scale") return Artifact::InputScale;
  return std::nullopt;
}

// Splits on '.', at most kMax parts; a longer name is unknown to this pack.
constexpr int kMaxParts = 6;

struct Parts {
  std::array<std::string_view, kMaxParts> part;
  int count = 0;
};

std::optional<Parts> split(std::string_view body) {
  Parts out;
  size_t begin = 0;
  while (true) {
    if (out.count == kMaxParts) return std::nullopt;
    const size_t dot = body.find('.', begin);
    if (dot == std::string_view::npos) {
      out.part[out.count++] = body.substr(begin);
      return out;
    }
    out.part[out.count++] = body.substr(begin, dot - begin);
    begin = dot + 1;
  }
}

std::optional<TensorName> dense(std::string_view artifact_text, Role role, Projection projection,
                                TensorName base) {
  const std::optional<Artifact> artifact = artifact_of(artifact_text);
  if (!artifact) return std::nullopt;
  base.role = role;
  base.projection = projection;
  base.artifact = *artifact;
  return base;
}

// Gate/up/down of a routed or shared expert, or the 27B dense MLP of the same names.
std::optional<Projection> mlp_projection(std::string_view name) {
  if (name == "gate_proj") return Projection::GateProj;
  if (name == "up_proj") return Projection::UpProj;
  if (name == "down_proj") return Projection::DownProj;
  return std::nullopt;
}

std::optional<TensorName> classify_body(std::string_view body, TensorName base) {
  const std::optional<Parts> parts = split(body);
  if (!parts) return std::nullopt;
  const auto& p = parts->part;
  const int n = parts->count;

  if (n == 2 && (p[0] == "input_layernorm" || p[0] == "post_attention_layernorm")) {
    return dense(p[1], Role::Norm, Projection::None, base);
  }

  if (p[0] == "self_attn" && n >= 2) {
    if (n == 3 && p[1] == "q_norm") return dense(p[2], Role::Norm, Projection::QNorm, base);
    if (n == 3 && p[1] == "k_norm") return dense(p[2], Role::Norm, Projection::KNorm, base);
    if (n != 3) return std::nullopt;
    if (p[1] == "q_proj") return dense(p[2], Role::Dense, Projection::QProj, base);
    if (p[1] == "k_proj") return dense(p[2], Role::Dense, Projection::KProj, base);
    if (p[1] == "v_proj") return dense(p[2], Role::Dense, Projection::VProj, base);
    if (p[1] == "o_proj") return dense(p[2], Role::Dense, Projection::OProj, base);
    return std::nullopt;
  }

  if (p[0] == "linear_attn" && n >= 2) {
    if (n == 2) {
      if (p[1] == "A_log") return dense("weight", Role::GdnParam, Projection::ALog, base);
      if (p[1] == "dt_bias") return dense("weight", Role::GdnParam, Projection::DtBias, base);
      return std::nullopt;
    }
    if (n != 3) return std::nullopt;
    if (p[1] == "in_proj_qkv") return dense(p[2], Role::Dense, Projection::InQkv, base);
    if (p[1] == "in_proj_z") return dense(p[2], Role::Dense, Projection::InZ, base);
    if (p[1] == "in_proj_a") return dense(p[2], Role::Dense, Projection::InA, base);
    if (p[1] == "in_proj_b") return dense(p[2], Role::Dense, Projection::InB, base);
    if (p[1] == "out_proj") return dense(p[2], Role::Dense, Projection::OutProj, base);
    if (p[1] == "conv1d") return dense(p[2], Role::Dense, Projection::Conv1d, base);
    if (p[1] == "norm") return dense(p[2], Role::Norm, Projection::None, base);
    return std::nullopt;
  }

  if (p[0] == "embed_tokens" && n == 2) {
    return dense(p[1], Role::Embedding, Projection::None, base);
  }

  if (p[0] == "mlp" && n >= 2) {
    if (p[1] == "gate" && n == 3) return dense(p[2], Role::Router, Projection::Router, base);
    if (p[1] == "shared_expert_gate" && n == 3) {
      return dense(p[2], Role::SharedExpert, Projection::SharedGate, base);
    }
    if (p[1] == "shared_expert" && n == 4) {
      const std::optional<Projection> proj = mlp_projection(p[2]);
      if (!proj) return std::nullopt;
      return dense(p[3], Role::SharedExpert, *proj, base);
    }
    // Dense MLP of the 27B sibling; harmless here and shared by the same classifier.
    if (n == 3) {
      const std::optional<Projection> proj = mlp_projection(p[1]);
      if (proj) return dense(p[2], Role::Dense, *proj, base);
    }
    if (p[1] != "experts") return std::nullopt;
    if (n == 3) {  // packed: experts.gate_up_proj, experts.down_proj (no artifact suffix)
      if (p[2] == "gate_up_proj") return dense("weight", Role::Expert, Projection::GateUpProj, base);
      if (p[2] == "down_proj") return dense("weight", Role::Expert, Projection::DownProj, base);
      return std::nullopt;
    }
    if (n == 5) {  // per expert: experts.<id>.<proj>.<artifact>
      int expert = -1;
      if (!parse_int(p[2], &expert)) return std::nullopt;
      const std::optional<Projection> proj = mlp_projection(p[3]);
      if (!proj) return std::nullopt;
      base.expert = expert;
      return dense(p[4], Role::Expert, *proj, base);
    }
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<TensorName> parse_mtp(std::string_view body) {
  TensorName base;
  base.mtp = true;
  if (body.starts_with(kLayersPrefix)) {
    body.remove_prefix(kLayersPrefix.size());
    const size_t dot = body.find('.');
    if (dot == std::string_view::npos) return std::nullopt;
    int index = -1;
    if (!parse_int(body.substr(0, dot), &index) || index >= MTP_LAYERS) return std::nullopt;
    base.layer = LAYERS;
    return classify_body(body.substr(dot + 1), base);
  }
  // MTP glue: fc, pre_fc_norm_embedding, pre_fc_norm_hidden, norm.
  const std::optional<Parts> parts = split(body);
  if (!parts || parts->count != 2) return std::nullopt;
  const std::optional<Artifact> artifact = artifact_of(parts->part[1]);
  if (!artifact) return std::nullopt;
  base.role = Role::Mtp;
  base.artifact = *artifact;
  return base;
}

}  // namespace

std::optional<TensorName> parse_tensor_name(std::string_view name) {
  if (name.starts_with("model.visual.")) {
    TensorName vision;
    vision.role = Role::Vision;
    return vision;
  }
  if (name.starts_with("lm_head.")) {
    TensorName base;
    base.role = Role::Head;
    base.projection = Projection::LmHead;
    return dense(name.substr(name.rfind('.') + 1), Role::Head, Projection::LmHead, base);
  }
  if (name.starts_with(kMtpPrefix)) return parse_mtp(name.substr(kMtpPrefix.size()));
  if (!name.starts_with(kModelPrefix)) return std::nullopt;
  name.remove_prefix(kModelPrefix.size());
  if (name.starts_with("embed_tokens.")) {
    TensorName base;
    return classify_body(name, base);
  }
  if (name == "norm.weight") {  // final layer norm of the text tower
    TensorName base;
    base.role = Role::Norm;
    return base;
  }
  if (!name.starts_with(kLayersPrefix)) return std::nullopt;
  name.remove_prefix(kLayersPrefix.size());
  const size_t dot = name.find('.');
  if (dot == std::string_view::npos) return std::nullopt;
  int layer = -1;
  if (!parse_int(name.substr(0, dot), &layer) || layer >= LAYERS) return std::nullopt;
  TensorName base;
  base.layer = layer;
  return classify_body(name.substr(dot + 1), base);
}

Dialect detect_dialect(std::span<const std::string> names) {
  // Both exports number the trunk's routed experts, and both carry `input_scale` somewhere — Ornith's real
  // 94,393-tensor export has 130 of them on the GDN and attention projections — so neither fact tells the two
  // apart. What does: the MTP block. NVIDIA packs it (`mtp.layers.0.mlp.experts.gate_up_proj`), Ornith numbers
  // it like the trunk. Deciding on `input_scale` first labelled Ornith's real export Nvidia.
  bool numbered_mtp = false, packed_mtp = false, input_scale = false, packed_any = false;
  for (const std::string& name : names) {
    if (!input_scale && name.ends_with(".input_scale")) input_scale = true;
    const size_t at = name.find(".mlp.experts.");
    if (at == std::string::npos) continue;
    const std::string_view rest(name.c_str() + at + std::string_view(".mlp.experts.").size(),
                                name.size() - at - std::string_view(".mlp.experts.").size());
    const bool is_mtp = name.starts_with("mtp.");
    if (!rest.empty() && rest.front() >= '0' && rest.front() <= '9') {
      if (is_mtp) numbered_mtp = true;
      continue;
    }
    if (rest.starts_with("gate_up_proj") || rest.starts_with("down_proj")) {
      packed_any = true;
      if (is_mtp) packed_mtp = true;
    }
  }
  if (numbered_mtp && packed_mtp) return Dialect::Unknown;  // contradictory: placement will reject it
  if (numbered_mtp) return Dialect::Ornith;
  if (packed_mtp || packed_any || input_scale) return Dialect::Nvidia;
  return Dialect::Unknown;
}

}  // namespace ingine::arch::qwen35moe
