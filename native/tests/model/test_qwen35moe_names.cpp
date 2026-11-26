// The Qwen3.6-35B-A3B tensor-name classifier against both published export dialects: NVIDIA (packed
// gate_up_proj/down_proj experts) and Ornith (numbered per-expert projections). Both exports carry
// `input_scale` on the dense projections — Ornith's real 94,393-tensor export has 130 of them — so the
// dialect is decided by the expert layout, not by that tensor. The test asserts both dialects map onto
// the same typed keys, which is what the loader and packing code rely on. Host only: no GPU, no model
// files.
#include "arch/qwen35moe/model/tensor_names.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace qwen = ingine::arch::qwen35moe;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

namespace {

using Key = std::tuple<int, qwen::Projection, int>;  // layer, projection, expert

std::vector<std::string> names(bool ornith) {
  std::vector<std::string> out;
  auto add = [&](std::string name) { out.push_back(std::move(name)); };
  const std::string layer_prefix = "model.language_model.layers.";

  for (int l : {0, 3}) {
    add(layer_prefix + std::to_string(l) + ".input_layernorm.weight");
    add(layer_prefix + std::to_string(l) + ".post_attention_layernorm.weight");
    if (l == 0) {
      for (const char* proj : {"in_proj_qkv", "in_proj_z", "out_proj"}) {
        const std::string base = layer_prefix + std::to_string(l) + ".linear_attn." + proj;
        add(base + ".weight");
        add(base + ".weight_scale");
        if (!ornith) add(base + ".input_scale");
      }
      add(layer_prefix + std::to_string(l) + ".linear_attn.in_proj_a.weight");
      add(layer_prefix + std::to_string(l) + ".linear_attn.in_proj_b.weight");
      add(layer_prefix + std::to_string(l) + ".linear_attn.conv1d.weight");
      add(layer_prefix + std::to_string(l) + ".linear_attn.A_log");
      add(layer_prefix + std::to_string(l) + ".linear_attn.dt_bias");
      add(layer_prefix + std::to_string(l) + ".linear_attn.norm.weight");
    } else {
      for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        const std::string base = layer_prefix + std::to_string(l) + ".self_attn." + proj;
        add(base + ".weight");
        add(base + ".weight_scale");
        if (!ornith) add(base + ".input_scale");
      }
      add(layer_prefix + std::to_string(l) + ".self_attn.q_norm.weight");
      add(layer_prefix + std::to_string(l) + ".self_attn.k_norm.weight");
    }
    add(layer_prefix + std::to_string(l) + ".mlp.gate.weight");
    add(layer_prefix + std::to_string(l) + ".mlp.shared_expert_gate.weight");
    for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
      const std::string base = layer_prefix + std::to_string(l) + ".mlp.shared_expert." + proj;
      add(base + ".weight");
      add(base + ".weight_scale");
      add(base + ".weight_scale_2");
      if (!ornith) add(base + ".input_scale");
    }
    for (int e = 0; e < 2; ++e) {
      for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
        const std::string base =
            layer_prefix + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj;
        add(base + ".weight");
        add(base + ".weight_scale");
        add(base + ".weight_scale_2");
        if (!ornith) add(base + ".input_scale");
      }
    }
  }

  add("model.language_model.embed_tokens.weight");
  add("lm_head.weight");
  add("lm_head.weight_scale");
  add("lm_head.weight_scale_2");
  if (!ornith) add("lm_head.input_scale");
  add("model.visual.blocks.0.attn.qkv.weight");

  add("mtp.fc.weight");
  add("mtp.pre_fc_norm_embedding.weight");
  add("mtp.pre_fc_norm_hidden.weight");
  add("mtp.norm.weight");
  add("mtp.layers.0.input_layernorm.weight");
  add("mtp.layers.0.post_attention_layernorm.weight");
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    add(std::string("mtp.layers.0.self_attn.") + proj + ".weight");
  }
  add("mtp.layers.0.mlp.gate.weight");
  for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
    const std::string base = std::string("mtp.layers.0.mlp.shared_expert.") + proj;
    add(base + ".weight");
    add(base + ".weight_scale");
    add(base + ".weight_scale_2");
  }
  if (ornith) {
    for (int e = 0; e < qwen::EXPERTS; ++e) {
      for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
        const std::string base =
            "mtp.layers.0.mlp.experts." + std::to_string(e) + "." + proj;
        add(base + ".weight");
        add(base + ".weight_scale");
        add(base + ".weight_scale_2");
      }
    }
  } else {
    add("mtp.layers.0.mlp.experts.gate_up_proj");
    add("mtp.layers.0.mlp.experts.down_proj");
  }
  return out;
}

// Unpacked (layer, projection, expert) weight keys. A packed GateUpProj covers all experts and both
// the gate and up projections; a per-expert tensor covers its own expert.
std::set<Key> expert_weight_keys(const std::vector<std::string>& all) {
  std::set<Key> out;
  for (const std::string& name : all) {
    const auto parsed = qwen::parse_tensor_name(name);
    CHECK(parsed.has_value());
    if (parsed->role != qwen::Role::Expert || parsed->artifact != qwen::Artifact::Weight) continue;
    const bool packed = parsed->projection == qwen::Projection::GateUpProj ||
                        parsed->projection == qwen::Projection::DownProj;
    const int first = parsed->expert >= 0 ? parsed->expert : 0;
    const int last = parsed->expert >= 0 ? parsed->expert : qwen::EXPERTS - 1;
    for (int e = first; e <= last; ++e) {
      if (parsed->projection == qwen::Projection::GateUpProj) {
        out.emplace(parsed->layer, qwen::Projection::GateProj, e);
        out.emplace(parsed->layer, qwen::Projection::UpProj, e);
      } else {
        out.emplace(parsed->layer, parsed->projection, e);
      }
    }
  }
  return out;
}

}  // namespace

// A file of real tensor names, one per line, classifies every line and reports the ones the loader would not
// recognise. This is how a new export is checked against the pack before anyone downloads its weights.
int check_names_file(const char* path) {
  std::FILE* f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  std::vector<std::string> all;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    std::string name(line);
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) name.pop_back();
    if (!name.empty()) all.push_back(std::move(name));
  }
  std::fclose(f);

  std::set<std::string> unknown_roots;
  int rejected = 0;
  for (const std::string& name : all) {
    if (!qwen::parse_tensor_name(name)) {
      ++rejected;
      // Group by prefix so a whole missing family shows as one line.
      const std::size_t keep = name.find(".layers.");
      unknown_roots.insert(keep == std::string::npos ? name : name.substr(0, keep));
    }
  }
  std::printf("%zu names, dialect %s, %d rejected\n", all.size(),
              qwen::detect_dialect(all) == qwen::Dialect::Ornith    ? "ornith"
              : qwen::detect_dialect(all) == qwen::Dialect::Nvidia  ? "nvidia"
                                                                    : "unknown",
              rejected);
  for (const std::string& root : unknown_roots) std::printf("  unclassified family: %s\n", root.c_str());
  return rejected == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc > 1) return check_names_file(argv[1]);
  const std::vector<std::string> nvidia = names(false);
  const std::vector<std::string> ornith = names(true);

  CHECK(qwen::detect_dialect(nvidia) == qwen::Dialect::Nvidia);
  CHECK(qwen::detect_dialect(ornith) == qwen::Dialect::Ornith);

  // The real Ornith export carries input_scale on its dense projections, unlike the synthetic set above:
  // deciding the dialect on it labelled the real export Nvidia. The expert layout has to win.
  std::vector<std::string> ornith_real = ornith;
  ornith_real.push_back("model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale");
  ornith_real.push_back("model.language_model.layers.0.self_attn.q_proj.input_scale");
  CHECK(qwen::detect_dialect(ornith_real) == qwen::Dialect::Ornith);

  // The MTP block is the discriminator, because both exports number the trunk's experts. A set that both
  // numbers and packs its MTP is contradictory, and the detector must not pick a side.
  std::vector<std::string> mixed = ornith;
  mixed.push_back("mtp.layers.0.mlp.experts.gate_up_proj");
  CHECK(qwen::detect_dialect(mixed) == qwen::Dialect::Unknown);

  for (const std::string& name : nvidia) {
    if (!qwen::parse_tensor_name(name)) std::fprintf(stderr, "nvidia name rejected: %s\n", name.c_str());
    CHECK(qwen::parse_tensor_name(name).has_value());
  }
  for (const std::string& name : ornith) {
    if (!qwen::parse_tensor_name(name)) std::fprintf(stderr, "ornith name rejected: %s\n", name.c_str());
    CHECK(qwen::parse_tensor_name(name).has_value());
  }

  // Model-level roles agree across dialects.
  for (const char* name : {"model.language_model.embed_tokens.weight", "lm_head.weight", "lm_head.weight_scale_2",
                           "mtp.fc.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight",
                           "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.self_attn.q_proj.weight"}) {
    const auto a = qwen::parse_tensor_name(name);
    CHECK(a.has_value());
    const auto b = qwen::parse_tensor_name(name);
    CHECK(b.has_value());
    CHECK(*a == *b);
  }

  // Q and K norms parse as norms with their own projection.
  const auto q_norm = qwen::parse_tensor_name("model.language_model.layers.3.self_attn.q_norm.weight");
  CHECK(q_norm && q_norm->role == qwen::Role::Norm && q_norm->projection == qwen::Projection::QNorm &&
        q_norm->layer == 3);
  const auto dt_bias = qwen::parse_tensor_name("model.language_model.layers.0.linear_attn.dt_bias");
  CHECK(dt_bias && dt_bias->role == qwen::Role::GdnParam && dt_bias->projection == qwen::Projection::DtBias);

  // Expert weights normalise to the same (layer, projection, expert) set in both dialects.
  CHECK(expert_weight_keys(nvidia) == expert_weight_keys(ornith));

  // The MTP block lands on layer LAYERS with the same coverage after unpacking.
  int mtp_layer_experts = 0;
  for (const Key& key : expert_weight_keys(ornith)) {
    if (std::get<0>(key) == qwen::LAYERS) ++mtp_layer_experts;
  }
  CHECK(mtp_layer_experts == qwen::EXPERTS * 3);
  CHECK(expert_weight_keys(nvidia) == expert_weight_keys(ornith));

  // The vision tower is recognised and carries no layer or expert.
  const auto vision = qwen::parse_tensor_name("model.visual.blocks.0.mlp.linear_fc1.weight");
  CHECK(vision && vision->role == qwen::Role::Vision);

  // Unknown names are rejected rather than guessed.
  for (const char* name : {
           "model.language_model.layers.0.mlp.experts.0.bogus.weight",
           "model.language_model.layers.0.self_attn.q_proj.bogus",
           "model.language_model.layers.40.self_attn.q_proj.weight",
           "model.language_model.layers.0.mlp.experts.0.gate_proj",
           "mtp.layers.3.fc.weight",
           "other.model.tensor",
           "model.language_model.layers.0.a.b.c.d.e.f",
       }) {
    if (qwen::parse_tensor_name(name)) std::fprintf(stderr, "accepted unknown name: %s\n", name);
    CHECK(!qwen::parse_tensor_name(name).has_value());
  }

  std::printf("qwen35moe tensor names: %zu nvidia, %zu ornith names, %zu expert keys\n", nvidia.size(),
              ornith.size(), expert_weight_keys(nvidia).size());
  return 0;
}
