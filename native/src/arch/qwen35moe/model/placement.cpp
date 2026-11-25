#include "arch/qwen35moe/model/placement.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace ingine::arch::qwen35moe {
namespace {

int projection_rank(Projection p) {
  switch (p) {
    case Projection::GateProj: return 0;
    case Projection::GateUpProj: return 0;
    case Projection::UpProj: return 1;
    case Projection::DownProj: return 2;
    default: return 9;
  }
}

int artifact_rank(Artifact a) {
  switch (a) {
    case Artifact::Weight: return 0;
    case Artifact::WeightScale: return 1;
    case Artifact::WeightScale2: return 2;
    case Artifact::InputScale: return 3;
  }
  return 9;
}

using SlotKey = std::pair<Projection, Artifact>;

struct ExpertTemplate {
  // One entry per (projection, artifact) with the byte size taken from its first occurrence.
  std::map<SlotKey, std::pair<std::string, std::uint64_t>> slots;
  std::set<SlotKey> all_keys;          // keys every expert must carry
  std::set<std::pair<int, SlotKey>> seen_pairs;  // (expert, key) to catch duplicates
  bool packed = false;                 // GateUpProj/DownProj dialect: one tensor covers every expert
  bool active = false;
};

}  // namespace

std::optional<PlacementPlan> plan_placement(std::span<const TensorSize> tensors, std::string* error) {
  auto fail = [&](const std::string& what) -> std::optional<PlacementPlan> {
    if (error) *error = what;
    return std::nullopt;
  };

  PlacementPlan plan;
  std::vector<std::string> names;
  names.reserve(tensors.size());
  for (const TensorSize& t : tensors) names.push_back(t.name);
  plan.dialect = detect_dialect(names);

  std::set<std::string> seen;
  std::vector<LayerPlan> layers(LAYERS + MTP_LAYERS);
  for (int l = 0; l < LAYERS + MTP_LAYERS; ++l) layers[l].layer = l;

  std::map<int, ExpertTemplate> templates;
  std::uint64_t model_bytes = 0;

  for (const TensorSize& t : tensors) {
    if (!seen.insert(t.name).second) return fail("duplicate tensor: " + t.name);
    const std::optional<TensorName> parsed = parse_tensor_name(t.name);
    if (!parsed) return fail("unknown tensor name: " + t.name);
    if (parsed->role == Role::Vision) continue;  // loaded separately by the vision path when enabled

    if (parsed->role != Role::Expert) {
      TensorSlot slot{t.name, parsed->projection, parsed->artifact, 0};
      if (parsed->layer < 0) {
        slot.offset = model_bytes;
        model_bytes += t.bytes;
        plan.model.push_back(std::move(slot));
      } else {
        LayerPlan& lp = layers[parsed->layer];
        slot.offset = lp.dense_bytes;
        lp.dense_bytes += t.bytes;
        lp.dense.push_back(std::move(slot));
      }
      plan.total_bytes += t.bytes;
      continue;
    }

    LayerPlan& lp = layers[parsed->layer];
    lp.has_experts = true;
    ExpertTemplate& tmpl = templates[parsed->layer];
    // Packing is a property of the name shape: only `mlp.experts.<gate_up_proj|down_proj>` parses without an
    // expert id, and only those two projections can be packed.
    const bool packed = parsed->expert < 0;
    if (packed && parsed->projection != Projection::GateUpProj && parsed->projection != Projection::DownProj) {
      return fail("expert tensor without an expert id for a per-expert projection: " + t.name);
    }
    if (packed) {
      if (tmpl.active && !tmpl.packed) return fail("mixed packed and per-expert tensors in layer " +
                                                   std::to_string(parsed->layer));
      tmpl.packed = true;
    } else if (tmpl.active && tmpl.packed) {
      return fail("mixed packed and per-expert tensors in layer " + std::to_string(parsed->layer));
    }
    tmpl.active = true;

    const SlotKey key{parsed->projection, parsed->artifact};
    if (!tmpl.seen_pairs.insert({packed ? -1 : parsed->expert, key}).second) {
      return fail("duplicate expert artifact: " + t.name);
    }
    const auto it = tmpl.slots.find(key);
    if (it == tmpl.slots.end()) {
      tmpl.slots.emplace(key, std::make_pair(t.name, t.bytes));
    } else if (it->second.second != t.bytes) {
      return fail("expert artifact size differs between experts: " + t.name);
    }
    tmpl.all_keys.insert(key);
  }

  for (const auto& [layer, tmpl] : templates) {
    if (!tmpl.active) continue;
    LayerPlan& lp = layers[layer];
    std::vector<SlotKey> keys(tmpl.slots.size());
    std::size_t i = 0;
    for (const auto& [key, value] : tmpl.slots) keys[i++] = key;
    std::sort(keys.begin(), keys.end(), [](const SlotKey& a, const SlotKey& b) {
      return std::make_tuple(projection_rank(a.first), artifact_rank(a.second)) <
             std::make_tuple(projection_rank(b.first), artifact_rank(b.second));
    });

    std::uint64_t offset = 0;
    for (const SlotKey& key : keys) {
      const auto& [source, bytes] = tmpl.slots.at(key);
      lp.experts.slots.push_back(TensorSlot{source, key.first, key.second, offset});
      offset += bytes;
    }
    lp.experts.stride = offset;

    if (tmpl.packed) {
      // The packed tensors already hold every expert.
      lp.experts.bytes = offset;
    } else {
      // Every expert must carry every key, or the stride would mix layouts.
      for (int e = 0; e < EXPERTS; ++e) {
        for (const SlotKey& key : keys) {
          if (!tmpl.seen_pairs.contains({e, key})) {
            return fail("layer " + std::to_string(layer) + " expert " + std::to_string(e) + " is missing a " +
                        std::to_string(int(key.second)) + " artifact of projection " +
                        std::to_string(int(key.first)));
          }
        }
      }
      lp.experts.bytes = offset * std::uint64_t(EXPERTS);
    }
    plan.total_bytes += lp.experts.bytes;
  }

  for (int l = 0; l < LAYERS; ++l) {
    if (layers[l].dense.empty() && !layers[l].has_experts) {
      return fail("layer " + std::to_string(l) + " has no tensors");
    }
  }

  // Arena layout: model area first, then each layer's dense slots followed by its expert block.
  std::uint64_t cursor = model_bytes;
  for (LayerPlan& lp : layers) {
    lp.dense_arena_offset = cursor;
    cursor += lp.dense_bytes;
    if (lp.has_experts) {
      lp.experts.arena_offset = cursor;
      cursor += lp.experts.bytes;
    }
  }
  plan.total_bytes = cursor;
  plan.layers = std::move(layers);
  return plan;
}

}  // namespace ingine::arch::qwen35moe
