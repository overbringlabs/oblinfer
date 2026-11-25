// Tensor names of the Qwen3.6-35B-A3B family, parsed into typed keys.
//
// Two exporters publish this architecture with different dialects of the same names:
//   NVIDIA: model.language_model.*, experts per expert with `input_scale`, MTP experts packed as
//           `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}` (no artifact suffix).
//   Ornith: the same per-expert main layers without `input_scale`, and MTP experts per expert.
// The loader maps both onto one set of keys, so the packing and kernel code sees a single shape.
//
// Host-only: no CUDA, no core/ headers, so the name tests run everywhere. Header-only packs compile
// this translation unit on every build (see CMakeLists.txt).
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "arch/qwen35moe/model/model_data.h"

namespace ingine::arch::qwen35moe {

enum class Dialect : std::uint8_t { Unknown, Nvidia, Ornith };

enum class Role : std::uint8_t {
  Norm,       // input/post attention layer norms, q/k norm, GDN norm
  GdnParam,   // A_log, dt_bias
  Dense,      // a quantized or plain dense projection (attention, GDN, dense MLP)
  Router,     // mlp.gate: the router scores
  SharedExpert,
  Expert,     // one routed expert, or a packed expert tensor (expert == -1)
  Embedding,
  Head,
  Mtp,        // MTP glue: fc, pre_fc_norm_embedding, pre_fc_norm_hidden, norm
  Vision,
};

enum class Projection : std::uint8_t {
  None,
  InQkv, InZ, InA, InB, OutProj, Conv1d, ALog, DtBias,
  QProj, KProj, VProj, OProj, QNorm, KNorm,
  GateProj, UpProj, DownProj, GateUpProj,  // GateUpProj: packed gate and up rows in one tensor
  Router, SharedGate, LmHead,
};

enum class Artifact : std::uint8_t { Weight, WeightScale, WeightScale2, InputScale };

struct TensorName {
  Role role = Role::Dense;
  Projection projection = Projection::None;
  Artifact artifact = Artifact::Weight;
  int layer = -1;   // LAYERS for the MTP block, -1 for model-level tensors
  int expert = -1;  // routed expert id; -1 for a packed tensor or a non-expert tensor
  bool mtp = false;

  bool operator==(const TensorName&) const = default;
};

// Nullopt for a name this pack does not know: an unknown projection, a missing or unknown artifact,
// or a prefix that belongs to another architecture.
std::optional<TensorName> parse_tensor_name(std::string_view name);

// Nvidia when any expert tensor carries `input_scale`; Ornith when experts are per expert and no
// `input_scale` appears; Unknown otherwise. Cheap, O(names) with one pass.
Dialect detect_dialect(std::span<const std::string> names);

}  // namespace ingine::arch::qwen35moe
