// Weights in the decoder's device layouts, read from the Hugging Face checkpoint: weights.load / load_layer,
// plus fast.FastDecoder._concat_projections (one [R, H] projection matrix per layer, the per-projection
// matrices are row views of it), _stack_layer_params (stacked norms and gate weights) and _fp8_dense
// (wproj, wo and the unembedding as FP8 rows + FP32 row scales, the BF16 copies freed), and
// model.Ablation.load / none.
//
// Large tensors are uploaded straight from the file with the core DeviceUploader; small per-channel
// parameters are read on the host, widened to FP32 (exact from BF16) and copied.
// Expert residency (EP step 2): a layer's four expert tensors and their factors go whole onto the second
// device when its ExpertResidency split is 0, through a second uploader; everything else stays here.
#include "model/runtime.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ingine::model {
namespace {

namespace fs = std::filesystem;

float bf16_bits_to_float(std::uint16_t b) {
  const std::uint32_t w = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

std::uint16_t float_to_bf16_bits(float f) {  // round to nearest even (torch .to(bfloat16))
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  if ((w & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((w >> 16) | 0x40);
  return std::uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

// Checkpoint values widened to FP32 on the host (BF16 -> FP32 exact, F32 as is).
std::vector<float> host_f32(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef ref = m.at(name);
  const std::vector<std::byte> raw = m.read(name);
  const std::size_t n = std::size_t(ref->numel());
  std::vector<float> out(n);
  if (ref->dtype == "F32") {
    std::memcpy(out.data(), raw.data(), n * 4);
  } else if (ref->dtype == "BF16") {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      out[i] = bf16_bits_to_float(b);
    }
  } else {
    throw std::runtime_error("weights: " + name + " has unsupported dtype " + ref->dtype);
  }
  return out;
}

// Makes `device` the calling thread's current device for the scope, restoring the previous one on exit.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    CUDA_CHECK(cudaGetDevice(&saved_));
    if (saved_ != device) CUDA_CHECK(cudaSetDevice(device));
  }
  ~DeviceScope() { (void)cudaSetDevice(saved_); }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int saved_ = 0;
};

// Loads checkpoint tensors into this device; with `second`, a layer's experts can be placed on that device
// instead. Every allocation and every upload runs with its own device current.
class Loader {
 public:
  Loader(const SafetensorsModel& m, DeviceUploader& up, cudaStream_t stream, std::vector<DeviceBuffer>& storage,
         const DeviceContext* second = nullptr, SecondStorage* storage_hi = nullptr)
      : m_(m), up_(up), stream_(stream), storage_(storage), second_(second), storage_hi_(storage_hi) {
    CUDA_CHECK(cudaGetDevice(&primary_));
    if (second_ == nullptr) return;
    if (storage_hi_ == nullptr) throw std::invalid_argument("Loader: a second device needs its own storage");
    storage_hi_->device = second_->device;
    // 16 MiB x 4 staging buffers: the primary uploader already pins 384 MiB and this host has 16 GB of RAM.
    const DeviceScope scope(second_->device);
    up_hi_.emplace(UploadOptions{std::size_t(16) << 20, 4, 4}, second_->device);
  }

  ~Loader() {
    if (!up_hi_) return;
    const DeviceScope scope(second_->device);  // the uploader's events belong to that device
    up_hi_.reset();
  }

  // Device memory for `bytes` behind the views: this loader's device, or (device 1) the second one.
  void* alloc(std::size_t bytes, int device = 0) {
    const DeviceScope scope(device_id(device));
    std::vector<DeviceBuffer>& s = device == 0 ? storage_ : storage_hi_->buffers;
    s.emplace_back(bytes);
    allocated_[device] += bytes;
    return s.back().p;
  }

  Tensor upload_f32(const std::vector<float>& host, std::initializer_list<std::int32_t> shape, int device = 0) {
    const DeviceScope scope(device_id(device));
    void* d = alloc(host.size() * 4, device);
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    return rm(d, DType::FP32, shape);
  }

  Tensor upload_bf16(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    std::vector<std::uint16_t> b(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) b[i] = float_to_bf16_bits(host[i]);
    void* d = alloc(b.size() * 2);
    CUDA_CHECK(cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    return rm(d, DType::BF16, shape);
  }

  // f32(name): checkpoint tensor as FP32 (weights.load_layer's `f32`), viewed with `shape`.
  Tensor f32(const std::string& name, std::initializer_list<std::int32_t> shape) {
    return upload_f32(host_f32(m_, name), shape);
  }

  // Queues the raw bytes of `name` (rows [0, rows) if rows >= 0) for upload to `dst` on `device`.
  void queue(const std::string& name, void* dst, std::int64_t rows = -1, int device = 0) {
    const SafetensorsRef ref = m_.at(name);
    UploadRequest r = upload_request(ref, dst);
    if (rows >= 0) r.bytes = ref->bytes / std::uint64_t(ref->shape[0]) * std::uint64_t(rows);
    pending_[device].push_back(r);
  }

  // Raw checkpoint tensor (any dtype) in new device memory, uploaded at flush().
  Tensor raw(const std::string& name, int device = 0) {
    const SafetensorsRef ref = m_.at(name);
    void* d = alloc(ref->bytes, device);
    queue(name, d, -1, device);
    return ref->view(d);
  }

  // Uploads every queued range: this device's on `stream_`, the second device's on its transfer stream.
  void flush() {
    if (!pending_[0].empty()) up_.upload(pending_[0], stream_);
    pending_[0].clear();
    if (!pending_[1].empty()) {
      const DeviceScope scope(device_id(1));
      up_hi_->upload(pending_[1], second_->transfer_stream);
    }
    pending_[1].clear();
  }

  const SafetensorsModel& model() const { return m_; }
  cudaStream_t stream() const { return stream_; }
  std::size_t bytes(int device) const { return allocated_[device]; }

 private:
  // The CUDA device of a residency slot: 0 is this loader's device, 1 the second one.
  int device_id(int device) const {
    if (device == 0) return primary_;
    if (second_ == nullptr) throw std::logic_error("Loader: no second device for an expert shard");
    return second_->device;
  }

  const SafetensorsModel& m_;
  DeviceUploader& up_;  // this device's uploads
  cudaStream_t stream_;
  std::vector<DeviceBuffer>& storage_;
  const DeviceContext* second_ = nullptr;  // options.second_gpu; null: one device
  SecondStorage* storage_hi_ = nullptr;
  std::optional<DeviceUploader> up_hi_;  // the second device's uploads, built only for a second device
  int primary_ = 0;                      // the device this loader's tensors live on
  std::array<std::size_t, 2> allocated_{};
  std::array<std::vector<UploadRequest>, 2> pending_;
};

std::array<ExpertFormat, LAYERS> expert_formats(const fs::path& model_dir) {
  std::ifstream in(model_dir / "hf_quant_config.json");
  if (!in) throw std::runtime_error("weights: cannot open " + (model_dir / "hf_quant_config.json").string());
  const nlohmann::json q = nlohmann::json::parse(in);
  std::array<ExpertFormat, LAYERS> fmt{};
  std::array<bool, LAYERS> seen{};
  const std::string suffix = "mlp.experts.w13_weight";
  for (const auto& group : q.at("model").at("parameter_groups")) {
    const std::string dt = group.at("compute_dtype").get<std::string>();
    for (const auto& t : group.at("tensors")) {
      const std::string name = t.get<std::string>();
      if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        continue;
      // model.llm.layers.<l>.mlp.experts.w13_weight
      const std::size_t a = std::string("model.llm.layers.").size();
      const int l = std::stoi(name.substr(a, name.find('.', a) - a));
      if (l < 0 || l >= LAYERS) continue;
      if (dt == "nvfp4") fmt[l] = ExpertFormat::NVFP4;
      else if (dt == "mxfp8") fmt[l] = ExpertFormat::MXFP8;
      else throw std::runtime_error("weights: unsupported expert format " + dt);
      seen[l] = true;
    }
  }
  for (int l = 0; l < LAYERS; ++l)
    if (!seen[l]) throw std::runtime_error("weights: no expert format for layer " + std::to_string(l));
  return fmt;
}

// Step 2 loads a layer's experts whole on one device: moe_up/moe_down take one base pointer per expert
// tensor and address rows e*R, so a split inside a layer needs per-expert bases (step 3).
void validate_residency(const ExpertResidency& residency, bool second_open) {
  for (int l = 0; l < LAYERS; ++l) {
    const int split = int(residency.split[l]);
    if (split == EXPERTS) continue;
    if (split != 0)
      throw std::invalid_argument("expert residency: split " + std::to_string(split) + " in layer " +
                                  std::to_string(l) + " needs per-expert base pointers (step 3)");
    if (!second_open)
      throw std::invalid_argument("expert residency: layer " + std::to_string(l) +
                                  " has its experts on the second device, which is not open");
  }
}

// FP8 rows + scales of a BF16 matrix (dense fp8_rows), into new storage.
Dense to_fp8(Loader& ld, const Tensor& bf16, int rows, int cols) {
  void* q = ld.alloc(std::size_t(rows) * cols);
  float* s = static_cast<float*>(ld.alloc(std::size_t(rows) * 4));
  kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(bf16.data), rows, cols, q, s, ld.stream());
  return Dense{rm(q, DType::FP8_E4M3FN, {rows, cols}), rm(s, DType::FP32, {rows})};
}

Dense row_view(const Dense& d, int begin, int end) {
  const int cols = d.cols();
  const std::size_t esize = d.fp8() ? 1 : 2;
  Dense v;
  v.w = rm(static_cast<char*>(d.w.data) + std::size_t(begin) * cols * esize, d.w.dtype, {end - begin, cols});
  if (d.fp8()) v.scale = rm(static_cast<float*>(d.scale.data) + begin, DType::FP32, {end - begin});
  return v;
}

Layer load_layer_into(Loader& ld, int l, ExpertFormat fmt, bool dense_fp8, int split) {
  const SafetensorsModel& m = ld.model();
  const std::string p = "model.llm.layers." + std::to_string(l) + ".";
  Layer layer;
  layer.idx = l;
  layer.full = is_full_layer(l);
  layer.loop = is_loop_layer(l);
  const int H = HIDDEN;

  if (layer.full) {
    layer.q_norm = ld.f32(p + "q_input_norm.weight", {H});
    layer.k_norm = ld.f32(p + "k_input_norm.weight", {H});
    layer.v_norm = ld.f32(p + "v_input_norm.weight", {H});
  } else {
    layer.attn_norm = ld.f32(p + "attn_norm.weight", {H});
  }
  layer.mlp_norm = ld.f32(p + "mlp_norm.weight", {H});

  // projections: one [R, H] BF16 matrix in SEG order (fast._concat_projections)
  static const char* seg_names[NUM_SEGS] = {"attn.wq_du.weight",        "attn.gate_attn.weight",
                                            "attn.wr_du.weight",        "attn.attn_sink_proj.weight",
                                            "attn.wk_dv.weight",        "attn.wv_dv.weight",
                                            "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"};
  const int R = proj_rows(layer.full);
  DeviceBuffer proj_tmp, wo_tmp;
  Tensor wproj_bf16, wo_bf16;
  auto queue_bf16_rows = [&](const std::string& name, void* dst, int rows, int cols) {
    const SafetensorsRef ref = m.at(name);
    if (ref->dtype != "BF16" || ref->numel() != std::int64_t(rows) * cols)
      throw std::runtime_error("weights: " + name + " must be BF16 [" + std::to_string(rows) + ", " +
                               std::to_string(cols) + "]");
    ld.queue(name, dst);
  };
  {
    void* d = dense_fp8 ? (proj_tmp = DeviceBuffer(std::size_t(R) * H * 2)).p : ld.alloc(std::size_t(R) * H * 2);
    for (int s = 0; s < num_segs(layer.full); ++s)
      queue_bf16_rows(p + seg_names[s], static_cast<char*>(d) + std::size_t(SEG_BEGIN[s]) * H * 2, SEG_ROWS[s], H);
    wproj_bf16 = rm(d, DType::BF16, {R, H});
    void* o = dense_fp8 ? (wo_tmp = DeviceBuffer(std::size_t(H) * 3072 * 2)).p : ld.alloc(std::size_t(H) * 3072 * 2);
    queue_bf16_rows(p + "attn.wo_ud.weight", o, H, HEADS * HEAD_DIM);
    wo_bf16 = rm(o, DType::BF16, {H, HEADS * HEAD_DIM});
  }

  layer.q_scale = ld.f32(p + "attn.q_l2_clip.scale", {HEADS, HEAD_DIM});
  layer.r_scale = ld.f32(p + "attn.rel_l2_clip.scale", {HEADS, D_REL});
  layer.rel_proj = ld.f32(p + "attn.rel_logits_proj.proj", {D_REL, REL_EXTENT});
  if (layer.full) {
    layer.k_static = ld.f32(p + "attn.k_sconv.static_w", {CONV_K, KV_DIM});
    layer.v_static = ld.f32(p + "attn.v_sconv.static_w", {CONV_K, KV_DIM});
  } else {
    layer.k_conv = ld.f32(p + "attn.k_sconv.weight", {KV_DIM, CONV_K});  // [C, 1, K] squeezed
    layer.v_conv = ld.f32(p + "attn.v_sconv.weight", {KV_DIM, CONV_K});
  }
  layer.attn_conv = ld.f32(p + "attn_sconv.weight", {H, CONV_K});
  layer.mlp_conv = ld.f32(p + "mlp_sconv.weight", {H, CONV_K});
  layer.router = ld.f32(p + "mlp.gate.weight", {EXPERTS, H});
  layer.router_bias = ld.f32(p + "mlp.gate.bias", {EXPERTS});
  layer.router_bf16 = ld.upload_bf16(host_f32(m, p + "mlp.gate.weight"), {EXPERTS, H});  // exact
  if (l > 0) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(l - 1);
    const std::int32_t nq = std::int32_t(m.at(qn)->numel() / H);
    layer.attn_query = ld.f32(qn, {nq, H});
  }
  layer.mlp_query = ld.f32("model.llm.attn_res.mlp_queries." + std::to_string(l), {1, H});

  // stacked norm weights (fast._stack_layer_params)
  {
    std::vector<float> n3(std::size_t(3) * H);
    if (layer.full) {
      const char* names[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
      for (int g = 0; g < 3; ++g) {
        const std::vector<float> v = host_f32(m, p + names[g]);
        std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
      }
    } else {
      const std::vector<float> v = host_f32(m, p + "attn_norm.weight");
      for (int g = 0; g < 3; ++g) std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
    }
    layer.attn_norm3 = ld.upload_f32(n3, {3, H});
    layer.attn_norm_mix = layer.full ? layer.attn_norm3 : rm(layer.attn_norm3.data, DType::FP32, {1, H});
    layer.mlp_norm1 = rm(layer.mlp_norm.data, DType::FP32, {1, H});
  }

  if (layer.loop) {
    // checkpoint controllers: attn (q), k, v, mlp; stacks [q, k, v] and [mlp]
    const char* ckpt[4] = {"attn", "k", "v", "mlp"};
    std::vector<float> down[4], forget[4], gain[4];
    for (int g = 0; g < 4; ++g) {
      const std::string c = p + "controllers." + ckpt[g] + ".";
      down[g] = host_f32(m, c + "down_weight");
      forget[g] = host_f32(m, c + "forget_weight");
      gain[g] = host_f32(m, c + "gain_weight");
    }
    auto stack = [](std::vector<float>* parts, int n) {
      std::vector<float> out;
      for (int i = 0; i < n; ++i) out.insert(out.end(), parts[i].begin(), parts[i].end());
      return out;
    };
    const int D2 = 2 * H;
    layer.qkv_gate_down = ld.upload_bf16(stack(down, 3), {3, GATE_RANK, D2});  // exact (BF16 checkpoint)
    layer.qkv_gate_forget = ld.upload_f32(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain = ld.upload_f32(stack(gain, 3), {3, H, GATE_RANK});
    layer.qkv_gate_forget16 = ld.upload_bf16(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain16 = ld.upload_bf16(stack(gain, 3), {3, H, GATE_RANK});
    layer.mlp1_gate_down = ld.upload_bf16(down[3], {1, GATE_RANK, D2});
    layer.mlp1_gate_forget = ld.upload_f32(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain = ld.upload_f32(gain[3], {1, H, GATE_RANK});
    layer.mlp1_gate_forget16 = ld.upload_bf16(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain16 = ld.upload_bf16(gain[3], {1, H, GATE_RANK});
    auto at = [](const Tensor& stack, int g, DType dt, std::int32_t a, std::int32_t b) {
      const std::size_t es = dt == DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(stack.data) + std::size_t(g) * a * b * es, dt, {a, b});
    };
    Tensor* downs[3] = {&layer.q_gate_down, &layer.k_gate_down, &layer.v_gate_down};
    Tensor* forgets[3] = {&layer.q_gate_forget, &layer.k_gate_forget, &layer.v_gate_forget};
    Tensor* gains[3] = {&layer.q_gate_gain, &layer.k_gate_gain, &layer.v_gate_gain};
    for (int g = 0; g < 3; ++g) {
      *downs[g] = at(layer.qkv_gate_down, g, DType::BF16, GATE_RANK, D2);
      *forgets[g] = at(layer.qkv_gate_forget, g, DType::FP32, H, GATE_RANK);
      *gains[g] = at(layer.qkv_gate_gain, g, DType::FP32, H, GATE_RANK);
    }
    layer.mlp_gate_down = at(layer.mlp1_gate_down, 0, DType::BF16, GATE_RANK, D2);
    layer.mlp_gate_forget = at(layer.mlp1_gate_forget, 0, DType::FP32, H, GATE_RANK);
    layer.mlp_gate_gain = at(layer.mlp1_gate_gain, 0, DType::FP32, H, GATE_RANK);
  }

  // routed experts in checkpoint encoding, whole on the device the residency table names
  const std::string e = p + "mlp.experts.";
  Experts& ex = layer.experts;
  ex.fmt = fmt;
  ex.split = static_cast<std::int16_t>(split);
  const int shard = split == EXPERTS ? 0 : 1;  // ExpertResidency: 0 or EXPERTS only
  if (shard == 0) {
    ex.w13 = ld.raw(e + "w13_weight");
    ex.w13_scale = ld.raw(e + "w13_weight_scale");
    ex.w2 = ld.raw(e + "w2_weight");
    ex.w2_scale = ld.raw(e + "w2_weight_scale");
  } else {
    ex.w13_hi = ld.raw(e + "w13_weight", 1);
    ex.w13_scale_hi = ld.raw(e + "w13_weight_scale", 1);
    ex.w2_hi = ld.raw(e + "w2_weight", 1);
    ex.w2_scale_hi = ld.raw(e + "w2_weight_scale", 1);
  }
  const bool nv = fmt == ExpertFormat::NVFP4;
  const std::string want_codes = nv ? "U8" : "F8_E4M3", want_scales = nv ? "F8_E4M3" : "U8";
  if (m.at(e + "w13_weight")->dtype != want_codes || m.at(e + "w13_weight_scale")->dtype != want_scales ||
      m.at(e + "w2_weight")->dtype != want_codes || m.at(e + "w2_weight_scale")->dtype != want_scales)
    throw std::runtime_error("weights: layer " + std::to_string(l) + " experts do not match their format");
  if (nv) {
    for (const char* w : {"w13", "w2"}) {
      const std::vector<float> amax = host_f32(m, e + w + "_weight_amax");
      const std::vector<float> den = host_f32(m, e + w + "_weight_outer_scale_denominator");
      std::vector<float> f(EXPERTS);
      for (int i = 0; i < EXPERTS; ++i) f[i] = amax[i] / den[den.size() == 1 ? 0 : i];  // FP32 division
      (std::string(w) == "w13" ? ex.w13_factor : ex.w2_factor) = ld.upload_f32(f, {EXPERTS}, shard);
    }
  }

  ld.flush();
  if (dense_fp8) {
    layer.wproj = to_fp8(ld, wproj_bf16, R, H);
    layer.wo = to_fp8(ld, wo_bf16, H, HEADS * HEAD_DIM);
    CUDA_CHECK(cudaStreamSynchronize(ld.stream()));  // before the BF16 temporaries are freed
  } else {
    layer.wproj = Dense{wproj_bf16, {}};
    layer.wo = Dense{wo_bf16, {}};
  }
  Dense* views[NUM_SEGS] = {&layer.wq, &layer.wgate, &layer.wr, &layer.wsink,
                            &layer.wk, &layer.wv,    &layer.wk_dyn, &layer.wv_dyn};
  for (int s = 0; s < num_segs(layer.full); ++s) *views[s] = row_view(layer.wproj, SEG_BEGIN[s], SEG_BEGIN[s + 1]);
  return layer;
}

}  // namespace

Weights load_weights(const fs::path& model_dir, bool dense_fp8, cudaStream_t stream, const DeviceContext* second,
                     const ExpertResidency& residency) {
  validate_residency(residency, second != nullptr);
  int primary = 0;
  CUDA_CHECK(cudaGetDevice(&primary));
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Weights w;
  w.dense_fp8 = dense_fp8;
  w.residency = residency;
  w.expert_formats = expert_formats(model_dir);
  w.storage.reserve(4096);
  w.storage_hi.buffers.reserve(2 * LAYERS);
  Loader ld(m, up, stream, w.storage, second, &w.storage_hi);
  const int H = HIDDEN;
  for (int l = 0; l < LAYERS; ++l)
    w.layers.push_back(load_layer_into(ld, l, w.expert_formats[l], dense_fp8, residency.split[l]));

  // message_queries [12, 4, H] = cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1) with queries and
  // kv_queries [24, H] unflattened to [12, 2, H]
  {
    const std::vector<float> q = host_f32(m, "model.llm.middle_cycle_cross_pass_queries");
    const std::vector<float> kv = host_f32(m, "model.llm.middle_cycle_cross_pass_kv_queries");
    const int n = int(q.size() / H / 2);
    std::vector<float> mq(std::size_t(n) * 4 * H);
    for (int i = 0; i < n; ++i) {
      const float* rows[4] = {&q[std::size_t(2 * i) * H], &kv[std::size_t(2 * i) * H], &kv[std::size_t(2 * i + 1) * H],
                              &q[std::size_t(2 * i + 1) * H]};
      for (int j = 0; j < 4; ++j) std::copy(rows[j], rows[j] + H, mq.begin() + (std::ptrdiff_t(i) * 4 + j) * H);
    }
    w.message_queries = ld.upload_f32(mq, {n, 4, H});
  }
  w.embed = ld.raw("model.llm.embed.weight");
  if (w.embed.dtype != DType::BF16) throw std::runtime_error("weights: embed must be BF16");
  w.embed_norm = ld.f32("model.llm.embed_norm.weight", {H});
  w.final_query = ld.f32("model.llm.attn_res.output_query", {1, H});
  w.final_norm = ld.f32("model.llm.norm.weight", {H});
  DeviceBuffer unembed_tmp;
  {
    const std::string name = "model.llm.unembed.weight";
    if (m.at(name)->dtype != "BF16") throw std::runtime_error("weights: unembed must be BF16");
    const std::size_t bytes = std::size_t(UNPADDED_VOCAB) * H * 2;
    void* d = dense_fp8 ? (unembed_tmp = DeviceBuffer(bytes)).p : ld.alloc(bytes);
    ld.queue(name, d, UNPADDED_VOCAB);  // unembed[:UNPADDED_VOCAB]
    ld.flush();
    const Tensor bf = rm(d, DType::BF16, {UNPADDED_VOCAB, H});
    if (dense_fp8) {
      w.unembed = to_fp8(ld, bf, UNPADDED_VOCAB, H);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      w.unembed = Dense{bf, {}};
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  w.bytes_primary = ld.bytes(0);
  w.bytes_second = ld.bytes(1);
  // the park budget reads cudaMemGetInfo on whatever is current (generator.cpp)
  int current = 0;
  CUDA_CHECK(cudaGetDevice(&current));
  if (current != primary)
    throw std::logic_error("load_weights: current device " + std::to_string(current) + " is not the load's device " +
                           std::to_string(primary));
  return w;
}

Layer load_layer(const fs::path& model_dir, int l, bool dense_fp8, cudaStream_t stream,
                 std::vector<DeviceBuffer>& storage) {
  if (l < 0 || l >= LAYERS) throw std::invalid_argument("load_layer: layer out of range");
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Loader ld(m, up, stream, storage);
  Layer layer = load_layer_into(ld, l, expert_formats(model_dir)[l], dense_fp8, EXPERTS);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return layer;
}

// File layouts (FP32 or BF16 tensors):
//   attn_dirs, mlp_dirs [48, H]; attn_w, mlp_w [48]                                  plain projection
//   attn_dirs, mlp_dirs [48, H]; attn_weight, mlp_weight, attn_offset, mlp_offset [48]; gate [1];
//       optionally attn_det, mlp_det [48, H] and attn_threshold, mlp_threshold [48]
// The second form removes w * r * (r . y - offset), limited to components above the offset when gate is
// non-zero and, with detectors, to writes with det . y above the threshold. Its weights carry other names so
// that engines that only know the first form reject the file.
Ablation load_ablation(const fs::path& path) {
  Ablation a;
  const int H = HIDDEN;
  const std::size_t nd = std::size_t(NUM_VISITS) * H;
  // rows 0..47: directions, rows 48..95: detectors (zero without them)
  std::vector<float> attn_dirs(2 * nd, 0.f), mlp_dirs(2 * nd, 0.f);
  std::vector<float> attn_w(std::size_t(NUM_VISITS) * 4, 0.f), mlp_w(attn_w.size(), 0.f);
  if (!path.empty()) {
    const SafetensorsModel m(path);
    auto read = [&](const std::string& name, std::size_t n) {
      std::vector<float> v = host_f32(m, name);
      if (v.size() != n) throw std::runtime_error("ablation: " + name + " has the wrong size");
      return v;
    };
    // F.normalize(dirs, dim=-1): x / max(||x||, 1e-12) in FP32
    auto normalize = [&](std::vector<float> v) {
      for (int r = 0; r < NUM_VISITS; ++r) {
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += double(v[std::size_t(r) * H + c]) * v[std::size_t(r) * H + c];
        const float norm = std::max(float(std::sqrt(ss)), 1e-12f);
        for (int c = 0; c < H; ++c) v[std::size_t(r) * H + c] = v[std::size_t(r) * H + c] / norm;
      }
      return v;
    };
    const bool plain = m.contains("attn_w");
    a.detect = !plain && m.contains("attn_det");
    const float floor = !plain && read("gate", 1)[0] != 0.f ? 0.f : -std::numeric_limits<float>::max();
    auto load = [&](const std::string& kind, std::vector<float>& dirs, std::vector<float>& rows) {
      const std::vector<float> r = normalize(read(kind + "_dirs", nd));
      std::copy(r.begin(), r.end(), dirs.begin());
      const std::vector<float> w = read(kind + (plain ? "_w" : "_weight"), NUM_VISITS);
      const std::vector<float> c = plain ? std::vector<float>(NUM_VISITS, 0.f) : read(kind + "_offset", NUM_VISITS);
      std::vector<float> tau(NUM_VISITS, 0.f);
      if (a.detect) {
        const std::vector<float> d = read(kind + "_det", nd);  // as stored: the threshold is in its units
        std::copy(d.begin(), d.end(), dirs.begin() + std::ptrdiff_t(nd));
        tau = read(kind + "_threshold", NUM_VISITS);
      }
      for (int v = 0; v < NUM_VISITS; ++v)
        rows[4 * v] = w[v], rows[4 * v + 1] = c[v], rows[4 * v + 2] = floor, rows[4 * v + 3] = tau[v];
    };
    load("attn", attn_dirs, attn_w);
    load("mlp", mlp_dirs, mlp_w);
    a.active = true;
  }
  auto up = [&](const std::vector<float>& h, std::initializer_list<std::int32_t> shape) {
    a.storage.emplace_back(h.size() * 4);
    CUDA_CHECK(cudaMemcpy(a.storage.back().p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return rm(a.storage.back().p, DType::FP32, shape);
  };
  a.attn_dirs = up(attn_dirs, {2 * NUM_VISITS, H});
  a.attn_w = up(attn_w, {NUM_VISITS, 4});
  a.mlp_dirs = up(mlp_dirs, {2 * NUM_VISITS, H});
  a.mlp_w = up(mlp_w, {NUM_VISITS, 4});
  return a;
}

}  // namespace ingine::model
