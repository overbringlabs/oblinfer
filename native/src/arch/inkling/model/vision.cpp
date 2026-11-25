#include "model/vision.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/vision/vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace {

namespace vk = kernels::vision;

constexpr int H = vk::HIDDEN;

int round_up(int v, int to) { return (v + to - 1) / to * to; }

// Loads tensors of the checkpoint into device memory owned by `storage`: matrices as the BF16 they are
// stored in, straight from the file; vectors widened to FP32 on the host (exact).
class Loader {
 public:
  Loader(const SafetensorsModel& m, std::vector<DeviceBuffer>& storage, std::size_t& bytes)
      : m_(m), storage_(storage), bytes_(bytes) {}

  void* alloc(std::size_t bytes) {
    storage_.emplace_back(bytes);
    bytes_ += bytes;
    return storage_.back().p;
  }

  std::vector<float> host(const std::string& name, std::int64_t numel) const {
    const SafetensorsRef ref = m_.at(name);
    if (ref->dtype != "BF16" || ref->numel() != numel) {
      throw std::runtime_error("vision weights: " + name + " is not BF16 with " + std::to_string(numel) + " values");
    }
    const std::vector<std::byte> raw = m_.read(name);
    std::vector<float> out(static_cast<std::size_t>(numel));
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      const std::uint32_t w = std::uint32_t(b) << 16;
      std::memcpy(&out[i], &w, 4);
    }
    return out;
  }

  const float* upload(const std::vector<float>& values) {
    void* d = alloc(values.size() * 4);
    CUDA_CHECK(cudaMemcpy(d, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
    return static_cast<const float*>(d);
  }

  const float* f32(const std::string& name, std::int64_t numel) { return upload(host(name, numel)); }

  // Queues the BF16 matrix `name` ([rows, cols]) for upload to dst.
  void queue(const std::string& name, std::int64_t rows, std::int64_t cols, void* dst) {
    const SafetensorsRef ref = m_.at(name);
    if (ref->dtype != "BF16" || ref->shape != std::vector<std::int64_t>{rows, cols}) {
      throw std::runtime_error("vision weights: " + name + " is not BF16 [" + std::to_string(rows) + ", " +
                               std::to_string(cols) + "]");
    }
    pending_.push_back(upload_request(ref, dst));
  }

  const __nv_bfloat16* matrix(const std::string& name, std::int64_t rows, std::int64_t cols) {
    void* d = alloc(std::size_t(rows) * cols * 2);
    queue(name, rows, cols, d);
    return static_cast<const __nv_bfloat16*>(d);
  }

  void flush(cudaStream_t stream) {
    DeviceUploader uploader(UploadOptions{std::size_t(16) << 20, 4, 4});
    uploader.upload(pending_, stream);
    pending_.clear();
  }

 private:
  const SafetensorsModel& m_;
  std::vector<DeviceBuffer>& storage_;
  std::size_t& bytes_;
  std::vector<UploadRequest> pending_;
};

}  // namespace

VisionWeights load_vision_weights(const std::filesystem::path& model_dir, cudaStream_t stream) {
  const SafetensorsModel m(model_dir);
  if (!m.contains("model.visual.proj_out.weight")) {
    throw std::runtime_error("the checkpoint in " + model_dir.string() + " has no vision tower (model.visual.*)");
  }
  VisionWeights w;
  Loader ld(m, w.storage, w.bytes);
  const std::string emb = "model.visual.backbone.embeddings.";
  w.patch_w = ld.matrix(emb + "patch_embedding.weight", H, vk::PATCH_DIM);
  w.pos = ld.f32(emb + "position_embedding", std::int64_t(vk::POS_GRID) * vk::POS_GRID * H);
  std::vector<float> cls = ld.host(emb + "class_embedding", H);
  const std::vector<float> cls_pos = ld.host(emb + "cls_position_embedding", H);
  for (int c = 0; c < H; ++c) cls[std::size_t(c)] += cls_pos[std::size_t(c)];
  w.cls = ld.upload(cls);
  w.ln_pre_w = ld.f32("model.visual.backbone.ln_pre.weight", H);
  w.ln_pre_b = ld.f32("model.visual.backbone.ln_pre.bias", H);
  for (int i = 0; i < VISION_LAYERS; ++i) {
    const std::string p = "model.visual.backbone.layers." + std::to_string(i) + ".";
    VisionLayer& l = w.layers[std::size_t(i)];
    l.ln1_w = ld.f32(p + "ln_1.weight", H), l.ln1_b = ld.f32(p + "ln_1.bias", H);
    l.ln2_w = ld.f32(p + "ln_2.weight", H), l.ln2_b = ld.f32(p + "ln_2.bias", H);
    l.ls1 = ld.f32(p + "ls_1.gamma", H), l.ls2 = ld.f32(p + "ls_2.gamma", H);
    // q, k and v stacked: one projection of 3072 outputs
    auto* wqkv = static_cast<__nv_bfloat16*>(ld.alloc(std::size_t(3) * H * H * 2));
    std::vector<float> bqkv;
    int part = 0;
    for (const char* name : {"q_proj", "k_proj", "v_proj"}) {
      ld.queue(p + "self_attn." + name + ".weight", H, H, wqkv + std::size_t(part++) * H * H);
      const std::vector<float> b = ld.host(p + "self_attn." + name + ".bias", H);
      bqkv.insert(bqkv.end(), b.begin(), b.end());
    }
    l.wqkv = wqkv;
    l.bqkv = ld.upload(bqkv);
    l.wo = ld.matrix(p + "self_attn.out_proj.weight", H, H);
    l.bo = ld.f32(p + "self_attn.out_proj.bias", H);
    l.fc = ld.matrix(p + "mlp.c_fc.weight", vk::MLP_DIM, H);
    l.bfc = ld.f32(p + "mlp.c_fc.bias", vk::MLP_DIM);
    l.proj = ld.matrix(p + "mlp.c_proj.weight", H, vk::MLP_DIM);
    l.bproj = ld.f32(p + "mlp.c_proj.bias", H);
  }
  w.proj_in = ld.matrix("model.visual.proj_in.weight", vk::MERGED, vk::MERGED);
  w.proj_out = ld.matrix("model.visual.proj_out.weight", VISION_OUT, vk::MERGED);
  std::vector<float> inv_freq(vk::ROPE_FREQS);  // 1 / 10000^(2i / 32), in FP32 as the reference computes it
  for (int i = 0; i < vk::ROPE_FREQS; ++i) {
    inv_freq[std::size_t(i)] = 1.0f / std::pow(10000.0f, float(2 * i) / float(2 * vk::ROPE_FREQS));
  }
  w.inv_freq = ld.upload(inv_freq);
  ld.flush(stream);
  return w;
}

VisionEncoder::VisionEncoder(const std::filesystem::path& model_dir, cudaStream_t stream)
    : stream_(stream), weights_(load_vision_weights(model_dir, stream)) {}

namespace {

struct Layout {
  int N, S, Sp, Np, Qc;
  std::size_t pixels, h, xb, big32, big16, attn, y, scores, probs, total;
};

Layout layout(int height, int width, std::size_t score_bytes) {
  Layout L{};
  const int rows = (height + vk::TILE - 1) / vk::TILE, cols = (width + vk::TILE - 1) / vk::TILE;
  L.N = rows * cols;
  L.S = 4 * L.N + 1;
  L.Sp = round_up(L.S, 64);
  L.Np = VisionEncoder::padded_tokens(L.N);
  // queries per attention GEMM: each holds Sp scores (FP32) and Sp probabilities (BF16)
  L.Qc = int(std::clamp<std::size_t>(score_bytes / (std::size_t(L.Sp) * 6) / 64 * 64, 64, std::size_t(L.Sp)));
  std::size_t at = 0;
  auto take = [&](std::size_t bytes) {
    const std::size_t off = at;
    at += (bytes + 255) / 256 * 256;
    return off;
  };
  const std::size_t Sp = std::size_t(L.Sp);
  L.pixels = take(std::size_t(height) * width * 3);
  L.h = take(Sp * H * 4);                 // residual stream f32 [Sp, 1024]
  L.xb = take(Sp * H * 2);                // LayerNorm output bf16 [Sp, 1024]
  L.big32 = take(Sp * vk::MLP_DIM * 4);   // f32 [Sp, 4096]: fused q|k|v, MLP hidden, merger hidden
  L.big16 = take(Sp * vk::MLP_DIM * 2);   // bf16 [Sp, 4096]: patches, q/k/vt, GELU outputs, merger rows
  L.attn = take(Sp * H * 2);              // attention output bf16 [Sp, 1024]
  L.y = take(Sp * H * 4);                 // projection outputs f32 [Sp, 1024]
  L.scores = take(std::size_t(L.Qc) * Sp * 4);
  L.probs = take(std::size_t(L.Qc) * Sp * 2);
  L.total = at;
  return L;
}

}  // namespace

std::size_t VisionEncoder::workspace_bytes(int height, int width) const {
  return layout(height, width, score_bytes_).total;
}

DeviceBuffer VisionEncoder::encode(const std::uint8_t* rgb, int height, int width) {
  if (!rgb || height < 1 || width < 1) throw std::invalid_argument("VisionEncoder::encode: empty image");
  const int rows = (height + vk::TILE - 1) / vk::TILE, cols = (width + vk::TILE - 1) / vk::TILE;
  const Layout L = layout(height, width, score_bytes_);
  const int Sp = L.Sp;
  const cudaStream_t s = stream_;
  const VisionWeights& w = weights_;

  DeviceBuffer out(std::size_t(L.Np) * VISION_OUT * 2);
  DeviceBuffer work(L.total);
  auto at = [&](std::size_t off) { return static_cast<std::byte*>(work.p) + off; };
  auto* pixels = reinterpret_cast<std::uint8_t*>(at(L.pixels));
  auto* h = reinterpret_cast<float*>(at(L.h));
  auto* xb = reinterpret_cast<__nv_bfloat16*>(at(L.xb));
  auto* big32 = reinterpret_cast<float*>(at(L.big32));
  auto* big16 = reinterpret_cast<__nv_bfloat16*>(at(L.big16));
  auto* attn = reinterpret_cast<__nv_bfloat16*>(at(L.attn));
  auto* y = reinterpret_cast<float*>(at(L.y));
  auto* scores = reinterpret_cast<float*>(at(L.scores));
  auto* probs = reinterpret_cast<__nv_bfloat16*>(at(L.probs));

  // patches -> patch embedding + position embeddings, class token in front, ln_pre
  CUDA_CHECK(cudaMemcpyAsync(pixels, rgb, std::size_t(height) * width * 3, cudaMemcpyHostToDevice, s));
  vk::patchify(pixels, height, width, rows, cols, big16, Sp, s);
  gemm_.run(big16, vk::PATCH_DIM, w.patch_w, vk::PATCH_DIM, y, H, true, Sp, H, vk::PATCH_DIM, s);
  vk::embed_assemble(y, w.pos, w.cls, rows, cols, h, Sp, s);
  if (observer) observer(-1, h, L.S);
  vk::layernorm(h, w.ln_pre_w, w.ln_pre_b, Sp, h, true, s);

  __nv_bfloat16* q = big16;                                         // [16, Sp, 64]
  __nv_bfloat16* k = q + std::size_t(vk::HEADS) * Sp * vk::HEAD_DIM;  // [16, Sp, 64]
  __nv_bfloat16* vt = k + std::size_t(vk::HEADS) * Sp * vk::HEAD_DIM; // [16, 64, Sp]
  const float scale = 1.0f / std::sqrt(float(vk::HEAD_DIM));
  for (int i = 0; i < VISION_LAYERS; ++i) {
    const VisionLayer& l = w.layers[std::size_t(i)];
    // h += ls1 * out_proj(attention(ln1(h)))
    vk::layernorm(h, l.ln1_w, l.ln1_b, Sp, xb, false, s);
    gemm_.run(xb, H, l.wqkv, H, big32, 3 * H, true, Sp, 3 * H, H, s);
    vk::qkv_post(big32, l.bqkv, w.inv_freq, rows, cols, Sp, q, k, vt, s);
    for (int head = 0; head < vk::HEADS; ++head) {
      const __nv_bfloat16* qh = q + std::size_t(head) * Sp * vk::HEAD_DIM;
      const __nv_bfloat16* kh = k + std::size_t(head) * Sp * vk::HEAD_DIM;
      const __nv_bfloat16* vth = vt + std::size_t(head) * vk::HEAD_DIM * Sp;
      for (int q0 = 0; q0 < Sp; q0 += L.Qc) {
        const int Q = std::min(L.Qc, Sp - q0);
        gemm_.run(qh + std::size_t(q0) * vk::HEAD_DIM, vk::HEAD_DIM, kh, vk::HEAD_DIM, scores, Sp, true, Q, Sp,
                  vk::HEAD_DIM, s);
        vk::softmax_rows(scores, Q, Sp, L.S, scale, probs, s);
        gemm_.run(probs, Sp, vth, Sp, attn + std::size_t(q0) * H + head * vk::HEAD_DIM, H, false, Q, vk::HEAD_DIM, Sp,
                  s);
      }
    }
    gemm_.run(attn, H, l.wo, H, y, H, true, Sp, H, H, s);
    vk::bias_scale_residual(y, l.bo, l.ls1, Sp, h, s);
    // h += ls2 * c_proj(gelu(c_fc(ln2(h))))
    vk::layernorm(h, l.ln2_w, l.ln2_b, Sp, xb, false, s);
    gemm_.run(xb, H, l.fc, H, big32, vk::MLP_DIM, true, Sp, vk::MLP_DIM, H, s);
    vk::bias_act(big32, l.bfc, Sp, vk::MLP_DIM, true, big16, s);
    gemm_.run(big16, vk::MLP_DIM, l.proj, vk::MLP_DIM, y, H, true, Sp, H, vk::MLP_DIM, s);
    vk::bias_scale_residual(y, l.bproj, l.ls2, Sp, h, s);
    if (observer) observer(i, h, L.S);
  }

  // merger: a tile's four patch states side by side -> proj_in -> GELU -> proj_out
  vk::merge_rows(h, L.N, L.Np, big16, s);
  gemm_.run(big16, vk::MERGED, w.proj_in, vk::MERGED, big32, vk::MERGED, true, L.Np, vk::MERGED, vk::MERGED, s);
  vk::bias_act(big32, nullptr, L.Np, vk::MERGED, true, big16, s);
  gemm_.run(big16, vk::MERGED, w.proj_out, vk::MERGED, out.p, VISION_OUT, false, L.Np, VISION_OUT, vk::MERGED, s);
  CUDA_CHECK(cudaStreamSynchronize(s));  // the workspace is released on return
  return out;
}

}  // namespace ingine::model
