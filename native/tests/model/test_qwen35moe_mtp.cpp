// MTP draft head (src/arch/qwen35moe/model/mtp.{h,cu}) on the real checkpoint, in five parts:
//   1. the load: all 19 tensors' shapes against the safetensors header, three 64-byte samples per tensor against
//      the file bytes, and the refusal of a directory that does not hold the head;
//   2. the whole block at seven positions against an FP64 host reference written here from the checkpoint's own
//      bytes (no device kernel takes part in it), so the drafter cache accumulates;
//   3. the Q4_0 cache: the store against the documented quantiser byte for byte, the split/combine kernels
//      against an FP64 attention decoded from the cache bytes the kernel was handed, and what the format costs
//      against BF16 over the same k/v stream;
//   4. the module on all three cache types, on the same seven positions;
//   5. the cost: milliseconds per drafted token at 1k / 32k / 256k positions split into its five stages, the
//      bytes it reads, and what the module allocates at 262144;
//   6. the batched chain: a row of draft_chain() against the serial draft of the same token, hidden and position
//      (tokens, h' and logits), and the cost of a 1/3/7-draft chain, serial against one batched head, at 1k and
//      262k positions.
// Skipped unless QWEN35MOE_SNAPSHOT is set.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/mtp/mtp_attn.h"
#include "arch/qwen35moe/model/mtp.h"
#include "core/safetensors.h"
#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ingine::test;

namespace qwen = ingine::arch::qwen35moe;
namespace attn = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;

namespace {

constexpr int HIDDEN = qwen::HIDDEN;
constexpr int QDIM = qwen::Q_DIM;
constexpr int KVDIM = qwen::KV_DIM;
constexpr int VOCAB = qwen::VOCAB;
constexpr int HEADS = qwen::HEADS;
constexpr int KVH = qwen::KV_HEADS;
constexpr int DIM = qwen::HEAD_DIM;
constexpr int INTER = qwen::MOE_INTER;
constexpr int SHARED = qwen::SHARED_INTER;
constexpr int TOPK = qwen::TOP_K;
constexpr int EXPERTS = qwen::EXPERTS;
constexpr int ROT = qwen::ROTARY_DIM;

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  return p.major == 12 && p.minor == 0;
}

void fail(const std::string& what) { std::printf("FAIL %s\n", what.c_str()); }

// A device row of HIDDEN BF16 values from a fixed seed; the reference is given the same rounded values.
std::vector<float> synthetic_hidden(std::uint64_t seed) {
  Rng rng(seed);
  std::vector<float> h(HIDDEN);
  for (float& v : h) v = round_bf16(float(1.5 * rng.normal()));
  return h;
}

// ------------------------------------------------------------------------------------------------- the reference

// Row-sequential reader over one safetensors tensor: one pread per 8 MB chunk, so the FP64 reference walks the
// checkpoint's 1.6 GB of weights without ever holding them.
class Rows {
 public:
  // `row_bytes` is the stored bytes of one row, so a tensor's row count times this must be its byte size.
  Rows(const ingine::SafetensorsModel& model, const char* name, int rows, int row_bytes)
      : rows_(rows), row_bytes_(std::size_t(row_bytes)) {
    const ingine::SafetensorsRef ref = model.at(name);
    file_ = ref.file;
    base_ = ref->offset;
    if (std::uint64_t(ref->bytes) != std::uint64_t(rows) * row_bytes_) {
      throw std::runtime_error(std::string("row view mismatch for ") + name);
    }
  }
  const std::byte* row(int r) const {
    if (r < chunk_first_ || r >= chunk_first_ + chunk_count_) {
      const int max_rows = std::max(1, int((std::size_t(8) << 20) / row_bytes_));
      chunk_first_ = std::max(0, std::min(r, rows_ - max_rows));
      chunk_count_ = std::min(max_rows, rows_ - chunk_first_);
      chunk_.resize(std::size_t(chunk_count_) * row_bytes_);
      file_->read_range(base_ + std::uint64_t(chunk_first_) * row_bytes_, chunk_.size(), chunk_.data());
    }
    return chunk_.data() + std::size_t(r - chunk_first_) * row_bytes_;
  }
  int rows() const { return rows_; }

 private:
  const ingine::SafetensorsFile* file_ = nullptr;
  std::uint64_t base_ = 0;
  int rows_ = 0;
  std::size_t row_bytes_ = 0;
  mutable std::vector<std::byte> chunk_;
  mutable int chunk_first_ = -1;
  mutable int chunk_count_ = 0;
};

std::vector<double> read_bf16_vector(const ingine::SafetensorsModel& model, const char* name, int n) {
  const std::vector<std::byte> bytes = model.read(name);
  std::vector<__nv_bfloat16> bf(static_cast<std::size_t>(n), __nv_bfloat16(0));
  std::memcpy(bf.data(), bytes.data(), bytes.size());
  std::vector<double> out(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) out[std::size_t(i)] = bf16_to_float(bf[std::size_t(i)]);
  return out;
}

// y[r] = sum_c W[r][c] * x[c], W read row by row from the checkpoint in FP64.
void ref_gemv(const Rows& w, const std::vector<double>& x, std::vector<double>* y) {
  y->assign(std::size_t(w.rows()), 0.0);
  const int cols = int(x.size());
  for (int r = 0; r < w.rows(); ++r) {
    const auto* row = reinterpret_cast<const __nv_bfloat16*>(w.row(r));
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) acc += double(bf16_to_float(row[c])) * x[std::size_t(c)];
    (*y)[std::size_t(r)] = acc;
  }
}

// Qwen3NextRMSNorm: x / rms(x) * weight, with the checkpoint's (1 + w) weight already applied.
void ref_norm(const std::vector<double>& x, const std::vector<double>& weight, std::vector<double>* y) {
  double sum = 0.0;
  for (double v : x) sum += v * v;
  const double inv = 1.0 / std::sqrt(sum / double(x.size()) + double(qwen::EPS));
  y->resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) (*y)[i] = x[i] * inv * weight[i];
}

double ref_silu(double v) { return v / (1.0 + std::exp(-v)); }
double as_bf16(double v) { return double(round_bf16(float(v))); }

// The whole MTP block in FP64 over the checkpoint's bytes. Its attention keeps the K/V history itself and rounds
// it to BF16 exactly where the device's BF16 cache does, so the two are comparable to FP32-vs-FP64 width.
class Reference {
 public:
  explicit Reference(const std::filesystem::path& snapshot)
      : model_(snapshot),
        fc_(model_, "mtp.fc.weight", HIDDEN, 4 * HIDDEN),  // [HIDDEN, 2 * HIDDEN] BF16
        q_(model_, "mtp.layers.0.self_attn.q_proj.weight", qwen::Q_PROJ_ROWS, 2 * HIDDEN),
        k_(model_, "mtp.layers.0.self_attn.k_proj.weight", KVDIM, 2 * HIDDEN),
        v_(model_, "mtp.layers.0.self_attn.v_proj.weight", KVDIM, 2 * HIDDEN),
        o_(model_, "mtp.layers.0.self_attn.o_proj.weight", HIDDEN, 2 * QDIM),
        router_(model_, "mtp.layers.0.mlp.gate.weight", EXPERTS, 2 * HIDDEN),
        gate_up_(model_, "mtp.layers.0.mlp.experts.gate_up_proj", EXPERTS * 2 * INTER, 2 * HIDDEN),
        down_(model_, "mtp.layers.0.mlp.experts.down_proj", EXPERTS * HIDDEN, 2 * INTER),
        sh_gate_(model_, "mtp.layers.0.mlp.shared_expert.gate_proj.weight", SHARED, 2 * HIDDEN),
        sh_up_(model_, "mtp.layers.0.mlp.shared_expert.up_proj.weight", SHARED, 2 * HIDDEN),
        sh_down_(model_, "mtp.layers.0.mlp.shared_expert.down_proj.weight", HIDDEN, 2 * SHARED),
        sh_scalar_(model_, "mtp.layers.0.mlp.shared_expert_gate.weight", 1, 2 * HIDDEN),
        head_w_(model_, "lm_head.weight", VOCAB, HIDDEN / 2),  // U8 e2m1 pairs
        head_s_(model_, "lm_head.weight_scale", VOCAB, HIDDEN / 16),
        embed_(model_, "model.language_model.embed_tokens.weight", VOCAB, 2 * HIDDEN) {
    auto plus_one = [&](const char* name, int n) {
      std::vector<double> w = read_bf16_vector(model_, name, n);
      for (double& v : w) v += 1.0;
      return w;
    };
    n_embed_ = plus_one("mtp.pre_fc_norm_embedding.weight", HIDDEN);
    n_hidden_ = plus_one("mtp.pre_fc_norm_hidden.weight", HIDDEN);
    n_input_ = plus_one("mtp.layers.0.input_layernorm.weight", HIDDEN);
    n_post_ = plus_one("mtp.layers.0.post_attention_layernorm.weight", HIDDEN);
    n_q_ = plus_one("mtp.layers.0.self_attn.q_norm.weight", DIM);
    n_k_ = plus_one("mtp.layers.0.self_attn.k_norm.weight", DIM);
    n_final_ = plus_one("mtp.norm.weight", HIDDEN);
    head_scale_ = scalar("lm_head.weight_scale_2");
    ck_.assign(std::size_t(kMaxPos) * KVDIM, 0.0);
    cv_.assign(std::size_t(kMaxPos) * KVDIM, 0.0);
  }

  // One step at `pos`; `hidden` is the BF16 row the device was given, as floats.
  void step(int token, const std::vector<float>& hidden, int pos, std::vector<double>* h_out,
            std::vector<double>* logits) {
    if (pos >= kMaxPos) throw std::runtime_error("reference position beyond its cache");

    std::vector<double> e(HIDDEN), tmp, x, h;
    const auto* erow = reinterpret_cast<const __nv_bfloat16*>(embed_.row(token));
    for (int i = 0; i < HIDDEN; ++i) e[std::size_t(i)] = double(bf16_to_float(erow[i]));
    ref_norm(e, n_embed_, &tmp);
    std::vector<double> hn(HIDDEN), hidden_norm;
    for (int i = 0; i < HIDDEN; ++i) hn[std::size_t(i)] = double(hidden[std::size_t(i)]);
    ref_norm(hn, n_hidden_, &hidden_norm);

    std::vector<double> cat = tmp;
    cat.insert(cat.end(), hidden_norm.begin(), hidden_norm.end());
    ref_gemv(fc_, cat, &x);  // embedding half first, then the hidden half

    const std::vector<double> sa = x;
    ref_norm(x, n_input_, &h);
    std::vector<double> qgate, kproj, vproj;
    ref_gemv(q_, h, &qgate);
    ref_gemv(k_, h, &kproj);
    ref_gemv(v_, h, &vproj);

    // Per head [query | gate], then q_norm/k_norm and the partial rotary over the leading 64 dims.
    std::vector<double> q(QDIM), gate(QDIM);
    for (int head = 0; head < HEADS; ++head) {
      for (int d = 0; d < DIM; ++d) {
        q[std::size_t(head * DIM + d)] = qgate[std::size_t(head * 2 * DIM + d)];
        gate[std::size_t(head * DIM + d)] = qgate[std::size_t(head * 2 * DIM + d + DIM)];
      }
    }
    auto head_norm_rope = [&](std::vector<double>* vec, int heads, const std::vector<double>& weight) {
      for (int head = 0; head < heads; ++head) {
        double* base = vec->data() + std::size_t(head) * DIM;
        double sum = 0.0;
        for (int d = 0; d < DIM; ++d) sum += base[d] * base[d];
        const double inv = 1.0 / std::sqrt(sum / double(DIM) + double(qwen::EPS));
        for (int d = 0; d < DIM; ++d) base[d] = base[d] * inv * weight[std::size_t(d)];
        for (int i = 0; i < ROT / 2; ++i) {
          const double angle = double(pos) * std::pow(double(qwen::ROPE_THETA), -2.0 * double(i) / double(ROT));
          const double c = std::cos(angle), s = std::sin(angle);
          const double x0 = base[i], x1 = base[i + ROT / 2];
          base[i] = x0 * c - x1 * s;
          base[i + ROT / 2] = x0 * s + x1 * c;
        }
      }
    };
    head_norm_rope(&q, HEADS, n_q_);
    head_norm_rope(&kproj, KVH, n_k_);

    // The cache row of this position, rounded where the device's BF16 cache rounds it.
    for (int head = 0; head < KVH; ++head) {
      for (int d = 0; d < DIM; ++d) {
        const std::size_t at = std::size_t(pos) * KVDIM + std::size_t(head) * DIM + std::size_t(d);
        ck_[at] = as_bf16(kproj[std::size_t(head * DIM + d)]);
        cv_[at] = as_bf16(vproj[std::size_t(head * DIM + d)]);
      }
    }

    const double scale = 1.0 / std::sqrt(double(DIM));
    std::vector<double> attn(QDIM, 0.0), score(std::size_t(pos) + 1);
    for (int head = 0; head < HEADS; ++head) {
      const int kh = head / (HEADS / KVH);
      double m = -std::numeric_limits<double>::infinity();
      for (int i = 0; i <= pos; ++i) {
        double dot = 0.0;
        for (int d = 0; d < DIM; ++d) {
          dot += q[std::size_t(head * DIM + d)] * ck_[std::size_t(i) * KVDIM + std::size_t(kh) * DIM + std::size_t(d)];
        }
        score[std::size_t(i)] = dot * scale;
        m = std::max(m, score[std::size_t(i)]);
      }
      double sum = 0.0;
      for (int i = 0; i <= pos; ++i) {
        score[std::size_t(i)] = std::exp(score[std::size_t(i)] - m);
        sum += score[std::size_t(i)];
      }
      for (int d = 0; d < DIM; ++d) {
        double acc = 0.0;
        for (int i = 0; i <= pos; ++i) {
          acc += score[std::size_t(i)] * cv_[std::size_t(i) * KVDIM + std::size_t(kh) * DIM + std::size_t(d)];
        }
        const double g = gate[std::size_t(head * DIM + d)];
        attn[std::size_t(head * DIM + d)] = acc / sum / (1.0 + std::exp(-g));
      }
    }
    std::vector<double> proj;
    ref_gemv(o_, attn, &proj);
    for (int i = 0; i < HIDDEN; ++i) x[std::size_t(i)] = sa[std::size_t(i)] + proj[std::size_t(i)];
    std::vector<double> moe_in;
    ref_norm(x, n_post_, &moe_in);

    // Router: 256 logits, top-8 by logit (ties to the lower id), softmax over the eight.
    std::vector<double> logits256;
    ref_gemv(router_, moe_in, &logits256);
    std::vector<int> order(EXPERTS);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + TOPK, order.end(), [&](int a, int b) {
      const double la = logits256[std::size_t(a)], lb = logits256[std::size_t(b)];
      if (la != lb) return la > lb;
      return a < b;
    });
    ids_.assign(order.begin(), order.begin() + TOPK);
    id_gap_ = logits256[std::size_t(ids_.back())] - logits256[std::size_t(order[TOPK])];
    std::vector<double> wsel(TOPK);
    double maxl = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < TOPK; ++k) maxl = std::max(maxl, logits256[std::size_t(ids_[std::size_t(k)])]);
    double wsum = 0.0;
    for (int k = 0; k < TOPK; ++k) {
      wsel[std::size_t(k)] = std::exp(logits256[std::size_t(ids_[std::size_t(k)])] - maxl);
      wsum += wsel[std::size_t(k)];
    }
    for (double& v : wsel) v /= wsum;

    // Routed experts: gate rows 0..511, up rows 512..1023 of the packed tensor, then down.
    std::vector<double> expert_out(HIDDEN, 0.0);
    for (int k = 0; k < TOPK; ++k) {
      const int expert = ids_[std::size_t(k)];
      const std::vector<double> hk = expert_hidden(expert, moe_in);
      for (int r = 0; r < HIDDEN; ++r) {
        const auto* row = reinterpret_cast<const __nv_bfloat16*>(down_.row(expert * HIDDEN + r));
        double acc = 0.0;
        for (int c = 0; c < INTER; ++c) acc += double(bf16_to_float(row[c])) * hk[std::size_t(c)];
        expert_out[std::size_t(r)] += wsel[std::size_t(k)] * acc;
      }
    }

    // The shared expert with its sigmoid gate.
    std::vector<double> sh_g, sh_u, sh_h, sh_out;
    ref_gemv(sh_gate_, moe_in, &sh_g);
    ref_gemv(sh_up_, moe_in, &sh_u);
    sh_h.resize(SHARED);
    for (int i = 0; i < SHARED; ++i) sh_h[std::size_t(i)] = ref_silu(sh_g[std::size_t(i)]) * sh_u[std::size_t(i)];
    ref_gemv(sh_down_, sh_h, &sh_out);
    const auto* gate_row = reinterpret_cast<const __nv_bfloat16*>(sh_scalar_.row(0));
    double gate_dot = 0.0;
    for (int i = 0; i < HIDDEN; ++i) gate_dot += double(bf16_to_float(gate_row[i])) * moe_in[std::size_t(i)];
    const double shared_gate = 1.0 / (1.0 + std::exp(-gate_dot));
    for (int i = 0; i < HIDDEN; ++i) {
      x[std::size_t(i)] += expert_out[std::size_t(i)] + shared_gate * sh_out[std::size_t(i)];
    }

    ref_norm(x, n_final_, h_out);

    // The trunk's NVFP4 head in FP64: e2m1 pairs, one e4m3 block scale per 16 columns, one tensor scale.
    logits->assign(VOCAB, 0.0);
    for (int r = 0; r < VOCAB; ++r) {
      const auto* wrow = reinterpret_cast<const std::uint8_t*>(head_w_.row(r));
      const auto* srow = reinterpret_cast<const std::uint8_t*>(head_s_.row(r));
      double acc = 0.0;
      for (int block = 0; block < HIDDEN / 16; ++block) {
        const double s = double(quant::fp8_e4m3_to_float(srow[block]));
        for (int j = 0; j < 16; ++j) {
          const int c = block * 16 + j;
          const std::uint8_t byte = wrow[c / 2];
          const int code = (c & 1) ? (byte >> 4) : (byte & 0xF);
          acc += double(quant::e2m1_to_float(std::uint8_t(code))) * s * (*h_out)[std::size_t(c)];
        }
      }
      (*logits)[std::size_t(r)] = acc * head_scale_;
    }
  }

  const std::vector<int>& ids() const { return ids_; }
  double id_gap() const { return id_gap_; }
  static int argmax(const std::vector<double>& v) {
    int best = 0;
    for (int i = 1; i < int(v.size()); ++i) {
      if (v[std::size_t(i)] > v[std::size_t(best)]) best = i;
    }
    return best;
  }

 private:
  static constexpr int kMaxPos = 4096;

  // gate rows 0..INTER-1 then up rows INTER..2*INTER-1 of one expert.
  std::vector<double> expert_hidden(int expert, const std::vector<double>& x) const {
    std::vector<double> h(static_cast<std::size_t>(INTER), 0.0);
    for (int j = 0; j < INTER; ++j) {
      const auto* g = reinterpret_cast<const __nv_bfloat16*>(gate_up_.row(expert * 2 * INTER + j));
      const auto* u = reinterpret_cast<const __nv_bfloat16*>(gate_up_.row(expert * 2 * INTER + INTER + j));
      double ag = 0.0, au = 0.0;
      for (int c = 0; c < HIDDEN; ++c) {
        ag += double(bf16_to_float(g[c])) * x[std::size_t(c)];
        au += double(bf16_to_float(u[c])) * x[std::size_t(c)];
      }
      h[std::size_t(j)] = ref_silu(ag) * au;
    }
    return h;
  }

  float scalar(const char* name) const {
    const ingine::SafetensorsRef ref = model_.at(name);
    std::vector<std::byte> four(4);
    ref.file->read_range(ref->offset, 4, four.data());
    float v = 1.0f;
    std::memcpy(&v, four.data(), 4);
    return v;
  }

  ingine::SafetensorsModel model_;
  Rows fc_, q_, k_, v_, o_, router_, gate_up_, down_, sh_gate_, sh_up_, sh_down_, sh_scalar_, head_w_, head_s_,
      embed_;
  std::vector<double> n_embed_, n_hidden_, n_input_, n_post_, n_q_, n_k_, n_final_;
  std::vector<double> ck_, cv_;
  double head_scale_ = 1.0;
  std::vector<int> ids_;
  double id_gap_ = 0.0;
};

// --------------------------------------------------------------------------------------- the borrowed trunk

struct Trunk {
  void* embed = nullptr;
  void* head_w = nullptr;
  void* head_s = nullptr;
  qwen::MtpTrunkHead view;
};

bool upload_trunk(const std::filesystem::path& snapshot, Trunk* trunk) {
  try {
    ingine::SafetensorsModel model(snapshot);
    struct Item {
      const char* name;
      void** dst;
    };
    const Item items[] = {{"model.language_model.embed_tokens.weight", &trunk->embed},
                          {"lm_head.weight", &trunk->head_w},
                          {"lm_head.weight_scale", &trunk->head_s}};
    std::vector<ingine::UploadRequest> requests;
    for (const Item& item : items) {
      const ingine::SafetensorsRef ref = model.at(item.name);
      INGINE_TEST_CUDA(cudaMalloc(item.dst, std::size_t(ref->bytes)));
      requests.push_back(ingine::upload_request(ref, *item.dst));
    }
    ingine::DeviceUploader uploader({}, 0);
    uploader.upload(requests, nullptr);
    const ingine::SafetensorsRef scale = model.at("lm_head.weight_scale_2");
    std::vector<std::byte> four(4);
    scale.file->read_range(scale->offset, 4, four.data());
    float s = 1.0f;
    std::memcpy(&s, four.data(), 4);
    trunk->view.embed_tokens = static_cast<const __nv_bfloat16*>(trunk->embed);
    trunk->view.head_w = static_cast<const std::uint8_t*>(trunk->head_w);
    trunk->view.head_scales = static_cast<const std::uint8_t*>(trunk->head_s);
    trunk->view.head_scale = s;
    return true;
  } catch (const std::exception& e) {
    fail(std::string("trunk upload: ") + e.what());
    return false;
  }
}

// ------------------------------------------------------------------------------------------ 1. the load check

bool check_tensors(const qwen::MtpDrafter& drafter, const ingine::SafetensorsModel& model) {
  bool ok = true;
  std::uint64_t bytes = 0;
  std::vector<std::byte> file_bytes(64), dev_bytes(64);
  const auto* base = static_cast<const std::byte*>(drafter.weights());
  for (const qwen::MtpTensors& t : drafter.tensors()) {
    const ingine::SafetensorsRef ref = model.find(t.name);
    if (!ref) {
      fail(t.name + ": not in the checkpoint");
      ok = false;
      continue;
    }
    bytes += ref->bytes;
    if (ref->shape != t.shape || ref->bytes != t.bytes || ref->dtype != "BF16") {
      std::printf("FAIL %s: header %s of %llu bytes does not match the module's %llu\n", t.name.c_str(),
                  ref->dtype.c_str(), static_cast<unsigned long long>(ref->bytes),
                  static_cast<unsigned long long>(t.bytes));
      ok = false;
      continue;
    }
    const std::uint64_t samples[] = {0, (ref->bytes / 2) & ~std::uint64_t(63),
                                     (ref->bytes - 64) & ~std::uint64_t(63)};
    for (std::uint64_t s : samples) {
      ref.file->read_range(ref->offset + s, 64, file_bytes.data());
      INGINE_TEST_CUDA(cudaMemcpy(dev_bytes.data(), base + t.offset + s, 64, cudaMemcpyDeviceToHost));
      if (std::memcmp(file_bytes.data(), dev_bytes.data(), 64) != 0) {
        std::printf("FAIL %s: the 64 bytes at %llu differ from the file\n", t.name.c_str(),
                    static_cast<unsigned long long>(s));
        ok = false;
        break;
      }
    }
  }
  if (drafter.tensors().size() != 19) {
    std::printf("FAIL %zu tensors loaded, the head is 19\n", drafter.tensors().size());
    ok = false;
  }
  if (drafter.weight_bytes() != bytes) {
    std::printf("FAIL the weight block is %llu bytes for 1689281536 of tensors: padding where 256-alignment of "
                "every tensor leaves none\n",
                static_cast<unsigned long long>(drafter.weight_bytes()));
    ok = false;
  }
  if (bytes != 1689281536ull) {
    std::printf("FAIL the 19 tensors total %llu bytes, the checkpoint's header says 1689281536\n",
                static_cast<unsigned long long>(bytes));
    ok = false;
  }
  std::printf("%s load: 19 tensors, shapes and 3x64-byte samples against the header, %llu bytes (%.4f GB), "
              "block %.4f GB\n",
              ok ? "PASS" : "FAIL", static_cast<unsigned long long>(bytes), double(bytes) / 1e9,
              double(drafter.weight_bytes()) / 1e9);
  return ok;
}

// ---------------------------------------------------------------------------------- 3. the Q4_0 cache kernels

constexpr int kQ4MaxPos = 1024;

// FP64 value of one Q4_0 cache element, from the bytes the kernel wrote (kernels/mtp/mtp_attn.h).
double q4_read(const std::uint8_t* base, std::uint64_t values, std::size_t index) {
  const std::uint8_t byte = base[index / 2];
  const int nibble = (index & 1) ? int(byte >> 4) : int(byte & 0x0F);
  const __half* scales = reinterpret_cast<const __half*>(base + values / 2);
  return double(__half2float(scales[index / 32])) * double(nibble - 8);
}

// Position `pos` of a synthetic Q4_0 history, plus the q/k/v the launch is handed.
struct Q4Case {
  int pos = 0;
  std::vector<float> q, k_new, v_new;
  std::vector<std::uint8_t> kc, vc;
  std::uint64_t values = 0;
};

Q4Case make_q4_case(int pos, unsigned seed) {
  Q4Case c;
  c.pos = pos;
  c.values = std::uint64_t(kQ4MaxPos) * KVH * DIM;
  c.q.resize(std::size_t(HEADS) * DIM);
  c.k_new.resize(std::size_t(KVH) * DIM);
  c.v_new.resize(std::size_t(KVH) * DIM);
  Rng rng(seed);
  for (float& v : c.q) v = float(rng.normal());
  for (float& v : c.k_new) v = float(rng.normal());
  for (float& v : c.v_new) v = float(rng.normal());
  // A plausible history: arbitrary nibbles with small positive block scales, so the reader is what is tested.
  c.kc.resize(std::size_t(attn::mtp_kv_bytes(attn::MtpKvType::Q4_0, kQ4MaxPos)));
  c.vc.resize(c.kc.size());
  for (auto* side : {&c.kc, &c.vc}) {
    for (auto& b : *side) b = std::uint8_t(rng.next() & 0xFF);
    __half* scales = reinterpret_cast<__half*>(side->data() + c.values / 2);
    for (std::uint64_t i = 0; i < c.values / 32; ++i) scales[i] = __float2half(0.01f + 0.04f * float(rng.uniform()));
  }
  return c;
}

// The store against the documented quantiser, byte for byte.
bool q4_store_case(int pos) {
  Q4Case c = make_q4_case(pos, 21);
  DeviceArray<float> q(c.q), k(c.k_new), v(c.v_new);
  DeviceArray<std::uint8_t> kc(c.kc), vc(c.vc);
  attn::DecodeAttnArgs args;
  args.q = q.get();
  args.k_new = k.get();
  args.v_new = v.get();
  args.k_cache = kc.get();
  args.v_cache = vc.get();
  args.pos = pos;
  args.max_pos = kQ4MaxPos;
  if (attn::mtp_kv_store_q4(args, nullptr) != cudaSuccess) {
    fail("q4 store launch");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));

  bool ok = true;
  const std::uint64_t values = c.values;
  const std::vector<std::uint8_t> got_k = kc.download(), got_v = vc.download();
  for (int side = 0; side < 2; ++side) {
    const std::vector<std::uint8_t>& got = side ? got_v : got_k;
    const std::vector<float>& src = side ? c.v_new : c.k_new;
    const __half* scales = reinterpret_cast<const __half*>(got.data() + values / 2);
    for (int head = 0; head < KVH; ++head) {
      for (int block = 0; block < DIM / 32; ++block) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[std::size_t(head * DIM + block * 32 + i)]));
        const float d = amax > 0.0f ? amax / 8.0f : 1.0f;
        const std::size_t base = std::size_t(pos) * KVH * DIM + std::size_t(head) * DIM + std::size_t(block) * 32;
        const __half want_scale = __float2half(d);
        if (std::memcmp(&scales[base / 32], &want_scale, sizeof(__half)) != 0) {
          std::printf("FAIL q4 store: block scale at %zu is %g, want %g\n", base / 32,
                      double(__half2float(scales[base / 32])), double(d));
          ok = false;
        }
        for (int i = 0; i < 32; ++i) {
          const int want = std::min(15, std::max(0, int(std::nearbyint(src[std::size_t(head * DIM + block * 32 + i)] / d)) + 8));
          const std::uint8_t byte = got[base / 2 + std::size_t(i / 2)];
          const int nibble = (i & 1) ? int(byte >> 4) : int(byte & 0x0F);
          if (nibble != want) {
            std::printf("FAIL q4 store: nibble %d of block at %zu is %d, want %d\n", i, base / 2, nibble, want);
            ok = false;
          }
        }
      }
    }
  }
  std::printf("%s q4_0 store at pos %d: nibbles and block scales byte for byte\n", ok ? "PASS" : "FAIL", pos);
  return ok;
}

// The split and combine kernels against an FP64 attention decoded from the cache bytes they were handed.
bool q4_attention_case(int pos, int splits) {
  Q4Case c = make_q4_case(pos, 31 + unsigned(pos));
  DeviceArray<float> q(c.q), k(c.k_new), v(c.v_new);
  DeviceArray<std::uint8_t> kc(c.kc), vc(c.vc);
  DeviceArray<float> out(std::size_t(HEADS) * DIM);
  std::vector<attn::SplitState> partial(std::size_t(HEADS) * attn::kMaxSplits);
  DeviceArray<attn::SplitState> dpartial(partial);

  attn::DecodeAttnArgs args;
  args.q = q.get();
  args.k_new = k.get();
  args.v_new = v.get();
  args.k_cache = kc.get();
  args.v_cache = vc.get();
  args.out = out.get();
  args.pos = pos;
  args.max_pos = kQ4MaxPos;
  args.partial = dpartial.get();
  args.splits = splits;
  if (attn::mtp_kv_store_q4(args, nullptr) != cudaSuccess ||
      attn::mtp_decode_split_q4(args, splits, nullptr) != cudaSuccess ||
      attn::decode_combine(args, nullptr) != cudaSuccess) {
    fail("q4 attention launches");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> got = out.download();
  const std::vector<std::uint8_t> kc_host = kc.download(), vc_host = vc.download();

  // Reference: attention over the bytes the kernel itself wrote.
  std::vector<double> ref(std::size_t(HEADS) * DIM, 0.0), score(std::size_t(pos) + 1);
  const double scale = 1.0 / std::sqrt(double(DIM));
  for (int head = 0; head < HEADS; ++head) {
    const int kh = head / (HEADS / KVH);
    double m = -std::numeric_limits<double>::infinity();
    for (int i = 0; i <= pos; ++i) {
      double dot = 0.0;
      for (int d = 0; d < DIM; ++d) {
        dot += double(c.q[std::size_t(head * DIM + d)]) *
               q4_read(kc_host.data(), c.values, std::size_t(i) * KVH * DIM + std::size_t(kh) * DIM + std::size_t(d));
      }
      score[std::size_t(i)] = dot * scale;
      m = std::max(m, score[std::size_t(i)]);
    }
    double sum = 0.0;
    for (int i = 0; i <= pos; ++i) {
      score[std::size_t(i)] = std::exp(score[std::size_t(i)] - m);
      sum += score[std::size_t(i)];
    }
    for (int d = 0; d < DIM; ++d) {
      double acc = 0.0;
      for (int i = 0; i <= pos; ++i) {
        acc += score[std::size_t(i)] *
               q4_read(vc_host.data(), c.values, std::size_t(i) * KVH * DIM + std::size_t(kh) * DIM + std::size_t(d));
      }
      ref[std::size_t(head * DIM + d)] = acc / sum;
    }
  }
  char tag[96];
  std::snprintf(tag, sizeof tag, "q4_0 attention pos %d splits %d", pos, splits);
  return report(tag, compare(got.data(), ref.data(), got.size()), 1e-5);
}

// FP64 reference of one full-layer step, and the fp32 device result, compared as max_abs / rms.
bool compare_step(const char* tag, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
  return report(tag, compare(got.data(), ref.data(), got.size()), tol);
}

// What a Q4_0 cache costs in quality, measured where the number means something: the same FP32 k/v stream stored
// through decode_kv_store into a BF16 cache and through mtp_kv_store_q4 into a Q4_0 one, then one attention over
// each. (Part 4's module comparison cannot carry this number: a small cache error moves the router's top-8, and
// an expert id flipping moves h' by more than the cache error did.)
bool cache_quality_case() {
  constexpr int kPos = 31;
  constexpr int kCapacity = 64;
  Rng rng(55);
  std::vector<std::vector<float>> hist_k(std::size_t(kPos) + 1), hist_v(hist_k.size());
  for (std::size_t p = 0; p < hist_k.size(); ++p) {
    hist_k[p].resize(KVDIM);
    hist_v[p].resize(KVDIM);
    for (float& x : hist_k[p]) x = float(rng.normal());
    for (float& x : hist_v[p]) x = float(rng.normal());
  }
  std::vector<float> q(std::size_t(HEADS) * DIM);
  for (float& x : q) x = float(rng.normal());

  const std::size_t out_values = std::size_t(HEADS) * DIM;
  DeviceArray<float> dq(q), dk(KVDIM), dv(KVDIM), out_bf(out_values), out_q4(out_values);
  DeviceArray<__nv_bfloat16> bkc(std::size_t(kCapacity) * KVDIM), bvc(std::size_t(kCapacity) * KVDIM);
  DeviceArray<std::uint8_t> qkc(attn::mtp_kv_bytes(attn::MtpKvType::Q4_0, kCapacity));
  DeviceArray<std::uint8_t> qvc(attn::mtp_kv_bytes(attn::MtpKvType::Q4_0, kCapacity));
  std::vector<attn::SplitState> partial(std::size_t(HEADS) * attn::kMaxSplits);
  DeviceArray<attn::SplitState> dpartial(partial);

  attn::DecodeAttnArgs a;
  a.q = dq.get();
  a.k_new = dk.get();
  a.v_new = dv.get();
  a.max_pos = kCapacity;
  a.partial = dpartial.get();
  a.splits = 4;
  for (int p = 0; p <= kPos; ++p) {
    dk.upload(hist_k[std::size_t(p)]);
    dv.upload(hist_v[std::size_t(p)]);
    a.pos = p;
    a.k_cache = bkc.get();
    a.v_cache = bvc.get();
    if (attn::decode_kv_store(a, nullptr) != cudaSuccess) {
      fail("bf16 store");
      return false;
    }
    a.k_cache = qkc.get();
    a.v_cache = qvc.get();
    if (attn::mtp_kv_store_q4(a, nullptr) != cudaSuccess) {
      fail("q4 store");
      return false;
    }
  }
  a.pos = kPos;
  a.k_cache = bkc.get();
  a.v_cache = bvc.get();
  a.out = out_bf.get();
  if (attn::decode_attention(a, nullptr) != cudaSuccess) {
    fail("bf16 attention");
    return false;
  }
  a.k_cache = qkc.get();
  a.v_cache = qvc.get();
  a.out = out_q4.get();
  if (attn::decode_kv_store(a, nullptr) != cudaSuccess || attn::mtp_decode_split_q4(a, a.splits, nullptr) != cudaSuccess ||
      attn::decode_combine(a, nullptr) != cudaSuccess) {
    fail("q4 attention");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const std::vector<float> bf = out_bf.download(), q4 = out_q4.download();
  double worst = 0.0, rms = 0.0;
  for (std::size_t i = 0; i < bf.size(); ++i) {
    worst = std::max(worst, std::fabs(double(q4[i]) - double(bf[i])));
    rms += double(bf[i]) * double(bf[i]);
  }
  rms = std::sqrt(rms / double(bf.size()));
  // The gate is a sanity bound, not a quality bar: four bits per value across a 32-value block give an error of
  // about max/16 per element, and the measured cost over 32 positions is the input the alpha measurement (K0)
  // needs, not something this test can pass or fail.
  const double rel = worst / rms;
  const bool ok = rel < 1.0;
  std::printf("%s q4_0 vs bf16 cache, %d positions: max_abs=%.3e rms=%.3e rel=%.3e (sanity bound 1.0)\n",
              ok ? "PASS" : "FAIL", kPos + 1, worst, rms, rel);
  return ok;
}

}  // namespace

int main() {
  const std::string snapshot = env_or_skip("QWEN35MOE_SNAPSHOT");
  if (snapshot.empty()) return 0;
  if (!have_matching_gpu()) {
    std::printf("skip: needs sm_120\n");
    return 0;
  }
  const std::filesystem::path dir(snapshot);
  bool ok = true;

  Trunk trunk;
  if (!upload_trunk(dir, &trunk)) return 1;

  // Every step of part 2 is the same seven (position, token) pairs, so the cache history a cache type is judged
  // on is the one the BF16 run built.
  const int positions[] = {0, 1, 2, 3, 4, 33, 2048};
  const int tokens[] = {1879, 314, 3377, 13, 271, 71093, 3237};
  constexpr int kSteps = int(sizeof(positions) / sizeof(positions[0]));

  // The hidden-state rows, and the fp32 device copy of the one being drafted.
  std::vector<std::vector<float>> hidden(kSteps);
  for (int i = 0; i < kSteps; ++i) hidden[std::size_t(i)] = synthetic_hidden(1000 + std::uint64_t(i));
  DeviceArray<__nv_bfloat16> d_hidden(static_cast<std::size_t>(HIDDEN));
  auto upload_hidden = [&](int i) {
    std::vector<__nv_bfloat16> bf(hidden[std::size_t(i)].size());
    for (std::size_t k = 0; k < bf.size(); ++k) bf[k] = __float2bfloat16(hidden[std::size_t(i)][k]);
    INGINE_TEST_CUDA(cudaMemcpy(d_hidden.get(), bf.data(), bf.size() * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
  };

  // ---- 1. load check ------------------------------------------------------------------------------------
  {
    qwen::MtpDrafter drafter;
    std::string error;
    if (!drafter.init(dir, trunk.view, 4096, attn::MtpKvType::Bf16, 0, &error)) {
      fail("init: " + error);
      return 1;
    }
    ingine::SafetensorsModel model(dir);
    ok &= check_tensors(drafter, model);

    // No fallback: a checkpoint without the head must be refused, not run on something else.
    qwen::MtpDrafter refused;
    std::string why;
    const bool started = refused.init(dir / "no-checkpoint-here", trunk.view, 64, attn::MtpKvType::Q4_0, 0, &why);
    const bool refused_ok = !started && !why.empty() && refused.weights() == nullptr && refused.max_pos() == 0;
    std::printf("%s a directory without the head is refused at init: \"%s\"\n", refused_ok ? "PASS" : "FAIL",
                why.c_str());
    ok &= refused_ok;
    std::printf("     cache bf16 at 4096: %.3f MB; at 262144 the same type is %.1f MB per side\n",
                double(drafter.cache_bytes()) / 1e6,
                double(attn::mtp_kv_bytes(attn::MtpKvType::Bf16, qwen::MAX_POS)) / 1e6);
  }

  // ---- 2. the block against the FP64 reference, BF16 cache ----------------------------------------------
  std::vector<std::vector<float>> bf16_h(kSteps);
  {
    qwen::MtpDrafter drafter;
    std::string error;
    if (!drafter.init(dir, trunk.view, 4096, attn::MtpKvType::Bf16, 0, &error)) {
      fail("init: " + error);
      return 1;
    }
    Reference reference(dir);
    std::vector<double> ref_h, ref_logits;
    for (int i = 0; i < kSteps; ++i) {
      const int pos = positions[i];
      upload_hidden(i);
      int next = -1;
      if (!drafter.draft(tokens[i], d_hidden.get(), pos, &next, &error)) {
        fail("draft: " + error);
        ok = false;
        break;
      }
      reference.step(tokens[i], hidden[std::size_t(i)], pos, &ref_h, &ref_logits);
      std::vector<float> got_h(HIDDEN), got_logits(VOCAB);
      INGINE_TEST_CUDA(cudaMemcpy(got_h.data(), drafter.last_hidden(), HIDDEN * sizeof(float),
                                  cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(got_logits.data(), drafter.logits(), VOCAB * sizeof(float),
                                  cudaMemcpyDeviceToHost));
      char tag[96];
      std::snprintf(tag, sizeof tag, "h' pos %d", pos);
      ok &= compare_step(tag, got_h, ref_h, 1e-4);
      std::snprintf(tag, sizeof tag, "logits pos %d", pos);
      ok &= compare_step(tag, got_logits, ref_logits, 1e-4);

      const int want = Reference::argmax(ref_logits);
      double top1 = -std::numeric_limits<double>::infinity(), top2 = top1;
      for (double v : ref_logits) {
        if (v > top1) {
          top2 = top1;
          top1 = v;
        } else if (v > top2) {
          top2 = v;
        }
      }
      const bool same = want == next;
      std::printf("%s draft token pos %d: %d, reference %d (top-1 margin %.4f logits)\n", same ? "PASS" : "FAIL", pos,
                  next, want, top1 - top2);
      ok &= same;

      std::vector<int> ids(TOPK);
      INGINE_TEST_CUDA(cudaMemcpy(ids.data(), drafter.router_ids(), TOPK * sizeof(int), cudaMemcpyDeviceToHost));
      const bool routed = ids == reference.ids();
      std::printf("%s router top-8 pos %d (8th/9th logit gap %.3e)\n", routed ? "PASS" : "FAIL", pos,
                  reference.id_gap());
      ok &= routed;
      bf16_h[std::size_t(i)] = got_h;
    }
  }

  // ---- 3. the Q4_0 cache kernels ------------------------------------------------------------------------
  ok &= q4_store_case(0);
  ok &= q4_store_case(17);
  ok &= cache_quality_case();
  for (int pos : {0, 7, 511, 600, 1023}) {
    for (int splits : {1, 5, 64}) {
      ok &= q4_attention_case(pos, splits);
    }
  }

  // ---- 4. the module on every cache type: it must run the block and keep the output magnitude ------------
  // The h' difference between cache types is reported, not gated: the router's top-8 turns a small cache error
  // into an expert id flip, and the flip moves h' far more than the cache did. What a Q4_0 cache costs is
  // measured in cache_quality_case() where nothing is chaotic.
  for (attn::MtpKvType type : {attn::MtpKvType::Q8_0, attn::MtpKvType::Q4_0}) {
    qwen::MtpDrafter drafter;
    std::string error;
    if (!drafter.init(dir, trunk.view, 4096, type, 0, &error)) {
      fail("init: " + error);
      return 1;
    }
    for (int i = 0; i < kSteps; ++i) {
      upload_hidden(i);
      int next = -1;
      if (!drafter.draft(tokens[i], d_hidden.get(), positions[i], &next, &error)) {
        fail("draft: " + error);
        ok = false;
        break;
      }
      std::vector<float> got_h(HIDDEN);
      INGINE_TEST_CUDA(cudaMemcpy(got_h.data(), drafter.last_hidden(), HIDDEN * sizeof(float),
                                  cudaMemcpyDeviceToHost));
      double worst = 0.0, got_rms = 0.0, ref_rms = 0.0;
      for (int d = 0; d < HIDDEN; ++d) {
        const double g = got_h[std::size_t(d)], r = bf16_h[std::size_t(i)][std::size_t(d)];
        worst = std::max(worst, std::fabs(g - r));
        got_rms += g * g;
        ref_rms += r * r;
      }
      got_rms = std::sqrt(got_rms / HIDDEN);
      ref_rms = std::sqrt(ref_rms / HIDDEN);
      const double magnitude = got_rms / ref_rms;  // stable under an expert flip, unlike any per-dim difference
      const bool same_shape = magnitude > 0.8 && magnitude < 1.25;
      std::printf("%s %s cache at pos %d: token %d, h' rms %.4f against bf16's %.4f (%.3f), worst dim off by "
                  "%.4f of rms\n",
                  same_shape ? "PASS" : "FAIL", attn::mtp_kv_type_name(type), positions[i], next, got_rms, ref_rms,
                  magnitude, worst / ref_rms);
      ok &= same_shape;
    }
  }

  // ---- 5. cost and footprint at depth -------------------------------------------------------------------
  {
    qwen::MtpDrafter drafter;
    std::string error;
    cudaSetDevice(0);
    std::size_t free_before = 0, total_mem = 0;
    INGINE_TEST_CUDA(cudaMemGetInfo(&free_before, &total_mem));
    if (!drafter.init(dir, trunk.view, qwen::MAX_POS, qwen::kMtpKvDefault, 0, &error)) {
      fail("init at 262144: " + error);
      return 1;
    }
    std::size_t free_after = 0, total_after = 0;
    INGINE_TEST_CUDA(cudaMemGetInfo(&free_after, &total_after));
    std::printf("     footprint at 262144 (%s cache): weights %.4f GB, cache %.4f GB (both sides), scratch "
                "%.4f GB, total %.4f GB; cudaMemGetInfo says %.4f GB more is gone\n",
                attn::mtp_kv_type_name(drafter.kv_type()), double(drafter.weight_bytes()) / 1e9,
                double(drafter.cache_bytes()) / 1e9,
                double(drafter.total_bytes() - drafter.weight_bytes() - drafter.cache_bytes()) / 1e9,
                double(drafter.total_bytes()) / 1e9, double(free_before - free_after) / 1e9);

    // Bytes one draft reads: the weights it touches plus the whole cache it attends over.
    const double dense_mb = (2.0 * HIDDEN * 2 * HIDDEN + QDIM * HIDDEN + 2.0 * KVDIM * HIDDEN + HIDDEN * QDIM) * 2 /
                            1e6;
    const double routed_mb = (TOPK * (2.0 * INTER * HIDDEN + HIDDEN * INTER)) * 2 / 1e6 + EXPERTS * HIDDEN * 2.0 / 1e6;
    const double shared_mb = (2.0 * SHARED * HIDDEN + HIDDEN * SHARED) * 2 / 1e6 + HIDDEN * 2.0 / 1e6;
    const double head_mb = double(VOCAB) * HIDDEN / 2 / 1e6 + double(VOCAB) * HIDDEN / 16 / 1e6;
    std::printf("     bytes per draft: dense %.1f MB, routed %.1f MB, shared %.1f MB, lm_head %.1f MB\n", dense_mb,
                routed_mb, shared_mb, head_mb);

    double deepest_ms = 0.0;
    const int depths[] = {1023, 32767, 262143};
    for (int depth : depths) {
      upload_hidden(0);
      int next = -1;
      for (int warm = 0; warm < 3; ++warm) {
        if (!drafter.draft(tokens[0], d_hidden.get(), depth, &next, &error)) {
          fail("draft at depth: " + error);
          ok = false;
        }
      }
      constexpr int kIters = 20;
      qwen::MtpTiming avg{};
      const auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < kIters; ++it) {
        qwen::MtpTiming timing;
        if (!drafter.draft(tokens[0], d_hidden.get(), depth, &next, &error, &timing)) {
          fail("draft at depth: " + error);
          ok = false;
          break;
        }
        avg.dense_ms += timing.dense_ms;
        avg.attn_ms += timing.attn_ms;
        avg.routed_ms += timing.routed_ms;
        avg.shared_ms += timing.shared_ms;
        avg.head_ms += timing.head_ms;
      }
      const double wall_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / kIters;
      const double cache_mb = 2.0 * double(attn::mtp_kv_bytes(drafter.kv_type(), depth + 1)) / 1e6;
      const double gpu_ms = (avg.dense_ms + avg.attn_ms + avg.routed_ms + avg.shared_ms + avg.head_ms) / kIters;
      deepest_ms = gpu_ms;
      std::printf(
          "     depth %6d: %.3f ms/draft (wall %.3f) = dense %.3f + attn %.3f + routed %.3f + shared %.3f + "
          "head %.3f | %.0f MB read, %.0f GB/s\n",
          depth, gpu_ms, wall_ms, avg.dense_ms / kIters, avg.attn_ms / kIters, avg.routed_ms / kIters,
          avg.shared_ms / kIters, avg.head_ms / kIters, dense_mb + routed_mb + shared_mb + head_mb + cache_mb,
          gpu_ms > 0 ? (dense_mb + routed_mb + shared_mb + head_mb + cache_mb) / gpu_ms : 0.0);
    }
    std::printf("     trunk reference (plans/20261126_qwen36-spec-decode.md, 258k): 21.66 ms/token over 40 "
                "layers = 0.54 ms per layer, so one draft (%.3f ms) is %.1fx a quantised trunk layer at the same "
                "depth\n",
                deepest_ms, deepest_ms / 0.5415);
  }

  // ---- 6. the batched chain ------------------------------------------------------------------------------
  // draft_chain() runs the same forward per draft and the trunk's NVFP4 head once over all n rows, so a row of
  // a chain has to be the row the serial draft of the same token, hidden and position produces. h' is required at
  // 1e-6 of its rms (the forwards are the same launches, so the measured difference is zero) and the logits at
  // 2e-5, which is the batch module's own bar against its FP64 mirror: the batched kernel is a different
  // reduction from the shipped single-row head, and a chain compares a max over n * VOCAB elements, more samples
  // than that test's n = 1 check, so its 1e-6 bar would be measuring the sample count (the measured value is
  // printed either way). The greedy tokens are required to be identical, which is the property the head exists
  // for.
  {
    qwen::MtpDrafter drafter;
    std::string error;
    cudaSetDevice(0);
    if (!drafter.init(dir, trunk.view, qwen::MAX_POS, qwen::kMtpKvDefault, 0, &error)) {
      fail("chain init: " + error);
      return 1;
    }
    const int chain_tokens[] = {1879, 314, 3377, 13, 271, 71093, 3237, 1043};
    constexpr int kChainTokens = int(sizeof(chain_tokens) / sizeof(chain_tokens[0]));
    std::vector<__nv_bfloat16> bf_row(HIDDEN);

    auto row_ratio = [](const float* got, const float* ref, int width) {
      double worst = 0.0, sq = 0.0;
      for (int i = 0; i < width; ++i) {
        worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])));
        sq += double(ref[i]) * double(ref[i]);
      }
      const double rms = std::sqrt(sq / double(width));
      return rms > 0.0 ? worst / rms : worst;
    };

    // One serial chain of k drafts from `base`: draft(), then the BF16 rounding of h' between the drafts, which
    // is what the decoder path does and what the chain does for itself. h_out / logits_out are optional.
    auto run_serial = [&](int k, int base, std::vector<int>* tokens_out, std::vector<std::vector<float>>* h_out,
                          std::vector<std::vector<float>>* logits_out) {
      upload_hidden(0);
      std::vector<float> h(HIDDEN);
      for (int i = 0; i < k; ++i) {
        if (i > 0) {
          for (int d = 0; d < HIDDEN; ++d) bf_row[std::size_t(d)] = float_to_bf16(h[std::size_t(d)]);
          INGINE_TEST_CUDA(cudaMemcpy(d_hidden.get(), bf_row.data(), bf_row.size() * sizeof(__nv_bfloat16),
                                      cudaMemcpyHostToDevice));
        }
        int next = -1;
        if (!drafter.draft(chain_tokens[i], d_hidden.get(), base + i, &next, &error)) return false;
        (*tokens_out)[std::size_t(i)] = next;
        INGINE_TEST_CUDA(cudaMemcpy(h.data(), drafter.last_hidden(), HIDDEN * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        if (h_out) {
          (*h_out)[std::size_t(i)] = h;
          (*logits_out)[std::size_t(i)].resize(VOCAB);
          INGINE_TEST_CUDA(cudaMemcpy((*logits_out)[std::size_t(i)].data(), drafter.logits(),
                                      VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
        }
      }
      return true;
    };

    // 6a: the rows. Position 0, so the cache is empty on both sides and the comparison is of the kernels alone.
    upload_hidden(0);
    for (int k : {1, 3, 7}) {
      std::vector<int> serial(k), chained(k);
      std::vector<std::vector<float>> serial_h(k), serial_logits(k);
      if (!run_serial(k, 0, &serial, &serial_h, &serial_logits)) {
        fail("serial drafts: " + error);
        ok = false;
        continue;
      }
      upload_hidden(0);
      qwen::MtpChainTiming timing;
      if (!drafter.draft_chain(chain_tokens, d_hidden.get(), k, 0, chained.data(), &error, &timing)) {
        fail("draft_chain: " + error);
        ok = false;
        continue;
      }
      std::vector<float> got_h(std::size_t(k) * HIDDEN), got_logits(std::size_t(k) * VOCAB);
      INGINE_TEST_CUDA(cudaMemcpy(got_h.data(), drafter.chain_hidden(), got_h.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(got_logits.data(), drafter.chain_logits(), got_logits.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
      std::vector<int> ids(k);
      INGINE_TEST_CUDA(cudaMemcpy(ids.data(), drafter.chain_ids(), std::size_t(k) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

      bool same_tokens = true;
      double worst_h = 0.0, worst_logits = 0.0;
      int worst_logits_row = 0;
      for (int i = 0; i < k; ++i) {
        same_tokens = same_tokens && ids[std::size_t(i)] == serial[std::size_t(i)] &&
                      chained[std::size_t(i)] == serial[std::size_t(i)];
        worst_h = std::max(worst_h, row_ratio(&got_h[std::size_t(i) * HIDDEN], serial_h[std::size_t(i)].data(),
                                              HIDDEN));
        const double logits_ratio =
            row_ratio(&got_logits[std::size_t(i) * VOCAB], serial_logits[std::size_t(i)].data(), VOCAB);
        if (logits_ratio > worst_logits) {
          worst_logits = logits_ratio;
          worst_logits_row = i;
        }
      }
      const bool row_ok = same_tokens && worst_h <= 1e-6 && worst_logits <= 2e-5;
      std::printf("%s chain k=%d against %d serial drafts: tokens %s, h' ratio %.3e (tol 1e-6), logits ratio "
                  "%.3e at row %d (tol 2e-5)\n",
                  row_ok ? "PASS" : "FAIL", k, k, same_tokens ? "identical" : "differ", worst_h, worst_logits,
                  worst_logits_row);
      ok &= row_ok;
    }

    // 6b: what the one head pass buys. "serial" is k draft() calls with the host hidden rebuild between them,
    // "chain" is one draft_chain(k); both run the same k candidate tokens at the same positions.
    const int depths[] = {1023, qwen::MAX_POS - 1};
    constexpr int kIters = 10;
    for (int depth : depths) {
      std::printf("\n     chain cost at depth %d, Q4_0 cache: k candidate tokens, one verify position each\n", depth);
      std::printf("       k   serial ms   chain ms   chain ms/draft   serial ms/draft   forward ms   head ms   "
                  "head ms/draft\n");
      std::vector<int> tokens_buf(32);
      for (int k = 1; k <= 7; ++k) {
        const int base = depth + 1 - k;
        std::vector<int> out_tokens(k);
        auto chain_once = [&] {
          return drafter.draft_chain(chain_tokens, d_hidden.get(), k, base, out_tokens.data(), &error);
        };
        bool ran = true;
        for (int warm = 0; warm < 2 && ran; ++warm) {
          ran = run_serial(k, base, &tokens_buf, nullptr, nullptr) && chain_once();
        }
        if (!ran) {
          fail("chain cost: " + error);
          ok = false;
          break;
        }
        auto serial_ms = [&] {
          const auto t0 = std::chrono::steady_clock::now();
          for (int it = 0; it < kIters; ++it) run_serial(k, base, &tokens_buf, nullptr, nullptr);
          return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / kIters;
        }();
        upload_hidden(0);
        qwen::MtpChainTiming timing{};
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < kIters; ++it) {
          if (!chain_once()) {
            fail("chain cost: " + error);
            ok = false;
            break;
          }
        }
        const double chain_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / kIters;
        // One more chain with the stage events on, for the split (events cost a few microseconds).
        if (!drafter.draft_chain(chain_tokens, d_hidden.get(), k, base, out_tokens.data(), &error, &timing)) {
          fail("chain timing: " + error);
          ok = false;
          break;
        }
        std::printf("     %2d   %8.3f   %8.3f   %12.3f   %15.3f   %10.3f   %7.3f   %12.3f\n", k, serial_ms, chain_ms,
                    chain_ms / k, serial_ms / k, timing.forward_ms, timing.head_ms, timing.head_ms / k);
      }
    }
  }

  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
