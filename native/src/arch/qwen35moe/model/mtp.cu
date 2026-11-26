#include "arch/qwen35moe/model/mtp.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/kernels/attn/rope_norm.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"
#include "arch/qwen35moe/kernels/moe/moe_apply.h"
#include "arch/qwen35moe/kernels/moe/router.h"
#include "arch/qwen35moe/kernels/moe/shared_expert.h"
#include "arch/qwen35moe/kernels/mtp/mtp_experts.h"
#include "core/safetensors.h"
#include "kernels/dense/dense_ops.h"
#include "kernels/quant/gemv.h"

namespace ingine::arch::qwen35moe {
namespace {

namespace attn = ingine::arch::qwen35moe::kernels;
namespace dense = ingine::kernels::dense;
namespace quant = ingine::kernels::quant;

// The head as the NVIDIA NVFP4 export ships it (model-00003-of-00003.safetensors, rev 1355db6a0524): 19 BF16
// tensors, 1,689,281,536 bytes. Ornith exports the same block per expert; that is not this shape.
struct Inventory {
  const char* name;
  std::vector<std::int64_t> shape;
};

const std::vector<Inventory>& inventory() {
  static const std::vector<Inventory> kInventory = {
      {"mtp.fc.weight", {HIDDEN, 2 * HIDDEN}},
      {"mtp.pre_fc_norm_embedding.weight", {HIDDEN}},
      {"mtp.pre_fc_norm_hidden.weight", {HIDDEN}},
      {"mtp.norm.weight", {HIDDEN}},
      {"mtp.layers.0.input_layernorm.weight", {HIDDEN}},
      {"mtp.layers.0.post_attention_layernorm.weight", {HIDDEN}},
      {"mtp.layers.0.self_attn.q_proj.weight", {Q_PROJ_ROWS, HIDDEN}},
      {"mtp.layers.0.self_attn.k_proj.weight", {KV_DIM, HIDDEN}},
      {"mtp.layers.0.self_attn.v_proj.weight", {KV_DIM, HIDDEN}},
      {"mtp.layers.0.self_attn.o_proj.weight", {HIDDEN, Q_DIM}},
      {"mtp.layers.0.self_attn.q_norm.weight", {HEAD_DIM}},
      {"mtp.layers.0.self_attn.k_norm.weight", {HEAD_DIM}},
      {"mtp.layers.0.mlp.gate.weight", {EXPERTS, HIDDEN}},
      {"mtp.layers.0.mlp.experts.gate_up_proj", {EXPERTS, 2 * MOE_INTER, HIDDEN}},
      {"mtp.layers.0.mlp.experts.down_proj", {EXPERTS, HIDDEN, MOE_INTER}},
      {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", {SHARED_INTER, HIDDEN}},
      {"mtp.layers.0.mlp.shared_expert.up_proj.weight", {SHARED_INTER, HIDDEN}},
      {"mtp.layers.0.mlp.shared_expert.down_proj.weight", {HIDDEN, SHARED_INTER}},
      {"mtp.layers.0.mlp.shared_expert_gate.weight", {1, HIDDEN}},
  };
  return kInventory;
}

// The seven norms, in the order the forward applies them, with the width each one normalises.
struct NormEntry {
  const char* name;
  int count;
};

constexpr NormEntry kNorms[] = {
    {"mtp.pre_fc_norm_embedding.weight", HIDDEN},
    {"mtp.pre_fc_norm_hidden.weight", HIDDEN},
    {"mtp.layers.0.input_layernorm.weight", HIDDEN},
    {"mtp.layers.0.post_attention_layernorm.weight", HIDDEN},
    {"mtp.layers.0.self_attn.q_norm.weight", HEAD_DIM},
    {"mtp.layers.0.self_attn.k_norm.weight", HEAD_DIM},
    {"mtp.norm.weight", HIDDEN},
};

// Every tensor starts on a 256-byte boundary, so the vector loads of the kernels are safe for any of them.
constexpr std::uint64_t kAlign = 256;
constexpr std::uint64_t align_up(std::uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void* alloc(std::size_t bytes) {
  void* p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

// The trunk hands over its hidden state, which the head's first norm wants in FP32.
__global__ void bf16_row_to_f32_kernel(const __nv_bfloat16* src, float* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

// The reverse, for the row drafts 1.. of a chain consume: draft i's h' as the BF16 row draft i + 1 takes.
__global__ void f32_row_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// One flash-decode column per this many positions, capped like the trunk's Decoder::splits_at: at 262144
// positions the drafter runs 64 columns over 16 query heads.
constexpr int kSplitPositions = 2048;
int splits_at(int pos) {
  return std::max(1, std::min(attn::kMaxSplits, (pos + kSplitPositions) / kSplitPositions));
}

}  // namespace

MtpDrafter::MtpDrafter() = default;

MtpDrafter::~MtpDrafter() { release(); }

void MtpDrafter::release() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (stream_) cudaStreamDestroy(stream_);
  stream_ = nullptr;
  for (cudaEvent_t& ev : events_) {
    if (ev) cudaEventDestroy(ev);
    ev = nullptr;
  }
  for (float* p : norm_storage_) cudaFree(p);
  norm_storage_.clear();
  if (weights_) cudaFree(weights_);
  weights_ = nullptr;
  float** const floats[] = {&e_,  &hf_,  &cat_, &x_,    &h_,    &qgate_,       &q_,          &gate_,
                            &k_,  &v_,   &attn_, &y_,   &h2_,   &moe_,         &shared_,     &sh_,
                            &su_, &expert_h_, &out_, &logits_, &router_logits_, &weights_r_};
  for (float** p : floats) {
    if (*p) cudaFree(*p);
    *p = nullptr;
  }
  if (ids_) cudaFree(ids_);
  ids_ = nullptr;
  if (argmax_) cudaFree(argmax_);
  argmax_ = nullptr;
  if (partial_) cudaFree(partial_);
  partial_ = nullptr;
  if (chain_hidden_) cudaFree(chain_hidden_);
  chain_hidden_ = nullptr;
  if (chain_logits_) cudaFree(chain_logits_);
  chain_logits_ = nullptr;
  if (chain_bf16_) cudaFree(chain_bf16_);
  chain_bf16_ = nullptr;
  if (chain_ids_) cudaFree(chain_ids_);
  chain_ids_ = nullptr;
  if (k_cache_) cudaFree(k_cache_);
  k_cache_ = nullptr;
  if (v_cache_) cudaFree(v_cache_);
  v_cache_ = nullptr;
  w_ = Weights{};
  n_ = Norms{};
  weight_bytes_ = 0;
  scratch_bytes_ = 0;
  max_pos_ = 0;
  device_ = -1;
}

bool MtpDrafter::load_weights(const std::filesystem::path& snapshot_dir, std::string* error) {
  SafetensorsModel model(snapshot_dir);
  std::vector<SafetensorsRef> refs;
  std::uint64_t offset = 0;
  for (const Inventory& t : inventory()) {
    const SafetensorsRef ref = model.find(t.name);
    if (!ref) {
      *error = std::string("missing tensor: ") + t.name;
      return false;
    }
    if (ref->dtype != "BF16") {
      *error = std::string(t.name) + " is " + ref->dtype + ", the MTP head is BF16";
      return false;
    }
    if (ref->shape != t.shape) {
      std::string got, want;
      for (std::int64_t d : ref->shape) got += (got.empty() ? "" : ",") + std::to_string(d);
      for (std::int64_t d : t.shape) want += (want.empty() ? "" : ",") + std::to_string(d);
      *error = std::string(t.name) + " has shape [" + got + "], expected [" + want + "]";
      return false;
    }
    tensors_.push_back(MtpTensors{t.name, ref->shape, ref->bytes, offset});
    refs.push_back(ref);
    offset = align_up(offset + ref->bytes);
  }

  weight_bytes_ = offset;
  weights_ = alloc(weight_bytes_);
  DeviceUploader uploader({}, device_);
  std::vector<UploadRequest> requests;
  requests.reserve(refs.size());
  for (std::size_t i = 0; i < refs.size(); ++i) {
    requests.push_back(upload_request(refs[i], static_cast<std::byte*>(weights_) + tensors_[i].offset));
  }
  uploader.upload(requests, stream_);
  check(cudaStreamSynchronize(stream_), "weight upload");

  auto* base = static_cast<std::byte*>(weights_);
  auto at = [&](const char* name) -> const __nv_bfloat16* {
    for (const MtpTensors& t : tensors_) {
      if (t.name == name) return reinterpret_cast<const __nv_bfloat16*>(base + t.offset);
    }
    throw std::runtime_error(std::string("unresolved tensor: ") + name);
  };
  w_.fc = at("mtp.fc.weight");
  w_.q_proj = at("mtp.layers.0.self_attn.q_proj.weight");
  w_.k_proj = at("mtp.layers.0.self_attn.k_proj.weight");
  w_.v_proj = at("mtp.layers.0.self_attn.v_proj.weight");
  w_.o_proj = at("mtp.layers.0.self_attn.o_proj.weight");
  w_.router = at("mtp.layers.0.mlp.gate.weight");
  w_.gate_up = at("mtp.layers.0.mlp.experts.gate_up_proj");
  w_.down = at("mtp.layers.0.mlp.experts.down_proj");
  w_.shared_gate = at("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
  w_.shared_up = at("mtp.layers.0.mlp.shared_expert.up_proj.weight");
  w_.shared_down = at("mtp.layers.0.mlp.shared_expert.down_proj.weight");
  w_.shared_gate_scalar = at("mtp.layers.0.mlp.shared_expert_gate.weight");
  return true;
}

bool MtpDrafter::load_norms(const std::filesystem::path& snapshot_dir, std::string* error) {
  SafetensorsModel model(snapshot_dir);
  std::vector<float*> storage;
  for (const NormEntry& entry : kNorms) {
    const std::vector<std::byte> bytes = model.read(entry.name);
    if (bytes.size() != std::size_t(entry.count) * 2) {
      *error = std::string("unexpected size for ") + entry.name;
      for (float* p : storage) cudaFree(p);
      return false;
    }
    std::vector<__nv_bfloat16> bf(std::size_t(entry.count));
    std::memcpy(bf.data(), bytes.data(), bytes.size());
    std::vector<float> host(std::size_t(entry.count));
    // Qwen3NextRMSNorm's weight starts at zero and scales (1 + w), the convention the trunk root-caused.
    for (int i = 0; i < entry.count; ++i) host[std::size_t(i)] = __bfloat162float(bf[std::size_t(i)]) + 1.0f;
    float* device = static_cast<float*>(alloc(host.size() * sizeof(float)));
    storage.push_back(device);
    check(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
          "norm upload");
  }
  check(cudaStreamSynchronize(stream_), "norm upload");
  norm_storage_ = storage;
  n_.embed = norm_storage_[0];
  n_.hidden = norm_storage_[1];
  n_.input = norm_storage_[2];
  n_.post = norm_storage_[3];
  n_.q = norm_storage_[4];
  n_.k = norm_storage_[5];
  n_.final = norm_storage_[6];
  return true;
}

bool MtpDrafter::allocate(int max_pos, MtpKvType kv_type, std::string* error) {
  max_pos_ = max_pos;
  kv_type_ = kv_type;
  const std::uint64_t side = kernels::mtp_kv_bytes(kv_type, max_pos);
  if (side == 0) {
    *error = "cache type or capacity invalid";
    return false;
  }
  k_cache_ = alloc(side);
  v_cache_ = alloc(side);
  check(cudaMemsetAsync(k_cache_, 0, side, stream_), "cache zero");
  check(cudaMemsetAsync(v_cache_, 0, side, stream_), "cache zero");

  auto take = [&](float** out, std::size_t count) {
    *out = static_cast<float*>(alloc(count * sizeof(float)));
    scratch_bytes_ += count * sizeof(float);
  };
  take(&e_, HIDDEN);
  take(&hf_, HIDDEN);
  take(&cat_, 2 * HIDDEN);
  take(&x_, HIDDEN);
  take(&h_, HIDDEN);
  take(&qgate_, Q_PROJ_ROWS);
  take(&q_, Q_DIM);
  take(&gate_, Q_DIM);
  take(&k_, KV_DIM);
  take(&v_, KV_DIM);
  take(&attn_, Q_DIM);
  take(&y_, HIDDEN);
  take(&h2_, HIDDEN);
  take(&moe_, HIDDEN);
  take(&shared_, HIDDEN);
  take(&sh_, SHARED_INTER);
  take(&su_, SHARED_INTER);
  take(&expert_h_, std::size_t(TOP_K) * MOE_INTER);
  take(&out_, HIDDEN);
  take(&logits_, VOCAB);
  take(&router_logits_, EXPERTS);
  take(&weights_r_, TOP_K);
  ids_ = static_cast<int*>(alloc(std::size_t(TOP_K) * sizeof(int)));
  argmax_ = static_cast<int*>(alloc(sizeof(int)));
  scratch_bytes_ += std::size_t(TOP_K) * sizeof(int) + sizeof(int);
  partial_ = static_cast<attn::SplitState*>(
      alloc(std::size_t(attn::ATTN_HEADS) * attn::kMaxSplits * sizeof(attn::SplitState)));
  scratch_bytes_ += std::size_t(attn::ATTN_HEADS) * attn::kMaxSplits * sizeof(attn::SplitState);
  take(&chain_hidden_, std::size_t(kMtpChainMax) * HIDDEN);
  take(&chain_logits_, std::size_t(kMtpChainMax) * VOCAB);
  chain_bf16_ = static_cast<__nv_bfloat16*>(alloc(std::size_t(HIDDEN) * sizeof(__nv_bfloat16)));
  chain_ids_ = static_cast<int*>(alloc(std::size_t(kMtpChainMax) * sizeof(int)));
  scratch_bytes_ += std::size_t(HIDDEN) * sizeof(__nv_bfloat16) + std::size_t(kMtpChainMax) * sizeof(int);
  return true;
}

bool MtpDrafter::init(const std::filesystem::path& snapshot_dir, const MtpTrunkHead& trunk, int max_pos,
                      MtpKvType kv_type, int cuda_device, std::string* error) {
  if (weights_ || stream_) {
    *error = "already initialised";
    return false;
  }
  if (!trunk.embed_tokens || !trunk.head_w || !trunk.head_scales) {
    *error = "the trunk's embedding table and lm_head are required";
    return false;
  }
  if (max_pos <= 0) {
    *error = "max_pos must be positive";
    return false;
  }
  device_ = cuda_device;
  try {
    check(cudaSetDevice(device_), "set device");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    for (cudaEvent_t& ev : events_) check(cudaEventCreate(&ev), "event");
    if (!load_weights(snapshot_dir, error)) return false;
    if (!load_norms(snapshot_dir, error)) return false;
    if (!allocate(max_pos, kv_type, error)) return false;
    embed_tokens_ = trunk.embed_tokens;
    head_w_ = trunk.head_w;
    head_scales_ = trunk.head_scales;
    head_scale_ = trunk.head_scale;
    check(cudaStreamSynchronize(stream_), "init");
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    release();
    return false;
  }
}

bool MtpDrafter::forward(int token, const __nv_bfloat16* hidden, int pos, float* out_row, bool timed,
                         std::string* error) {
  const cudaStream_t s = stream_;
  auto mark = [&](int index) {
    if (timed) check(cudaEventRecord(events_[index], s), "event record");
  };
  try {
    mark(0);

    // ---- dense, up to the cache store: embedding, both pre-fc norms, fc, the attention projections ----
    check(dense::embedding_row(embed_tokens_, token, e_, HIDDEN, s), "embedding");
    check(dense::rms_norm(e_, n_.embed, cat_, HIDDEN, EPS, s), "embedding norm");
    bf16_row_to_f32_kernel<<<HIDDEN / 256, 256, 0, s>>>(hidden, hf_, HIDDEN);
    check(cudaGetLastError(), "hidden to f32");
    check(dense::rms_norm(hf_, n_.hidden, cat_ + HIDDEN, HIDDEN, EPS, s), "hidden norm");
    check(dense::gemv_bf16(w_.fc, cat_, x_, HIDDEN, 2 * HIDDEN, s), "fc");
    check(dense::rms_norm(x_, n_.input, h_, HIDDEN, EPS, s), "input norm");
    check(dense::gemv_bf16(w_.q_proj, h_, qgate_, Q_PROJ_ROWS, HIDDEN, s), "q_proj");
    check(dense::gemv_bf16(w_.k_proj, h_, k_, KV_DIM, HIDDEN, s), "k_proj");
    check(dense::gemv_bf16(w_.v_proj, h_, v_, KV_DIM, HIDDEN, s), "v_proj");
    attn::SplitQGateArgs split;
    split.qgate = qgate_;
    split.q = q_;
    split.gate = gate_;
    split.heads = HEADS;
    split.head_dim = HEAD_DIM;
    check(attn::split_q_gate(split, s), "split q gate");
    attn::NormRopeArgs rope;
    rope.q = q_;
    rope.k = k_;
    rope.q_weight = n_.q;
    rope.k_weight = n_.k;
    rope.pos = pos;
    rope.eps = EPS;
    check(attn::qk_norm_rope(rope, s), "qk norm rope");
    mark(1);

    // ---- the layer's own attention over its cache ----
    attn::DecodeAttnArgs dec;
    dec.q = q_;
    dec.k_new = k_;
    dec.v_new = v_;
    dec.k_cache = k_cache_;
    dec.v_cache = v_cache_;
    dec.out = attn_;
    dec.pos = pos;
    dec.max_pos = max_pos_;
    dec.partial = partial_;
    dec.splits = splits_at(pos);
    if (kv_type_ == MtpKvType::Bf16) {
      dec.kv_type_k = attn::KvType::Bf16;
      dec.kv_type_v = attn::KvType::Bf16;
      check(attn::decode_attention(dec, s), "attention bf16");
    } else if (kv_type_ == MtpKvType::Q8_0) {
      dec.kv_type_k = attn::KvType::Q8_0;
      dec.kv_type_v = attn::KvType::Q8_0;
      check(attn::decode_attention(dec, s), "attention q8_0");
    } else {
      // Q4_0: the store and the split are this module's, the merge is the shipped combine kernel.
      check(kernels::mtp_kv_store_q4(dec, s), "kv store q4_0");
      check(kernels::mtp_decode_split_q4(dec, dec.splits, s), "attention q4_0 split");
      check(attn::decode_combine(dec, s), "attention q4_0 combine");
    }
    mark(2);

    // ---- the rest of the dense block: output gate, o_proj, residual, post-attention norm ----
    attn::OutputGateArgs gate;
    gate.x = attn_;
    gate.gate = gate_;
    gate.count = Q_DIM;
    check(attn::output_gate(gate, s), "output gate");
    check(dense::gemv_bf16(w_.o_proj, attn_, y_, HIDDEN, Q_DIM, s), "o_proj");
    check(attn::axpy(x_, y_, 1.0f, HIDDEN, s), "attention residual");
    check(dense::rms_norm(x_, n_.post, h2_, HIDDEN, EPS, s), "post attention norm");
    mark(3);

    // ---- routed experts ----
    attn::RouterArgs router;
    router.weight = w_.router;
    router.x = h2_;
    router.ids = ids_;
    router.weights = weights_r_;
    router.logits = router_logits_;
    check(attn::route(router, s), "router");
    kernels::Bf16MoeArgs moe;
    moe.gate_up = w_.gate_up;
    moe.down = w_.down;
    moe.x = h2_;
    moe.ids = ids_;
    moe.weights = weights_r_;
    moe.y = moe_;
    moe.scratch_h = expert_h_;
    check(kernels::bf16_moe_apply(moe, s), "routed experts");
    mark(4);

    // ---- shared expert, then the two residual adds that close the MoE block ----
    check(dense::gemv_bf16(w_.shared_gate, h2_, sh_, SHARED_INTER, HIDDEN, s), "shared gate");
    check(dense::gemv_bf16(w_.shared_up, h2_, su_, SHARED_INTER, HIDDEN, s), "shared up");
    check(attn::silu_mul(sh_, sh_, su_, SHARED_INTER, s), "shared silu");
    check(dense::gemv_bf16(w_.shared_down, sh_, shared_, HIDDEN, SHARED_INTER, s), "shared down");
    check(attn::scale_sigmoid_dot(shared_, w_.shared_gate_scalar, h2_, HIDDEN, s), "shared gate scalar");
    check(attn::axpy(x_, moe_, 1.0f, HIDDEN, s), "moe residual");
    check(attn::axpy(x_, shared_, 1.0f, HIDDEN, s), "shared residual");
    mark(5);

    // ---- mtp.norm: the row the head is run on ----
    check(dense::rms_norm(x_, n_.final, out_row, HIDDEN, EPS, s), "mtp norm");
    mark(6);
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool MtpDrafter::draft(int token, const __nv_bfloat16* hidden, int pos, int* next_token, std::string* error,
                       MtpTiming* timing) {
  if (!weights_ || !stream_) {
    *error = "not initialised";
    return false;
  }
  if (!hidden || !next_token) {
    *error = "hidden state row and token output are required";
    return false;
  }
  if (pos < 0 || pos >= max_pos_) {
    *error = "pos " + std::to_string(pos) + " outside [0, " + std::to_string(max_pos_) + ")";
    return false;
  }
  if (token < 0 || token >= VOCAB) {
    *error = "token " + std::to_string(token) + " outside the vocabulary";
    return false;
  }
  const cudaStream_t s = stream_;
  float stages[6] = {};
  try {
    check(cudaSetDevice(device_), "set device");
    if (!forward(token, hidden, pos, out_, timing != nullptr, error)) return false;

    // ---- the trunk's NVFP4 head, one row ----
    quant::Nvfp4GemvArgs head;
    head.w = head_w_;
    head.scales = head_scales_;
    head.scale = head_scale_;
    head.x = out_;
    head.y = logits_;
    head.rows = VOCAB;
    head.cols = HIDDEN;
    check(quant::nvfp4_gemv(head, s), "lm_head");
    check(dense::argmax(logits_, VOCAB, argmax_, nullptr, s), "argmax");
    if (timing) check(cudaEventRecord(events_[7], s), "event record");
    check(cudaMemcpyAsync(next_token, argmax_, sizeof(int), cudaMemcpyDeviceToHost, s), "argmax readback");
    check(cudaStreamSynchronize(s), "draft");

    if (timing) {
      stage_times(stages);
      float tail_ms = 0.0f;
      check(cudaEventElapsedTime(&tail_ms, events_[6], events_[7]), "elapsed");
      timing->dense_ms = stages[0] + stages[2];  // the projections before the store and the ones after the merge
      timing->attn_ms = stages[1];
      timing->routed_ms = stages[3];
      timing->shared_ms = stages[4];
      timing->head_ms = stages[5] + tail_ms;  // mtp.norm, the head, argmax and the readback
    }
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

void MtpDrafter::stage_times(float stages[6]) {
  for (int k = 0; k < 6; ++k) check(cudaEventElapsedTime(&stages[k], events_[k], events_[k + 1]), "elapsed");
}

bool MtpDrafter::chain_head(int n, std::string* error) {
  const cudaStream_t s = stream_;
  try {
    // kernels::BATCH_MAX rows per pass over the weights: a seven-draft chain is two passes, not seven.
    for (int first = 0; first < n; first += kernels::BATCH_MAX) {
      const int rows = std::min(kernels::BATCH_MAX, n - first);
      kernels::Nvfp4GemvBatchArgs head;
      head.w = head_w_;
      head.scales = head_scales_;
      head.scale = head_scale_;
      head.x = chain_hidden_ + std::size_t(first) * HIDDEN;
      head.y = chain_logits_ + std::size_t(first) * VOCAB;
      head.rows = VOCAB;
      head.cols = HIDDEN;
      head.n = rows;
      check(kernels::nvfp4_gemv_batch(head, s), "lm_head batch");
      for (int r = 0; r < rows; ++r) {
        check(dense::argmax(chain_logits_ + std::size_t(first + r) * VOCAB, VOCAB, chain_ids_ + first + r, nullptr,
                            s),
              "argmax row");
      }
    }
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool MtpDrafter::draft_chain(const int* tokens, const __nv_bfloat16* hidden, int n, int pos, int* out_tokens,
                             std::string* error, MtpChainTiming* timing) {
  if (!weights_ || !stream_) {
    *error = "not initialised";
    return false;
  }
  if (!tokens || !hidden || !out_tokens) {
    *error = "candidate tokens, hidden state row and token output are required";
    return false;
  }
  if (n < 1 || n > kMtpChainMax) {
    *error = "chain length " + std::to_string(n) + " outside [1, " + std::to_string(kMtpChainMax) + "]";
    return false;
  }
  if (pos < 0 || std::int64_t(pos) + n > max_pos_) {
    *error = "positions " + std::to_string(pos) + ".." + std::to_string(pos + n - 1) + " outside [0, " +
             std::to_string(max_pos_) + ")";
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (tokens[i] < 0 || tokens[i] >= VOCAB) {
      *error = "token " + std::to_string(tokens[i]) + " at chain index " + std::to_string(i) +
               " outside the vocabulary";
      return false;
    }
  }
  const cudaStream_t s = stream_;
  try {
    check(cudaSetDevice(device_), "set device");
    float sums[6] = {};
    for (int i = 0; i < n; ++i) {
      if (!forward(tokens[i], i == 0 ? hidden : chain_bf16_, pos + i, chain_hidden_ + std::size_t(i) * HIDDEN,
                   timing != nullptr, error)) {
        return false;
      }
      if (timing) {
        // The six phase events are one set per forward, so a timed chain waits for each forward before reading
        // them. Untimed chains, which is every production call, never record or wait for an event.
        check(cudaStreamSynchronize(s), "chain stage sync");
        float stages[6] = {};
        stage_times(stages);
        for (int k = 0; k < 6; ++k) sums[k] += stages[k];
      }
      if (i + 1 < n) {
        // h' is FP32 on the device; the next draft takes the BF16 row draft() would have been handed.
        f32_row_to_bf16_kernel<<<HIDDEN / 256, 256, 0, s>>>(chain_hidden_ + std::size_t(i) * HIDDEN, chain_bf16_,
                                                            HIDDEN);
        check(cudaGetLastError(), "hidden row to bf16");
      }
    }
    if (!chain_head(n, error)) return false;
    if (timing) check(cudaEventRecord(events_[7], s), "event record");
    check(cudaMemcpyAsync(out_tokens, chain_ids_, std::size_t(n) * sizeof(int), cudaMemcpyDeviceToHost, s),
          "chain token readback");
    check(cudaStreamSynchronize(s), "draft chain");

    if (timing) {
      float head_ms = 0.0f;
      check(cudaEventElapsedTime(&head_ms, events_[6], events_[7]), "elapsed");
      timing->dense_ms = sums[0] + sums[2];
      timing->attn_ms = sums[1];
      timing->routed_ms = sums[3];
      timing->shared_ms = sums[4] + sums[5];  // mtp.norm is per position inside a chain
      timing->head_ms = head_ms;
      timing->forward_ms = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5];
    }
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

}  // namespace ingine::arch::qwen35moe
