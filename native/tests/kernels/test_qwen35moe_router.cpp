// Router of the Qwen3.6-35B-A3B MoE layers: 256 bf16 logits, top 8, softmax over the selected logits.
// The CPU reference is independent; a one-hot input pins the selection itself.
#include "common/test_util.h"

#include "arch/qwen35moe/kernels/moe/router.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace moe = ingine::arch::qwen35moe::kernels;
constexpr int E = moe::ROUTER_EXPERTS;
constexpr int K = moe::ROUTER_TOP_K;
constexpr int H = moe::ROUTER_HIDDEN;

bool have_matching_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
  if (p.major != 12 || p.minor != 0) {
    std::printf("skip CUDA comparison: %s is sm_%d%d, the kernel builds for sm_120a\n", p.name, p.major, p.minor);
    return false;
  }
  return true;
}

bool one_hot_case() {
  Rng rng(5);
  std::vector<__nv_bfloat16> w(std::size_t(E) * H);
  std::vector<float> column(E);
  for (int e = 0; e < E; ++e) {
    for (int i = 0; i < H; ++i) w[std::size_t(e) * H + i] = __float2bfloat16(float(rng.normal()));
    column[e] = __bfloat162float(w[std::size_t(e) * H]);
  }
  std::vector<float> x(H, 0.0f);
  x[0] = 1.0f;  // logit[e] == w[e][0]
  std::vector<int> ids(K);
  std::vector<float> weights(K);
  std::vector<float> logits(K);
  moe::RouterArgs a;
  a.weight = w.data();
  a.x = x.data();
  a.ids = ids.data();
  a.weights = weights.data();
  a.logits = logits.data();
  moe::route_cpu(a);

  // Expected: the top 8 of the first column, descending.
  std::vector<int> order(E);
  for (int i = 0; i < E; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a_, int b_) {
    if (column[a_] != column[b_]) return column[a_] > column[b_];
    return a_ < b_;
  });
  bool ok = true;
  for (int k = 0; k < K; ++k) {
    if (ids[k] != order[k]) {
      std::printf("FAIL one-hot id %d: got %d want %d\n", k, ids[k], order[k]);
      ok = false;
    }
  }
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) sum += weights[k];
  if (std::fabs(sum - 1.0f) > 1e-5f) {
    std::printf("FAIL one-hot weights sum %f\n", sum);
    ok = false;
  }
  for (int k = 1; k < K; ++k) {
    if (weights[k] > weights[k - 1] + 1e-7f) ok = false;  // descending
  }
  std::printf("%s one-hot selection and normalisation\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool random_case(bool cuda) {
  Rng rng(6);
  std::vector<__nv_bfloat16> w(std::size_t(E) * H);
  for (auto& v : w) v = __float2bfloat16(float(rng.normal()));
  std::vector<float> x(H);
  for (auto& v : x) v = float(rng.normal());

  std::vector<int> ref_ids(K);
  std::vector<float> ref_w(K), got_w(K), ref_logits(K);
  moe::RouterArgs r;
  r.weight = w.data();
  r.x = x.data();
  r.ids = ref_ids.data();
  r.weights = ref_w.data();
  r.logits = ref_logits.data();
  moe::route_cpu(r);
  if (!cuda) return true;

  std::vector<int> got_ids(K);
  DeviceArray<__nv_bfloat16> dw(w);
  DeviceArray<float> dx(x);
  DeviceArray<int> di(K);
  DeviceArray<float> dwn(K), dl(E);
  moe::RouterArgs d;
  d.weight = dw.get();
  d.x = dx.get();
  d.ids = di.get();
  d.weights = dwn.get();
  d.logits = dl.get();
  if (moe::route(d, nullptr) != cudaSuccess) {
    std::printf("FAIL route launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  got_ids = di.download();
  got_w = dwn.download();

  bool ok = true;
  for (int k = 0; k < K; ++k) {
    if (got_ids[k] != ref_ids[k]) {
      std::printf("FAIL id %d: got %d want %d\n", k, got_ids[k], ref_ids[k]);
      ok = false;
    }
    if (std::fabs(got_w[k] - ref_w[k]) > 1e-5f) {
      std::printf("FAIL weight %d: got %f want %f\n", k, got_w[k], ref_w[k]);
      ok = false;
    }
  }
  std::printf("%s cuda route vs reference\n", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = one_hot_case();
  const bool cuda = have_matching_gpu();
  if (!cuda) std::printf("skip CUDA comparison: needs sm_120\n");
  ok &= random_case(cuda);
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
