// Unquantized dense ops against FP64 references: bf16 GEMV, RMSNorm (in and out of place), embedding lookup.
// CPU halves run everywhere; the CUDA comparison runs on sm_120.
#include "common/test_util.h"

#include "kernels/dense/dense_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;
namespace dense = ingine::kernels::dense;

namespace {

bool close(const char* tag, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
  double worst = 0.0, rms = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - ref[i]));
    rms += ref[i] * ref[i];
  }
  rms = std::sqrt(rms / double(ref.size()));
  const double ratio = worst / (rms > 1e-12 ? rms : 1.0);
  const bool ok = ratio <= tol;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", tag, worst, rms,
              ratio, tol);
  return ok;
}

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

int rms_cases() {
  Rng rng(11);
  const int n = 2048;
  std::vector<float> x(n), w(n);
  for (auto& v : x) v = float(rng.normal());
  for (auto& v : w) v = float(0.5 + rng.normal() * 0.1);

  double sum = 0.0;
  for (float v : x) sum += double(v) * double(v);
  const double inv = 1.0 / std::sqrt(sum / n + 1e-6);
  std::vector<double> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = double(x[i]) * inv * double(w[i]);

  std::vector<float> out(n);
  dense::rms_norm_cpu(x.data(), w.data(), out.data(), n, 1e-6f);
  bool ok = close("rms_norm", out, ref, 1e-6);

  // In place must match.
  std::vector<float> in_place = x;
  dense::rms_norm_cpu(in_place.data(), w.data(), in_place.data(), n, 1e-6f);
  ok &= close("rms_norm in place", in_place, ref, 1e-6);
  return ok ? 0 : 1;
}

int cuda_cases() {
  Rng rng(12);
  const int rows = 8192, cols = 2048;  // q_proj shape
  std::vector<__nv_bfloat16> w(std::size_t(rows) * cols);
  for (auto& v : w) v = __float2bfloat16(float(rng.normal()));
  std::vector<float> x(cols), y(rows);
  for (auto& v : x) v = float(rng.normal());

  std::vector<double> ref(rows, 0.0);
  for (int r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) acc += double(__bfloat162float(w[std::size_t(r) * cols + c])) * double(x[c]);
    ref[r] = acc;
  }

  DeviceArray<__nv_bfloat16> dw(w);
  DeviceArray<float> dx(x), dy(rows);
  if (dense::gemv_bf16(dw.get(), dx.get(), dy.get(), rows, cols, nullptr) != cudaSuccess) {
    std::printf("FAIL gemv launch\n");
    return 1;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  bool ok = close("cuda gemv_bf16", dy.download(), ref, 1e-5);

  // RMSNorm on the device.
  const int n = 2048;
  std::vector<float> nx(n), nw(n), ny(n);
  for (auto& v : nx) v = float(rng.normal());
  for (auto& v : nw) v = float(0.5 + rng.normal() * 0.1);
  double ssum = 0.0;
  for (float v : nx) ssum += double(v) * double(v);
  const double ninv = 1.0 / std::sqrt(ssum / n + 1e-6);
  std::vector<double> nref(n);
  for (int i = 0; i < n; ++i) nref[i] = double(nx[i]) * ninv * double(nw[i]);
  DeviceArray<float> dnx(nx), dnw(nw), dny(n);
  if (dense::rms_norm(dnx.get(), dnw.get(), dny.get(), n, 1e-6f, nullptr) != cudaSuccess) {
    std::printf("FAIL rms launch\n");
    return 1;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  ok &= close("cuda rms_norm", dny.download(), nref, 1e-5);

  // Embedding row lookup against the table itself.
  const int vocab = 64;
  std::vector<__nv_bfloat16> table(std::size_t(vocab) * n);
  for (auto& v : table) v = __float2bfloat16(float(rng.normal()));
  std::vector<float> erow(n);
  DeviceArray<__nv_bfloat16> dtable(table);
  DeviceArray<float> de(n);
  if (dense::embedding_row(dtable.get(), 17, de.get(), n, nullptr) != cudaSuccess) {
    std::printf("FAIL embedding launch\n");
    return 1;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  std::vector<double> eref(n);
  for (int i = 0; i < n; ++i) eref[i] = double(__bfloat162float(table[std::size_t(17) * n + i]));
  ok &= close("cuda embedding_row", de.download(), eref, 1e-7);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int failures = rms_cases();
  if (have_matching_gpu()) {
    failures += cuda_cases();
  } else {
    std::printf("skip CUDA comparison: needs sm_120\n");
  }
  if (failures == 0) std::printf("ALL PASS\n");
  return failures == 0 ? 0 : 1;
}
