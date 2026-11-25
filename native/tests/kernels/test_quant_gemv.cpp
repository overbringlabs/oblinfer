// Decode GEMV over the checkpoint's quantized weights: FP8 e4m3 with a per-tensor scale, and NVFP4 e2m1 with
// e4m3 block scales every 16 columns. The CPU references decode the same bytes, so the comparison pins both
// the byte order and the maths. Host halves run everywhere; the CUDA comparison runs on sm_120.
#include "common/test_util.h"

#include "kernels/quant/gemv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;

namespace {

namespace quant = ingine::kernels::quant;

bool close(const char* tag, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
  double worst = 0.0, rms = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])));
    rms += double(ref[i]) * double(ref[i]);
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

// Closed-form checks of the two CPU decoders against values of the formats.
bool decoder_checks() {
  bool ok = true;
  // e4m3: 1.0 (0x38), 2.0 (0x40), 0.5 (0x30), 448 (0x7E), -1.0 (0xB8), smallest subnormal (0x01 = 2^-9).
  const std::pair<std::uint8_t, float> cases[] = {
      {0x38, 1.0f}, {0x40, 2.0f}, {0x30, 0.5f}, {0x7E, 448.0f}, {0xB8, -1.0f}, {0x01, 1.0f / 512.0f}};
  for (const auto& [code, want] : cases) {
    if (std::fabs(quant::fp8_e4m3_to_float(code) - want) > 1e-7f) {
      std::printf("FAIL e4m3 0x%02X: got %f want %f\n", code, quant::fp8_e4m3_to_float(code), want);
      ok = false;
    }
  }
  // e2m1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.
  const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int c = 0; c < 8; ++c) {
    if (quant::e2m1_to_float(std::uint8_t(c)) != mags[c]) ok = false;
    if (quant::e2m1_to_float(std::uint8_t(c | 0x8)) != -mags[c]) ok = false;
  }
  std::printf("%s decoder tables\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool fp8_case(bool cuda) {
  const int rows = 64, cols = 2048;  // in_proj_qkv rows scaled down; the column count is the real one
  Rng rng(21);
  std::vector<std::uint8_t> w(std::size_t(rows) * cols);
  for (auto& b : w) b = std::uint8_t(rng.uniform() * 255.0);
  std::vector<float> x(cols), ref(rows), got(rows);
  for (auto& v : x) v = float(rng.normal());
  const float scale = 0.0123f;

  quant::Fp8GemvArgs a;
  a.w = w.data();
  a.scale = scale;
  a.x = x.data();
  a.y = ref.data();
  a.rows = rows;
  a.cols = cols;
  quant::fp8_gemv_cpu(a);
  if (!cuda) return true;

  DeviceArray<std::uint8_t> dw(w);
  DeviceArray<float> dx(x), dy(rows);
  quant::Fp8GemvArgs d;
  d.w = dw.get();
  d.scale = scale;
  d.x = dx.get();
  d.y = dy.get();
  d.rows = rows;
  d.cols = cols;
  if (quant::fp8_gemv(d, nullptr) != cudaSuccess) {
    std::printf("FAIL fp8 launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  got = dy.download();
  return close("cuda fp8 gemv", got, ref, 1e-3);
}

bool nvfp4_case(bool cuda) {
  const int rows = 512, cols = 2048;  // one expert's gate_proj
  Rng rng(22);
  std::vector<std::uint8_t> w(std::size_t(rows) * cols / 2);
  for (auto& b : w) b = std::uint8_t(rng.uniform() * 255.0);
  std::vector<std::uint8_t> s(std::size_t(rows) * cols / 16);
  for (auto& b : s) b = std::uint8_t(0x30 + rng.uniform() * 0x0F);  // e4m3 codes near 0.5..~1.9
  std::vector<float> x(cols), ref(rows), got(rows);
  for (auto& v : x) v = float(rng.normal());
  const float scale2 = 0.0034f;

  quant::Nvfp4GemvArgs a;
  a.w = w.data();
  a.scales = s.data();
  a.scale = scale2;
  a.x = x.data();
  a.y = ref.data();
  a.rows = rows;
  a.cols = cols;
  quant::nvfp4_gemv_cpu(a);
  if (!cuda) return true;

  DeviceArray<std::uint8_t> dw(w), ds(s);
  DeviceArray<float> dx(x), dy(rows);
  quant::Nvfp4GemvArgs d;
  d.w = dw.get();
  d.scales = ds.get();
  d.scale = scale2;
  d.x = dx.get();
  d.y = dy.get();
  d.rows = rows;
  d.cols = cols;
  if (quant::nvfp4_gemv(d, nullptr) != cudaSuccess) {
    std::printf("FAIL nvfp4 launch\n");
    return false;
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  got = dy.download();
  return close("cuda nvfp4 gemv", got, ref, 1e-3);
}

}  // namespace

int main() {
  bool ok = decoder_checks();
  const bool cuda = have_matching_gpu();
  if (!cuda) std::printf("skip CUDA comparison: needs sm_120\n");
  ok &= fp8_case(cuda);
  ok &= nvfp4_case(cuda);
  if (ok) std::printf("ALL PASS\n");
  return ok ? 0 : 1;
}
