// MoE decode bench: the token-major routed kernels, the router and the shared expert, side by side at the
// shapes the decoder actually runs. Needs sm_120a to exercise the hardware e2m1 decode.
//
//   bench_fused_experts [device] [layers] [experts] [tokens]
//
// Defaults are 0, 12, 16, 100 — cheap, and the historical comparison. The production shape is
// `0 40 256 64` on the fast card; `0 40 128 64` keeps the arena inside one 16 GB card. Id and weight sets
// rotate per token, so no token re-reads the previous token's experts and the reported GB/s is DRAM, not L2.
// The routed cases bill the bytes the kernel really reads: half of them when every second expert is remote.
#include "arch/qwen35moe/kernels/moe/fused_experts.h"
#include "arch/qwen35moe/kernels/moe/router.h"
#include "arch/qwen35moe/kernels/moe/shared_expert.h"
#include "kernels/quant/gemv.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace moe = ingine::arch::qwen35moe::kernels;
namespace quant = ingine::kernels::quant;

namespace {

constexpr int K = moe::MOE_TOP_K, INTER = moe::MOE_INTER, HID = moe::MOE_HIDDEN, E = 16, L = 12;

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e_ = (x);                                                                                      \
    if (e_ != cudaSuccess) {                                                                                   \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                  \
      std::exit(1);                                                                                            \
    }                                                                                                          \
  } while (0)

// ------------------------------------------------------------------ previous kernels (sm_120a build)
namespace legacy {
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kColChunk = 8;
constexpr int kWarpCols = kColChunk * 32;
constexpr int kGateUpSteps = HID / kWarpCols;
constexpr int kDownSteps = INTER / kWarpCols;
constexpr int kRowsPerBlock = 8;
constexpr int kGateUpGroups = INTER / kRowsPerBlock;
constexpr int kGateUpRowsPerWarp = kRowsPerBlock / kWarps;
constexpr int kDownBlocks = HID / kWarps;

__device__ __forceinline__ float e4m3_decode(std::uint8_t code) {
  const int sign = (code >> 7) & 1;
  const int exp = (code >> 3) & 0xF;
  const int mant = code & 0x7;
  if (exp == 0xF && mant == 0x7) return 0.0f;
  float value;
  if (exp == 0) {
    value = std::ldexp(float(mant), -9);
  } else {
    value = std::ldexp(1.0f + float(mant) / 8.0f, exp - 7);
  }
  return sign ? -value : value;
}

__device__ __forceinline__ std::uint32_t f16x2_from_e2m1x2(std::uint32_t word, int i) {
  const std::uint32_t w = word >> (8 * i);
  std::uint32_t r;
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %1;\ncvt.rn.f16x2.e2m1x2 %0, b0;\n}"
      : "=r"(r)
      : "r"(w));
  return r;
}

__device__ __forceinline__ float2 halves(std::uint32_t h2) {
  const __half2 v = *reinterpret_cast<const __half2*>(&h2);
  return make_float2(__low2float(v), __high2float(v));
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float chunk_dot(std::uint32_t code, const float2* x) {
  float part = 0.0f;
#pragma unroll
  for (int i = 0; i < kColChunk / 2; ++i) {
    const float2 v = halves(f16x2_from_e2m1x2(code, i));
    const float2 xv = x[i];
    part += v.x * xv.x + v.y * xv.y;
  }
  return part;
}

__global__ __launch_bounds__(kThreads) void fused_gate_up_kernel(const moe::FusedMoeArgs args) {
  const int k = blockIdx.x / kGateUpGroups;
  const int row0 = (blockIdx.x % kGateUpGroups) * kRowsPerBlock;
  const int eid = args.ids[k];
  const moe::FusedExpertLayer& L = args.layers[eid];
  if (L.stride_bytes == 0) {
    float* zg = args.scratch_gate + std::size_t(k) * INTER;
    float* zu = args.scratch_up + std::size_t(k) * INTER;
    for (int r = row0 + (threadIdx.x >> 5) * kGateUpRowsPerWarp, i = 0; i < kGateUpRowsPerWarp; ++r, ++i) {
      if (r < INTER && threadIdx.x % 32 == 0) {
        zg[r] = 0.0f;
        zu[r] = 0.0f;
      }
    }
    return;
  }
  const std::uint8_t* wb = L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes);
  const std::uint8_t* sb = L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes);
  const std::uint8_t* gw = wb + L.gate_weight_off;
  const std::uint8_t* uw = wb + L.up_weight_off;
  const std::uint8_t* gs = sb + L.gate_scale_off;
  const std::uint8_t* us = sb + L.up_scale_off;
  float* outg = args.scratch_gate + std::size_t(k) * INTER;
  float* outu = args.scratch_up + std::size_t(k) * INTER;
  const int lane = threadIdx.x & 31;
  const int col = lane * kColChunk;
#pragma unroll
  for (int r = 0; r < kGateUpRowsPerWarp; ++r) {
    const int row = row0 + (threadIdx.x >> 5) * kGateUpRowsPerWarp + r;
    const std::uint8_t* gwr = gw + std::size_t(row) * (HID / 2);
    const std::uint8_t* uwr = uw + std::size_t(row) * (HID / 2);
    const std::uint8_t* gsr = gs + std::size_t(row) * (HID / 16);
    const std::uint8_t* usr = us + std::size_t(row) * (HID / 16);
    float ag = 0.0f, au = 0.0f;
#pragma unroll
    for (int step = 0; step < kGateUpSteps; ++step) {
      const int c = col + step * kWarpCols;
      const float2* x = reinterpret_cast<const float2*>(args.x + c);
      const std::uint32_t gc = *reinterpret_cast<const std::uint32_t*>(gwr + c / 2);
      const std::uint32_t uc = *reinterpret_cast<const std::uint32_t*>(uwr + c / 2);
      ag += chunk_dot(gc, x) * e4m3_decode(gsr[c / 16]);
      au += chunk_dot(uc, x) * e4m3_decode(usr[c / 16]);
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
      outg[row] = ag * L.gate_scale2;
      outu[row] = au * L.up_scale2;
    }
  }
}

__global__ __launch_bounds__(kThreads) void fused_silu_mul_kernel(float* h, const float* u, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = h[i];
  h[i] = (v / (1.0f + __expf(-v))) * u[i];
}

struct DownSlot {
  const std::uint8_t* w;
  const std::uint8_t* s;
  const float* h;
  float scale;
};

__global__ __launch_bounds__(kThreads) void fused_down_kernel(const moe::FusedMoeArgs args) {
  __shared__ int s_ids[K];
  __shared__ DownSlot s_slot[K];
  if (threadIdx.x < K) s_ids[threadIdx.x] = args.ids[threadIdx.x];
  __syncthreads();
  if (threadIdx.x < K) {
    const int eid = s_ids[threadIdx.x];
    const moe::FusedExpertLayer& L = args.layers[eid];
    const bool here = L.stride_bytes != 0;
    s_slot[threadIdx.x] = {here ? L.w_base + std::size_t(eid) * std::size_t(L.stride_bytes) + L.down_weight_off
                               : nullptr,
                           here ? L.s_base + std::size_t(eid) * std::size_t(L.scale_stride_bytes) + L.down_scale_off
                                : nullptr,
                           args.scratch_gate + std::size_t(threadIdx.x) * INTER,
                           here ? L.down_scale2 * args.weights[threadIdx.x] : 0.0f};
  }
  __syncthreads();
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int col = lane * kColChunk;
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const DownSlot d = s_slot[k];
    if (d.w == nullptr) continue;
    const std::uint8_t* wr = d.w + std::size_t(row) * (INTER / 2);
    const std::uint8_t* sr = d.s + std::size_t(row) * (INTER / 16);
    float acc = 0.0f;
#pragma unroll
    for (int step = 0; step < kDownSteps; ++step) {
      const int c = col + step * kWarpCols;
      const std::uint32_t code = *reinterpret_cast<const std::uint32_t*>(wr + c / 2);
      acc += chunk_dot(code, reinterpret_cast<const float2*>(d.h + c)) * e4m3_decode(sr[c / 16]);
    }
    total += d.scale * warp_sum(acc);
  }
  if (lane == 0) args.y[row] = total;
}

void apply(const moe::FusedMoeArgs& a, cudaStream_t s) {
  fused_gate_up_kernel<<<K * kGateUpGroups, kThreads, 0, s>>>(a);
  fused_silu_mul_kernel<<<(K * INTER + kThreads - 1) / kThreads, kThreads, 0, s>>>(a.scratch_gate, a.scratch_up,
                                                                                    K * INTER);
  fused_down_kernel<<<kDownBlocks, kThreads, 0, s>>>(a);
}
}  // namespace legacy

// ------------------------------------------------------------------ ceiling
__global__ void read_only(const uint4* p, std::size_t n4, float* out) {
  std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  std::uint32_t acc = 0;
  for (; i < n4; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0xdeadbeefu) *out = 1.0f;
}

// ------------------------------------------------------------------ host
std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

int main(int argc, char** argv) {
  int dev = argc > 1 ? std::atoi(argv[1]) : 0;
  // Production shape is 40 layers x 256 experts; the small default keeps the old comparison cheap and lets
  // the arena sit on one card. argv: <device> [layers] [experts] [tokens] [remote-percentage]
  int LAYERS = argc > 2 ? std::atoi(argv[2]) : L;
  int EXP = argc > 3 ? std::atoi(argv[3]) : E;
  int TOK = argc > 4 ? std::atoi(argv[4]) : 100;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, dev));
  if (prop.major != 12) {
    std::printf("bench_fused_experts: needs sm_120a (got sm_%d%d)\n", prop.major, prop.minor);
    return 77;
  }
  std::printf("device %d: %s, %d SMs, L2 %d MB - %d layers x %d experts x %d tokens\n", dev, prop.name,
              prop.multiProcessorCount, prop.l2CacheSize >> 20, LAYERS, EXP, TOK);

  const int GW = INTER * HID / 2, DW = HID * INTER / 2;
  const int W_STRIDE = 2 * GW + DW, GS = INTER * HID / 16, S_STRIDE = 3 * GS;
  std::uint8_t *dw, *ds;
  CK(cudaMalloc(&dw, std::size_t(LAYERS) * EXP * W_STRIDE));
  CK(cudaMalloc(&ds, std::size_t(LAYERS) * EXP * S_STRIDE));
  {
    std::vector<std::uint8_t> hw(std::size_t(LAYERS) * EXP * W_STRIDE), hs(std::size_t(LAYERS) * EXP * S_STRIDE);
    std::uint64_t s = 41;
    for (auto& b : hw) b = std::uint8_t(splitmix(s) >> 17);
    for (auto& b : hs) b = std::uint8_t(0x30 + splitmix(s) % 15);
    CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice));
  }
  std::vector<moe::FusedExpertLayer> hl(std::size_t(LAYERS) * EXP);
  std::vector<float> hsc(std::size_t(LAYERS) * EXP * 3);
  std::uint64_t s2 = 7;
  for (auto& v : hsc) v = 0.001f + float(splitmix(s2) % 1000) / 1e6f;
  for (int l = 0; l < LAYERS; ++l)
    for (int e = 0; e < EXP; ++e) {
      moe::FusedExpertLayer& Lx = hl[std::size_t(l) * EXP + e];
      Lx.w_base = dw + std::size_t(l) * EXP * W_STRIDE;
      Lx.s_base = ds + std::size_t(l) * EXP * S_STRIDE;
      Lx.stride_bytes = W_STRIDE;
      Lx.scale_stride_bytes = S_STRIDE;
      Lx.up_weight_off = GW;
      Lx.down_weight_off = 2 * GW;
      Lx.up_scale_off = GS;
      Lx.down_scale_off = 2 * GS;
      Lx.gate_scale2 = hsc[std::size_t(l) * EXP * 3 + e * 3 + 0];
      Lx.up_scale2 = hsc[std::size_t(l) * EXP * 3 + e * 3 + 1];
      Lx.down_scale2 = hsc[std::size_t(l) * EXP * 3 + e * 3 + 2];
    }
  moe::FusedExpertLayer* d_layers;
  CK(cudaMalloc(&d_layers, sizeof(moe::FusedExpertLayer) * LAYERS * EXP));
  CK(cudaMemcpy(d_layers, hl.data(), sizeof(moe::FusedExpertLayer) * LAYERS * EXP, cudaMemcpyHostToDevice));
  float *dx, *dwt, *dy, *dsg, *dsu, *dsd;
  int* d_ids;
  // One id/weight set per token, so a token walks different experts than the previous one: the production
  // access pattern, where a fixed set would sit in L2 and report a bandwidth the decoder never sees.
  const int ISETS = TOK > 0 ? TOK : 1;
  CK(cudaMalloc(&dx, sizeof(float) * LAYERS * HID));
  CK(cudaMalloc(&dwt, sizeof(float) * ISETS * LAYERS * K));
  CK(cudaMalloc(&d_ids, sizeof(int) * ISETS * LAYERS * K));
  {
    std::vector<float> hx(std::size_t(LAYERS) * HID), hw(std::size_t(ISETS) * LAYERS * K);
    std::vector<int> hi(std::size_t(ISETS) * LAYERS * K);
    std::uint64_t rs = 123;
    for (int t = 0; t < ISETS; ++t)
      for (int l = 0; l < LAYERS; ++l) {
        std::vector<int> pool(EXP);
        for (int i = 0; i < EXP; ++i) pool[i] = i;
        for (int i = 0; i < K; ++i) std::swap(pool[i], pool[int(splitmix(rs) % EXP)]);
        for (int i = 0; i < K; ++i) hi[(std::size_t(t) * LAYERS + l) * K + i] = pool[i];
        float sum = 0;
        for (int i = 0; i < K; ++i) {
          hw[(std::size_t(t) * LAYERS + l) * K + i] = float(splitmix(rs) % 1000) / 1000.f + 0.01f;
          sum += hw[(std::size_t(t) * LAYERS + l) * K + i];
        }
        for (int i = 0; i < K; ++i) hw[(std::size_t(t) * LAYERS + l) * K + i] /= sum;
      }
    for (auto& v : hx) v = float(std::sin(double(splitmix(rs) % 100000) * 0.001));
    CK(cudaMemcpy(dx, hx.data(), sizeof(float) * LAYERS * HID, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwt, hw.data(), sizeof(float) * ISETS * LAYERS * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ids, hi.data(), sizeof(int) * ISETS * LAYERS * K, cudaMemcpyHostToDevice));
  }
  CK(cudaMalloc(&dy, sizeof(float) * HID));
  CK(cudaMalloc(&dsg, sizeof(float) * K * INTER));
  CK(cudaMalloc(&dsu, sizeof(float) * K * INTER));
  CK(cudaMalloc(&dsd, sizeof(float) * K * HID));

  std::vector<moe::FusedMoeArgs> args(LAYERS);
  for (int l = 0; l < LAYERS; ++l) {
    moe::FusedMoeArgs& a = args[l];
    a.layers = d_layers + std::size_t(l) * EXP;
    a.x = dx + std::size_t(l) * HID;
    a.ids = d_ids + std::size_t(l) * K;  // token 0 set
    a.weights = dwt + std::size_t(l) * K;
    a.y = dy;
    a.scratch_gate = dsg;
    a.scratch_up = dsu;
    a.scratch_down = dsd;
  }
  // Half-remote variant: even experts of every layer get stride_bytes == 0, so roughly 4 of 8 routed ids
  // skip, the production shape when a pack spreads the selected experts over both cards.
  std::vector<moe::FusedExpertLayer> half = hl;
  for (int l = 0; l < LAYERS; ++l)
    for (int e = 0; e < EXP; e += 2) half[std::size_t(l) * EXP + e].stride_bytes = 0;
  moe::FusedExpertLayer* d_half;
  CK(cudaMalloc(&d_half, sizeof(moe::FusedExpertLayer) * LAYERS * EXP));
  CK(cudaMemcpy(d_half, half.data(), sizeof(moe::FusedExpertLayer) * LAYERS * EXP, cudaMemcpyHostToDevice));
  std::vector<moe::FusedMoeArgs> args_half = args;
  for (int l = 0; l < LAYERS; ++l) args_half[l].layers = d_half + std::size_t(l) * EXP;

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  const double bytes = double(LAYERS) * K * (W_STRIDE + S_STRIDE);

  // `read` is the traffic the case actually moves: half of it when every second expert is remote and skipped.
  // The id set rotates per token, so no token re-reads the previous token's experts.
  auto run = [&](const char* name, auto fn, std::vector<moe::FusedMoeArgs> as, double read) {
    auto pass = [&](int t) {
      for (int l = 0; l < LAYERS; ++l) {
        as[l].ids = d_ids + (std::size_t(t) * LAYERS + l) * K;
        as[l].weights = dwt + (std::size_t(t) * LAYERS + l) * K;
        fn(as[l]);
      }
    };
    for (int t = 0; t < 10; ++t) pass(t % ISETS);
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int tr = 0; tr < 9; ++tr) {
      cudaEventRecord(ev0);
      for (int t = 0; t < TOK; ++t) pass(t % ISETS);
      cudaEventRecord(ev1);
      CK(cudaEventSynchronize(ev1));
      float ms;
      cudaEventElapsedTime(&ms, ev0, ev1);
      best = std::min(best, ms * 1000.f / TOK);
    }
    std::printf("%-34s %9.1f us/token  %7.1f GB/s\n", name, best, read / best / 1000.0);
    return best;
  };

  auto legacy_run = [](const moe::FusedMoeArgs& a) { legacy::apply(a, nullptr); };
  run("legacy kernels", legacy_run, args, bytes);
  const float shipped = run("fused_moe_apply (shipped)",
                            [](const moe::FusedMoeArgs& a) {
                              if (moe::fused_moe_apply(a, nullptr) != cudaSuccess)
                                std::fprintf(stderr, "apply failed\n");
                            },
                            args, bytes);
  const float shipped_half = run("fused_moe_apply, half remote",
                                 [](const moe::FusedMoeArgs& a) {
                                   if (moe::fused_moe_apply(a, nullptr) != cudaSuccess)
                                     std::fprintf(stderr, "apply failed\n");
                                 },
                                 args_half, bytes / 2);
  std::printf("fused_moe_apply per layer: %.2f us all-local, %.2f us half-remote\n", shipped / LAYERS,
              shipped_half / LAYERS);


  {  // Pure-read ceiling over the same byte count as one token's routed traffic: half of each weight and scale
     // buffer, i.e. the eight selected experts of every layer. Reads once per token, not once per layer.
    const std::size_t n4w = std::size_t(LAYERS) * K * W_STRIDE / 16;
    const std::size_t n4s = std::size_t(LAYERS) * K * S_STRIDE / 16;
    float* dummy;
    CK(cudaMalloc(&dummy, sizeof(float)));
    auto pass = [&]() {
      read_only<<<4096, 256>>>(reinterpret_cast<const uint4*>(dw), n4w, dummy);
      read_only<<<2048, 256>>>(reinterpret_cast<const uint4*>(ds), n4s, dummy);
    };
    for (int t = 0; t < 10; ++t) pass();
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int tr = 0; tr < 9; ++tr) {
      cudaEventRecord(ev0);
      for (int t = 0; t < TOK; ++t) pass();
      cudaEventRecord(ev1);
      CK(cudaEventSynchronize(ev1));
      float ms;
      cudaEventElapsedTime(&ms, ev0, ev1);
      best = std::min(best, ms * 1000.f / TOK);
    }
    std::printf("%-34s %9.1f us/token  %7.1f GB/s\n", "read_only ceiling", best, bytes / best / 1000.0);
    CK(cudaFree(dummy));
  }

  // ---------------------------------------------------------------- stage attribution
  // The number that produced "289 us per token" timed fused_moe_apply alone. The decoder's MoE block also
  // runs the router (two launches), the shared expert (two launches) and, under expert parallelism, a
  // host-staged exchange per layer. Time each part at the same shape so the production cost is reconstructable.
  auto timefn = [&](const char* name, auto fn, double read) {
    for (int t = 0; t < 10; ++t) fn(t % ISETS);
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int tr = 0; tr < 9; ++tr) {
      cudaEventRecord(ev0);
      for (int t = 0; t < TOK; ++t) fn(t % ISETS);
      cudaEventRecord(ev1);
      CK(cudaEventSynchronize(ev1));
      float ms;
      cudaEventElapsedTime(&ms, ev0, ev1);
      best = std::min(best, ms * 1000.f / TOK);
    }
    std::printf("%-34s %9.1f us/token  %7.1f GB/s\n", name, best, read / best / 1000.0);
  };

  {
    std::vector<__nv_bfloat16> hw(std::size_t(moe::ROUTER_EXPERTS) * moe::ROUTER_HIDDEN);
    std::uint64_t s3 = 99;
    for (auto& v : hw) v = __float2bfloat16(float(std::sin(double(splitmix(s3) % 10000) * 0.001)));
    __nv_bfloat16* dw3;
    CK(cudaMalloc(&dw3, hw.size() * 2));
    CK(cudaMemcpy(dw3, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    int* did3;
    float *dwt3, *dl3;
    CK(cudaMalloc(&did3, sizeof(int) * moe::ROUTER_TOP_K));
    CK(cudaMalloc(&dwt3, sizeof(float) * moe::ROUTER_TOP_K));
    CK(cudaMalloc(&dl3, sizeof(float) * moe::ROUTER_EXPERTS));
    moe::RouterArgs ra;
    ra.weight = dw3;
    ra.x = dx;
    ra.ids = did3;
    ra.weights = dwt3;
    ra.logits = dl3;
    timefn("router (logits + select)",
           [&](int) {
             for (int l = 0; l < LAYERS; ++l)
               if (moe::route(ra, nullptr) != cudaSuccess) std::fprintf(stderr, "route failed\n");
           },
           double(LAYERS) * moe::ROUTER_EXPERTS * moe::ROUTER_HIDDEN * 2);
    CK(cudaFree(dw3));
    CK(cudaFree(did3));
    CK(cudaFree(dwt3));
    CK(cudaFree(dl3));
  }

  {  // Shared expert: one routed-expert-sized NVFP4 block, read once per layer.
    namespace quant = ingine::kernels::quant;
    std::vector<std::uint8_t> gw(std::size_t(INTER) * HID / 2), gs(std::size_t(INTER) * HID / 16);
    std::vector<std::uint8_t> uw(gw.size()), us(gs.size()), dwv(std::size_t(HID) * INTER / 2), dsv(gs.size());
    std::uint64_t s4 = 55;
    for (auto* v : {&gw, &gs, &uw, &us, &dwv, &dsv})
      for (auto& b : *v) b = std::uint8_t(splitmix(s4) >> 17);
    for (auto& b : gs) b = std::uint8_t(0x30 + splitmix(s4) % 15);
    for (auto& b : us) b = std::uint8_t(0x30 + splitmix(s4) % 15);
    for (auto& b : dsv) b = std::uint8_t(0x30 + splitmix(s4) % 15);
    std::vector<__nv_bfloat16> shw(HID);
    for (auto& v : shw) v = __float2bfloat16(float(std::sin(double(splitmix(s4) % 1000) * 0.01)));
    std::uint8_t *bgw, *bgs, *buw, *bus, *bdw, *bds;
    __nv_bfloat16* bg2;
    float *bsh, *bsu;
    CK(cudaMalloc(&bgw, gw.size()));
    CK(cudaMalloc(&bgs, gs.size()));
    CK(cudaMalloc(&buw, uw.size()));
    CK(cudaMalloc(&bus, us.size()));
    CK(cudaMalloc(&bdw, dwv.size()));
    CK(cudaMalloc(&bds, dsv.size()));
    CK(cudaMalloc(&bg2, shw.size() * 2));
    CK(cudaMalloc(&bsh, sizeof(float) * INTER));
    CK(cudaMalloc(&bsu, sizeof(float) * INTER));
    CK(cudaMemcpy(bgw, gw.data(), gw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bgs, gs.data(), gs.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(buw, uw.data(), uw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bus, us.data(), us.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bdw, dwv.data(), dwv.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bds, dsv.data(), dsv.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bg2, shw.data(), shw.size() * 2, cudaMemcpyHostToDevice));
    moe::SharedExpertArgs sa;
    sa.gate = quant::Nvfp4GemvArgs{bgw, bgs, 0.001f, dx, bsh, INTER, HID};
    sa.up = quant::Nvfp4GemvArgs{buw, bus, 0.001f, dx, bsu, INTER, HID};
    sa.down = quant::Nvfp4GemvArgs{bdw, bds, 0.001f, bsh, dy, HID, INTER};
    sa.gate_weight = bg2;
    sa.x = dx;
    sa.y = dy;
    sa.scratch_h = bsh;
    sa.scratch_up = bsu;
    timefn("shared_expert (gate+up, down)",
           [&](int) {
             for (int l = 0; l < LAYERS; ++l)
               if (moe::shared_expert(sa, nullptr) != cudaSuccess) std::fprintf(stderr, "shared failed\n");
           },
           double(LAYERS) * 3 * (W_STRIDE / 2) * (1.0 + 1.0 / 8.0));
    CK(cudaFree(bgw)); CK(cudaFree(bgs)); CK(cudaFree(buw)); CK(cudaFree(bus)); CK(cudaFree(bdw));
    CK(cudaFree(bds)); CK(cudaFree(bg2)); CK(cudaFree(bsh)); CK(cudaFree(bsu));
  }
  return 0;
}
