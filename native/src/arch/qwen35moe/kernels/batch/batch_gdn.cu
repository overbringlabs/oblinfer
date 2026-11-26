#include "arch/qwen35moe/kernels/batch/batch_gdn.h"

#include <cmath>
#include <cstddef>

#include "arch/qwen35moe/kernels/batch/batch_gemv.h"

namespace ingine::arch::qwen35moe::kernels {
namespace {

namespace gdn = ingine::kernels::gdn;

constexpr int kK = gdn::GDN_K_DIM;      // 128 key dims
constexpr int kV = gdn::GDN_V_DIM;      // 128 value dims
constexpr int kKHeadDim = gdn::GDN_K_HEADS;
constexpr int kThreads = 128;
// A block owns 32 value dims of one head and splits the 128 key dims over its four warps, so a thread keeps
// 32 state elements in registers: the n positions then run back to back without touching the state in DRAM.
constexpr int kColGroup = 32;
constexpr int kColGroups = kV / kColGroup;             // 4 blocks per head
constexpr int kRowChunks = kThreads / kColGroup;       // 4 (one per warp)
constexpr int kRowsPerThread = kK / kRowChunks;        // 32
// q = l2norm(q) * k_head_dim**-0.5, the same constant the single-token step uses.
constexpr float kQScale = 0.08838834764831845f;

static_assert(kRowChunks * kRowsPerThread == kK, "the row split must cover the key dimension");
static_assert(kColGroups * kColGroup == kV, "the column split must cover the value dimension");

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

template <int V_HEADS>
__global__ __launch_bounds__(kThreads) void gdn_steps_kernel(const GdnStepsArgs args) {
  static_assert(V_HEADS % kKHeadDim == 0, "key heads must divide value heads");
  constexpr int GROUP = V_HEADS / kKHeadDim;

  const int h = blockIdx.x;
  const int kh = h / GROUP;
  const int lane = threadIdx.x & 31;
  const int rc = threadIdx.x >> 5;      // row chunk, and the warp index
  const int cg = threadIdx.x & (kColGroup - 1);
  const int d = blockIdx.y * kColGroup + cg;
  const int i0 = rc * kRowsPerThread;

  __shared__ float q_hat[BATCH_MAX][kK], k_hat[BATCH_MAX][kK];
  __shared__ float s_dec[BATCH_MAX], s_beta[BATCH_MAX], s_qsc[BATCH_MAX], s_ksc[BATCH_MAX];
  __shared__ float w_sum[2][kRowChunks];
  __shared__ float sh_kv[kRowChunks][kColGroup], sh_o[kRowChunks][kColGroup], sh_delta[kColGroup];

  float* s_head = args.state + std::size_t(h) * kK * kV + d;
  float sreg[kRowsPerThread];
#pragma unroll
  for (int j = 0; j < kRowsPerThread; ++j) sreg[j] = s_head[std::size_t(i0 + j) * kV];

  // Normalise q and k for every position first: the four warps each cover 32 key dims, so the l2 norm is a
  // warp reduction plus a four-way merge, and the results stay in shared memory for the recurrence.
  for (int p = 0; p < args.n; ++p) {
    const float* qp = args.q + std::size_t(p) * kKHeadDim * kK + kh * kK;
    const float* kp = args.k + std::size_t(p) * kKHeadDim * kK + kh * kK;
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
      const float qv = qp[i0 + j], kv = kp[i0 + j];
      sq += qv * qv;
      sk += kv * kv;
    }
    // All 32 lanes of the warp walked the same rows, so the partial is already complete: no shuffle.
    if (lane == 0) {
      w_sum[0][rc] = sq;
      w_sum[1][rc] = sk;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float aq = 0.0f, ak = 0.0f;
#pragma unroll
      for (int w = 0; w < kRowChunks; ++w) {
        aq += w_sum[0][w];
        ak += w_sum[1][w];
      }
      s_dec[p] = expf(args.g[std::size_t(p) * args.v_heads + h]);
      s_beta[p] = args.beta[std::size_t(p) * args.v_heads + h];
      s_qsc[p] = rsqrtf(aq + 1e-6f) * kQScale;
      s_ksc[p] = rsqrtf(ak + 1e-6f);
    }
    __syncthreads();
    if (lane == 0) {
      const float qsc = s_qsc[p], ksc = s_ksc[p];
#pragma unroll
      for (int j = 0; j < kRowsPerThread; ++j) {
        q_hat[p][i0 + j] = qp[i0 + j] * qsc;
        k_hat[p][i0 + j] = kp[i0 + j] * ksc;
      }
    }
    __syncthreads();
  }

  for (int p = 0; p < args.n; ++p) {
    const float dec = s_dec[p];
    const float* khp = k_hat[p];
    float kv = 0.0f;
#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
      const float s = sreg[j] * dec;
      sreg[j] = s;
      kv += s * khp[i0 + j];
    }
    sh_kv[rc][cg] = kv;
    __syncthreads();
    if (rc == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kRowChunks; ++w) total += sh_kv[w][cg];
      const float v = args.v[(std::size_t(p) * args.v_heads + h) * kV + d];
      sh_delta[cg] = (v - total) * s_beta[p];
    }
    __syncthreads();
    const float delta = sh_delta[cg];
    const float* qhp = q_hat[p];
    float o = 0.0f;
#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
      const float s = sreg[j] + khp[i0 + j] * delta;
      sreg[j] = s;
      o += s * qhp[i0 + j];
    }
    sh_o[rc][cg] = o;
    __syncthreads();
    if (rc == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kRowChunks; ++w) total += sh_o[w][cg];
      args.out[(std::size_t(p) * args.v_heads + h) * kV + d] = total;
    }
    // The next position rewrites sh_kv, and warp 0 is still reading sh_o here.
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < kRowsPerThread; ++j) s_head[std::size_t(i0 + j) * kV] = sreg[j];
}

}  // namespace

cudaError_t gdn_recurrent_steps(const GdnStepsArgs& args, cudaStream_t stream) {
  if (!args.q || !args.k || !args.v || !args.beta || !args.g || !args.state || !args.out) {
    return cudaErrorInvalidValue;
  }
  if (args.n < 1 || args.n > BATCH_MAX) return cudaErrorInvalidValue;
  if (args.v_heads != 32) return cudaErrorInvalidValue;  // this pack's linear_value_num_heads
  const dim3 grid(args.v_heads, kColGroups);
  gdn_steps_kernel<32><<<grid, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

void gdn_recurrent_steps_cpu(const GdnStepsArgs& args, double* state, double* out) {
  const int group = args.v_heads / gdn::GDN_K_HEADS;
  for (int p = 0; p < args.n; ++p) {
    for (int h = 0; h < args.v_heads; ++h) {
      const int kh = h / group;
      double q_hat[gdn::GDN_K_DIM], k_hat[gdn::GDN_K_DIM];
      double q_norm = 0.0, k_norm = 0.0;
      for (int i = 0; i < gdn::GDN_K_DIM; ++i) {
        const double q = double(args.q[std::size_t(p) * gdn::GDN_K_HEADS * gdn::GDN_K_DIM + kh * gdn::GDN_K_DIM + i]);
        const double k = double(args.k[std::size_t(p) * gdn::GDN_K_HEADS * gdn::GDN_K_DIM + kh * gdn::GDN_K_DIM + i]);
        q_norm += q * q;
        k_norm += k * k;
      }
      const double q_scale = 1.0 / std::sqrt(q_norm + 1e-6) / std::sqrt(double(gdn::GDN_K_DIM));
      const double k_scale = 1.0 / std::sqrt(k_norm + 1e-6);
      for (int i = 0; i < gdn::GDN_K_DIM; ++i) {
        q_hat[i] = double(args.q[std::size_t(p) * gdn::GDN_K_HEADS * gdn::GDN_K_DIM + kh * gdn::GDN_K_DIM + i]) * q_scale;
        k_hat[i] = double(args.k[std::size_t(p) * gdn::GDN_K_HEADS * gdn::GDN_K_DIM + kh * gdn::GDN_K_DIM + i]) * k_scale;
      }
      const double decay = std::exp(double(args.g[std::size_t(p) * args.v_heads + h]));
      const double beta = double(args.beta[std::size_t(p) * args.v_heads + h]);
      double* s_head = state + std::size_t(h) * gdn::GDN_K_DIM * gdn::GDN_V_DIM;
      double* out_head = out + (std::size_t(p) * args.v_heads + h) * gdn::GDN_V_DIM;
      for (int d = 0; d < gdn::GDN_V_DIM; ++d) {
        double kv = 0.0;
        for (int i = 0; i < gdn::GDN_K_DIM; ++i) {
          double& s = s_head[std::size_t(i) * gdn::GDN_V_DIM + d];
          s *= decay;
          kv += s * k_hat[i];
        }
        const double delta = (double(args.v[(std::size_t(p) * args.v_heads + h) * gdn::GDN_V_DIM + d]) - kv) * beta;
        double o = 0.0;
        for (int i = 0; i < gdn::GDN_K_DIM; ++i) {
          double& s = s_head[std::size_t(i) * gdn::GDN_V_DIM + d];
          s += k_hat[i] * delta;
          o += s * q_hat[i];
        }
        out_head[d] = o;
      }
    }
  }
}

}  // namespace ingine::arch::qwen35moe::kernels
