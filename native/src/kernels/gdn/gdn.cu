// Decode step (S = 1) of the Qwen3.8 gated delta-rule linear attention. Contract: kernels/gdn/gdn.h, math in
// docs/arch/gdn-recurrent.md.
//
// One block per value head (v_heads blocks), GDN_V_DIM threads. Thread d owns output dim d and walks the state
// column S[:, d] (stride GDN_V_DIM, so each wave of loads/stores is coalesced across the block). The state is
// FP32 and small at decode (3.1 MB at 48 heads), so the two column passes are DRAM/L2 bound but far below the
// step cost: decay + kv_mem, then update + out. The l2 norms of q and k are one block reduction; q, k and the
// per-head scalars are then broadcast from shared memory so every k step reads one shared float.
//
// The state column is decayed in place first and re-read for the update, which keeps the order of the reference
// exactly (S = S * exp(g), then kv_mem from that state, then the rank-1 update, then out from the updated state).
#include "kernels/gdn/gdn.h"

#include <cmath>
#include <cstddef>

namespace ingine::kernels::gdn {
namespace {

constexpr int kBlock = GDN_V_DIM;  // one thread per value dim
constexpr int kWarps = kBlock / 32;
// q = l2norm(q) * k_head_dim**-0.5 (Python float 128**-0.5 rounded to FP32).
constexpr float kQScale = 0.08838834764831845f;

__device__ __forceinline__ float block_sum(float v, float* warp_sums) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = v;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) total += warp_sums[w];
  return total;
}

template <int V_HEADS>
__global__ __launch_bounds__(kBlock) void gdn_step_kernel(GdnStepArgs args) {
  static_assert(V_HEADS % GDN_K_HEADS == 0);
  constexpr int kGroup = V_HEADS / GDN_K_HEADS;

  const int h = blockIdx.x;
  const int d = threadIdx.x;
  const int kh = h / kGroup;

  __shared__ float q_hat[kBlock];
  __shared__ float k_hat[kBlock];
  __shared__ float warp_q[kWarps];
  __shared__ float warp_k[kWarps];
  __shared__ float q_inv, k_inv, decay, beta;

  const float q_raw = args.q[kh * GDN_K_DIM + d];
  const float k_raw = args.k[kh * GDN_K_DIM + d];
  const float q_norm = block_sum(q_raw * q_raw, warp_q);
  const float k_norm = block_sum(k_raw * k_raw, warp_k);
  if (threadIdx.x == 0) {
    q_inv = rsqrtf(q_norm + 1e-6f);
    k_inv = rsqrtf(k_norm + 1e-6f);
    decay = expf(args.g[h]);
    beta = args.beta[h];
  }
  __syncthreads();
  q_hat[d] = q_raw * q_inv * kQScale;
  k_hat[d] = k_raw * k_inv;
  __syncthreads();  // every thread reads all of q_hat/k_hat below

  float* s_col = args.state + std::size_t(h) * GDN_K_DIM * GDN_V_DIM + d;
  const float v = args.v[h * GDN_V_DIM + d];

  float kv_mem = 0.0f;
  for (int i = 0; i < GDN_K_DIM; ++i) {
    const float s = s_col[std::size_t(i) * GDN_V_DIM] * decay;
    s_col[std::size_t(i) * GDN_V_DIM] = s;
    kv_mem += s * k_hat[i];
  }
  const float delta = (v - kv_mem) * beta;

  float o = 0.0f;
  for (int i = 0; i < GDN_K_DIM; ++i) {
    const float s = s_col[std::size_t(i) * GDN_V_DIM] + k_hat[i] * delta;
    s_col[std::size_t(i) * GDN_V_DIM] = s;
    o += s * q_hat[i];
  }
  args.out[h * GDN_V_DIM + d] = o;
}

}  // namespace

cudaError_t gdn_recurrent_step(const GdnStepArgs& args, cudaStream_t stream) {
  switch (args.v_heads) {
    case 48: gdn_step_kernel<48><<<48, kBlock, 0, stream>>>(args); break;
    case 32: gdn_step_kernel<32><<<32, kBlock, 0, stream>>>(args); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

void gdn_recurrent_step_cpu(const GdnStepArgs& args, double* state, double* out) {
  const int group = args.v_heads / GDN_K_HEADS;
  for (int h = 0; h < args.v_heads; ++h) {
    const int kh = h / group;
    double q_hat[GDN_K_DIM], k_hat[GDN_K_DIM];
    double q_norm = 0.0, k_norm = 0.0;
    for (int i = 0; i < GDN_K_DIM; ++i) {
      const double q = args.q[kh * GDN_K_DIM + i], k = args.k[kh * GDN_K_DIM + i];
      q_norm += q * q;
      k_norm += k * k;
    }
    const double q_inv = 1.0 / std::sqrt(q_norm + 1e-6), k_inv = 1.0 / std::sqrt(k_norm + 1e-6);
    const double q_scale = q_inv / std::sqrt(double(GDN_K_DIM));
    for (int i = 0; i < GDN_K_DIM; ++i) {
      q_hat[i] = double(args.q[kh * GDN_K_DIM + i]) * q_scale;
      k_hat[i] = double(args.k[kh * GDN_K_DIM + i]) * k_inv;
    }

    const double decay = std::exp(double(args.g[h]));
    const double beta = double(args.beta[h]);
    double* s_head = state + std::size_t(h) * GDN_K_DIM * GDN_V_DIM;
    double* out_head = out + std::size_t(h) * GDN_V_DIM;
    for (int d = 0; d < GDN_V_DIM; ++d) {
      double kv_mem = 0.0;
      for (int i = 0; i < GDN_K_DIM; ++i) {
        double& s = s_head[std::size_t(i) * GDN_V_DIM + d];
        s *= decay;
        kv_mem += s * k_hat[i];
      }
      const double delta = (double(args.v[h * GDN_V_DIM + d]) - kv_mem) * beta;
      double o = 0.0;
      for (int i = 0; i < GDN_K_DIM; ++i) {
        double& s = s_head[std::size_t(i) * GDN_V_DIM + d];
        s += k_hat[i] * delta;
        o += s * q_hat[i];
      }
      out_head[d] = o;
    }
  }
}

}  // namespace ingine::kernels::gdn
