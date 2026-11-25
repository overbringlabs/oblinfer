// Elementwise kernels of the vision tower; see vision.h for the contracts and the token order.
#include "kernels/vision/vision.h"

#include <cmath>

namespace ingine::kernels::vision {

namespace {

constexpr int ROW_T = 256;          // threads of a row kernel: 4 of the 1024 columns each
constexpr int ELEM_T = 256;         // threads of an elementwise kernel

__device__ __forceinline__ std::uint16_t bf16_bits(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// Sum over the block's ROW_T threads (8 warps); every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* smem) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < ROW_T / 32; ++w) total += smem[w];
  __syncthreads();
  return total;
}

// Row and column (0-based, in patches) of patch p of an image cols tiles wide.
__device__ __forceinline__ void patch_position(int p, int cols, int& row, int& col) {
  const int tile = p >> 2, in = p & 3;
  row = (tile / cols) * 2 + (in >> 1);
  col = (tile % cols) * 2 + (in & 1);
}

__global__ void patchify_kernel(const std::uint8_t* __restrict__ rgb, int height, int width, int cols, int patches,
                                std::uint16_t* __restrict__ out, std::int64_t total) {
  const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int p = int(i / PATCH_DIM), e = int(i % PATCH_DIM);
  if (p >= patches) {
    out[i] = 0;
    return;
  }
  int row, col;
  patch_position(p, cols, row, col);
  const int y = row * PATCH + e / (PATCH * 3), x = col * PATCH + (e / 3) % PATCH, c = e % 3;
  const float v = (y < height && x < width) ? float(rgb[(std::size_t(y) * width + x) * 3 + c]) : -1.f;
  out[i] = bf16_bits(__fdiv_rn(__fsub_rn(v, 127.5f), 127.5f));
}

// One axis of the bilinear resampling of the 32-entry table to `size` positions: the two taps and the
// weight of the second, in the reference's FP32 arithmetic.
__device__ __forceinline__ void axis_taps(int index, int size, int& t0, int& t1, float& w1) {
  const float src = __fsub_rn(__fdiv_rn(__fmul_rn(__fadd_rn(float(index), 0.5f), float(POS_GRID)), float(size)), 0.5f);
  const float low = floorf(src);
  w1 = __fsub_rn(src, low);
  const int l = int(low);
  t0 = min(max(l, 0), POS_GRID - 1);
  t1 = min(max(l + 1, 0), POS_GRID - 1);
}

__global__ void __launch_bounds__(ROW_T) embed_kernel(const float* __restrict__ patches,
                                                      const float* __restrict__ pos_table,
                                                      const float* __restrict__ cls, int rows, int cols,
                                                      float* __restrict__ h) {
  const int token = blockIdx.x, c = threadIdx.x * 4;
  const int S = 1 + 4 * rows * cols;
  float4* out = reinterpret_cast<float4*>(h + std::size_t(token) * HIDDEN + c);
  if (token >= S) {
    *out = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  if (token == 0) {
    *out = *reinterpret_cast<const float4*>(cls + c);
    return;
  }
  const int p = token - 1;
  int row, col;
  patch_position(p, cols, row, col);
  int r0, r1, c0, c1;
  float wr, wc;
  axis_taps(row, rows * 2, r0, r1, wr);
  axis_taps(col, cols * 2, c0, c1, wc);
  const float w00 = __fmul_rn(1.f - wr, 1.f - wc), w01 = __fmul_rn(1.f - wr, wc);
  const float w10 = __fmul_rn(wr, 1.f - wc), w11 = __fmul_rn(wr, wc);
  auto entry = [&](int r, int cc) {
    return *reinterpret_cast<const float4*>(pos_table + std::size_t(r * POS_GRID + cc) * HIDDEN + c);
  };
  const float4 a = entry(r0, c0), b = entry(r0, c1), d = entry(r1, c0), e = entry(r1, c1);
  const float4 x = *reinterpret_cast<const float4*>(patches + std::size_t(p) * HIDDEN + c);
  auto blend = [&](float xa, float pa, float pb, float pd, float pe) {
    return xa + (((pa * w00 + pb * w01) + pd * w10) + pe * w11);
  };
  *out = make_float4(blend(x.x, a.x, b.x, d.x, e.x), blend(x.y, a.y, b.y, d.y, e.y), blend(x.z, a.z, b.z, d.z, e.z),
                     blend(x.w, a.w, b.w, d.w, e.w));
}

__global__ void __launch_bounds__(ROW_T) layernorm_kernel(const float* x, const float* __restrict__ w,
                                                          const float* __restrict__ b, void* out, bool out_fp32) {
  __shared__ float smem[ROW_T / 32];
  const int c = threadIdx.x * 4;
  const std::size_t off = std::size_t(blockIdx.x) * HIDDEN + c;
  const float4 v = *reinterpret_cast<const float4*>(x + off);
  const float mean = block_sum((v.x + v.y) + (v.z + v.w), smem) * (1.f / HIDDEN);
  const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
  const float var = block_sum((d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3), smem) * (1.f / HIDDEN);
  const float r = 1.f / sqrtf(var + LN_EPS);
  const float4 w4 = *reinterpret_cast<const float4*>(w + c), b4 = *reinterpret_cast<const float4*>(b + c);
  const float y0 = d0 * r * w4.x + b4.x, y1 = d1 * r * w4.y + b4.y, y2 = d2 * r * w4.z + b4.z, y3 = d3 * r * w4.w + b4.w;
  if (out_fp32) {
    *reinterpret_cast<float4*>(static_cast<float*>(out) + off) = make_float4(y0, y1, y2, y3);
  } else {
    std::uint16_t* o = static_cast<std::uint16_t*>(out) + off;
    o[0] = bf16_bits(y0), o[1] = bf16_bits(y1), o[2] = bf16_bits(y2), o[3] = bf16_bits(y3);
  }
}

// One block per token; thread j handles rotary pairs j and j + 256 of the 512 (16 heads x 32 pairs).
__global__ void __launch_bounds__(ROW_T) qkv_post_kernel(const float* __restrict__ qkv, const float* __restrict__ bias,
                                                         const float* __restrict__ inv_freq, int cols, int S, int Sp,
                                                         std::uint16_t* __restrict__ q, std::uint16_t* __restrict__ k,
                                                         std::uint16_t* __restrict__ vt) {
  const int token = blockIdx.x;
  const bool valid = token < S;
  float pos_row = 0.f, pos_col = 0.f;  // the class token sits at (0, 0)
  if (valid && token > 0) {
    int row, col;
    patch_position(token - 1, cols, row, col);
    pos_row = float(row + 1), pos_col = float(col + 1);
  }
  const float* x = qkv + std::size_t(token) * (3 * HIDDEN);
  for (int pair = threadIdx.x; pair < HIDDEN / 2; pair += ROW_T) {
    const int head = pair / (HEAD_DIM / 2), j = pair % (HEAD_DIM / 2);
    const int c = head * HEAD_DIM + 2 * j;  // column of the pair's first element within q, k or v
    const std::size_t qk = (std::size_t(head) * Sp + token) * HEAD_DIM + 2 * j;
    const std::size_t v0 = (std::size_t(head) * HEAD_DIM + 2 * j) * Sp + token;
    if (!valid) {
      q[qk] = q[qk + 1] = k[qk] = k[qk + 1] = 0;
      vt[v0] = vt[v0 + Sp] = 0;
      continue;
    }
    const float angle = __fmul_rn(j < ROPE_FREQS ? pos_col : pos_row, inv_freq[j % ROPE_FREQS]);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    const float qa = x[c] + bias[c], qb = x[c + 1] + bias[c + 1];
    const float ka = x[HIDDEN + c] + bias[HIDDEN + c], kb = x[HIDDEN + c + 1] + bias[HIDDEN + c + 1];
    q[qk] = bf16_bits(__fadd_rn(__fmul_rn(qa, cs), __fmul_rn(-qb, sn)));
    q[qk + 1] = bf16_bits(__fadd_rn(__fmul_rn(qb, cs), __fmul_rn(qa, sn)));
    k[qk] = bf16_bits(__fadd_rn(__fmul_rn(ka, cs), __fmul_rn(-kb, sn)));
    k[qk + 1] = bf16_bits(__fadd_rn(__fmul_rn(kb, cs), __fmul_rn(ka, sn)));
    vt[v0] = bf16_bits(x[2 * HIDDEN + c] + bias[2 * HIDDEN + c]);
    vt[v0 + Sp] = bf16_bits(x[2 * HIDDEN + c + 1] + bias[2 * HIDDEN + c + 1]);
  }
}

// One block per row. Pass 1: each thread keeps a running maximum and the sum of exponentials relative to
// it over its columns; the block combines them. Pass 2 writes the probabilities.
__global__ void __launch_bounds__(ROW_T) softmax_kernel(const float* __restrict__ scores, int ld, int S, float scale,
                                                        std::uint16_t* __restrict__ p) {
  __shared__ float smem[ROW_T / 32];
  __shared__ float s_max[ROW_T / 32];
  const float* row = scores + std::size_t(blockIdx.x) * ld;
  std::uint16_t* out = p + std::size_t(blockIdx.x) * ld;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < S; c += ROW_T) m = fmaxf(m, row[c] * scale);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) s_max[threadIdx.x >> 5] = m;
  __syncthreads();
  m = s_max[0];
#pragma unroll
  for (int w = 1; w < ROW_T / 32; ++w) m = fmaxf(m, s_max[w]);
  __syncthreads();
  float sum = 0.f;
  for (int c = threadIdx.x; c < S; c += ROW_T) sum += expf(row[c] * scale - m);
  const float inv = 1.f / block_sum(sum, smem);
  for (int c = threadIdx.x; c < ld; c += ROW_T) out[c] = c < S ? bf16_bits(expf(row[c] * scale - m) * inv) : 0;
}

__global__ void bias_act_kernel(const float* __restrict__ x, const float* __restrict__ bias, std::int64_t total, int N,
                                bool gelu, std::uint16_t* __restrict__ out) {
  const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = x[i];
  if (bias) v += bias[i % N];
  if (gelu) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752440f));
  out[i] = bf16_bits(v);
}

__global__ void __launch_bounds__(ROW_T) residual_kernel(const float* __restrict__ y, const float* __restrict__ bias,
                                                         const float* __restrict__ gamma, float* __restrict__ h) {
  const int c = threadIdx.x * 4;
  const std::size_t off = std::size_t(blockIdx.x) * HIDDEN + c;
  const float4 v = *reinterpret_cast<const float4*>(y + off);
  const float4 b = *reinterpret_cast<const float4*>(bias + c), g = *reinterpret_cast<const float4*>(gamma + c);
  float4* out = reinterpret_cast<float4*>(h + off);
  const float4 r = *out;
  *out = make_float4(r.x + g.x * (v.x + b.x), r.y + g.y * (v.y + b.y), r.z + g.z * (v.z + b.z), r.w + g.w * (v.w + b.w));
}

__global__ void merge_kernel(const float* __restrict__ h, std::int64_t valid, std::int64_t total,
                             std::uint16_t* __restrict__ out) {
  const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  out[i] = i < valid ? bf16_bits(h[HIDDEN + i]) : 0;  // skip the class token's row
}

unsigned blocks(std::int64_t n) { return unsigned((n + ELEM_T - 1) / ELEM_T); }

}  // namespace

void patchify(const std::uint8_t* rgb, int height, int width, int rows, int cols, __nv_bfloat16* out, int out_rows,
              cudaStream_t stream) {
  const std::int64_t total = std::int64_t(out_rows) * PATCH_DIM;
  patchify_kernel<<<blocks(total), ELEM_T, 0, stream>>>(rgb, height, width, cols, 4 * rows * cols,
                                                        reinterpret_cast<std::uint16_t*>(out), total);
}

void embed_assemble(const float* patches, const float* pos_table, const float* cls, int rows, int cols, float* h,
                    int out_rows, cudaStream_t stream) {
  embed_kernel<<<out_rows, ROW_T, 0, stream>>>(patches, pos_table, cls, rows, cols, h);
}

void layernorm(const float* x, const float* w, const float* b, int M, void* out, bool out_fp32, cudaStream_t stream) {
  layernorm_kernel<<<M, ROW_T, 0, stream>>>(x, w, b, out, out_fp32);
}

void qkv_post(const float* qkv, const float* bias, const float* inv_freq, int rows, int cols, int Sp,
              __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* vt, cudaStream_t stream) {
  qkv_post_kernel<<<Sp, ROW_T, 0, stream>>>(qkv, bias, inv_freq, cols, 1 + 4 * rows * cols, Sp,
                                            reinterpret_cast<std::uint16_t*>(q), reinterpret_cast<std::uint16_t*>(k),
                                            reinterpret_cast<std::uint16_t*>(vt));
}

void softmax_rows(const float* scores, int M, int ld, int S, float scale, __nv_bfloat16* p, cudaStream_t stream) {
  softmax_kernel<<<M, ROW_T, 0, stream>>>(scores, ld, S, scale, reinterpret_cast<std::uint16_t*>(p));
}

void bias_act(const float* x, const float* bias, std::int64_t M, int N, bool gelu, __nv_bfloat16* out,
              cudaStream_t stream) {
  const std::int64_t total = M * N;
  bias_act_kernel<<<blocks(total), ELEM_T, 0, stream>>>(x, bias, total, N, gelu, reinterpret_cast<std::uint16_t*>(out));
}

void bias_scale_residual(const float* y, const float* bias, const float* gamma, int M, float* h, cudaStream_t stream) {
  residual_kernel<<<M, ROW_T, 0, stream>>>(y, bias, gamma, h);
}

void merge_rows(const float* h, int N, int out_rows, __nv_bfloat16* out, cudaStream_t stream) {
  const std::int64_t total = std::int64_t(out_rows) * MERGED;
  merge_kernel<<<blocks(total), ELEM_T, 0, stream>>>(h, std::int64_t(N) * MERGED, total,
                                                     reinterpret_cast<std::uint16_t*>(out));
}

}  // namespace ingine::kernels::vision
