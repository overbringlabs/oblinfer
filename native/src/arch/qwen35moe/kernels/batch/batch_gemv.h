// Verify-width (n-row) GEMVs over the checkpoint's own weight formats: one read of a weight row serves all
// n positions of a speculative verify or a prefill chunk. This is the term that makes multi-token steps pay
// for the dense part of the network — GDN `in_proj_qkv` / `in_proj_z` / `out_proj`, the attention
// projections, the bf16 routers, the shared expert and `lm_head` are streamed once per n tokens instead of
// once per token.
//
// A weight base that is not vector aligned (the dense tensors are placed by size, so alignment is not
// guaranteed anywhere except in the packed expert blocks) is read through byte loads instead: the launcher
// tests the pointer and picks the instantiation. Activations and outputs must be normally aligned.
//
// Contract, shared by every launcher here:
//   x is [n, cols] row-major, y is [n, rows] row-major, y is overwritten;
//   n is 1..BATCH_MAX and is a runtime value that selects a templated instantiation, so the n-fold
//   accumulation is fully unrolled and stays in registers;
//   nothing is allocated, nothing is synchronised, so a launch is stream-capturable.
//
// Arithmetic matches `kernels/quant/gemv.cu` per row: a lane owns eight columns, the products are summed in
// fp32 inside the warp and multiplied by the per-tensor scale at the end. With n = 1 the results agree with
// the single-token kernels to the last bit of the reduction order they use (fp32 warp butterfly), which is
// what the n = 1 test asserts against.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ingine::arch::qwen35moe::kernels {

// Upper bound of the verify width these kernels are instantiated for (1..BATCH_MAX).
inline constexpr int BATCH_MAX = 4;

struct Fp8GemvBatchArgs {
  const std::uint8_t* w = nullptr;  // [rows, cols] e4m3
  float scale = 1.0f;               // per-tensor weight_scale
  const float* x = nullptr;         // [n, cols]
  float* y = nullptr;               // [n, rows]
  int rows = 0;
  int cols = 0;
  int n = 0;
};

// FP8 e4m3 weights: the GDN and attention projections.
cudaError_t fp8_gemv_batch(const Fp8GemvBatchArgs& args, cudaStream_t stream);
void fp8_gemv_batch_cpu(const Fp8GemvBatchArgs& args);  // FP64 over the same bytes

struct Nvfp4GemvBatchArgs {
  const std::uint8_t* w = nullptr;       // [rows, cols/2] two e2m1 codes per byte, even column low
  const std::uint8_t* scales = nullptr;  // [rows, cols/16] e4m3 block scales
  float scale = 1.0f;                    // per-tensor weight_scale_2
  const float* x = nullptr;              // [n, cols]
  float* y = nullptr;                    // [n, rows]
  int rows = 0;
  int cols = 0;
  int n = 0;
};

// NVFP4 weights: the shared expert and lm_head at verify width.
cudaError_t nvfp4_gemv_batch(const Nvfp4GemvBatchArgs& args, cudaStream_t stream);
void nvfp4_gemv_batch_cpu(const Nvfp4GemvBatchArgs& args);

struct Bf16GemvBatchArgs {
  const __nv_bfloat16* w = nullptr;  // [rows, cols]
  const float* x = nullptr;          // [n, cols]
  float* y = nullptr;                // [n, rows]
  int rows = 0;
  int cols = 0;
  int n = 0;
};

// Unquantized bf16 weights: the router (rows = 256) and the small in_proj_a/b.
cudaError_t bf16_gemv_batch(const Bf16GemvBatchArgs& args, cudaStream_t stream);
void bf16_gemv_batch_cpu(const Bf16GemvBatchArgs& args);

// h[i] = silu(g[i]) * u[i] over `count` flat elements, g in place.
cudaError_t silu_mul_batch(float* g, const float* u, int count, cudaStream_t stream);

// ---- row-wise passes ----
//
// What surrounds the GEMVs at verify width: every one of them is per position, so a launch per position would
// put back the per-token overhead the batch step exists to remove. Each takes `rows` = n rows of `width` floats,
// laid out row-major and contiguous, and runs one block per row. The arithmetic per row is the single-token
// kernel's, in the same order, so a row of the batched call is bit-identical to the single-row kernel.

// y[t] = x[t] / rms(x[t]) * weight, the pack's RMSNorm over one row. Same 256-thread walk as dense::rms_norm.
cudaError_t rms_norm_rows(const float* x, const float* weight, float* y, int rows, int width, float eps,
                          cudaStream_t stream);

// x[t] = float(table[row_ids[t], 0..width)); `row_ids` is a device array of n token ids.
cudaError_t embedding_rows(const __nv_bfloat16* table, const int* row_ids, float* x, int rows, int width,
                           cudaStream_t stream);

// index[t] = argmax(x[t, 0..width)), ties to the lower index, as dense::argmax.
cudaError_t argmax_rows(const float* x, int rows, int width, int* index, cudaStream_t stream);

}  // namespace ingine::arch::qwen35moe::kernels
