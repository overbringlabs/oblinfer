// Routed-expert kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `moe`) and the expert decode to BF16. Launchers run on `stream`, never
// allocate or synchronise, read routing from device memory and are CUDA-graph capturable. Shapes are
// row-major (Python order); H = 2304, I = 1152, 128 experts, top-8. Invalid arguments throw
// std::invalid_argument.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::moe {

enum class Format : int { NVFP4 = 0, MXFP8 = 1 };

// moe_up: W13 GEMV of the selected experts + softcapped SwiGLU. x bf16 [M, H], idx int32 [M, 8],
// out bf16 [M, 8, I]. NVFP4: w13 u8 [128, 2I, H/2], w13_scale e4m3 [128, 2I, H/16], w13_factor fp32 [128];
// MXFP8: w13 e4m3 [128, 2I, H], w13_scale u8 E8M0 [128, 2I, H/32], factor unused (may be null).
// pdl: the kernel waits for its predecessor before reading anything.
void moe_up(const __nv_bfloat16* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, __nv_bfloat16* out, bool pdl, cudaStream_t stream);

// moe_down: W2 GEMV + routing-weighted sum + post-MLP conv with state update. act bf16 [M, 8, I],
// wts fp32 [M, 8], w2 [128, H, I(/2)] + scales [128, H, I/16 | I/32] (+ factor), conv_w fp32 [H, 4],
// conv_state fp32 [3, H], out fp32 [M, H], conv_hist fp32 [M, H] (BF16-rounded conv inputs) or null.
// pdl: valid only right after moe_up (idx is read before the PDL wait).
void moe_down(const __nv_bfloat16* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream);

// moe_down_rows: moe_down's expert pass alone, for experts that run on another device (EP). Every slot
// (token n, k) gets its expert's row bf16(W2_e . act[n, k]) in rows[(n * 8 + k) * H + c] — the value
// moe_down's FP32 accumulation consumes, before its routing-weighted sum. Same weight arguments as
// moe_down; wts, conv_w, conv_state, out and conv_hist are not read.
void moe_down_rows(const __nv_bfloat16* act, int M, const int* idx, const void* w2, const void* w2_scale,
                   const float* w2_factor, Format fmt, __nv_bfloat16* rows, bool pdl, cudaStream_t stream);

// moe_sum: moe_down's epilogue over rows returned by moe_down_rows (the device that owns `out`): the
// routing-weighted sum in slot order k = 0..7, its BF16 rounding and the post-MLP conv with the state
// update (same arguments and rounding as moe_down). s[n, c] = bf16(sum_k wts[n, k] * rows[(n * 8 + k) * H + c]).
// Bit-identical to moe_down when the rows came from the same act / idx / weights.
void moe_sum(const __nv_bfloat16* rows, int M, const float* wts, const float* conv_w, float* conv_state,
             float* out, float* conv_hist, cudaStream_t stream);

// dequant_experts: decode experts[0..n) (int32 device) of one weight ([128, rows, cols] in checkpoint
// encoding: NVFP4 codes u8 [128, rows, cols/2] + e4m3 scales [128, rows, cols/16] + factor fp32 [128];
// MXFP8 codes e4m3 [128, rows, cols] + E8M0 scales [128, rows, cols/32]) into out bf16 [n, rows, cols],
// exactly as model.decode_experts (FP32 product, one BF16 rounding).
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ingine::kernels::moe
