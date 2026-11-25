// Kernels of Inkling-2's vision tower (transformers Inkling2VisionModel): a 23-layer pre-LayerNorm ViT over
// 20x20-pixel patches with a class token, learned position embeddings resampled to the patch grid, 2-D
// rotary attention (16 heads of 64) and layer scale, followed by a 2x2 patch merger. One image at a time.
//
// Token order. An image of rows x cols tiles (40x40 pixels, one language-model token each, raster order) has
// 4 * rows * cols patches: tile t owns patches 4t .. 4t+3 = its (0,0), (0,1), (1,0), (1,1) quadrants, so a
// tile's patches are contiguous and the merger is a reshape. Token 0 is the class token, token 1 + p is
// patch p; S = 1 + 4 * rows * cols. Buffers hold Sp >= S rows (Sp a multiple of 64 keeps GEMM operands
// aligned and their shapes few); rows >= S are padding that no valid row reads.
//
// Matrix products run through kernels::prefill::Gemm (BF16 operands, FP32 accumulation); these kernels
// are the elementwise work between them. The residual stream, statistics and activations are FP32.
// Launchers follow KERNELS.md: preallocated outputs, a stream, no allocation, no synchronisation.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::vision {

inline constexpr int HIDDEN = 1024;        // residual width
inline constexpr int HEADS = 16;
inline constexpr int HEAD_DIM = 64;
inline constexpr int PATCH = 20;           // pixels per side of a patch
inline constexpr int TILE = 40;            // pixels per side of a tile (2x2 patches)
inline constexpr int PATCH_DIM = 1200;     // 20 * 20 * 3 values per patch, (y, x, channel)
inline constexpr int POS_GRID = 32;        // the learned position table is 32 x 32
inline constexpr int MLP_DIM = 4096;
inline constexpr int MERGED = 4096;        // 4 patches of 1024
inline constexpr int ROPE_FREQS = 16;      // per axis: pairs 0-15 turn with the column, 16-31 with the row
inline constexpr float LN_EPS = 1e-5f;

// out[p] = bf16((v - 127.5) / 127.5) of patch p's 1200 values, v = the pixel byte or -1 outside the image
// (the processor pads right and bottom to whole tiles with -1). rgb u8 [height, width, 3]; out bf16
// [out_rows, 1200], rows >= 4 * rows * cols are zero.
void patchify(const std::uint8_t* rgb, int height, int width, int rows, int cols, __nv_bfloat16* out, int out_rows,
              cudaStream_t stream);

// h[0] = cls; h[1 + p] = patches[p] + the position table resampled bilinearly (half-pixel centres, edge
// clamp) to the 2 rows x 2 cols patch grid at patch p; rows >= S are zero. patches f32 [>= 4 rows cols, 1024]
// (the patch embedding), pos_table f32 [32 * 32, 1024], cls f32 [1024] (class + class position embedding),
// h f32 [out_rows, 1024].
void embed_assemble(const float* patches, const float* pos_table, const float* cls, int rows, int cols, float* h,
                    int out_rows, cudaStream_t stream);

// LayerNorm over 1024 columns: out[m] = (x[m] - mean) / sqrt(var + eps) * w + b, FP32 statistics; out is
// bf16 [M, 1024] or, with out_fp32, f32 [M, 1024] (may be x itself).
void layernorm(const float* x, const float* w, const float* b, int M, void* out, bool out_fp32, cudaStream_t stream);

// Attention inputs of one layer from the fused projection qkv f32 [Sp, 3072] = [q | k | v] (before bias):
// adds bias f32 [3072], turns q and k by the 2-D rotary angles of the token (class token (0, 0), patch at
// (row + 1, col + 1); angle = position * inv_freq, pair j < 16 by the column, pair 16 + j by the row;
// (a, b) -> (a cos - b sin, b cos + a sin)) and writes
//   q, k bf16 [16, Sp, 64]   head-major, the GEMM operands of the scores
//   vt   bf16 [16, 64, Sp]   values transposed, the operand of probabilities x values
// Rows >= S = 1 + 4 rows cols are zero. inv_freq f32 [16].
void qkv_post(const float* qkv, const float* bias, const float* inv_freq, int rows, int cols, int Sp,
              __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* vt, cudaStream_t stream);

// p[m] = bf16(softmax(scale * scores[m, 0:S])) with zeros in columns >= S. scores f32 [M, ld], p bf16 [M, ld].
void softmax_rows(const float* scores, int M, int ld, int S, float scale, __nv_bfloat16* p, cudaStream_t stream);

// out = bf16(act(x + bias)), act = exact GELU (x Phi(x)) or identity; bias f32 [N] or null. x f32 [M, N].
void bias_act(const float* x, const float* bias, std::int64_t M, int N, bool gelu, __nv_bfloat16* out,
              cudaStream_t stream);

// Layer scale and residual: h += gamma * (y + bias). y, h f32 [M, 1024]; bias, gamma f32 [1024].
void bias_scale_residual(const float* y, const float* bias, const float* gamma, int M, float* h, cudaStream_t stream);

// Merger input: out[t] = bf16 of tile t's four patch states, h[1 + 4t .. 1 + 4t + 3] (the class token is
// dropped), for t < N; rows N .. out_rows are zero. h f32 [>= 1 + 4N, 1024], out bf16 [out_rows, 4096].
void merge_rows(const float* h, int N, int out_rows, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ingine::kernels::vision
