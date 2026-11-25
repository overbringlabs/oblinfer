// Decode-width GEMV over the checkpoint's quantized weights, unquantized activations:
//   FP8 layers:   W e4m3 [rows, cols], one F32 tensor scale;      y = scale * W @ x
//   NVFP4 layers: W e2m1 [rows, cols] packed two per byte, e4m3 block scales every 16 columns, one F32
//                 tensor scale;                                    y = scale * (W * block) @ x
// `input_scale` in the checkpoint is a calibration artifact for quantized-activation eval (W4A16 / W8A16 run
// activations at full precision), so it is not applied here.
//
// Layout facts read from the checkpoint headers on 2026-11-25:
//   in_proj_qkv.weight    F8_E4M3 [8192, 2048]   weight_scale F32 []            input_scale F32 []
//   experts.0.gate_proj   U8      [ 512, 1024]   weight_scale F8_E4M3 [512,128] weight_scale_2 F32 []
//   lm_head.weight        U8      [248320, 1024] weight_scale F8_E4M3 [248320,128]
// One warp per row; lanes take two columns each so every wave reads 64 contiguous weight bytes.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace ingine::kernels::quant {

struct Fp8GemvArgs {
  const std::uint8_t* w = nullptr;  // [rows, cols], e4m3, low byte first
  float scale = 1.0f;               // per-tensor weight scale
  const float* x = nullptr;         // [cols]
  float* y = nullptr;               // [rows]
  int rows = 0;
  int cols = 0;
};

struct Nvfp4GemvArgs {
  const std::uint8_t* w = nullptr;      // [rows, cols / 2], two e2m1 codes per byte, even column low
  const std::uint8_t* scales = nullptr;  // [rows, cols / 16], e4m3 block scales
  float scale = 1.0f;                   // per-tensor scale (weight_scale_2)
  const float* x = nullptr;             // [cols]
  float* y = nullptr;                   // [rows]
  int rows = 0;
  int cols = 0;
};

cudaError_t fp8_gemv(const Fp8GemvArgs& args, cudaStream_t stream);
cudaError_t nvfp4_gemv(const Nvfp4GemvArgs& args, cudaStream_t stream);
void fp8_gemv_cpu(const Fp8GemvArgs& args);
void nvfp4_gemv_cpu(const Nvfp4GemvArgs& args);

// Scalar decoders shared by the CPU references and tests. e4m3 is the fn variant (max 448, 0x7F NaN -> 0).
float fp8_e4m3_to_float(std::uint8_t code);
float e2m1_to_float(std::uint8_t code);

}  // namespace ingine::kernels::quant
