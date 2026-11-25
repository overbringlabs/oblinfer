# Decode-width quantized GEMV shared by the hybrid packs: FP8 e4m3 with a per-tensor scale, NVFP4 e2m1 with
# e4m3 block scales every 16 columns. Compiled on every build; the quant_gemv test links it.
add_library(ingine_kernels_quant OBJECT gemv.cu)
ingine_cuda_target(ingine_kernels_quant)
target_include_directories(ingine_kernels_quant PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_quant PUBLIC CUDA::cudart)
