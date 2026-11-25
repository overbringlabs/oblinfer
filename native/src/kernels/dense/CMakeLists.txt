# Unquantized dense ops shared by the packs: bf16 GEMV, RMSNorm, embedding rows. Compiled on every build.
add_library(ingine_kernels_dense_ops OBJECT dense_ops.cu)
ingine_cuda_target(ingine_kernels_dense_ops)
target_include_directories(ingine_kernels_dense_ops PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_dense_ops PUBLIC CUDA::cudart)
