# Decode step of the Qwen3.x gated delta-rule linear attention, shared by every hybrid pack (qwen38 and
# qwen35moe publish the same linear-attention shapes). Compiled on every build so it cannot rot; the kernels_gdn
# test links it, and a pack's model target will once one exists.
add_library(ingine_kernels_gdn OBJECT gdn.cu conv.cu gates.cu)
ingine_cuda_target(ingine_kernels_gdn)
target_include_directories(ingine_kernels_gdn PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_gdn PUBLIC CUDA::cudart)
