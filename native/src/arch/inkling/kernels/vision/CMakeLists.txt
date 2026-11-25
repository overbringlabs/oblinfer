# ingine_kernels_vision: elementwise kernels of the vision tower (its matrix products use
# kernels::prefill::Gemm).
add_library(ingine_kernels_vision STATIC
  vision.cu
)
ingine_cuda_target(ingine_kernels_vision)
ingine_internal_includes(ingine_kernels_vision)
target_include_directories(ingine_kernels_vision PUBLIC
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/arch/inkling)
target_link_libraries(ingine_kernels_vision PUBLIC ingine_core CUDA::cudart)
