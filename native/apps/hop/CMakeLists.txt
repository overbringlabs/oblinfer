# ingine-hop: EP step 3 probe. Measures the pinned-host expert hop between two devices; needs two GPUs.
add_executable(ingine-hop main.cu)
ingine_internal_includes(ingine-hop)
ingine_cuda_target(ingine-hop)
target_link_libraries(ingine-hop PRIVATE CUDA::cudart)
