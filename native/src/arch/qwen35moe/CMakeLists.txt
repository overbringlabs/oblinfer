# qwen35moe: header-only for now. The INTERFACE target carries the pack's include root (so sources keep including
# "model/..." and "kernels/...") and the constants header; the OBJECT target compiles that header on every build.
# Nothing here is linked into ingine-serve unless INGINE_ARCH_PACK=qwen35moe.
add_library(qwen35moe_data INTERFACE)
target_include_directories(qwen35moe_data INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_sources(qwen35moe_data INTERFACE model/model_data.h)

add_library(qwen35moe_data_check OBJECT model/model_data_check.cpp)
target_link_libraries(qwen35moe_data_check PRIVATE qwen35moe_data)

# Host-only tensor-name classifier (and, later, the placement planner): compiled on every build so it cannot rot,
# and not linked into ingine-serve. The model_qwen35moe_names test links it.
add_library(qwen35moe_names OBJECT model/tensor_names.cpp)
target_include_directories(qwen35moe_names PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_names PRIVATE qwen35moe_data)

# Placement planner: dense slots and one packed expert block per MoE layer, from the checkpoint's real tensor
# sizes. Host only, like the name classifier.
add_library(qwen35moe_placement OBJECT model/placement.cpp)
target_include_directories(qwen35moe_placement PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_placement PRIVATE qwen35moe_data qwen35moe_names)

# Decode attention of the full-attention layers (head 256, 2 KV heads, BF16 cache) and the position-band plan
# that reads it from both cards at once. Compiled on every build like the other shared kernels; the
# kernels_qwen35moe_attn tests link it.
add_library(qwen35moe_attn OBJECT kernels/attn/decode_attn.cu kernels/attn/rope_norm.cu kernels/attn/kv_bands.cu)
ingine_cuda_target(qwen35moe_attn)
target_include_directories(qwen35moe_attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_attn PRIVATE qwen35moe_data CUDA::cudart)

# MoE decode: router (top-8 over 256 experts) and, next, the grouped expert kernels.
add_library(qwen35moe_moe OBJECT kernels/moe/router.cu kernels/moe/moe_apply.cu kernels/moe/shared_expert.cu
                                  kernels/moe/fused_experts.cu)
ingine_cuda_target(qwen35moe_moe)
target_include_directories(qwen35moe_moe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_moe PUBLIC ingine_kernels_quant PRIVATE qwen35moe_data CUDA::cudart)

# Streaming upload into the placement plan: index + headers from core/safetensors, chunked reads, pinned
# staging, one H2D per chunk. Needs CUDA and ingine_core.
add_library(qwen35moe_uploader OBJECT model/uploader.cpp)
ingine_cuda_target(qwen35moe_uploader)
target_include_directories(qwen35moe_uploader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_uploader PUBLIC ingine_core PRIVATE qwen35moe_data qwen35moe_placement CUDA::cudart)

# The decoder driver: layer dispatch over the verified kernels, KV caches, greedy sampling.
add_library(qwen35moe_decoder OBJECT model/decoder.cu model/decode_graph.cu)
ingine_cuda_target(qwen35moe_decoder)
target_include_directories(qwen35moe_decoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_decoder PUBLIC qwen35moe_uploader qwen35moe_moe qwen35moe_attn qwen35moe_batch
                                            ingine_kernels_gdn
                                            ingine_kernels_quant ingine_kernels_dense_ops ingine_core
                      PRIVATE qwen35moe_data CUDA::cudart)

# Verify-width (n-token) kernels of the S8 verify path: batched dense GEMVs, deduplicated routed experts,
# n-query full attention over one cache, and the register-staged GDN recurrence. Standalone by contract —
# nothing under kernels/moe, kernels/attn or model includes them, so agent C can build and test them while
# the decoder and the single-token kernels are being edited elsewhere.
add_library(qwen35moe_batch OBJECT kernels/batch/batch_gemv.cu kernels/batch/batch_moe.cu
                                    kernels/batch/batch_attn.cu kernels/batch/batch_gdn.cu)
ingine_cuda_target(qwen35moe_batch)
target_include_directories(qwen35moe_batch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_batch PUBLIC qwen35moe_attn ingine_kernels_quant
                      PRIVATE qwen35moe_data ingine_kernels_gdn CUDA::cudart)

# MTP drafter (spec-decode K4): the head's own Q4_0 K/V cache format on top of the shipped split/combine kernels
# and the BF16 routed-expert path the NVFP4 trunk kernels do not cover.
add_library(qwen35moe_mtp_kernels OBJECT kernels/mtp/mtp_attn.cu kernels/mtp/mtp_experts.cu)
ingine_cuda_target(qwen35moe_mtp_kernels)
target_include_directories(qwen35moe_mtp_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_mtp_kernels PUBLIC qwen35moe_attn qwen35moe_moe
                      PRIVATE qwen35moe_data CUDA::cudart)

# Loader, forward and cache of the MTP head, standalone: it takes the 19 tensors from a snapshot directory and
# borrows the trunk's embedding table and lm_head.
add_library(qwen35moe_mtp OBJECT model/mtp.cu)
ingine_cuda_target(qwen35moe_mtp)
target_include_directories(qwen35moe_mtp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_mtp PUBLIC qwen35moe_mtp_kernels qwen35moe_attn qwen35moe_moe
                      ingine_kernels_dense_ops ingine_kernels_quant ingine_core
                      PRIVATE qwen35moe_data CUDA::cudart)
