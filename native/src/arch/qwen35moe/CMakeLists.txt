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

# Decode attention of the full-attention layers (head 256, 2 KV heads, BF16 cache). Compiled on every build
# like the other shared kernels; the kernels_qwen35moe_attn test links it.
add_library(qwen35moe_attn OBJECT kernels/attn/decode_attn.cu kernels/attn/rope_norm.cu)
ingine_cuda_target(qwen35moe_attn)
target_include_directories(qwen35moe_attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_attn PRIVATE qwen35moe_data CUDA::cudart)

# MoE decode: router (top-8 over 256 experts) and, next, the grouped expert kernels.
add_library(qwen35moe_moe OBJECT kernels/moe/router.cu kernels/moe/moe_apply.cu kernels/moe/shared_expert.cu)
ingine_cuda_target(qwen35moe_moe)
target_include_directories(qwen35moe_moe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen35moe_moe PUBLIC ingine_kernels_quant PRIVATE qwen35moe_data CUDA::cudart)
