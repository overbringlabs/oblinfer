# Decode kernel tests against FP64 host oracles (synthetic weights at the model's shapes, < 1 GB of GPU memory).
ingine_add_test(NAME kernels_dense SOURCES test_dense.cpp LIBS ingine_kernels_dense GPU ORACLE TIMEOUT 600)
ingine_add_test(NAME kernels_moe SOURCES test_moe.cpp LIBS ingine_kernels_moe GPU ORACLE TIMEOUT 600)
ingine_add_test(NAME kernels_attn SOURCES test_attn.cpp LIBS ingine_kernels_attn GPU ORACLE TIMEOUT 900)
ingine_add_test(NAME kernels_head SOURCES test_head.cpp LIBS ingine_kernels_head GPU ORACLE TIMEOUT 600)
# every output bit of the M-wide weight-streaming and sliding-attention kernels (~1 GB of GPU memory)
ingine_add_test(NAME kernels_bitlock SOURCES test_bitlock.cpp
  LIBS ingine_kernels_dense ingine_kernels_moe ingine_kernels_attn GPU TIMEOUT 600)

# qwen38 GDN decode step: the FP64 reference runs everywhere (no GPU label, so ctest -LE gpu runs it); the
# CUDA comparison inside skips until an sm_120 device runs the sm_120a build.
ingine_add_test(NAME kernels_gdn SOURCES test_gdn.cpp LIBS ingine_kernels_gdn CUDA::cudart ORACLE TIMEOUT 300)
# conv1d + SiLU and the gated RMSNorm around the delta rule: FP64 halves run everywhere, the CUDA comparison
# skips off sm_120
ingine_add_test(NAME kernels_gdn_conv SOURCES test_gdn_conv.cpp LIBS ingine_kernels_gdn CUDA::cudart ORACLE TIMEOUT 300)

# Qwen3.6-35B-A3B decode attention (head 256, 2 KV heads, BF16 cache) against the FP64 reference; the CUDA
# comparison runs only on sm_120.
if(TARGET qwen35moe_attn)
  ingine_add_test(NAME kernels_qwen35moe_attn SOURCES test_qwen35moe_attn.cpp
                  LIBS qwen35moe_attn CUDA::cudart ORACLE TIMEOUT 300)
endif()

# q/k RMSNorm + partial rotary and the output gate of the Qwen3.6-35B-A3B attention layers
if(TARGET qwen35moe_attn)
  ingine_add_test(NAME kernels_qwen35moe_rope_norm SOURCES test_qwen35moe_rope_norm.cpp
                  LIBS qwen35moe_attn CUDA::cudart ORACLE TIMEOUT 300)
endif()

# FP8 e4m3 and NVFP4 e2m1 decode GEMV against byte-identical CPU references
if(TARGET ingine_kernels_quant)
  ingine_add_test(NAME kernels_quant_gemv SOURCES test_quant_gemv.cpp
                  LIBS ingine_kernels_quant CUDA::cudart ORACLE TIMEOUT 300)
endif()

# Qwen3.6-35B-A3B router: 256 bf16 logits, top 8, softmax over the selected logits
if(TARGET qwen35moe_moe)
  ingine_add_test(NAME kernels_qwen35moe_router SOURCES test_qwen35moe_router.cpp
                  LIBS qwen35moe_moe ingine_kernels_quant CUDA::cudart ORACLE TIMEOUT 300)
endif()

# Qwen3.6-35B-A3B routed-expert math: silu(gate x) * (up x), summed with the router weights
if(TARGET qwen35moe_moe)
  ingine_add_test(NAME kernels_qwen35moe_moe SOURCES test_qwen35moe_moe.cpp
                  LIBS qwen35moe_moe ingine_kernels_quant CUDA::cudart ORACLE TIMEOUT 300)
endif()

# Qwen3.6-35B-A3B shared expert: sigmoid(shared_expert_gate x) * down(silu(gate x) * up x)
if(TARGET qwen35moe_moe)
  ingine_add_test(NAME kernels_qwen35moe_shared_expert SOURCES test_qwen35moe_shared_expert.cpp
                  LIBS qwen35moe_moe ingine_kernels_quant CUDA::cudart ORACLE TIMEOUT 300)
endif()

# Unquantized dense ops: bf16 GEMV, RMSNorm, embedding lookup
if(TARGET ingine_kernels_dense_ops)
  ingine_add_test(NAME kernels_dense_ops SOURCES test_dense_ops.cpp
                  LIBS ingine_kernels_dense_ops CUDA::cudart ORACLE TIMEOUT 300)
endif()
