ingine_add_test(NAME model_data SOURCES test_model_data.cpp LIBS ingine_core)
# both Qwen3.6-35B-A3B export dialects map onto one set of typed tensor keys (host only)
ingine_add_test(NAME model_qwen35moe_names SOURCES test_qwen35moe_names.cpp LIBS qwen35moe_names TIMEOUT 120)
# dense slots and packed expert blocks from both export dialects (host only)
ingine_add_test(NAME model_qwen35moe_placement SOURCES test_qwen35moe_placement.cpp
                LIBS qwen35moe_placement qwen35moe_names TIMEOUT 120)
# one real layer at a time (~1.1 GB of GPU memory)
ingine_add_test(NAME model_layer SOURCES test_layer.cpp LIBS ingine_model GPU MODEL TIMEOUT 600)
# the decode runtime (graphs, verify widths, rollback, pipelined steps, device-selected verify) and the Generator's
# stop/cancel/budget exactness on a synthetic model (< 2 GB of GPU memory)
ingine_add_test(NAME model_runtime SOURCES test_runtime.cpp LIBS ingine_model GPU TIMEOUT 600)
# prompt checkpoints with speculative decoding on the full model (~22 GB of GPU memory)
ingine_add_test(NAME model_reuse SOURCES test_reuse.cpp LIBS ingine_model GPU MODEL TIMEOUT 900)
# expert residency across two devices: the split table, the shards on the second device, the same bytes as a
# plain load_layer (~16.5 GB of GPU memory across two cards; opt in with INGINE_RESIDENCY_TEST=1)
ingine_add_test(NAME model_residency SOURCES test_residency.cpp LIBS ingine_model GPU MODEL TIMEOUT 1200)
# DSpark over the eager two-device step: a greedy speculative generation walks the plain decode's tokens, and
# the prompt is prefilled one token per decode step (two model loads, ~16.5 GB across both cards;
# INGINE_RESIDENCY_TEST=1)
ingine_add_test(NAME model_ep_spec SOURCES test_ep_spec.cpp LIBS ingine_model GPU MODEL TIMEOUT 1800)
# full-model parity against the Python engine's goldens (tools/goldens_model.py); ~20 GB of GPU memory
if(TARGET ingine-validate)
  set(_validate ingine-validate --model-dir ${INGINE_TEST_MODEL_DIR} --golden ${PROJECT_SOURCE_DIR}/tests/golden/model)
  add_test(NAME model_parity COMMAND ${_validate})
  # exact greedy continuations only where the goldens have no near ties (long and FP8-cache cases flip at ties)
  set(_greedy_cases --cases kv16_chat,kv16_t300 --min-top1 1)
  add_test(NAME model_parity_generate COMMAND ${_validate} --generate ${_greedy_cases})
  add_test(NAME model_parity_spec COMMAND ${_validate} --generate --spec ${_greedy_cases})
  # long greedy paths of the Generator teacher-forced through a plain width-1 runtime: plain decoding exact;
  # speculative decoding as a rate over the four FP8-cache paths (12,000 tokens): single positions move by
  # several logits between kernel widths (30,000 tokens over all golden cases: 13-14 above 0.5 logits, 3 above
  # 2, in old and new verify kernels alike), a real bug moves most tokens
  add_test(NAME model_path COMMAND ${_validate} --path 1500 --cases abl_kv8_t300)
  add_test(NAME model_path_spec COMMAND ${_validate} --path 3000 --spec --cases abl_kv8 --max-gap-rate 15)
  set_tests_properties(model_parity model_parity_generate model_parity_spec model_path model_path_spec PROPERTIES
    LABELS "gpu;model" RESOURCE_LOCK gpu TIMEOUT 3600)
endif()
# image input on the full model: colours and text of images, prompt caches with images, an image over several
# prefill chunks (~22 GB of GPU memory)
if(TARGET ingine_serve AND TARGET ingine_kernels_vision)
  ingine_add_test(NAME model_vision SOURCES test_vision_generate.cpp LIBS ingine_model ingine_serve
                  LABELS vision GPU MODEL TIMEOUT 900)
endif()

# End-to-end load of the real 23 GB checkpoint through the placement plan; skipped unless QWEN35MOE_SNAPSHOT is set
ingine_add_test(NAME model_qwen35moe_upload SOURCES test_qwen35moe_upload.cpp
                LIBS qwen35moe_uploader qwen35moe_placement qwen35moe_names ingine_core CUDA::cudart
                TIMEOUT 900)

# One-token decoder smoke test on the real checkpoint (needs QWEN35MOE_SNAPSHOT)
# OBJECT libraries do not propagate their objects, so every one the decoder calls is listed here.
ingine_add_test(NAME model_qwen35moe_decoder SOURCES test_qwen35moe_decoder.cpp
                LIBS qwen35moe_decoder qwen35moe_uploader qwen35moe_moe qwen35moe_attn
                     qwen35moe_placement qwen35moe_names ingine_kernels_gdn ingine_kernels_quant
                     ingine_kernels_dense_ops ingine_core CUDA::cudart TIMEOUT 1800)

# Depth benchmark: prefill then decode, tok/s only (QWEN35MOE_DEPTH, QWEN35MOE_GEN)
ingine_add_test(NAME model_qwen35moe_bench SOURCES test_qwen35moe_bench.cpp
                LIBS qwen35moe_decoder qwen35moe_uploader qwen35moe_moe qwen35moe_attn
                     qwen35moe_placement qwen35moe_names ingine_kernels_gdn ingine_kernels_quant
                     ingine_kernels_dense_ops ingine_core CUDA::cudart TIMEOUT 5400)
