# Qwen3.6-35B-A3B pack

Pack identity: `Qwen3.6-35B-A3B` and `Ornith-1.5-35B-A3B-NVFP4` are the same architecture,
`Qwen3_5MoeForConditionalGeneration` (`text_config.model_type` `qwen3_5_moe_text`). The constants in
`model/model_data.h` come from the checkpoint config copied to
[`docs/arch/qwen3.6-35b-a3b.config.json`](../../../../docs/arch/qwen3.6-35b-a3b.config.json).

This is **not** Inkling. Inkling-2-Mini has 24 layers, 128 experts, hidden 2304 and a 48-visit loop plan; this
model has 40 plain layers, 256 experts, hidden 2048 and no loop. The Inkling visit plan, its `moe.cu` expert
format and its block layout do not apply here; the pack needs its own model and kernel sources.

## Status

Scaffolding only. What exists:

- `model/model_data.h` — constants, layer kind table (`LAYER_KIND`, `FULL_INDEX`, `LINEAR_INDEX`) and the MTP
  head shape. Header-only, no CUDA, no `core/` headers.
- `CMakeLists.txt` — `qwen35moe_data` INTERFACE target plus `qwen35moe_data_check` OBJECT target. The OBJECT
  target compiles `model_data.h` in the default build, so the constants are checked without linking the pack
  into `ingine-serve`.

## Constants

All values are `text_config` fields of `docs/arch/qwen3.6-35b-a3b.config.json` unless the note says derived or
gives another source.

| Constant | Value | Config key / note |
|---|---|---|
| `HIDDEN` | 2048 | `hidden_size` |
| `LAYERS` | 40 | `num_hidden_layers` |
| `VOCAB` | 248320 | `vocab_size` |
| `MAX_POS` | 262144 | `max_position_embeddings` |
| `EPS` | 1e-6 | `rms_norm_eps` |
| activation | SiLU | `hidden_act` |
| `FULL_ATTENTION_INTERVAL` | 4 (10 full, 30 linear) | `full_attention_interval`, `layer_types` |
| `HEADS` / `KV_HEADS` / `HEAD_DIM` | 16 / 2 / 256 | `num_attention_heads` / `num_key_value_heads` / `head_dim` |
| `Q_PROJ_ROWS` | 8192 | derived: `attn_output_gate` true, q_proj emits query + gate halves |
| `ROTARY_DIM` | 64 of 256 | derived from `partial_rotary_factor` 0.25 |
| `ROPE_THETA` | 1e7 | `rope_parameters.rope_theta` |
| `MROPE_SECTION` | 11, 11, 10 | `rope_parameters.mrope_section`, interleaved |
| `EXPERTS` / `TOP_K` | 256 / 8 | `num_experts` / `num_experts_per_tok` |
| `MOE_INTER` | 512 | `moe_intermediate_size` |
| `SHARED_INTER` | 512 | `shared_expert_intermediate_size` |
| `CONV_K` | 4 | `linear_conv_kernel_dim` |
| `LIN_KEY_HEADS` / `LIN_KEY_DIM` | 16 / 128 | `linear_num_key_heads` / `linear_key_head_dim` |
| `LIN_VALUE_HEADS` / `LIN_VALUE_DIM` | 32 / 128 | `linear_num_value_heads` / `linear_value_head_dim` |
| `LIN_CONV_DIM` | 8192 | derived: `2 * key_dim + value_dim` conv channels of the Qwen3-Next in_proj layout |
| recurrence dtype | FP32 | `mamba_ssm_dtype` |
| `MTP_LAYERS` | 1 | `mtp_num_hidden_layers`; `mtp_use_dedicated_embeddings` false |
| token ids | eos/bos 248044, image 248056, video 248057, vision start/end 248053/248054 | root and `text_config` |

Layer kinds: `layer_types` is a 3 linear / 1 full cycle, so `is_full_layer(l)` is `l % 4 == 3` (layers 3, 7, …,
39) and `FULL_INDEX` / `LINEAR_INDEX` give each layer its slot in the full-attention KV cache or the linear
attention state, `-1` on the other kind.

## Not implemented yet

- Linear attention (GDN) kernels: conv1d over `CONV_K = 4`, the gated delta-rule recurrence in FP32, per-layer
  state.
- Full attention kernels for head_dim 256, 2 KV heads, partial rotary (64 dims) and the output gate.
- MoE: router over 256 experts, top-8 dispatch, the 512-wide shared expert, expert weight loading.
- MTP head and any speculative decode path for it.
- NVFP4 weight loading for the Ornith checkpoint (`moe.cu` under `src/arch/inkling/` is Inkling's layout).
- Tokenizer, chat template, vision tower (`vision_config`: depth 27, hidden 1152, out_hidden 2048).
- Dual-GPU expert split and the KV hop; this pack has no residency layout yet.
- `src/arch/qwen35moe/model/` and `kernels/` targets: `src/arch/<pack>/model` and
  `src/arch/<pack>/kernels/<area>` are picked up by the top-level CMake only once those directories exist.

## Build wiring

The default pack stays `inkling`. `native/CMakeLists.txt` adds this directory unconditionally, so the constants
header is always compiled, but it links `qwen35moe_data` into `ingine-serve` only when the pack is selected:

```bash
# default: Inkling, ingine-serve links ingine_model
cmake --preset release

# the header-only pack: ingine-serve builds --fake only, ingine-bench is skipped, and ingine-validate does not
# build (it includes the Inkling pack's model headers) until this pack has a model target
cmake -S native -B native/build-qwen35moe -DINGINE_ARCH_PACK=qwen35moe
```
