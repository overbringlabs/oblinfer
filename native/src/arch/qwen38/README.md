# Qwen3.8-27B pack

Pack identity: `Qwen3.8-27B` is `Qwen3_5ForConditionalGeneration` (`text_config.model_type` `qwen3_5_text`). The
constants in `model/model_data.h` come from the checkpoint config copied to
[`docs/arch/qwen3.8-27b.config.json`](../../../../docs/arch/qwen3.8-27b.config.json).

`Qwen3.6-27B` publishes the same architecture and the same `text_config` numbers, so it is the **same pack**.
`DaisyRose`/`Swift` builds belong here too as long as they keep this config.

This is **not** Inkling and **not** the `qwen35moe` pack:

- Inkling-2-Mini has 24 layers, 128 experts, hidden 2304 and a 48-visit loop plan. No loop here.
- `qwen35moe` (Qwen3.6-35B-A3B) is the MoE sibling: 40 layers, 256 experts, hidden 2048. That pack's router,
  top-8 dispatch and 512-wide shared expert do not apply.
- This model has **no experts**. Every layer runs one dense SwiGLU MLP of `intermediate_size` 17408.

What the two Qwen packs share is the hybrid attention recipe: a 3 linear / 1 full cycle, GDN-style linear
attention with `conv_k` 4, and full attention with `head_dim` 256, partial rotary and an output gate. The kernel
shapes differ (`LIN_VALUE_HEADS` 48 here against 32 there, different hidden and head counts).

## Status

Scaffolding only. What exists:

- `model/model_data.h` — constants, layer kind table (`LAYER_KIND`, `FULL_INDEX`, `LINEAR_INDEX`) and the MTP
  head shape. Header-only, no CUDA, no `core/` headers.
- `CMakeLists.txt` — `qwen38_data` INTERFACE target plus `qwen38_data_check` OBJECT target. The OBJECT target
  compiles `model_data.h` in the default build, so the constants are checked without linking the pack into
  `ingine-serve`.

## Constants

All values are `text_config` fields of `docs/arch/qwen3.8-27b.config.json` unless the note says derived or gives
another source.

| Constant | Value | Config key / note |
|---|---|---|
| `HIDDEN` | 5120 | `hidden_size` |
| `LAYERS` | 64 | `num_hidden_layers` |
| `VOCAB` | 248320 | `vocab_size` |
| `MAX_POS` | 262144 | `max_position_embeddings` |
| `EPS` | 1e-6 | `rms_norm_eps` |
| activation | SiLU | `hidden_act` |
| `INTERMEDIATE` | 17408 | `intermediate_size`, dense MLP on every layer |
| `FULL_ATTENTION_INTERVAL` | 4 (16 full, 48 linear) | `full_attention_interval`, `layer_types` |
| `HEADS` / `KV_HEADS` / `HEAD_DIM` | 24 / 4 / 256 | `num_attention_heads` / `num_key_value_heads` / `head_dim` |
| `Q_PROJ_ROWS` | 12288 | derived: `attn_output_gate` true, q_proj emits query + gate halves |
| `ROTARY_DIM` | 64 of 256 | derived from `partial_rotary_factor` 0.25 |
| `ROPE_THETA` | 1e7 | `rope_parameters.rope_theta` |
| `MROPE_SECTION` | 11, 11, 10 | `rope_parameters.mrope_section`, interleaved |
| `CONV_K` | 4 | `linear_conv_kernel_dim` |
| `LIN_KEY_HEADS` / `LIN_KEY_DIM` | 16 / 128 | `linear_num_key_heads` / `linear_key_head_dim` |
| `LIN_VALUE_HEADS` / `LIN_VALUE_DIM` | 48 / 128 | `linear_num_value_heads` / `linear_value_head_dim` |
| `LIN_CONV_DIM` | 10240 | derived: `2 * key_dim + value_dim` conv channels of the Qwen3-Next in_proj layout |
| recurrence dtype | FP32 | `mamba_ssm_dtype` |
| `MTP_LAYERS` | 1 | `mtp_num_hidden_layers`; `mtp_use_dedicated_embeddings` false |
| token ids | eos/bos 248044, image 248056, video 248057, vision start/end 248053/248054 | root and `text_config` |

Layer kinds: `layer_types` is a 3 linear / 1 full cycle, so `is_full_layer(l)` is `l % 4 == 3` (layers 3, 7, …,
63) and `FULL_INDEX` / `LINEAR_INDEX` give each layer its slot in the full-attention KV cache or the linear
attention state, `-1` on the other kind.

## Same recipe, different sizes

`Qwen3.5-9B` is the same hybrid recipe (`qwen3_5`, interval 4, GDN plus full attention) at smaller sizes: hidden
4096, 32 layers, 16 / 4 attention heads, 32 linear value heads. None of those are hardcoded here; this pack's
constants are the 27B shape only. If the kernels are templated on the constants later, the 9B variant can reuse
them; this binary does not serve it.

## Not implemented yet

- Linear attention (GDN) kernels: conv1d over `CONV_K = 4`, the gated delta-rule recurrence in FP32, per-layer
  state.
- Full attention kernels for head_dim 256, 4 KV heads, partial rotary (64 dims) and the output gate.
- Dense MLP over `INTERMEDIATE` 17408, weight loading.
- MTP head and any speculative decode path for it.
- Tokenizer, chat template, vision tower (`vision_config`: depth 27, hidden 1152, out_hidden 5120, intermediate
  4304, spatial merge 2).
- Any residency layout for two 16 GB cards.
- `src/arch/qwen38/model/` and `kernels/` targets: `src/arch/<pack>/model` and `src/arch/<pack>/kernels/<area>`
  are picked up by the top-level CMake only once those directories exist.

## Build wiring

The default pack stays `inkling`. `native/CMakeLists.txt` adds this directory unconditionally, so the constants
header is always compiled, but it links `qwen38_data` into `ingine-serve` only when the pack is selected:

```bash
# default: Inkling, ingine-serve links ingine_model
cmake --preset release

# the header-only pack: ingine-serve builds --fake only, ingine-bench is skipped, and ingine-validate does not
# build (it includes the Inkling pack's model headers) until this pack has a model target
cmake -S native -B native/build-qwen38 -DINGINE_ARCH_PACK=qwen38
```
