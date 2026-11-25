# Qwen3.5/3.8 dense-hybrid pack

Pack identity: `Qwen3.8-27B`, `Qwen3.6-27B`, `Qwen3.5-9B` and `Qwen3.5-4B` are the same architecture,
`Qwen3_5ForConditionalGeneration` (`text_config.model_type` `qwen3_5_text`). `Qwen3.8-27B` and `Qwen3.6-27B`
publish the same `text_config`, so they are the same shape and the same pack. `DaisyRose`/`Swift` builds belong
here too as long as they keep that config. The constants in `model/model_data.h` come from the checkpoint configs
copied to [`docs/arch/qwen3.8-27b.config.json`](../../../../docs/arch/qwen3.8-27b.config.json),
[`docs/arch/qwen3.5-9b.config.json`](../../../../docs/arch/qwen3.5-9b.config.json) and
[`docs/arch/qwen3.5-4b.config.json`](../../../../docs/arch/qwen3.5-4b.config.json).

This is **not** Inkling and **not** the `qwen35moe` pack:

- Inkling-2-Mini has 24 layers, 128 experts, hidden 2304 and a 48-visit loop plan. No loop here.
- `qwen35moe` (Qwen3.6-35B-A3B) is the MoE sibling: 40 layers, 256 experts, hidden 2048. That pack's router,
  top-8 dispatch and 512-wide shared expert do not apply.
- Every shape here has **no experts**. Every layer runs one dense SwiGLU MLP of `intermediate_size` width.

What the two Qwen packs share is the hybrid attention recipe: a 3 linear / 1 full cycle, GDN-style linear
attention with `conv_k` 4, and full attention with `head_dim` 256, partial rotary and an output gate. The kernel
shapes differ (`LIN_VALUE_HEADS` 48 on the 27B against 32 on both small shapes, different hidden and head counts).

## One pack, three shapes

`Shape<...>` in `model/model_data.h` derives every dependent constant from the checkpoint's per-shape fields.

| Shape | HIDDEN | LAYERS (linear + full) | HEADS / KV | INTERMEDIATE | LIN_VALUE_HEADS | tie_word_embeddings |
|---|---|---|---|---|---|---|
| `Shape27` (`Qwen3.8-27B` = `Qwen3.6-27B`) | 5120 | 64 (48 + 16) | 24 / 4 | 17408 | 48 | false |
| `Shape9` (`Qwen3.5-9B`) | 4096 | 32 (24 + 8) | 16 / 4 | 12288 | 32 | false |
| `Shape4` (`Qwen3.5-4B`) | 2560 | 32 (24 + 8) | 16 / 4 | 9216 | 32 | true |

The 9B and 4B share layer counts and GDN head counts; only hidden size and MLP width differ. All three share
`VOCAB` 248320, `HEAD_DIM` 256, `CONV_K` 4, `LIN_KEY_HEADS` 16 / `LIN_KEY_DIM` 128, `LIN_VALUE_DIM` 128, partial
rotary 0.25 (`ROTARY_DIM` 64), `ROPE_THETA` 1e7, mrope `[11, 11, 10]` and one MTP layer.

The derived widths therefore split in two:

| Derived | 27B | 9B | 4B |
|---|---|---|---|
| `Q_DIM` / `Q_PROJ_ROWS` | 6144 / 12288 | 4096 / 8192 | 4096 / 8192 |
| `KV_DIM` | 1024 | 1024 | 1024 |
| `LIN_KEY_DIM_TOT` | 2048 | 2048 | 2048 |
| `LIN_VALUE_DIM_TOT` | 6144 | 4096 | 4096 |
| `LIN_CONV_DIM` | 10240 | 8192 | 8192 |

The shapes are selected at compile time, not at runtime. This is **not a safetensors swap**: one binary compiles
for one `Qwen38Shape` and its kernels instantiate on that shape's constants. Serving a different size means a
rebuild with another `INGINE_QWEN38_SHAPE`, not pointing the same binary at other weights. Kernels are written as
`template <class S>` and read `S::HIDDEN`, `S::LAYERS`, `S::Q_DIM`, `S::LAYER_KIND` and friends; one templated
kernel set serves 9B and 4B (and 27B where the width is the only difference) without per-shape source forks.

## Status

Scaffolding only. What exists:

- `model/model_data.h` — shared recipe constants, the `Shape<...>` template with `Shape27` / `Shape9` / `Shape4`,
  layer kind table (`LAYER_KIND`, `FULL_INDEX`, `LINEAR_INDEX`), the MTP head shape and the `Qwen38Shape`
  typedef. Header-only, no CUDA, no `core/` headers.
- `CMakeLists.txt` — `qwen38_data` INTERFACE target plus `qwen38_data_check` OBJECT target. The OBJECT target
  compiles `model_data.h` and instantiates all three shapes in the default build, so a wrong constant or layer
  table fails without linking the pack into `ingine-serve`.
- Back-compat aliases: bare `HIDDEN`, `LAYERS`, `HEADS`, `LAYER_KIND`, … resolve to `Shape27` (`Defaults`) so
  the existing names keep compiling.

## Shared constants

All values are `text_config` fields of the configs above unless the note says derived or gives another source.

| Constant | Value | Config key / note |
|---|---|---|
| `VOCAB` | 248320 | `vocab_size`, all shapes |
| `MAX_POS` | 262144 | `max_position_embeddings` |
| `EPS` | 1e-6 | `rms_norm_eps` |
| activation | SiLU | `hidden_act` |
| `FULL_ATTENTION_INTERVAL` | 4 | `full_attention_interval`, `layer_types` |
| `HEAD_DIM` | 256 | `head_dim` |
| `ROTARY_DIM` | 64 of 256 | derived from `partial_rotary_factor` 0.25 |
| `ROPE_THETA` | 1e7 | `rope_parameters.rope_theta` |
| `MROPE_SECTION` | 11, 11, 10 | `rope_parameters.mrope_section`, interleaved |
| `CONV_K` | 4 | `linear_conv_kernel_dim` |
| `LIN_KEY_HEADS` / `LIN_KEY_DIM` | 16 / 128 | `linear_num_key_heads` / `linear_key_head_dim` |
| `LIN_VALUE_DIM` | 128 | `linear_value_head_dim` |
| recurrence dtype | FP32 | `mamba_ssm_dtype` |
| `MTP_LAYERS` | 1 | `mtp_num_hidden_layers`; `mtp_use_dedicated_embeddings` false |
| token ids | eos/bos 248044, image 248056, video 248057, vision start/end 248053/248054 | root and `text_config` |

Layer kinds: `layer_types` is a 3 linear / 1 full cycle in every shape, so `is_full_layer(l)` is `l % 4 == 3`
(layers 3, 7, …, `LAYERS - 1`) and each shape's `FULL_INDEX` / `LINEAR_INDEX` give the layer its slot in the
full-attention KV cache or the linear attention state, `-1` on the other kind.

## Not implemented yet

- Linear attention (GDN) kernels: conv1d over `CONV_K = 4`, the gated delta-rule recurrence in FP32, per-layer
  state.
- Full attention kernels for head_dim 256, 4 KV heads, partial rotary (64 dims) and the output gate.
- Dense MLP over each shape's `INTERMEDIATE`, weight loading.
- MTP head and any speculative decode path for it.
- Tokenizer, chat template, vision tower (`vision_config`: 27B/9B are depth 27, hidden 1152, intermediate 4304
  with out_hidden 5120/4096; the 4B is depth 24, hidden 1024, intermediate 4096, out_hidden 2560).
- Any residency layout for two 16 GB cards.
- `src/arch/qwen38/model/` and `kernels/` targets: `src/arch/<pack>/model` and `src/arch/<pack>/kernels/<area>`
  are picked up by the top-level CMake only once those directories exist.

## Build wiring

The default pack stays `inkling`. `native/CMakeLists.txt` adds this directory unconditionally, so the constants
header is always compiled, but it links `qwen38_data` into `ingine-serve` only when the pack is selected. The
default shape is the 27B.

```bash
# default: Inkling, ingine-serve links ingine_model
cmake --preset release

# the header-only pack, 27B shape (Shape27)
cmake -S native -B native/build-qwen38 -DINGINE_ARCH_PACK=qwen38

# the same pack built for 9B (Shape9) or 4B (Shape4); pick one shape per build
cmake -S native -B native/build-qwen38-9b -DINGINE_ARCH_PACK=qwen38 -DINGINE_QWEN38_SHAPE=9
cmake -S native -B native/build-qwen38-4b -DINGINE_ARCH_PACK=qwen38 -DINGINE_QWEN38_SHAPE=4
```

`INGINE_QWEN38_SHAPE` (27, 9 or 4) becomes the `INGINE_QWEN38_SHAPE` compile definition on `qwen38_data`;
`Qwen38Shape` then names `Shape27`, `Shape9` or `Shape4`. A plain `-DINGINE_QWEN38_SHAPE=9` compile definition
also works. With the header-only pack, `ingine-serve` builds `--fake` only, `ingine-bench` is skipped, and
`ingine-validate` does not build (it includes the Inkling pack's model headers) until this pack has a model
target.
