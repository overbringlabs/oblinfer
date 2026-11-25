# Nemotron 3.5 Lightning 30B-A3B pack

Pack identity: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` is `NemotronHForCausalLM`
(`model_type` `nemotron_h`). The constants in `model/model_data.h` come from the NVFP4 checkpoint config copied to
[`docs/arch/nemotron-lightning-30b-a3b.config.json`](../../../../docs/arch/nemotron-lightning-30b-a3b.config.json).
The NVFP4 card publishes `max_position_embeddings` 1048576; the BF16 card publishes 262144, so a BF16 build would
need a different `MAX_POS`.

This is **not** Inkling and **not** either Qwen pack:

- Inkling-2-Mini has 24 looped layers and no Mamba or expert blocks. This model has 52 single-visit layers and
  three block kinds.
- `qwen38` is dense with 3 linear / 1 full attention; `qwen35moe` is 40 layers, 256 experts, top-8, GDN linear
  attention. This model is 52 layers, 128 experts, top-6, Mamba-2 state-space layers and only six KV-cache layers.

## Not Bolt

Bolt E4B is `Gemma4ForConditionalGeneration` — a different family, not this pack.

Bolt 74B-A9B is the same *recipe* at other sizes: its LLM core is `NemotronHPuzzleForCausalLM` with hidden 4096,
88 layers (40 Mamba + 40 MoE + 8 attention), `mamba_num_heads` 128, and the same `conv_kernel` 4, `ssm_state_size`
128, 2 KV heads and `head_dim` 128, wrapped in a vision omni model. It would be a future `Shape74` like the
`qwen38` pack's `Shape9`: same `Shape<...>` and layer-kind machinery, different block table and sizes. Nothing in
this scaffold hardcodes 74B constants, and the default shape is Lightning 30B-A3B (`Shape30`).

## One pack, one shape

`Shape<...>` in `model/model_data.h` takes the fields that differ between checkpoints and derives the rest.
`layers_block_type` is passed as a table, not a formula: Lightning's block order is irregular.

| Shape | HIDDEN | LAYERS (Mamba/MoE/attn) | HEADS/KV/DIM | EXPERTS/TOP_K | MOE/SHARED_INTER | MAMBA HEADS x DIM |
|---|---|---|---|---|---|---|
| `Shape30` (Lightning 30B-A3B) | 2688 | 52 (23 + 23 + 6) | 32 / 2 / 128 | 128 / 6 | 1856 / 3712 | 64 x 64 |

Derived: `Q_DIM` 4096, `KV_DIM` 256, `ROTARY_DIM` 128 (full rotary), `MTP_LAYER_BEGIN` 52. Mamba-2 widths
follow the reference mixer (`mamba_num_heads * mamba_head_dim`): `MAMBA_INNER` 4096, `MAMBA_CONV_DIM` 6144
(`inner + 2 * n_groups * ssm_state_size`), `MAMBA_IN_PROJ_ROWS` 10304 (`gate + conv input + dt`) and
`MAMBA_NORM_GROUP` 512 (`inner / n_groups`). These match the checkpoint tensors: `in_proj` [10304, 2688],
`conv1d` [6144, 1, 4], `norm` [4096], `q_proj` [4096, 2688], `k_proj` / `v_proj` [256, 2688] and `o_proj`
[2688, 4096].

## Constants

All values are config fields of `docs/arch/nemotron-lightning-30b-a3b.config.json` unless the note says derived or
gives another source.

| Constant | Value | Config key / note |
|---|---|---|
| `VOCAB` | 131072 | `vocab_size` |
| `MAX_POS` | 1048576 | `max_position_embeddings`, NVFP4 card (BF16 card: 262144) |
| `EPS` | 1e-5 | `norm_eps` and `layer_norm_epsilon` |
| token ids | pad 0, bos 1, eos 2 | `pad_token_id` / `bos_token_id` / `eos_token_id` |
| `HEADS` / `KV_HEADS` / `HEAD_DIM` | 32 / 2 / 128 | `num_attention_heads` / `num_key_value_heads` / `head_dim` |
| `ROTARY_DIM` | 128 of 128 | derived from `partial_rotary_factor` 1.0 |
| `ROPE_THETA` | 1e4 | `rope_theta` |
| attention bias | false | `attention_bias`, `use_bias` |
| sliding window | none (full causal) | `sliding_window` null |
| `MAMBA_HEADS` / `MAMBA_HEAD_DIM` | 64 / 64 | `mamba_num_heads` / `mamba_head_dim` |
| `SSM_STATE_SIZE` | 128 | `ssm_state_size` |
| `CONV_KERNEL` | 4 | `conv_kernel` |
| `MAMBA_EXPAND` | 2 | `expand`; the reference mixer width is `MAMBA_HEADS * MAMBA_HEAD_DIM`, not this field |
| `MAMBA_CHUNK_SIZE` | 128 | `chunk_size` |
| Mamba biases | conv true, projection false | `use_conv_bias` / `mamba_proj_bias` |
| `MAMBA_GROUPS` | 8 | `n_groups`, Mamba evolution-matrix groups |
| `TIME_STEP_MIN` / `MAX` / `FLOOR` | 1e-3 / 1e-1 / 1e-4 | `time_step_min` / `time_step_max` / `time_step_floor` |
| Mamba activation / state | SiLU / FP32 | `mamba_hidden_act` / `mamba_ssm_cache_dtype` |
| `EXPERTS` / `TOP_K` | 128 / 6 | `n_routed_experts` / `num_experts_per_tok` |
| `SHARED_EXPERTS` | 1 | `n_shared_experts` |
| `MOE_INTER` | 1856 | `moe_intermediate_size`, per routed expert |
| `SHARED_INTER` | 3712 | `moe_shared_expert_intermediate_size` |
| `MOE_GROUPS` / `MOE_TOPK_GROUPS` | 1 / 1 | `n_group` / `topk_group` |
| `NORM_TOPK_PROB` | true | `norm_topk_prob` |
| `ROUTED_SCALING_FACTOR` | 2.5 | `routed_scaling_factor` |
| `SHARED_EXPERT_OVERLAP` | true | `moe_shared_expert_overlap` |
| MoE activation | relu2 | `mlp_hidden_act` |
| `RESCALE_PRE_NORM_RESIDUAL` / `RESIDUAL_IN_FP32` | true / false | `rescale_prenorm_residual` / `residual_in_fp32` |
| `NEXTN_PREDICT_LAYERS` | 1 | `num_nextn_predict_layers`, one MTP prediction step |
| `MTP_LAYERS` | 2 | physical `mtp.layers.0` attention + `mtp.layers.1` MoE; length of `mtp_layers_block_type` |
| MTP block types | attention, MoE | `mtp_layers_block_type` |
| `TIED_EMBEDDINGS` | false | `tie_word_embeddings` |

`intermediate_size` is also 1856, the same width as `moe_intermediate_size`; the model has no dense MLP block
kind. `n_groups` 8 is the Mamba evolution-matrix group count (`MAMBA_GROUPS`) in the reference config, not a MoE
routing field; MoE routing uses `n_group` 1 and `topk_group` 1. `moe_latent_size` is null, so experts use
`hidden_size` directly, and the reference mixer does not use `expand` for the Mamba width. The router gate is
[128, 2688] F32 with a 128-wide `e_score_correction_bias` buffer and sigmoid scores.

Quantization is `MIXED_PRECISION`: Mamba `in_proj` / `out_proj` are FP8, routed and shared experts plus `lm_head`
are NVFP4 W4A16 with group size 16, and the KV cache is FP8. Embeddings, `conv1d`, router gates, attention q/k/v/o
projections and the MTP tensors are unquantized (`ignore`).

## Layer kinds

`LayerKind` is `Mamba`, `Moe` or `Attention`. The 52-entry `LIGHTNING_BLOCK_TYPE` table is `layers_block_type`
verbatim; it starts and ends on non-attention blocks:

| Kind | Count | Layer indices | Slot table |
|---|---:|---|---|
| Mamba | 23 | all layers except the attention and MoE ones | `MAMBA_INDEX` |
| MoE | 23 | all layers except the attention and Mamba ones | `MOE_INDEX` |
| Attention | 6 | 5, 12, 19, 26, 33, 42 | `ATTENTION_INDEX` |

Each slot table gives a layer's position in that kind's weights/state and is `-1` on the other kinds, the way the
Inkling pack's `RING_INDEX` / `FULL_INDEX` map visits. Only `ATTENTION_INDEX` allocates KV cache; the Mamba layers
keep a recurrence state and the MoE layers keep no per-token state.

The MTP module is one prediction step after the decoder (logical `MTP_LAYER_BEGIN` 52) with two physical layers
named `mtp.layers.0` (attention) and `mtp.layers.1` (MoE) in the checkpoint.

## Status

Scaffolding only. What exists:

- `model/model_data.h` — shared recipe constants, the `Shape<...>` template with `Shape30`, the three-kind layer
  table plus slot maps, the MTP block shape and the `NemotronHShape` typedef. Header-only, no CUDA, no `core/`
  headers.
- `CMakeLists.txt` — `nemotron_h_data` INTERFACE target plus `nemotron_h_data_check` OBJECT target. The OBJECT
  target compiles `model_data.h` in the default build, so a wrong constant or layer table fails without linking
  the pack into `ingine-serve`.
- Default-shape aliases: bare `HIDDEN`, `LAYERS`, `LAYER_KIND`, `ATTENTION_INDEX`, … resolve to `Shape30` so early
  porting code can use the short names.

## Not implemented yet

- Mamba-2 mixer kernels: `in_proj` rows 10304 split gate 4096 / conv input 6144 / dt 64, causal `conv1d` over
  `CONV_KERNEL` 4 with bias, the chunked SSD recurrence in FP32 at `MAMBA_CHUNK_SIZE` 128, `time_step_*` handling
  and per-layer state.
- Full attention for `head_dim` 128, 2 KV heads, full rotary and no output gate; only six layers cache KV.
- MoE: router over 128 experts with one routing group, top-6 selection, `norm_topk_prob` normalization and the 2.5
  routed scaling; non-gated relu2 MLP experts at width 1856, plus the non-gated shared expert at width 3712.
- MTP head, its attention + MoE block and any speculative decode path for it.
- NVFP4 / FP8 weight loading for this checkpoint's tensor names (`backbone.layers.*.mixer.*`, `lm_head`).
- Tokenizer, chat template, and the Bolt vision wrapper (not part of this text-only checkpoint).
- Dual-GPU residency and the KV hop.

## Build wiring

The default pack stays `inkling`. `native/CMakeLists.txt` adds this directory unconditionally, so the constants
header is always compiled, but it links `nemotron_h_data` into `ingine-serve` only when the pack is selected:

```bash
# default: Inkling, ingine-serve links ingine_model
cmake --preset release

# the header-only pack: ingine-serve builds --fake only, ingine-bench is skipped, and ingine-validate does not
# build (it includes the Inkling pack's model headers) until this pack has a model target
cmake -S native -B native/build-nemotron-h -DINGINE_ARCH_PACK=nemotron_h
```

There is no shape selector yet: the binary is `Shape30`. A `Shape74` needs its own block table, an
`INGINE_NEMOTRON_H_SHAPE` selector and the Bolt wrapper's sizes before it becomes a build option.
