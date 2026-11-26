# Target checkpoints

The specific checkpoints oblinfer supports, grouped by pack. A pack runs when
`ingine-serve -m <dir-or-repo>` resolves the snapshot and the pack's `model_type` guard matches.

Status: **running** (serves today) · **first target** (current focus) · **scaffold** (constants only) ·
**drafter** (speculation artifact, no standalone serving) · **later**.

Provenance: *verified* means the checkpoint's `config.json`, `model.safetensors.index.json` and
`hf_quant_config.json` were read on 2026-11-25. *Snapshot* means the shape comes from
`docs/arch/*.config.json` in this repo and still needs confirmation against the HF repo. † marks a
repo that has not been inspected at all; its rows carry no size or precision claims.

| Pack | Family | Checkpoints in scope |
|---|---|---|
| `arch/inkling` | Inkling-2-Mini | base + abliterated |
| `arch/qwen35moe` | Qwen3.6-35B-A3B (`qwen3_5_moe`) | NVIDIA NVFP4, Ornith NVFP4 |
| `arch/qwen38` | Qwen3.8/3.6-27B dense hybrid (`qwen3_5_text`) | NVIDIA 3.6-27B, NVIDIA 3.8-27B QAD, DaisyRose, DSpark drafter |
| `arch/nemotron_h` | Nemotron 3.5 Lightning 30B-A3B | NVFP4, DFlash drafter |

## `arch/inkling` — running

| Checkpoint | What | Size | Precision / KV | Spec | Status |
|---|---|---|---|---|---|
| `thinkingmachines/Inkling-2-Mini` | base model, rev `c9d028302cd0`; `Inkling2LoopForConditionalGeneration`, packed experts (1244 tensors) | 20.16 GB, 2 shards | 24 looped layers, 128 experts top-8, hidden 2304; vision 422 tensors | DSpark via `mtp.safetensors` | verified |
| `satellitedown/Inkling-2-Mini-abliterated-ingine` | abliterated checkpoint served today, pinned at `cea73d6` in `runtime-manifest.json` | 20.2 GB | same | `mtp.safetensors` 0.64 GB | **running** |

## `arch/qwen35moe` — first target

| Checkpoint | What | Size | Precision / KV | Spec | Status |
|---|---|---|---|---|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | primary target, rev `1355db6a0524`, updated 2026-08-29 | 23.42 GB, 3 shards | modelopt `MIXED_PRECISION`, **KV FP8** | MTP head included, 19 bf16 tensors, **1.689 GB** | **first target** (verified) |
| `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | derivative, same architecture and shape | — | per-expert experts, no `input_scale` on experts | MTP exported per expert, 785 tensors | scaffold (verified index/config) |

Verified details of the NVIDIA checkpoint:

| Property | Value |
|---|---|
| Shape | 40 layers (30 GDN + 10 full), hidden 2048, 16/2 heads, head_dim 256, 256 experts top-8, moe_inter 512, shared 512, vocab 248320, 262144 ctx, conv_k 4, linear 16×128 key / 32×128 value, partial rotary 0.25, mrope `[11,11,10]`, θ=1e7, untied embeddings |
| Quantized | FP8 e4m3 with one FP32 tensor scale: GDN `in_proj_qkv` / `in_proj_z` / `out_proj` (30 each), attention q/k/v/o (10 each). NVFP4 W4A16: routed experts, shared-expert gate/up/down, and **`lm_head`** (U8 `[248320, 1024]`, e4m3 block scales `[248320, 128]`, per-tensor `weight_scale_2` = 1.1190e-4 and `input_scale` = 0.0277, both F32 scalars) |
| Scale layout (headers, 2026-11-25) | FP8 tensors: `weight_scale` F32 `[]` and `input_scale` F32 `[]`, both per tensor. NVFP4 experts: `weight` U8 `[rows, cols/2]`, `weight_scale` F8_E4M3 `[rows, cols/16]`, `weight_scale_2` F32 `[]`, `input_scale` F32 `[]` (unused at W4A16) |
| Unquantized | router `mlp.gate`, `shared_expert_gate`, norms, `conv1d`, `A_log`, `dt_bias`, `in_proj_a/b`, embeddings, MTP head, vision tower |
| Layout | experts shipped **per expert** (`experts.M.{gate,up,down}_proj`), 122,880 expert tensors of 124,468 total; `model.language_model.*` prefix |
| Extras | `chat_template.jinja`, `tokenizer.json`, `vocab.json`, vision tower (`model.visual.*`, ~1.5 GB bf16) |

Ornith differs from NVIDIA's export in ways the loader must normalise: no `input_scale` on expert
tensors, and the MTP block is per-expert instead of packed `gate_up_proj` / `down_proj`.

Fit on supra: 23.42 GB of weights, ~21.9 GB without the vision tower, plus 2.68 GB of FP8 KV at
262,144 = 10 layers × 2 KV heads × 256 × 2 × 1 B. Two 16 GB cards hold it with ~5 GB for
workspaces; one 32 GB card holds it comfortably.

## `arch/qwen38` — scaffold

| Checkpoint | What | Size | Precision / KV | Spec | Status |
|---|---|---|---|---|---|
| `nvidia/Qwen3.6-27B-NVFP4` | 27B dense hybrid, rev `0893e1606ff3` | 21.92 GB, 3 shards, 2194 tensors | FP8 208 + NVFP4 193, **KV FP8**, vision 333 | MTP head, 15 tensors incl. `mtp.fc` | verified |
| `nvidia/Qwen3.8-27B-NVFP4-QAD` | 27B dense hybrid, QAD export, rev `cd46ab575a5a` | 19.90 GB, **64 shards**, 2304 tensors | FP8 94 + NVFP4 307, **KV FP8**, vision 333 | MTP head, 11 tensors, **no `mtp.fc`** | verified |
| `RadixArk/DaisyRose-27B-NVFP4` | Qwen3.8-27B derivative, rev `d7a485994f1c` | 21.92 GB, 6 shards, 2194 tensors | FP8 208 + NVFP4 193, **no KV calibration**, vision 333 | MTP head, 15 tensors incl. `mtp.fc` | verified |
| `RadixArk/Qwen3.8-27B-DSpark` | DSpark drafter, rev `b9a5dbdf03bc`; ships `dspark.py` and `dflash.py` | 3.71 GB, single safetensors | 5 layers, hidden 5120, no vision, unquantized | `DSparkDraftModel`, `draft-dspark` | **drafter** |

Shape per `docs/arch/qwen3.8-27b.config.json`: 64 layers (48 GDN + 16 full), hidden 5120, 24 heads /
4 KV, head_dim 256, MLP 17408, 48 GDN value heads, 262,144 ctx, one MTP layer. All three 27B
checkpoints publish the same `text_config`, so one build shape serves them. Two export differences
matter: QAD's MTP block has no `mtp.fc` projection and its shards are 64 instead of 3–6, and
DaisyRose ships no FP8 KV calibration, so FP8 KV there is our choice rather than the publisher's.

Qwen3.5-9B and Qwen3.5-4B stay only as build shapes (`Shape9`, `Shape4`). They are de-emphasized:
no NVFP4 checkpoint with MTP from a publisher we trust.

## `arch/nemotron_h` — scaffold

| Checkpoint | What | Size | Precision / KV | Spec | Status |
|---|---|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 30B-A3B hybrid, rev `bee7596271d1` | 21.56 GB, 52 shards, 18,487 tensors | FP8 46 + NVFP4 5935, **KV FP8**, no vision | MTP head, 270 tensors, 1 nextn layer | verified |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` | DFlash drafter, rev `8abcc4db8f34` | 1.18 GB, single safetensors | 6 layers, hidden 2688, NVFP4, excludes embeddings and self-attn | `DFlashDraftModel`, `draft-dflash` | **drafter** |

Shape per `docs/arch/nemotron-lightning-30b-a3b.config.json`: 52 layers (23 Mamba-2 + 23 MoE +
6 attention), hidden 2688, 128 experts top-6, shared expert 3712, 1,048,576 ctx, `relu2` experts.
The config says **one** nextn layer; the earlier note of two physical MTP layers came from the pack
README and is wrong. Experts ship per expert: 17,920 expert tensors (`backbone.layers.*.mixer.*`).

## Later

| Checkpoint | What | Shape | Why not now |
|---|---|---|---|
| `RadixArk/Muse-Glimmer-NVFP4` | `MuseGlimmerForConditionalGeneration`, dense + vision | 52 layers (39 sliding-2048 + 13 full), hidden 6656, 32/2 heads, MLP 19968, vocab 202048, 131,072 ctx, untied | needs sliding-window attention and soft-capped logits; ~25B dense, no experts |
| `LiquidAI/LFM3-1.7B-NVFP4` | `Lfm3ForCausalLM` | 24 layers (conv k=7, sliding-256, full every 4th), hidden 2048, 16/4 heads, 262,144 ctx | short-conv + sliding-window kernels; small win |

## Verification state

Every row above is verified: `config.json`, `hf_quant_config.json` and the index `weight_map` were
read from the Hub. What still needs a read before its pack is implemented:

- the tokenizer pre-tokenizer regex and token counts per family (Qwen, Nemotron, Inkling);
- the safetensors header of the NVIDIA 35B MTP block, to confirm the packed `gate_up_proj` shape
  against Ornith's per-expert naming;
- the exact dtype of `weight_scale` / `weight_scale_2` per export (fp8 e4m3 with a per-tensor fp32
  scale is the modelopt default; the Ornith export omits `input_scale` entirely).
