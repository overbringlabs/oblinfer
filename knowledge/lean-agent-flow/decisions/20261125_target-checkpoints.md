---
id: D-20261125-01
date: 2026-11-25
status: standing
areas: [scope, serving, checkpoints]
question: Which specific checkpoints define oblinfer's support surface, per architecture pack?
---

# Target checkpoint set

Isaak: support these specific NVFP4 checkpoints, per architecture. `docs/checkpoints.md` is the
canonical list; this entry records the ruling behind it.

- `arch/qwen35moe` — the first target. `nvidia/Qwen3.6-35B-A3B-NVFP4` (rev `1355db6a0524`) and the
  derivative `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`. The two exports differ: NVIDIA ships per-expert
  tensors with `input_scale`, Ornith ships per-expert tensors without `input_scale` and a per-expert
  MTP block instead of NVIDIA's packed `gate_up_proj` / `down_proj`. One loader normalises both.
- `arch/qwen38` — `nvidia/Qwen3.6-27B-NVFP4`, `nvidia/Qwen3.8-27B-NVFP4-QAD`,
  `RadixArk/DaisyRose-27B-NVFP4`, and the drafter `RadixArk/Qwen3.8-27B-DSpark` (`draft-dspark`).
- `arch/nemotron_h` — `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` and its DFlash drafter
  (`draft-dflash`).
- `arch/inkling` — `thinkingmachines/Inkling-2-Mini` and the abliterated checkpoint already served,
  `satellitedown/Inkling-2-Mini-abliterated-ingine`.
- Later, not now: `RadixArk/Muse-Glimmer-NVFP4`, `LiquidAI/LFM3-1.7B-NVFP4`.

Qwen3.5-9B and Qwen3.5-4B stay as compile-time shapes only, de-emphasized because no NVFP4
checkpoint with MTP from a trusted publisher exists. Drafter repos are speculation artifacts, not
standalone serving targets.

Provenance rule: a checkpoint row counts as verified only after its `config.json`,
`hf_quant_config.json` and index `weight_map` were read. Shapes taken from `docs/arch/*.config.json`
stay marked as snapshots, and unread repos stay marked until inspected.
