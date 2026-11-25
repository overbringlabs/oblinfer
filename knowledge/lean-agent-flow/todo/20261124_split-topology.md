---
id: T-20261124-01
date: 2026-11-24
status: in-progress
trigger: Supra rerun of model_ep_spec + ingine-bench --long-prompt closes it
areas: [serving, gpu]
question: Pipeline-parallel, tensor-parallel, or Maya-like split for 256k at 100–120 tok/s on 2×16 GB?
---

# Split topology

Parked behind the speed-structure explore. Do not pick PP/TP/EP until we know which of ingine's shallow-depth wins (native sm_120a FP4, PDL graphs, native NVFP4 layouts, DSpark, …) a split would break. Maya already has `IK2_EP=1` expert-parallel — that is prior art, not a default.

**2026-11-25:** the hybrid is in the tree — EP (whole layers, GPU1) plus the 3/8 KV-head split, for decode and for the chunked prefill. Open until the supra numbers (tok/s at 256k) land.
