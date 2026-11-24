---
id: T-20261124-01
date: 2026-11-24
status: open
trigger: Explore report lands at plans/20261124_two-gpu-explore.md
areas: [serving, gpu]
question: Pipeline-parallel, tensor-parallel, or Maya-like split for 256k at 100–120 tok/s on 2×16 GB?
---

# Split topology

Parked behind the speed-structure explore. Do not pick PP/TP/EP until we know which of ingine's shallow-depth wins (native sm_120a FP4, PDL graphs, native NVFP4 layouts, DSpark, …) a split would break. Maya already has `IK2_EP=1` expert-parallel — that is prior art, not a default.
