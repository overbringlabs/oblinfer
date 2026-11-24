---
id: D-20261124-05
date: 2026-11-24
status: standing
areas: [serving]
question: Does the 100 tok/s bar include DSpark, and do we keep native NVFP4?
---

# DSpark on; keep NVFP4/MXFP8

Isaak: use DSpark. The 100–120 tok/s bar is user-visible tokens with the drafter, not plain decode. Serving profile already has `spec: true` (`runtime-manifest.json`).

Keep the original mixed NVFP4 (layers 0–19) + MXFP8 (20–23) safetensors. Maya on this box uses Unsloth `UD-Q4_K_XL` and a `__byte_perm` LUT, so the two Blackwell cards never issue `cvt.rn.f16x2.e2m1x2`. That conversion is already in `moe.cu:168-178`. Do not requantize to Q4_K to "match Maya".
