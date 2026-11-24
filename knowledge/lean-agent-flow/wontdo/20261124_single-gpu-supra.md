---
id: W-20261124-04
date: 2026-11-24
status: standing
areas: [serving, gpu]
rule: "Do not try to load ingine on one 16 GB card on supra. Weights alone exceed 16 GiB; host RAM is also 16 GB."
---

# No single-GPU run on supra

Isaak 2026-11-24. 5070 Ti is 16303 MiB. Checkpoint is mixed NVFP4/MXFP8 ~20 GB on disk; resident weights ~16.6 GiB plus DSpark plus KV. Host is 16 GB DDR5-6000, so RAM offload is also out. Split is mandatory from load. Dual-5070 later does not change this (still 16+16). Reopen only on a ≥32 GB card.
