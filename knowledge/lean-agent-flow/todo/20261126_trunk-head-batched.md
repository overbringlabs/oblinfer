---
id: T-20261126-02
date: 2026-11-26
status: parked
trigger: next edit to decoder.cu for any other reason
areas: [gpu]
question: Why does the trunk use the 470 GB/s head kernel when the batched one does 773 GB/s at one row?
---

# Trunk lm_head through the batched kernel

The batched NVFP4 head (`kernels/batch/batch_gemv.h`) measured **0.370 ms at one row against the shipped
single-row kernel's 0.608 ms** for the same 286 MB — 773 against 470 GB/s. The trunk calls the slow one.

Routing every token's head through the batched kernel at n=1 is **0.24 ms off every token, +1.8%**, one call site
in `decoder.cu`. It is the cheapest remaining win in the tree.

Caveat: the two kernels sum in different orders, so logits move by ~1e-7 relative. The ids must stay
`1879 314 3377 13 271 71093 3237 198`; if they move, the change is wrong, not the tolerance.
