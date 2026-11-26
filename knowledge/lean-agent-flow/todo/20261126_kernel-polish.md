---
id: T-20261126-08
date: 2026-11-26
status: parked
trigger: any of these files is open for another reason
areas: [gpu, kernels]
question: Which measured micro-wins are still unlanded?
---

# Measured micro-wins not yet landed

Each of these has a number attached. None is worth its own round; all are worth folding into the next edit of the
file they live in.

| win | measured | file |
|---|---|---|
| trunk head through the batched kernel at n=1 | 0.608 -> 0.370 ms, +1.8% on every token | `decoder.cu` (own item, T-02) |
| `BATCH_MAX` 4 -> 8 | 7-draft chain head 1.42 -> ~1.0 ms | `kernels/batch/batch_gemv.h` |
| q4_0 K identity `sum q*d*(n-8) = d*(sum q*n - 8*sum q)` | K-side bytes currently buy nothing (18.9 MB read in 104 us against 35.7 MB in 108), so this is what makes K4V0 pay | `decode_attn.cu` |
| MoE kernel 615 -> 851 GB/s ceiling | up to 0.25 ms/token | `kernels/moe/fused_experts.cu` |
| `HPAIR=4` with q staged in shared memory | untested; 186 registers at present is 1 block/SM | `decode_attn.cu` |
| MTP block + `embed_tokens` moved to span 1 | reaches the balanced expert split, ~0.46 ms/token; needs the MTP path to launch on card 1 and a card-1 embedding gather | `uploader.cpp`, `decoder.cu` |
