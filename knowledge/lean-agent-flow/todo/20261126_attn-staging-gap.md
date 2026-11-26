---
id: T-20261126-03
date: 2026-11-26
status: parked
trigger: the exchange or band plan is touched for any other reason
areas: [gpu, attention]
question: Where do the 1.7 ms/token of non-kernel time in the attention term go?
---

# The 1.7 ms staging gap in the attention term

At 258k the attention term is 6.39 ms. A perfectly overlapped term would be 4.63 ms: card 0 runs two thirds of the
positions at 694 us/layer, card 1 one third at 1317 us/layer (1.90x for 36/70 SMs, SM-proportional). The 1.7 ms
gap is **pinned staging, 30 launches and per-layer events**, and it is now 27% of the term against 20% before the
paired-head rewrite.

It is diagnosed only by subtraction. Changing it churns the banded bit-identity gates and the hop structure in
`decoder.cu`, so it wants a chunk of its own, with `nsys` per-layer timelines before and after — `nsys` exists on
supra, `ncu` does not.

Prize: ~1.7 ms at 258k, about +8 tok/s. Second prize: batched steps make the same staging carry n rows for one
copy, so this work pays twice once K3 lands.
