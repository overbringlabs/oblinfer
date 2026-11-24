# ingine on supra's asymmetric pair: EP vs PP vs hybrid at 262144

Read-only design model, 2026-11-24. Operator's hardware and bar are the truth (§2, §1). Trees: ingine (this repo),
Maya `src/ik2/` + its two docs (read only), card specs (cited). No code written, nothing compiled, nothing run.

The bar is 100 tok/s of plain decode at 262144 (120 is the stretch; 100 already wins); Maya on this same box is ~60.
The question is not "which split is elegant". It is: at 262144 the decode step must stream **7.70 GiB** per token,
the 5070 Ti can stream 896 GB/s and the 5060 Ti 448 GB/s, and the 1.7×10¹⁰-B weight set does not fit one 16 GiB
card. So the split is forced by VRAM; the only freedom is *which* bytes land on the slow bus and *whether the two
cards ever compute at the same time*.

---

## 1. Verdict

**Expert-parallel, not pipeline-parallel — the opposite of the parent's recommendation, and this time the arithmetic
says why.** Isaak's first instinct ("layer-split is a relay race") is the correct one. For a single request, PP is
not a pipeline: visit *v+1* reads visit *v*'s `y2` write (`native/src/model/decoder.cpp:119-130,238`), so the two
cards take turns and the token's time is the **sum** of both cards' read times, not the max. Every byte moved to the
5060 Ti under PP makes the token slower (448 < 896 GB/s), and the loop's three passes over layers 6-17 mean a cut
anywhere costs 8.3 GiB of duplicated weights: 30 of the 48 visits have to live on the slow card for the totals to fit. EP is the only
topology that lets the 5060 Ti work **concurrently** with the 5070 Ti, and it keeps the 4.16 GiB of KV on the fast card.

EP alone, modelled at a defensible 90% of DRAM bandwidth, lands at **~91 tok/s** — +52% over Maya, but *under* the
100 bar. One addition fixes that and is cheap: the eight full-attention visits are parallel over 8 KV heads, so 3 of
those heads (1.56 GiB of KV) can live and compute on the 5060 Ti **at the same time** as the 5070 Ti's 5 heads. That
hybrid models at **105–116 tok/s** at 85–95% DRAM efficiency with a 20 µs hop, drops under 100 only when the hop costs
40 µs *and* DRAM runs at 90%, and touches one kernel signature and one cache allocation.

| Rank | Topology | GPU0 (5070 Ti) GiB/token | GPU1 (5060 Ti) GiB/token | Serial cross-GPU waits | Modelled ms @90% | tok/s | vs bar 100 |
|:---:|---|---:|---:|:---:|---:|---:|---|
| **1** | **Hybrid: EP experts 44/56 + 3/8 KV heads on GPU1** | 5.39 | 2.31 | 56 | **9.02** | **111** | **hits 100**, near 120 at 95% |
| 2 | EP only (Maya placement: dense+KV+attn+head on GPU0, experts sharded) | 6.94 | 0.75 | 48 | 10.94 | 91 | misses by ~9; 97 with frequency-ordered experts |
| 3 | PP (visit cut; the loop's weight sharing forces a cut at visit 18) | 3.11 | 4.59 | **1** | 16.4 | 61 | misses; ties Maya |
| — | No split (one 5070 Ti) | 7.70 | — | 0 | 10.25 | 98¹ | impossible: 21.7 GiB won't fit 16 |

¹ At 90% efficiency. Shown to size the wall, not as an option — §3.

The rank is not close and it is not a matter of tuning: EP and PP differ by *where 4.16 GiB of KV lives*, and PP puts
2.0 of it on the slow card while serialising the fast card behind it. PP's best fitting configuration models **61 tok/s
— Maya's current number**, which is the cleanest way to say that PP buys nothing here.

---

## 2. Hardware model

| Item | Value | Source |
|---|---|---|
| GPU0 | RTX 5070 Ti, 16 GB GDDR7, 256-bit, 28 Gbps, **896 GB/s** | Gainward/MSI/PNY spec sheets, guru3d (`5070 Ti: 16 GB GDDR7 256-bit 28 Gbps = 896 GB/s`) |
| GPU1 | RTX 5060 Ti, 16 GB GDDR7, 128-bit, 28 Gbps, **448 GB/s** | Manli/zotac/notebookcheck/guru3d 16 GB review (`128-bit, 28 Gbps, 448 GB/s`) |
| Bandwidth ratio | exactly 2.0 (and ~1.94× the SM count: 70 vs 36 SMs) | card specs |
| PCIe | 5.0 x8/x8 ≈ 32 GB/s each direction, **no P2P** | operator brief; Maya's own EP box is in the same state — "the target box has no peer access (`cudaDeviceCanAccessPeer = 0`), so the h a visit sends to GPU 1 and the rows it sends back go through pinned host memory" (`maya src/ik2/ik2_main.cu:2287-2289`) |
| Host RAM | 16 GB DDR5-6000. **Expert offload forbidden** | operator brief |
| VRAM usable per card | ~15.7 GiB of the 16 GiB (CUDA context ~0.3 GiB) | estimate, unmeasured here |
| Effective DRAM efficiency to assume | **85–90%** on weight/KV streams | Maya measured 846 GB/s of 936 on attention and 745 GB/s on NVFP4 experts (`maya README.md:57-59`); ingine's own note: "measured 64% vs 89% of DRAM bandwidth" for the winning warp schedule (`native/src/kernels/moe/moe.cu:17-18`) |
| Cross-device hop cost | 10–20 µs per round trip, no host blocking (`cudaStreamWaitEvent`) | estimate; sensitivity in §4.4. Maya pays the same class of cost with pinned staging |

The 2:1 bandwidth asymmetry is the single most important fact here and both prior reports missed it (the parent's
"~448 GB/s per card" is the 5060 Ti's number applied to both). It does not change the ranking — it strengthens EP,
because EP puts almost nothing on the slow card.

---

## 3. Byte budget

### 3.1 Resident (VRAM), from the checkpoint encodings

| Kind | Bytes | GiB | Source |
|---|---:|---:|---|
| NVFP4 experts, layers 0–19 (128 exps × (w13 2304×1152 + e4m3 2304×144 + w2 2304×576 + e4m3 2304×72)) | 11.47e9 | 10.68 | `native/src/model/model_data.h:136-145`; per-expert 4.271 MiB, matches `KERNELS.md:75-76` |
| MXFP8 experts, layers 20–23 (128 exps × 7.83 MiB) | 4.20e9 | 3.92 | same; `KERNELS.md:56` |
| Dense attention/mix, all 24 layers, `--dense-fp8` (wproj 454 MiB + wo 162 MiB + 12 loop gate stacks 108 MiB + norms/conv/router/queries 46 MiB) | 0.81e9 | 0.75 | `weights.cpp:186-330`, `weights.cpp:319-327` (`to_fp8`), `model_data.h:158-191` |
| Embedding table (VOCAB 201024 × H, BF16) | 0.926e9 | 0.86 | `model_data.h:197`, `weights.cpp:360` |
| Unembedding (200058 × H, FP8 rows + FP32 scales) | 0.462e9 | 0.43 | `weights.cpp:366-380`, `KERNELS.md:158-159` |
| **Weights total resident** | **17.87e9** | **16.64** | sum above; on disk 19.526 GB + 0.59 GB drafter (`runtime-manifest.json:49-50,54-55`) |
| KV, full attention, FP8, 8 visits × 262144 × 8 heads × 128 | 4.295e9 | **4.00** | `KERNELS.md:108-110`; `state.cpp:48-49` |
| Sliding rings, BF16, 40 visits × 1024 × 8 × 128 × 2 (K+V) | 0.168e9 | 0.16 | `state.cpp:44-45`; **not** 1.0–1.5 GiB as the earlier report said |
| Conv states, 48 visits × 3 × (1024+1024+2304+2304) × 4 | 3.2e6 | 0.003 | `state.cpp:50-53` |
| Decode buffers, widths 1/2/4/8 (logits 6.4 MiB each is the big one) + undo + verify backup | ~0.16e9 | 0.15 | `state.cpp:65-92`, `runtime.cpp:29-40` |
| Drafter (mtp) weights 0.59 GiB + FP8 global ring at DRAFT_CTX 65536 (0.13 GiB) | 0.77e9 | 0.71 | `drafter.h:9-11,47`; `runtime-manifest.json:54-55` |
| **Total at 262144 with spec** | | **~21.7** | |

Two consequences, both load-bearing:

* **One card cannot hold it** (16.64 GiB of weights alone > 15.7), so a split is mandatory at any context.
* **The 5060 Ti must carry ≥ ~6.4 GiB of expert weights.** GPU0's fixed load is KV 4.16 + rings 0.16 + dense 0.75 +
  embed 0.86 + unembed 0.43 + drafter 0.71 + buffers 0.15 + context 0.30 = **7.53 GiB**, leaving 8.17 GiB for expert
  weights → GPU1 holds 14.59 − 8.17 = 6.42 GiB = **44% of the expert weights**. That number, not a preference, sets
  EP's expert split. (Maya reaches the same conclusion from the other direction: "expert e of every layer belongs to
  GPU e & 1 — GPU 0 also holds the attention, the dense weights and the KV cache", `ik2_main.cu:379-386`.)

### 3.2 Per token, at 262144 (plain decode, M = 1)

| Phase | Bytes | GiB | Where it comes from |
|---|---:|---:|---|
| Dense attention: 48 × (wproj fp8 19.8–20.0 MB + wo fp8 7.08 MB + q/r scales, rel_proj, conv, query rows) | 1.31e9 | 1.217 | `decoder.cpp:166-208`, `model_data.h:94-98` |
| KV read: 8 full visits × 512 MiB + 40 sliding visits × 4 MiB | 4.46e9 | **4.156** | `KERNELS.md:108-110` (`attn.cu:1207` launches `dim3(CS=16, HKV=8)` over the whole cached range per full visit) |
| Router (every visit) + `message_gates` (the 24 message visits read the four gate stacks) | 0.199e9 | 0.185 | `decoder.cpp:157-163,227-229`, `model_data.h:176-183` |
| Routed experts: 48 visits × 8 experts, read **once per visit** (layers 6–17 are visited 3×, so their weights are read 3× per token) | 1.84e9 | **1.713** | `moe.cu:514` (grid 144 × M·8 slots), `moe.cu:527` for `moe_down` |
| LM head, 200058 × 2304 fp8 | 0.462e9 | 0.430 | `KERNELS.md:158-159` |
| **Total** | **8.27e9** | **7.701** | = 9.23 ms at 100% of 896 GB/s = 108 tok/s ceiling on one 5070 Ti |

Two independent cross-checks that these byte counts are right, both from Maya's measurements rather than its prose:

* Shallow (drop the KV row) = 3.545 GiB = **3.81 GB**, against Maya's "a token costs what its weights weigh: 3.7 GB
  read from VRAM, half of it experts" (`maya README.md:57-58`). Experts are 48% of our number too.
* Depth term = 16 KB per context position (`KERNELS.md:108-110`; Maya says the same at `README.md:79-80`), so
  262144 × 16 KB = 4.0 GiB. Fitted against Maya's measured depth curve on a 3090 (128.7 → 105.6 → 90.6 → 62.3 tok/s
  at 47 / 63,981 / 127,910 / 340,000 tokens, `docs/INKLING2-MINI.md:118-124`) the slope is 24.4 ns per position,
  i.e. **70% of peak bandwidth on the depth term** and a 7.8 ms fixed part (51% of peak, launch/Q4_K-dequant bound).
  Our 90% assumption is therefore the *optimistic* end; §4.4 carries 85%.

---

## 4. EP at 256k

### 4.1 Placement (Maya's shape, on our VRAM)

| On GPU0 (5070 Ti) | On GPU1 (5060 Ti) |
|---|---|
| all 24 layers' dense attention weights + loop gate stacks (0.75) | expert shards: 44–56% of the 128 experts per layer, chosen by pick frequency |
| both KV tensors, opaques: 8 full caches (4.00) + 40 rings (0.16) | — |
| embed 0.86, unembed 0.43, drafter 0.71, buffers 0.15, context 0.30 | its own context 0.30 |
| **7.53 + 8.17 = 15.70 GiB** | **6.42 + 0.30 = 6.7 GiB** |

Both fit with ~0.2 and ~9 GiB spare. Nothing crosses except activations, because 32 GiB of VRAM holds all 16.64 GiB
of weights: **EP here is pure residency, not a cache.** That is a real difference from Maya's EP, which keeps a
per-GPU cache and copies misses over PCIe (`ik2_main.cu:379-386`); no expert weight byte ever touches PCIe in ours.

### 4.2 Bytes per card and across PCIe

| | GPU0 | GPU1 | PCIe payload |
|---|---:|---:|---:|
| GiB/token @262144 | **6.94** (dense 1.217 + KV 4.156 + gates/router 0.185 + 56% of experts 0.96 + head 0.430) | **0.75** (44% of experts) | **1.90 MiB/token** |
| Composition of the PCIe payload | send `xm` bf16 [1, 2304] 4.6 KiB + idx/wts 64 B per visit; receive 8 per-slot bf16 rows 36.0 KiB per visit, ×48 | | 62 µs at 32 GB/s peak; ~100 µs at a realistic 20 GB/s pinned copy — irrelevant next to the round-trip *latency*, §4.4 |

The 36 KiB back-channel is deliberately the *per-expert BF16 rows*, not a partial sum: `moe_down`'s contract is
`bf16(W2_e·act)` per expert then an FP32 sum in slot order `k = 0..7` (`KERNELS.md:66-67`, `moe.cu:31`), so if GPU1
returns its rows and GPU0 sums in slot order the result is **bit-identical** and the bit locks in
`tests/kernels/test_bitlock.cpp` keep their hashes. This is the one design decision that makes EP cheap to adopt
without touching the numerics contract.

### 4.3 Modelled time, and the bottleneck

| Step | GPU0 | GPU1 | Serial time @90% |
|---|---:|---:|---:|
| attention phase: dense + KV + gates/router (+ GPU1's KV slice in the hybrid) | 5.55 GiB → 6.65 ms | 0 | 7.39 ms |
| expert phase (both cards concurrently) | 0.96 GiB → 1.15 ms | 0.75 GiB → 1.80 ms | 2.00 ms |
| head | 0.43 GiB → 0.52 ms | 0 | 0.57 ms |
| 48 cross-device waits × 20 µs | | | 0.96 ms |
| **total** | | | **10.94 ms → 91 tok/s** |

**Bottleneck: GPU0's DRAM, then GPU1's DRAM in the expert phase, then hop latency.** GPU0 is the wall because 5.4 of
the 6.9 GiB it reads is the KV/dense attention block that only it can do under this placement.

Two cheap corrections, both worth modelling before coding:

* **Frequency-ordered expert placement** (Maya's `IK2_EP_SPLIT` + `IK2_POP_OUT`/`IK2_WARM`, `ik2_main.cu:395-399`).
  VRAM forces 44% of the *weights* onto GPU1 but the split that balances *time* is 33% of the *picks*: the 5060 Ti
  gets the experts it can read in the time the 5070 Ti needs for its share. Expert phase 2.00 → 1.37 ms,
  **91 → 97 tok/s**. Needs a popularity profile; ingine has none today.
* **Move the embed table and the drafter to GPU1** (0.86 + 0.71 GiB off GPU0): expert weights on GPU0 rise to 9.7 GiB,
  the hot-byte split improves slightly, and the drafter's 0.59 GiB weight read per draft lands on the slow card — a
  bad trade for spec, so leave the drafter on GPU0 and move only the embed (one 4.6 KiB row copy per token). ~+1 tok/s.
  Not worth the code. Listed for completeness, not recommended.

### 4.4 Sensitivity (this is the honest uncertainty band)

| DRAM efficiency | hop 10 µs | hop 20 µs | hop 40 µs | hop 80 µs |
|---|---:|---:|---:|---:|
| 95% | 101 | 96 | 88 | 75 |
| 90% | 96 | **91** | 84 | 72 |
| 85% | 91 | **87** | 80 | 70 |

EP is an 87–101 tok/s answer (87–96 across the defensible band). It beats Maya's 60 by half again, it beats PP's
modelled 61 by 40%, and it does **not** reach 100 unless the hop is cheap *and* the attention kernel runs at 95% of
DRAM. That gap is what decides §6.

---

## 5. PP at 256k

### 5.1 Placement is forced, and the loop makes it worse than it looks

The cut is a visit index, and the loop breaks the naive arithmetic: **a layer's weights are shared by its three
visits** (`model_data.h:56-62`), so any cut inside the loop needs those layers resident on *both* cards. At cut 18
(layers 0-17 on GPU0, 6-23 on GPU1) that duplicates layers 6-17 = **8.3 GiB**, and that duplication is what saves the
layout: it is the only family of cuts where both cards fit.

| Cut (visit) | GPU0 GiB/token | GPU1 GiB/token | GPU0 VRAM | GPU1 VRAM | ms @90% | tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 6 (layers 0-5 / 6-23) | 1.30 | 6.39 | 6.8 | 15.5 | 18.8 | 53 |
| 12 | 2.21 | 5.49 | 11.2 | 14.9 | 17.6 | 57 |
| **18 (layers 0-17 / 6-23)** | **3.11** | **4.59** | **15.5** | **14.4** | **16.4** | **61** |
| 24 | 4.01 | 3.69 | **16.0** over | 13.9 | 15.2 | 66 |
| 30 (loop pass boundary) | 4.91 | 2.79 | **16.6** over | 13.4 | 14.0 | 71 |
| 36 | 5.81 | 1.89 | **17.1** over | 9.0 | 12.8 | 78 |

Later cuts are faster and do not fit: every GiB of KV and weights moved to the fast card has to fit in 15.7 GiB that
already holds 0.86 embed + 0.43 unembed + 0.71 drafter. **The feasible PP ceiling is ~61 tok/s at cut 18**, i.e. about
what Maya already does on this box, with 8.3 GiB of weights duplicated and one hop per token.

| | GPU0 (cut 18) | GPU1 |
|---|---:|---:|
| resident | 12.48 weights + 1.5 KV + 0.07 rings + 2.45 misc = 15.5 GiB | 12.48 weights (duplicated) + 2.0 KV + 0.13 rings + 0.3 = 14.4 GiB |
| GiB/token | **3.11** | **4.59** |
| PCIe payload | 9.2 KiB per token (one `hid` handoff: fp32 [1, 2304]) | |
| Serial waits | **1** | |

### 5.2 Why it loses: the relay is the whole token

`decoder.cpp:119-130`: each visit's `hid` is the previous visit's `y2` write, produced by `moe_down` (`:238-239`) and
consumed by the next visit's `mix` (`:143-159`). There is no second token in flight (`scheduler.h:2-6`: "one worker
thread runs jobs strictly one at a time"), so GPU0 idles while GPU1 walks its 30 visits and vice versa. Time is the
**sum**: 3.11 GiB x 1.198 ms/GiB + 4.59 GiB x 2.397 ms/GiB = **16.4 ms, 61 tok/s**. Every visit handed to the slow
card costs ~2.4 ms/GiB against 1.2 ms/GiB on the fast one, and the loop's weight sharing is what stops you handing
over fewer.


### 5.3 What PP does to PDL and the graphs

This is the one place PP is clean: two CUDA graphs per token (one per card), one cross-device wait, and **PDL
survives intact inside each half** — `cudaLaunchAttributeProgrammaticStreamSerialization` is captured as a
programmatic graph edge (`moe.cu:493-503`, `common/pdl.cuh:1-22,45-46`), which is a per-device stream property. One
hop per token also means the host still launches ~2 objects per token, so the `chain_*` pipelined width-1 path
(`runtime.cpp:205-217`) keeps working unchanged. Cost (estimated): ~2 ms of lost PDL overlap at the single cut, plus the relay itself.

EP breaks PDL at 48 points and needs ~96 graph launches per token (§7.4) — and it is still 5.5 ms faster, because what
PDL buys (~1–2% end to end at shallow depth, `KERNELS.md:37-39`) is small next to parking 4 GiB of KV on 448 GB/s.

---

## 6. Hybrid (it wins, and it is small)

The full-attention kernel is parallel over **KV heads**: grid is `(CS = 16, HKV = 8)` — `attn.cu:101`, `attn.cu:1207`, `attn.cu:365`
(`blockIdx.y` is the KV head, 16 blocks per head own output dims `[8r, 8r+8)`), and the 8 GQA heads are independent
(separate `k_cache[fi]`/`v_cache[fi]` slices, separate softmax, separate sink logit, `KERNELS.md:99-108`). So the 8
full visits can be split **by KV head across the two cards with no partial-softmax combine and no new math**: GPU1
runs 3 heads (1.56 GiB of KV, 37.5% of the 4.156), GPU0 runs 5 (2.60 GiB), each writes its own disjoint output rows of
`attn_out` (`[m, 3072]`, head-major), and the only traffic is GPU1 receiving the k/v projection rows for its 3 heads
(~1.5 KiB) and the q rows for its 9 query heads (4.6 KiB) and returning 9 × 128 fp32 (4.6 KiB) — ~11 KiB per full
visit, 8 visits, one wait each.

| | GPU0 | GPU1 | @90% |
|---|---:|---:|---:|
| attention phase (GPU0 4.00 GiB / GPU1 1.56 GiB, concurrent) | 4.00 → 4.79 ms | 1.56 → 3.73 ms | 5.32 ms |
| expert phase (EP 56/44) | 1.15 ms | 1.80 ms | 2.00 ms |
| head | 0.52 ms | — | 0.57 ms |
| 56 waits × 20 µs | | | 1.12 ms |
| **total** | | | **9.02 ms → 111 tok/s** |

| DRAM efficiency | hop 10 µs | hop 20 µs | hop 40 µs |
|---|---:|---:|---:|
| 95% | 124 | 116 | 103 |
| 90% | 118 | **111** | 99 |
| 85% | 112 | **105** | 94 |

Add frequency-ordered expert placement (§4.3) and the same configuration models **119 tok/s at 90%**. Two variants are
worse and not worth building: 4/8 heads (108 — GPU1 over-subscribed), 2/8 heads (103 — GPU0 still the wall).

Why it beats both, in one line each: it beats EP by taking 1.56 GiB of the object that dominates the token (KV) off
the 5070 Ti and running it on the idle card *concurrently*; it beats PP because a KV-head split is real parallelism
inside one visit, while a visit cut is a dependency.

**Graphs:** two exec graphs per token (one per device), 104 segments (48 visits + 8 full-visit attention splits on
either side), 56 cross-device waits. PDL survives *inside* each segment — a programmatic edge is per-device — so what
is lost is the prefetch across segment boundaries, not the mechanism. If that cost measures badly, the fix is a
device-side spin-wait protocol (one graph per device per token, mapped-memory flags), which is where this design
eventually wants to go.

### Where the loop and `message_gates` sit on the cut

* **EP and the hybrid: nowhere.** The whole loop (layers 6–17 × 3 = 36 of 48 visits, **74% of the per-token bytes**)
  stays on GPU0, along with `B1`, `B2[0..2]`, `E[0..2]`, `msgs` and the 12 gate stacks. `message_gates` reads the
  previous pass's blocks (`decoder.cpp:134-163`: `B2[p-1]`, `E[p-1]`, `w.message_queries[i]`) — a 12-visit-long-lived
  dependency that would cross the wire on every visit under a PP cut inside the loop, and every pass boundary under
  a cut between passes. Under EP it is local, untouched.
* **PP: it is the worst place to cut.** A cut inside the loop puts pass-*p* blocks on one card and pass-*p+1*'s gate
  read on the other, so the gate read (`w_down` bf16 `[3, 64, 2H]`, 1.77 MiB per visit, `KERNELS.md:148-154`) becomes
  a cross-card read of a value written 12 visits ago — plus the per-visit relay on top. And the loop's three passes
  over each layer mean any cut inside it duplicates those layers' weights: 8.3 GiB at cut 18 (§5.1), which is the
  only placement family that fits at all.
* The taps (`TAP_VISITS = {5, 23, 41}`, `model_data.h:79`) must reach the drafter, which lives on GPU0. EP/hybrid:
  `b.taps` never leaves GPU0. PP: at least two taps cross the cut (3 × 4.6 KiB — cheap, but one more thing on the
  protocol).

---

## 7. Recommendation

**Implement expert-parallel first, and treat the KV-head split as step 5 of the same plan rather than a separate
project.** EP is the mandatory two-device plumbing (contexts, expert residency plan, handoff protocol, dual graphs)
and every later optimisation assumes it. Alone it models 87–96 tok/s: +50% over Maya, short of the bar. The KV-head
split reuses the same handoff and the same second device, adds ~11 KiB and one wait per full visit, and takes the same
configuration to 105–116. Pipeline-parallel is not on the list: its best fitting configuration is 33% slower than EP and
lands on Maya's current number, and it makes the loop and the gate stack cross the wire.

Why EP is the one most likely to clear 100: it is the only topology that (a) leaves the 5070 Ti's 896 GB/s as the
bottleneck for the 5.5 GiB of attention bytes only it can serve, and (b) lets the 5060 Ti's 448 GB/s work *during*
that phase instead of after it. Every other arrangement either serialises (PP) or parks hot bytes on the slow bus.

### Implementation order (baby steps, no rewrite)

1. **Two devices, no behaviour change.** A second `DeviceContext` (`core/device.cu:152`) and a `--second-gpu` option;
   with the flag off nothing differs. `DeviceUploader` (`core/safetensors.cpp:498`) gains a target device.
2. **Residency plan, still one-device execution.** Split `load_weights` (`weights.cpp:335-381`) by an explicit table
   (which layer's experts, which range) so the loader can place expert stacks on either card. Verify: same logits,
   same bit locks, one card.
3. **Handoff primitives with a stub.** Pinned staging buffers, one event pair, two streams; a stub kernel pair that
   pushes 4.6 KiB out and 36 KiB back per visit, 48 times. Measure the real round trip on supra. If it is > 40 µs,
   stop and re-rank (§4.4) before writing any MoE split.
4. **Expert split.** A per-device slot mask over `idx` (GPU1's slots keep their expert, GPU0's become −1), `moe_up`
   unchanged (it already early-exits non-first-occurrence slots, `moe.cu:21-22`), and one new `moe_down` variant that
   emits per-slot BF16 rows for the off-card slots so GPU0's slot-ordered FP32 sum stays bit-identical. Graphs: two
   per token (48 segments each), the eager path kept for validation (`runtime.cpp:195-198`).
   *Landed 2026-11-24 (`1a5a47b`) as whole layers on GPU1 with an eager step, and DSpark on that eager step (the
   verify width read back, `run_step_eager`, the acceptance graph unchanged) — Isaak's option A: dual graphs buy
   launch overhead only, so they are an optimisation after the rate is measured, not a prerequisite for spec.*
5. **KV-head split of the 8 full visits.** Parameterize `attn_decode`'s KV-head range (`attn.cu:1207`,
   `gridDim.y`) and slice `State`'s `k_full`/`v_full` per head range (`state.cpp:48-49`, `model_data.h:236`,
   `k_full_at`). Good news: full attention is **not** in the bit-lock set (`KERNELS.md:46-49` locks proj,
   proj_conv, head, moe_up/down and *sliding* attention), so the split has a free hand; sliding attention stays
   entirely on GPU0. Watch: `save_full_rows`/`restore_full_rows` and `save_verify` (`runtime.cpp:274-320`) walk the
   cache tensors and must learn the split.
6. **Frequency-ordered expert placement.** Export expert pick counts from a calibration run (the `pop_count` idea in
   Maya, `ik2_main.cu:364-378`) into an offline table; deal GPU1 the experts with the best bytes/time ratio. +5–8%.
7. **Verification at each step:** `ingine-validate` goldens + `tests/kernels/test_bitlock.cpp` hashes + a 4k/64k/256k
   decode sweep on supra. The bar is the token rate at 262144 with a cold cache.

### What to expect, and what would falsify it

| Checkpoint | Modelled | Falsified if |
|---|---:|---|
| step 4 (EP, one card per expert share, 90% eff) | 91 | measured < 80 → the hop is > 40 µs or attention is < 80% efficient; both are fixable, neither changes the ranking |
| step 5 (+ KV heads 3/8) | 111 | measured < 100 → the KV-head split's per-visit wait is not amortising; fall back to EP + frequency placement (97) |
| step 6 (+ frequency placement) | 119 | — |

---

## 8. Numbered questions for Isaak

1. **Is the 100 tok/s bar plain decode, or does the drafter count?** Plain decode at 262144 is a 7.70 GiB/token
   stream; a verify call of 8 rows reads the same KV once for 8 rows, so at ~3 accepted tokens/call the KV cost per
   token falls ~2.7× and both topologies clear 100 with room. If spec counts, EP alone (§4) is enough and step 5 is
   optional. **This changes the plan's length by ~a week.** Example: EP plain = 91 tok/s; EP with a working spec loop
   at 3 tokens/call = >200.
2. **Do you want step 5 (KV-head split) pre-authorized now, or is it gated on step 4's measurement?** Pre-authorizing
   lets the residency plan (§7.2) and the `State` slicing land once; gating it keeps the diff smaller but means the
   cache layout is touched twice.
3. **Two concurrent requests, ever?** Today's scheduler is strictly one job at a time (`scheduler.h:2-6`). PP becomes
   the right answer the moment two decode streams must overlap (163 tok/s of aggregate at 90%), and that decision
   would change the residency plan, not just the split. If batching is on the roadmap, say so before step 2.

---

## 9. Did not try

* No compile, no `cmake`/`nvcc`, no kernel tests — read-only as instructed.
* No run on supra: no `nvidia-smi`, no `cudaDeviceCanAccessPeer`, no PCIe or DRAM-efficiency measurement on either
  card. Every efficiency number is either Maya's measurement on a 3090 or an assumption; §4.4 carries the range.
* Did not touch the live `inkling-2-mini-maya` server, and did not re-measure Maya on supra. The ~60 tok/s at depth is
  the operator's figure; the 3090 depth curve (`docs/INKLING2-MINI.md:118-124`) is the only measured proxy.
* Did not model prefill. At 262144 a cold prompt is a separate problem (Maya: 1,473 tok/s at 340k) and the
  two-card prefill split has different (easier) economics — it is bandwidth-parallel over tokens.
* Did not model the spec loop's per-token economics in detail; §8.1 is a first-order estimate, not a model. In
  particular the drafter's 0.59 GiB weight read per draft and its 0.13 GiB ring are on GPU0 and are *serial* between
  verify calls — worth 0.7 ms per draft and unmodelled here.
* Did not model a device-side spin-wait protocol (persistent kernels + mapped-memory flags) that would collapse EP's
  48 hops into 1; it is the obvious next optimisation if §4.4's hop cost bites, but it is not a baby step.
* Did not crawl Maya's GLM / DeepSeek / Hy4 / mm3 trees; stayed in `src/ik2/`, `README.md`, `docs/INKLING2-MINI.md`.
* Did not verify ingine's exact kernel-launch count per token (counted ~390 statements in `decoder.cpp`'s visit loop,
  not measured from a trace).
* Did not decide whether `--dense-fp8` should stay on for the 5060 Ti's expert share; the dense weights are all on
  GPU0 either way.

---

## Parent check (grok-4.6)

Caught this completion. Numbers I re-checked: `attn.cu:1207` grid is `(CS, HKV=8)` so a KV-head split is real; `decoder.cpp:119-130` really is visit-serial; `scheduler.h` is one job at a time. The 2:1 bandwidth (896 vs 448) was missing from the Ornith pass and from my PP recommendation — that recommendation is withdrawn.

Agree with the rank: PP ~61 tok/s is a dead end on this box for single-stream decode. EP first, KV-head split as the thing that clears 100. Spec-on is already the serving profile; still do the KV-head split because prose spec acceptance is weak (Maya's own note). Two-request batching is not in v1.
