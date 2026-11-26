# Qwen3.6-35B-A3B — speculative decoding plan

Mandatory, by owner ruling 2026-11-26: spec decode has to work, because nothing else closes the gap to
llama.cpp. The design below is the only shape whose arithmetic reaches the 64.7 tok/s gate at 258k without
breaking the VRAM budget or the current feature set. Every number carries its source.

## Where we stand (sm_120, K8V8, `model_qwen35moe_bench`, commit `5eeb258`)

| case | today |
|---|---|
| 1k depth | 71.5 tok/s (13.98 ms) |
| 2k depth | 85.1 (11.75 ms) |
| 32k synthetic | 77.2 (12.95 ms) |
| 258k synthetic | 46.2 (21.66 ms) |
| prefill, 2k | 78 tok/s (one decode step per prompt token) |
| llama.cpp bar at 258k | 51.74 tok/s; gate is 64.7 |

Byte-proportional work at 258k is about 12.5 ms; the remaining **~9.2 ms/token is per-position overhead**
(router, cross-card hop staging, norms and elementwise launches, residuals). Source: agent C's block
measurements against the 21.66 ms token. Agent B is measuring the same overhead directly from `decoder.cu`.

## Why the first attempt lost (agent C, measured)

A four-token verify built naively came out at 27 tok/s, i.e. 0.59×. Four defects, all fixable:

1. **Per-position overhead multiplied by four.** The hop and the router ran once per token, so 4 × 9.2 ms
   swamped everything the batch saved.
2. **Batch attention underfilled.** One warp per split instead of the shipped eight warps per position:
   57 GB/s against 135.
3. **Expert dedup produced no time win.** 19.8 unique experts of 32 for n=4 (38% of expert bytes saved,
   measured on real weights through 40 layers), but five launches of small work ate it.
4. **No rollback existed**, so the GDN state advance made a rejected draft unrecoverable.

The parts of C's batch module that already beat the single-token path: dense GEMVs 1.82× (GDN) and 1.79×
(attention), `lm_head` 1.94×, GDN register-staged recurrence 2.80×. Those kernels are tested (180 checks,
`ALL PASS`, sanitizer clean) and are the foundation.

## The design

`step_batch(ids[0..n-1])` advances the trunk by n positions in one pass. Speculation drives it with
n = 1 + K drafts, K default 3.

- **Router once per layer** for all n tokens (one kernel, n rows).
- **One hop per layer per step.** The pinned exchange carries n rows of hidden state (n × 2048 bf16), the
  n × 8 expert ids and weights, and returns n rows. This is the defect-1 fix and it does not depend on the
  dtype of anything else.
- **Attention keeps the shipped block shape** (eight warps per position, 16 dims per lane, register
  `SplitState`) and carries n accumulator sets, so the cache is read once for n queries. Depth cost stays at
  one cache pass per step instead of n.
- **GDN projections batched** (C's kernel), then n sequential register-resident state updates in the same
  block. The recurrence is serial in position by construction and costs 0.075 ms/token.
- **Rollback by checkpoint, not replay.** Before the step, copy the GDN recurrent state (30 × 32 × 128 × 128
  FP32 = 62.9 MB) and the conv state (2.9 MB) to a scratch buffer: one 66 MB copy, about 80 µs at 800 GB/s.
  Trunk and drafter KV need no rollback at all — a rejected position's rows are overwritten by the next
  step, because the position counter only advances over accepted tokens.
- **Drafter** is the checkpoint's single BF16 MTP layer: `mtp.fc` over `concat([embed_norm, hidden_norm])`
  (embedding half first), one full-attention layer with its own K/V cache, MoE with 256 experts top-8,
  `mtp.norm`, then the trunk's NVFP4 `lm_head`. Tensor inventory and the exact forward are in agent C's
  round-1 report; the norm convention is `x/rms * (1 + w)`.

### VRAM at 262,144 (decimal GB, two 16,303 + 16,311 MiB cards)

| item | GB |
|---|---|
| trunk weights | 23.41 |
| trunk KV K8V8 | 2.85 |
| drafter KV K4V4 | 0.15 |
| drafter weights BF16 | 1.69 |
| GDN state + conv | 0.07 |
| **total / 34.20** | **28.17**, 6.03 spare |

The drafter stays BF16 (no quantization) and every KV type is user-selectable, so a 2×5090 host can run
BF16 trunk KV with a q8_0 drafter cache without a code change. Drafter experts follow the same free-VRAM
split as the trunk.

## Steps, in order

| step | work | gate |
|---|---|---|
| K0 | Measure the MTP acceptance rate before writing any CUDA: run the BF16 MTP layer in PyTorch on 6–8 real prompts of this family, greedy, and record α1, α2, α3. | a table of α per prompt; if α1 < 0.5 the plan changes shape |
| K1 | `KvType::Q4_0` (two-array layout: 16 bytes of nibbles per 32 values, scales in a second array, so the 16-byte read path carries over), per-cache type selection, drafter-cache spec through the decoder API. | numerical test vs FP64 for the new type; decoder parity unchanged; 262144 allocates at K8V8/K4V4 |
| K2 | `step_batch(ids, n)` over C's kernels, trunk only, no speculation. | n=1 bit-identical to `step()`; ids identical for n=2..4 against n single steps on the real weights; n=1 decode speed unchanged |
| K3 | Batched hop and batched router (defect 1), driven by B's breakdown. | 258k verify step ≤ 1.6× one token's time |
| K4 | MTP drafter module: load, own cache, forward, test against a PyTorch reference on the real weights. | draft ids match the PyTorch reference; measured α reproduces K0 |
| K5 | Accept/rollback loop with the GDN checkpoint. | greedy ids identical to non-speculative decoding on 16 tokens; tok/s at 1k, 2k, 32k, 258k |
| K6 | CLI: `--spec-type draft-mtp`, `-ctkd`/`-ctvd`, `--draft-max`, `-c`, `-m` resolving an HF hub cache dir; llama-server-shaped logging. | `-m <dir> -c 262144 -ctk q8_0 -ctv q8_0 -ctkd q4_0 -ctvd q4_0 -fa on --spec-type draft-mtp` runs the server |

K1–K3 are independent of K0 and K4 in file terms (decoder vs new drafter module), so two agents can run at
once; K2 and K3 both touch `decoder.cu`, so they are one agent's serial work.

## Expected arithmetic

E[tokens per step] = 1 + α + α² + α³. At α = 0.7 that is 2.53 tokens per step, at 0.8 it is 2.95.

With the hop batched and every block byte-proportional, C's model gives 258k ≈ 18.8 ms/step → **134 tok/s
(2.9×)** and 1k ≈ 17.4 ms/step → **112 tok/s (1.6×)**. At 1k the token is 77% per-position overhead, so
speculation helps least there; at depth it is the whole game.

## Risks

- **Acceptance is unmeasured.** Every number above is conditional on α. K0 exists to stop us building a
  drafter for a model whose MTP head does not accept.
- **The 9.2 ms overhead is not yet attributed.** If it lives in the hop, K3 fixes it. If it lives in
  per-layer launches that n does not amortise, the 134 tok/s figure is optimistic and the same K3 becomes the
  hard part.
- **Graph capture across n.** The capture path patches `pos` per replay; a batched step patches n positions.
  Expect one round of work on `decode_graph.cu` in K5.
- **Drafter KV quality at K4V4.** The drafter only has to propose tokens, so a coarse cache costs acceptance,
  not correctness. K0 should measure α1 at K4V4 as well as BF16 if the PyTorch harness can emulate it.
- **Prefill** stays at 78 tok/s until `step_batch` is reused for it. K2 unlocks that; it is not in this plan's
  gates but it is the next cheapest win after spec decode.

## K0 result: the acceptance rate (measured 2026-11-26, supra, HEAD `a8d18e3`)

6 prompts (4 prose, 2 code), 112-160 tokens of context, 32 greedy tokens each, Q4_0 drafter cache, real trunk and
real drafter, depth 4096.

| region | α1 | α2 | α3 | E(n=2) | E(n=3) | E(n=4) |
|---|---|---|---|---|---|---|
| all positions, pooled | 0.596 (589/989) | 0.613 (359/586) | 0.646 (230/356) | 1.60 | 1.96 | **2.20** |
| generated positions, pooled | **0.758** (150/198) | **0.728** (110/151) | **0.736** (81/110) | 1.76 | 2.31 | **2.72** |
| code only (generated) | 0.85-0.88 | 0.83-0.96 | 0.75-0.92 | 1.85-1.88 | 2.6-2.7 | **3.1-3.4** |
| prose only (generated) | 0.49-0.94 | 0.36-0.94 | 0.18-0.97 | 1.48-1.94 | 1.8-2.8 | 2.0-3.7 |

The abort condition (α1 < 0.5) does not fire. The plan's 0.8-α arithmetic was optimistic by about 8%, and the
number that matters for a coding workhorse — Ornith — is the code row, at 3.1-3.4 tokens per step.

The Q4_0 cache costs nothing measurable: BF16 gives 0.598/0.616/0.662 all-positions and 0.758/0.755/0.735
generated, i.e. within 0.02 of the chain above. The drafter's own attention is 0.015 ms at this depth, so K4V4
stays the default.

Cost side: one draft is 0.829 ms (mean over 1,931 chain calls), of which the trunk's `lm_head` is 0.608 — 72%.
Three drafts are 2.54-2.69 ms, about 38% of a trunk token before the verify step. A batched M-row head is the
obvious next cut, and it is K2/K3's business.

Sample size, honestly: 989 draft-1 samples pooled, but the prompt is the sampling unit and the spread across
prompts is 0.43-0.88. Six prompts is six samples, and the two easy ones (a verbatim paragraph restart, a
boilerplate `main()`) sit at the top of the range. Depth 144 only; acceptance at 258k is unmeasured.

## Ornith-1.5-35B-A3B is ready to load

Its published index has **94,393 tensor names, 23.407 GB**, including a 333-tensor vision tower, and 785 MTP
tensors in the per-expert dialect. The classifier takes every one of them (0 rejected), and the dialect detector
now finds `ornith` where it previously said `nvidia`: both exports number the trunk's routed experts, and Ornith's
real export carries 130 `input_scale` tensors on the dense projections, contrary to what `docs/checkpoints.md`
claimed. The MTP block is the discriminator — NVIDIA packs it, Ornith numbers it.

Nothing in the pack needs a new code path for Ornith beyond that detection fix, so every speed win in this plan
lands on Ornith-1.5 unchanged. The checkpoint itself is not downloaded on supra; the check ran on metadata only.

## Phase A result (2026-11-26): two of three pieces landed, K3 does not pass yet

**Attention is instruction-bound, not bandwidth-bound.** A head-range sweep on the committed kernel showed
per-head cost flat out to four heads (0.227 ns/pos-head at 16 heads, 0.260 at 4), so the eight query heads
re-reading the same rows were served from L2 for free. SASS: 170 warp instructions per two positions, 56 of them
`PRMT`/`I2F` on the half-rate conversion pipe against 58 `FFMA`/`FMUL` of real arithmetic. Pairing two query heads
per block amortises the decode across two dots. One layer at 258048, q8_0: 936.8 → 693.7 µs (1.35×); attention
term 7.87 → 6.39 ms; **258k decode 68.2 → 76.3 tok/s (13.11 ms)**, 98304 92.6 → 98.3.

What is left in that term is not the loop: card 0 does two thirds of the positions at 694 µs/layer and card 1 one
third at 1317 µs (1.90× for 36/70 SMs, SM-proportional), so a perfectly overlapped term would be 4.63 ms against
the measured 6.39. The 1.7 ms gap is pinned staging, 30 launches and per-layer events, and it is now 27% of the
attention term. It needs its own chunk.

**`step_batch` works and is 1.85× at 1k**, but K3's gate (a 258k verify step ≤ 1.6× one token) fails: n=4 costs
3.1×. The batch attention pays the cache about once per query at depth, 9.7 ms of marginal cost per query, even
though the kernel reads the rows once. The cause is register pressure: at the shipped 16-dim shape, n × (16
accumulators + 16 q) costs 181 registers, one block per SM, against the shipped kernel's four. At 8 dims per lane
it is 104 registers, two blocks, and the pass got ~3× cheaper — which is the shape that shipped. The fix, measured
not guessed: run the verify block as **groups of two queries at the shipped 16-dim, two-stream shape**, so n=4
costs two cache passes rather than four passes' worth of time. `redux.sync.add.f32` for the score reduction is the
second candidate (five shuffles saved per row-query).

**The drafter is no longer linear in draft count.** One batched head pass per chain instead of one per draft:
3 drafts 1.41 ms against 2.66 serial (−47% at 1k, −24% at 262k where the drafter's own attention dominates at 0.92
of 1.33 ms). The batched head also beats the shipped single-row head at one row — 0.370 against 0.608 ms for the
same 286 MB, 773 against 470 GB/s — so **routing the trunk's own head through it is 0.24 ms off every token**.

**α does not decay.** Generated-text pooled: 0.758, 0.728, 0.736, 0.779, 0.797, 0.826, 0.889, 0.900 for α1..α8.
The drafts that survive rejection are the easy stretches. So the count is set by content: code n=8 (E = 5.65 on one
prompt, 3.85 on the other), prose saturates near E = 2.0 by n=4 (three of four prompts have α4 = 0.0–0.4), and a
single setting lands flat at n=5-6. A draft pays for itself while α_i > 0.15-0.16, which argues for an adaptive
count driven by the drafter's own confidence rather than a constant.

Verified together on a clean build of HEAD plus all three units: eleven gates pass, ids unchanged, step_batch and
the drafter chain both pass, `258048` at **76.30 / 76.27 / 76.36 tok/s** over three runs, 1k 144.8, 32k 126.1.
