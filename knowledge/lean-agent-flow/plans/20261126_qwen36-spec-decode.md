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
