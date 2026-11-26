# TODO — generated index of pending and parked work

Built from the front-matter of each body in `todo/`, beside this index. Do not hand-edit a row:
amend the body and regenerate.

A decision log, not a task board. Status per row: 🟡 awaiting a decision (design
not free to assume) · ⚪ parked with a named revisit trigger · 🔴 in progress.
A finished item leaves this index the same turn; its record goes to
`PROGRESS.md`. Do not revive a parked item without a new decision entry.

Read at cold start: the 🔴 and 🟡 rows, one line each. Parked rows are read on
demand — they are the answer to "is anything already waiting on this?".

## Row shape

```
| **Title** | 🟡 | What decides it; the trigger that unparks it | `todo/YYYYMMDD_slug.md` |
```

## Body front-matter

```yaml
---
id: T-YYYYMMDD-NN
date: YYYY-MM-DD
status: parked          # open | parked | in-progress
trigger: The event that unparks this
areas: [money]
question: What is pending
---
```

## Active

| **Split topology** | 🔴 | Ranked: hybrid 111 / EP 91 / PP 61. EP + 3/8 KV split implemented (decode, chunked prefill); open until the supra tok/s numbers land. | `todo/20261124_split-topology.md` |

## Phase B of the speed push (2026-11-26)

Ordered. Each row names its prerequisite; none is free to start out of order.

| **Symmetric pair retune** | ⚪ | Second 5070 Ti installed. Band ratio 2.0 -> 1.0, split policy back to free-memory, re-measure every baseline; +40% memory throughput is worth 13.1 -> 10.5 ms at 258k on its own. | `todo/20261126_symmetric-pair-retune.md` |
| **Batch verify attention shape** | ⚪ | A 258k verify step costs 3.1x one token, so speculation loses at depth. Pairs of queries at the 16-dim shape; gate is 1.6x. | `todo/20261126_batch-attention-shape.md` |
| **Accept loop and draft count** | ⚪ | T-20261126-01 closes. GDN checkpoint and batched chain are landed; alpha to 8 is measured; the count should be adaptive (alpha_i > 0.15). | `todo/20261126_accept-loop.md` |
| **Trunk head through the batched kernel** | ⚪ | Next edit to decoder.cu. 0.608 -> 0.370 ms, +1.8% on every token. | `todo/20261126_trunk-head-batched.md` |
| **Attention staging gap** | ⚪ | Exchange or band plan touched anyway. 1.7 ms/token, 27% of the attention term, diagnosed only by subtraction. | `todo/20261126_attn-staging-gap.md` |
| **Card 1 idle 88%** | ⚪ | After the retune, if card 1's duty is under 50%. Pipeline layers first, route-on-card-1 second, dense move third. | `todo/20261126_card1-idle.md` |
| **DFlash as second draft type** | ⚪ | Accept loop landed. Block-16 drafter, 0.772 GB; measure against MTP on code before choosing. | `todo/20261126_dflash-drafter.md` |
| **Kernel micro-wins** | ⚪ | Any of those files open for another reason. Six measured items with numbers attached. | `todo/20261126_kernel-polish.md` |
| **Ornith on real weights** | ⚪ | Checkpoint on a local disk. Index verified (94,393 names, 0 rejected); loading and the llama.cpp anchor are not. | `todo/20261126_ornith-real-weights.md` |
