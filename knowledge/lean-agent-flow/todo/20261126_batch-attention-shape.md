---
id: T-20261126-01
date: 2026-11-26
status: parked
trigger: supra is back after the 5070 Ti swap, or any sm_120 pair is free
areas: [gpu, spec-decode]
question: Can a 4-token verify step cost no more than 1.6x one token at 258k?
---

# Batch verify attention shape (K3 gate)

`step_batch` is 1.85x at 1k (n=4: 3.72 ms/token against 6.87 for four single steps) but **fails at depth: n=4
costs 3.1x one token**, so speculative decoding is a net loss at 258k today (47.5 ms per 2.72 tokens = 57 tok/s
against the 76.3 tok/s we already have).

Cause, measured: the batch kernel pays the cache about once per query at 258k, 9.7 ms of marginal cost per query,
although it reads the rows once. Register pressure — n x (16 accumulators + 16 q) at the shipped 16-dim shape is
181 registers, one block per SM, against the shipped single-token kernel's four. At 8 dims per lane it is 104
registers, two blocks, and the pass got ~3x cheaper, which is the shape that shipped.

The fix to try first: run the verify block as **pairs of queries at the shipped 16-dim, two-stream shape**, so n=4
costs two cache passes instead of four passes' worth of time. Second candidate: `redux.sync.add.f32` for the score
reduction, which removes five shuffles per (row, query). Both are contained in `kernels/batch/batch_attn.cu`.

Gate: n=4 at 258048 at most 1.6x one token (about 21 ms), ids identical to n single steps, sanitizer clean.

Also worth folding in while that file is open: `BATCH_MAX` is 4, so a 5..8-draft chain pays two head passes;
raising it to 8 cuts the 7-draft head from 1.42 to ~1.0 ms (measured estimate, agent F).
