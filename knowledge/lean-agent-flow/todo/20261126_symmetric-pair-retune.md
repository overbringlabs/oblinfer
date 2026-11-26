---
id: T-20261126-04
date: 2026-11-26
status: parked
trigger: the second RTX 5070 Ti is installed and supra is back up
areas: [gpu, tuning]
question: What are the right split and band constants on 2x 5070 Ti, and what does the extra bandwidth buy?
---

# Retune for the symmetric pair

Today: 5070 Ti (851 GB/s measured) + 5060 Ti (427 GB/s measured) = 1278 GB/s combined. After the swap: 1792, which
is **+40% of memory throughput**, worth about 258k 13.1 ms -> 10.5 ms before any new work.

Constants that encode the old asymmetry and must be re-measured, not assumed:

- `KV_BAND_RATIO` is **2.0**, chosen against the 70/36 SM and 851/427 GB/s split. It wants 1.0 on a matching pair.
- The expert-split policy (agent D's) already falls back to the free-memory proportion when the two bandwidths are
  within a fifth, so it adapts; verify the log line says so, and sweep `QWEN35MOE_SPLIT` around the new default
  because the fit range moves (card 0 no longer carries 1786 MiB of bands against card 1's 894).
- Card 1's ceiling is 427 today and gates the band plan's balance point; re-run the band probe.

Every baseline in `plans/20261125_qwen36-35b-a3b.md` was taken on the asymmetric pair. Re-measure 16 / 1k / 2k /
32k / 98304 / 258048 / 262144 and rewrite the tables rather than scaling them.

What the swap does NOT change: total VRAM is still 32 GB, so K8V8 remains the 262k default and the 28.17 GB budget
stands. It also does not remove the per-layer serialisation (see T-20261126-07).
