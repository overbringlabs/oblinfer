---
id: T-20261126-09
date: 2026-11-26
status: parked
trigger: the Ornith-1.5-35B-A3B-NVFP4 checkpoint is on a local disk
areas: [ornith, verification]
question: Does the pack load and match a reference on Ornith's real weights, not just its index?
---

# Verify Ornith on real weights

Done so far, metadata only: all **94,393** tensor names classify (0 rejected) and the dialect detector now reports
`ornith` after the MTP-based fix (commit `1a0be91`). The export is 23.407 GB, carries a 333-tensor vision tower and
785 MTP tensors in the per-expert dialect.

Not done: loading it. The first run should use `QWEN35MOE_SNAPSHOT=<dir>` with the decoder test, then compare
greedy ids against a llama.cpp reference the way the NVIDIA checkpoint was anchored. Two things to watch: the
vision tower should be ignored rather than rejected, and Ornith numbers its MTP experts, so the MTP loader path is
exercised for the first time on real data.
