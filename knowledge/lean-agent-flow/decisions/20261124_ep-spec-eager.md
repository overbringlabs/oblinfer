---
id: D-20261124-07
date: 2026-11-24
status: standing
areas: [serving, gpu]
question: How DSpark verifies when experts live on the second GPU: eager steps or dual graphs?
---

# DSpark on the eager two-device step

Isaak: go, option A. With a residency table that moves whole layers' experts to GPU1, a decode step spans two
devices and is no CUDA graph. DSpark runs on that eager step instead of waiting for dual graphs: the verify width
the drafts' confidence picks is read back to the host (`verify_width`, the same device rule as `verify_select`),
`run_step_eager` runs that width, and the acceptance/rollback graph is launched unchanged. One extra host
synchronisation per round; the next draft no longer overlaps the host's emission work.

Dual CUDA graphs (one per device, chained by events) are an optimisation for later, not a prerequisite for spec:
they would buy launch overhead only, and the rate at 262144 is what the bar measures.

The prompt under EP is processed one token per decode step (`EngineGenerator::prefill_steps`): the grouped prefill
expert GEMMs take one base pointer per expert tensor on this device. It is exact, and it is slow — one hop per
remote visit per token — so hopping the prefill GEMM is the step that follows. Image prompts need the chunked
prefill and are refused under EP.

One device, or a residency table that moves nothing: unchanged (captured graphs and the captured verify switch).
