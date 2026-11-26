---
id: T-20261126-06
date: 2026-11-26
status: parked
trigger: the accept loop lands and the MTP path is measured end to end
areas: [spec-decode, ornith]
question: Does a block-diffusion drafter beat the MTP head on agentic coding, and what does its verify cost?
---

# DFlash as the second draft type (`--spec-type draft-dflash`)

`ornith-ai/Ornith-1.5-35B-A3B-DFlash` is 0.772 GB BF16, arch `DFlashDraftModel`: 6 layers, hidden 2048, 32 q /
8 kv heads, head_dim 128, intermediate 6144, **dense** MLP, 5 layers at sliding window 4096 plus 1 full. It proposes
a **block of 16** tokens in one parallel pass, conditioned on hidden states from target layers
`[1, 6, 11, 16, 22, 27, 32, 37]` (mask token 248077). KV at 262144 is about 325 MB.

So it is a different trade from MTP: sixteen proposals per pass against one, at a cost of a 16-token verify step
whose expert traffic rises from 19.8 unique experts of 32 (n=4) toward 28-30 (n=16), i.e. ~3.5x a single token.

The owner's concern is the right one: acceptance on agentic coding may be worse than MTP's 0.85-0.88. Measure both
on the same code prompts before choosing. Plan: implement as a second drafter behind the same accept loop, keep
MTP the default until numbers say otherwise, and consider selecting by content at run time.

Needs the eight layer taps available in serving, not just through the measurement dump.
