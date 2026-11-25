---
id: L-20261126-01
date: 2026-11-26
status: standing
areas: [testing, numerics, model-interpretation]
rule: A numpy reference written from the same reading as the kernel proves consistency, not correctness. Validate an interpretation against an independent implementation before trusting either side.
---

# Self-consistent references do not validate an interpretation

The Qwen3.6-35B-A3B decoder reproduced my own numpy reference exactly — 1e-7 for every GDN layer, 4e-4 for
the attention layers (BF16 KV rounding), the same expert outputs, the same head. It still produced garbage
tokens where llama.cpp produced ` Paris.`

Both implementations shared the same reading of the checkpoint: which rows of `in_proj_qkv` are q, k and v;
how q_proj splits query from gate; how the MTP/head tensors are arranged. Eight layout combinations (flat or
per-key-head qkv × flat or per-head query/gate) all produced different junk, which is the signature of a wrong
interpretation rather than a numeric bug.

Damage: hours spent confirming agreement between two copies of one assumption.

Rule for next time: an oracle must come from a different lineage. For a checkpoint, that means the publisher's
reference code path (transformers' module code, a vLLM/llama.cpp loader) or a converted artifact of the same
weights — not the same maths re-derived from the same tensor names. Use my own reference only to catch
regressions after the interpretation is anchored by something external.
