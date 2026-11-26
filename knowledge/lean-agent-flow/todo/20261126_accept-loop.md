---
id: T-20261126-05
date: 2026-11-26
status: parked
trigger: T-20261126-01 closes, i.e. a 258k verify step costs at most 1.6x one token
areas: [spec-decode]
question: What does the accept/rollback loop look like, and how many drafts should it run?
---

# Accept loop and draft count (K5)

Pieces already landed: `step_batch` with the GDN checkpoint/restore (62.9 MB recurrent + 2.9 MB conv, one D2D
copy), the batched draft chain, and the measured acceptance to alpha8.

Acceptance over generated text, pooled: 0.758 0.728 0.736 0.779 0.797 0.826 0.889 0.900. It rises, so the draft
count is set by content, not by a constant:

| content | recommended n | E(tokens/step) |
|---|---|---|
| code | 8 | 5.65 on the C prompt, 3.85 on Python |
| prose | 4-5 | 3 of 4 prompts saturate near 2.0 |
| one setting for both | 5-6 | pooled ms/token flat |

A draft pays for itself while alpha_i > 0.15-0.16 (a draft is 0.42 ms at 1k, 1.33 at 262k; a token is worth 2.7 ms
at 1k, 8.4 at depth). That argues for an **adaptive count driven by the drafter's own confidence**: stop drafting
when the proposal's margin collapses, instead of a fixed n.

The loop itself: accept the longest matching prefix, advance pos by that count, restore the GDN state for rejected
positions (KV rows need no rollback), and re-draft from the accepted token. Greedy only at first.

Gate: end-to-end greedy ids identical to non-speculative decoding over 16 tokens at four depths, then the tok/s
table. Graph capture stays unsupported on the batch path until someone makes the banded launch capture-legal; say
so rather than pretending.
