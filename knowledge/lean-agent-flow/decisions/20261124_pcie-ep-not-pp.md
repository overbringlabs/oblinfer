---
id: D-20261124-03
date: 2026-11-24
status: superseded-by:D-20261124-04
areas: [serving, gpu]
question: Pipeline-parallel layer split, or PCIe expert-parallel with no P2P?
---

# PCIe expert traffic, not a layer-split relay

Isaak overrules the parent PP recommendation. Layer-split is a relay race. Non-P2P PCIe copies between the two Blackwell cards are accepted. Topology to design around: Maya-style expert parallel (dense + KV + attn on the faster card, experts sharded), all experts resident in VRAM, no host-RAM expert cache.

Parent's KV-on-GPU0 depth warning stays as a number to beat, not as a reason to go back to PP.
