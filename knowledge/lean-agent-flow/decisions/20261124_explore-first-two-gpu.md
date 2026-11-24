---
id: D-20261124-01
date: 2026-11-24
status: standing
areas: [serving, process]
question: How do we start the two-GPU ingine work?
---

# Explore first, then split design

Isaak picked explore-before-code (option C). No engine code until the explore report is in the tree and a split design is approved.

The explore runs on a turbo subagent (`turbo/Ornith-1.5-35B-A3B-vision`) to keep SuperGrok off the grind. Parent synthesizes.

Isaak 2026-11-24 (same day): the **first** question is structural, not the split. Ingine is insanely faster than Maya even at shallow depth. Maya is a multi-model engine (Strata; GLM/DS/Qwen/Hy4/ik2). Ingine is Inkling-only. Find what ingine does that Maya does not, and do not destroy that when we go two-GPU.

Success bar for the *later* implementation (not this explore): load and run on supra's 2× RTX 5060 Ti 16 GB, 256k context, 100–120 tok/s at depth. Maya on the same box is ~80 tok/s at depth — that is the floor to beat. 5090-class 490–560 tok/s is not the bar (see W-20261124-01).
