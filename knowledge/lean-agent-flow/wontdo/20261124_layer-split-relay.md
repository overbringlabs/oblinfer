---
id: W-20261124-03
date: 2026-11-24
status: superseded-by:D-20261124-04
areas: [serving, gpu]
rule: "Do not pipeline-parallel / layer-split. Operator calls it a relay race; PCIe EP is the accepted 2-GPU shape."
---

# No layer-split relay

Reopen only if EP on this box cannot reach 100–120 tok/s at 256k after a measured attempt.
