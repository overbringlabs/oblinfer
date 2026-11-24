---
id: C-20261124-01
date: 2026-11-24
status: superseded-by:C-20261124-02
areas: [serving, gpu]
rule: "Target host is supra: 2× RTX 5060 Ti 16 GB, PCIe 5.0 x8/x8. Product bar is 256k context at 100–120 tok/s at depth, beating Maya (~80 tok/s)."
---

# Supra two-GPU depth bar

- Host: supra (`192.168.88.14`). Fleet inventory: 2× RTX 5060 Ti 16 GB, PCIe 5.0 x8/x8, 32 GB *split*, not one 32 GB 5090.
- Context: 256k (262144) must remain the serving window. Cutting max-ctx to "whatever fits" is not the product.
- Speed: 100–120 tok/s at that depth. Maya (`inkling-2-mini-maya`, already loaded on supra) is ~80 tok/s at depth — that is the floor.
- Non-bar: 5090-class 490–560 tok/s. Bandwidth on these cards will not match.
- Engine today: single `DeviceContext`, `sm_120a` kernels, fused decode graphs. Weights on disk ~19.5 GB (`model-00001`) + 0.64 GB MTP.
