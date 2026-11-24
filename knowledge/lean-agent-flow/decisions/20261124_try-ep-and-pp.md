---
id: D-20261124-04
date: 2026-11-24
status: standing
areas: [serving, gpu]
question: Which 2-GPU topology do we design first on supra?
---

# Try EP and PP; pick by 256k tok/s

Isaak 2026-11-24 (same day as D-20261124-03): pipeline parallelism is back on the table. He does not know which wins. The bar is deep decode: 100–120 tok/s, and **100 already beats Maya ~60 at depth**. PCIe copies without P2P are accepted. Host-RAM expert offload is still forbidden (16 GB DDR5-6000).

Design both. Rank by modelled tok/s at 262144 on 5070 Ti + 5060 Ti. Implement the winner first; keep the loser as the fallback if the winner misses 100.
