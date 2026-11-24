---
id: W-20261124-02
date: 2026-11-24
status: standing
areas: [serving, gpu]
rule: "Do not offload experts to system RAM. Supra has 16 GB DDR5-6000."
---

# No expert offload to host RAM

Reopen only if supra's host RAM becomes large enough to hold expert blobs without paging.
