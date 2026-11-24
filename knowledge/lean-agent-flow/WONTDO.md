# WONTDO — index of explicit rejections

Things decided against, so they never silently return. Bodies live in
`wontdo/{YYYYMMDD}_{slug}.md`, beside this index. Never delete an entry; a reversal is a new
decision that supersedes this row and says why.

## Standing

| **Chase 5090 tok/s** (2026-11-24, standing) | serving | Do not chase 490–560 tok/s on this dual-16 GB box. | `wontdo/20261124_chase-5090-toks.md` |
| **Expert RAM offload** (2026-11-24, standing) | serving, gpu | Do not offload experts to 16 GB DDR5-6000 host RAM. | `wontdo/20261124_ram-expert-offload.md` |
| **Single-GPU on supra** (2026-11-24, standing) | serving, gpu | Do not load ingine on one 16 GB card. Weights exceed 16 GiB. | `wontdo/20261124_single-gpu-supra.md` |
| **Layer-split relay** (2026-11-24, superseded by D-20261124-04) | serving, gpu | PP is back on the table. | `wontdo/20261124_layer-split-relay.md` |
