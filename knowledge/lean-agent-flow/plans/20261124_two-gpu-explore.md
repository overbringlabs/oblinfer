# Two-GPU explore: why ingine beats Maya even at shallow depth, and what survives a 2×16 GB split

Read-only exploration. Trees: ingine (this repo, editable deliverable), Maya (other repo, read-only), supra (read-only).
First question answered here: the structural reason ingine is faster at *shallow* depth. Second: which split keeps it.

---

## 1. Verdict — why ingine is faster at shallow depth

Shallow depth = KV cache is tiny, so the per-token cost is **moving the ~3.7 GB of weights**, not reading cache.
That isolates the real causes. Ranked biggest first.

| # | Reason | Mechanism in ingine | Maya counterpart | Evidence |
|---|--------|--------------------|------------------|----------|
| 0 | **~1.9× more DRAM bandwidth (hardware)** | RTX 5090 `sm_120a`, ~1.79 TB/s | RTX 3090 `sm_86`, ~936 GB/s | `native/KERNELS.md:3`; Maya `README.md:11`, `docs/INKLING2-MINI.md:104` |
| 1 | **PDL — overlap kernel N+1's weight loads with kernel N's compute** | `griddepcontrol` programmatic edges inside the graph; next kernel fetches weights before the current one finishes | none — Maya's graph is purely stream-ordered | `moe.cu:267-269`, `moe.cu:374-408`, `moe.cu:493-500`; `decoder.cpp:149-236` |
| 2 | **Fusion → ~half the launches** | `moe_down` fuses weighted-sum + post-MLP conv; `proj_conv` fuses proj + conv; `mix` writes side outputs so no separate copy nodes | fuses too, but keeps ~757 launches | `decoder.cpp`; `moe.cu:440-480`; `KERNELS.md:126,148-150`; Maya `docs/INKLING2-MINI.md:143-146` |
| 3 | **L2 evict-first for one-read weights** | weight loads use `L2::cache_hint` + `L1::no_allocate`, so they stop evicting the KV rings / activations | none in `ik2_kernels.cu` | `moe.cu:72-96`; `KERNELS.md:37-39` |
| 4 | **Interleaved weight layout** | `w13` rows interleaved (gate/up) → coalesced 128-B quad loads | GGUF layout | `KERNELS.md:19-20`, `moe.cu:290-299` |
| 5 | **Pipelined width-1 steps (no host sync)** | `chain_advance/token/undo` on a mapped host buffer + events: step N+1 launches before the host reads token N | host reads each token between steps | `runtime.cpp:205-217`; `DESIGN.md:233-236` |

**The single biggest shallow-depth factor is the 5090's bandwidth (#0).** It is hardware, not "what ingine does," and at
shallow depth both engines move the *same* ~3.7 GB of weight bytes, so a ~1.9× bandwidth gap is ~1.9× of the story on its own.
The structural wins that compound it — and the answer to "what does ingine do that Maya doesn't" — are **#1 PDL** (the big one),
then **#2 fusion** and **#3 L2 policy**.

### H1 is killed: ingine does NOT use native Blackwell FP4

The `sm_120a` tag does not mean native FP4 tensor cores. ingine's decode MoE decodes NVFP4 codes to **bf16 in registers** and runs
a **bf16** tensor-core MMA — the same register-dequant technique Maya uses with FP16.

- ingine decode MoE: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, weights decoded to bf16 in registers (`moe.cu:224-279`).
- ingine dense `gemv.cu:209`, attn `attn.cu:187-194` — all `mma.m16n8k16 bf16/f16 → f32`. No `wgmma`, no FP4 MMA anywhere.
- Maya: "decodes the 4-bit codes in registers and feeds the tensor cores" (`README.md:63-65`); register E2M1→FP16 decode `ik2_kernels.cu:2097-2135`.

Both read 4-bit codes from VRAM, decode in registers, sum in FP32. At shallow depth (bandwidth-bound on the same bytes) the
decode precision is not the differentiator. **H1 as stated is wrong.**

### H3 (weight format): partial, not a speed win
ingine ships its own NVFP4/MXFP8 safetensors; Maya loads Unsloth `UD-Q4_K_XL` GGUF. Both decode to the same bf16/f16 values and
both hit ~80% of DRAM bandwidth on the experts (`moe.cu` target; Maya `745 GB/s`), so the container differs but the decode
bandwidth does not. The interleaved `w13` layout (#4) is the only real edge here, and it is small.

### H4 (DSpark/spec): not a shallow-depth factor
Both have DSpark + a spec loop (verify-width switch graphs). Speculation changes tokens/second, not the per-token weight-bandwidth
cost that dominates shallow depth. Parity, not a differentiator.

### H5 (L2 / host sync / scheduler): partly confirmed
L2 evict-first (#3) and the pipelined `chain_` steps (#5) are real structural wins. Scheduler is single-request in both (no edge).

---

## 2. What Maya already does well (no strawman)

- **One CUDA graph per width** — 757–781 launches replayed as a single graph, GPU never waits for host (`README.md:57-59`). ingine does the same.
- **Register-dequant + tensor cores** — parity with ingine, not a Maya edge (`README.md:63-65`, `ik2_kernels.cu:2097-2135`).
- **Expert parallel (`IK2_EP`)** — the only 2-GPU path in either engine; dense + KV + attention on GPU 0, experts sharded (`ik2_main.cu:379-395`). ingine has no multi-GPU path at all.
- **FP8 KV, DSpark drafter, spec loop** — parity with ingine.
- **8 GB card path** — experts in pinned host RAM with a VRAM cache + CPU compute on miss (`docs/INKLING2-MINI.md:148-161`). ingine has no offload path.

---

## 3. What a 2-GPU split destroys

The shallow-depth win is a **single-GPU property**: all weights resident + one PDL graph + no cross-GPU hop. Every split trades
some of that. Which survives:

| Split | Keeps resident weights? | Keeps PDL single graph? | Handles the loop + message passes? | Verdict on 2×16 GB PCIe |
|-------|:----------------------:|:-----------------------:|-----------------------------------|-------------------------|
| **Pipeline (layers)** | yes | **no** — cross-GPU barrier every layer | **no** — `message_gates` straddle the cut, loop folds across GPUs | Reject. Fragments the decode graph, kills PDL overlap, breaks message passing. |
| **Tensor (matrix shard)** | partial | **no** | **no** — token routes to 8 experts → all-to-all per token | Reject. All-to-all expert routing on PCIe is catastrophic. |
| **Expert (IK2_EP)** | **yes (per shard)** | **partial** — cross-GPU fetch for off-GPU experts | yes (experts are per-layer, no message coupling) | Only viable. See §4–5. |

---

## 4. 256k @ 100–120 tok/s on 2×16 GB — VRAM math

Weights do **not** fit one 16 GB card, so a split is forced regardless of preference.

| Item | Bytes | GiB | Source |
|------|------:|----:|--------|
| Model safetensors (NVFP4/MXFP8 + dense) | 19,526,019,168 | 18.18 | `runtime-manifest.json:49-50` |
| Drafter (`mtp.safetensors`) | 635,801,416 | 0.59 | `runtime-manifest.json:54-55` |
| KV fp8, 8 full-attention visits @ 262144 | — | ~4.0 | `KERNELS.md:108-110` |
| Sliding BF16 rings (1024 keys) | — | ~1.0–1.5 | `KERNELS.md:86-92` (estimate) |
| Activations / workspace / graph overhead | — | ~1.0–2.0 | estimate |
| **Total** | | **~24–26** | |

- **One 16 GB card: impossible** — weights alone (18.2 GiB) exceed it, even at ctx = 0.
- **Two 16 GB = 32 GiB: fits under EP.** Shard the 128 experts ~64/64 → ~9 GiB experts/GPU, plus ~1.5 GiB dense/embed/head
  and ~2 GiB KV share on GPU 0. Both shards stay resident.
- **The catch (Maya's own warning):** if experts leave VRAM, decode collapses to 50–80 tok/s (`docs/INKLING2-MINI.md:80-96`,
  `README.md:60-66`). EP keeps them resident, but on 5060 Ti (no NVLink) an off-GPU expert is a **PCIe round trip**. With
  top-8 routing over 64 experts/GPU, almost every token's picks straddle both GPUs, so EP trades the PDL single-graph win for
  per-token PCIe expert traffic. The 100–120 tok/s target at depth is exactly what that penalty puts at risk — **unmeasured**.

---

## 5. Recommendation

**Split with expert parallelism (Maya-style `IK2_EP`), not PP or TP — but only as "the least-bad option," not because it
preserves the shallow-depth win.**

- PP and TP structurally destroy the decode (graph/PDL/message-passing, and all-to-all). Reject both outright.
- EP is the only split that keeps weights resident and the decode math intact, and it is the only one that fits on 2×16 GB.
- EP does **not** preserve #1 PDL or #0's full bandwidth advantage: off-GPU experts become PCIe fetches. Expect a depth-dependent
  penalty vs single-GPU shallow depth — which, on 16 GB cards, you cannot fully recover because the weights don't fit one card.
- Concretely: dense + KV + attention + drafter on GPU 0, experts sharded 64/64, `IK2_EP_SPLIT` tuned per card. Measure decode at
  256k before trusting the 100–120 tok/s number; if the PCIe expert traffic is too high, the honest result is "single-GPU
  shallow-depth speed is not reproducible on 16 GB cards — the weights won't fit."

---

## 6. Numbered questions for Isaak (answers change the build)

1. **Target: shallow-depth or depth?** The speed structure we found is shallow-depth (bandwidth on weights). On 2×16 GB you cannot
   match single-GPU shallow depth (weights don't fit one card). Do you want shallow-depth speed preserved, or is 256k-depth 100–120
   tok/s the only real target? — decides whether the whole split is worth it.
2. **Peer-to-peer between the two 5060 Ti on supra:** is `cudaDeviceCanAccessPeer = 1`, or must EP use host-mem copies (Maya's
   tested path, `ik2_main.cu:2287`)? — changes the EP cross-GPU copy path and its latency.
3. **Can `max_ctx` stay 262144, or lower it?** 18.2 GiB weights exceed 16 GB at any ctx, so single-card is out — but if a smaller
   context fits all weights on one card, that beats any split. Confirm you don't want single-card.

---

## 7. Did not try

- No compile, no build of ingine (`cmake`/`nvcc`), per the read-only rule.
- No run of Maya on supra; no live `nvidia-smi` / compute-cap check (would touch the live `inkling-2-mini-maya`). supra facts are
  unavailable without risk, so the 5060 Ti PCIe/no-NVLink assumption is from card specs, not measured.
- No ingine benchmark on supra — the 100–120 tok/s target and the EP penalty are modelled, not measured.
- Did not model the drafter's per-position context budget across the split at 256k.
- Did not crawl Maya's GLM / DeepSeek / Hy4 / mm3 trees — stayed in ik2 + docs.
- Did not verify Maya's exact launch count beyond its own docs (757 README / 781 guide); ingine's ~435 is a count of `decoder.cpp`.

---

## Parent check (grok-4.6, 2026-11-24)

Ornith's file:line on PDL, fusion, L2, graphs, and "no FP4 MMA" is good. Three corrections:

1. **H1 is only half-killed.** There is no FP4 *MMA* (`moe.cu:224-231` is `mma.m16n8k16.bf16`). There *is* Blackwell FP4 *conversion*: `cvt.rn.f16x2.e2m1x2` (`moe.cu:168-178`). Maya's ik2 path is a `__byte_perm` LUT, with an sm_86 special case (`ik2_kernels.cu:2097-2138`). README line 21 is this cvt, not tensor-core FP4. On a 5060 Ti (sm_120) that cvt is a real ingine edge if Maya is still the 3090 LUT.
2. **#0 hardware mix-up.** KERNELS.md 1.79 TB/s is a 5090. Maya README 936 GB/s is a 3090. Supra is 2× 5060 Ti. Do not use 1.9× as the supra story.
3. **EP is the wrong split for Isaak's bar.** Maya `IK2_EP` (and `IK2_TP`) keep attention + dense + **all KV on GPU 0** (`ik2_main.cu:379-390`). At 262144, full KV read is 4 GiB/token (`KERNELS.md:108-110`). EP leaves GPU0 holding that plus dense, GPU1 only experts. Depth ceiling is then one card. Pipeline-parallel puts half the full-attn visits (and their KV) on each card, so both 448 GB/s buses work at depth. That is the 100–120 tok/s arithmetic: ~3.7 GiB weights + 4 GiB KV, split, vs ~448 GB/s per card. EP cannot hit that. PDL across a layer cut is a real cost; it is smaller than parking 4 GiB/token of KV on one 16 GB card.

Q1 in §6 is already answered: 256k @ 100–120 tok/s is the bar; 5090 shallow speed is a wontdo. Q3 (cut ctx) is no. Q2 (P2P on supra) is still live.

**Parent recommendation: pipeline-parallel (layer/visit split), not Maya-style EP.** Measure `cudaDeviceCanAccessPeer` on supra before writing the bounce path.
