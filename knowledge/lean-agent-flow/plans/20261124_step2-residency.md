# Step 2 — expert residency table (expert load split across two devices)

Plan, 2026-11-24. Parent ruling D-20261124-06. Step 1 (second `DeviceContext` + `--second-gpu`) is in flight in
this tree; this plan is written against the API it has already put on disk and does not edit its files.
Scope: the weights loader. No kernel change, no decode change, no RAM offload, NVFP4/MXFP8 encodings unchanged.

What step 2 buys: one table that says where each layer's expert bytes live, a loader that honours it on two devices,
and the plumbing (device scoping, teardown, reporting) that a later expert step needs. What it does not buy: a decode
step that reads device 1. `moe_up`/`moe_down` take **one** base pointer per expert tensor and compute `row = e*R`
(`kernels/moe/moe.cu:315,408`); the prefill grouped GEMM does the same (`kernels/prefill/wgemm.cu:120-121`). So a
layer's experts are addressable only as one contiguous range, and GPU-0 kernels may not dereference a device-1 pointer.
Under the default table (all primary) every allocation, byte order and graph is exactly today's.

## 1. Current load path (one device, implicit)

| Step | Where | Device today |
|---|---|---|
| Allocate | `DeviceBuffer(bytes)` → `cudaMalloc` on the current device, freed with `cudaFree` (`core/arena.cu:98-126`); no device field (`core/arena.h:20-41`) | whatever is current |
| Own | `Loader::alloc` appends to `Weights::storage` (`model/weights.cpp:68-71`, `model/model_data.h:206`) | one vector, order = allocation order |
| Queue | `Loader::queue`/`raw` build `UploadRequest`s for the raw checkpoint bytes (`weights.cpp:93-107`) | none recorded |
| Upload | `Loader::flush` → `DeviceUploader::upload(pending_, stream_)`, once per layer (`weights.cpp:108-112,317`) | uploader's construction device |
| Uploader | binds its device at construction, refuses a different current device, worker threads bind again (`core/safetensors.cpp:376-381,429-437,498,532`) | one device per uploader |
| Factors | `host_f32` + `upload_f32` → plain `cudaMemcpy` (`weights.cpp:73-85,306-315`) | current device |
| Entry | `Runtime` creates `stream_`, then `load_weights(dir, dense_fp8, stream_)` (`model/runtime.cpp:66-69`); `open_second_device()` keeps the primary current (`runtime.cpp:66-101` after step 1) | primary |

Expert tensors are the four `ld.raw()` calls per layer (`weights.cpp:296-302`): `w13`, `w13_scale`, `w2`,
`w2_scale`, one contiguous allocation each. Everything else in a layer is read by GPU-0 kernels — `wproj`/`wo` and the
SEG row views (`weights.cpp:160-166,327-329`), norms, router (`model_data.h:152-193`), loop gate stacks — and
`embed`/`unembed` (`weights.cpp:360-380`). Decode passes `Tensor::data` as a bare pointer (`decoder.cpp:232-238`), so
placement is invisible until a pointer is dereferenced on the wrong device.

## 2. Proposed table

```cpp
// model_data.h, beside Experts/Weights
struct ExpertResidency {
  // Experts [0, split[l]) on the primary device, [split[l], EXPERTS) on the second one.
  std::array<std::int16_t, LAYERS> split;      // ctor fills every entry with EXPERTS: all primary
  ExpertResidency();                           // NOT `{}`: zero-initialising the array means all-second
  bool all_primary() const;
  int device_of(int layer, int expert) const;  // 0 / 1
};
```

- **Default is all primary.** `split[l] == EXPERTS` for every layer; no second shard, no second uploader, no second
  allocation. `Experts` gains `split` plus second-shard views (`w13_hi`, `w13_scale_hi`, `w2_hi`, `w2_scale_hi`); the
  existing fields keep meaning "primary shard, rows `[0, split)`", so `model_data.h:136-144` keeps its shape and every
  call site compiles.
- **Granule is one layer's four tensors plus its factors** (546.75 MiB NVFP4 / 1002.4 MiB MXFP8). `w13`/`w2` must be
  on the same device or no kernel can run them, and the factors are read as `wfactor[e]` next to the codes
  (`moe.cu:318,409`), so they travel with their shard. 24 layers × {0,128} is a 546.75 MiB-granular partition — enough
  to balance the two cards (§5).
- **Index ranges are carried, not honourable yet.** The loader accepts `split[l] ∈ {0, EXPERTS}` and throws
  `std::invalid_argument("expert residency: split N needs per-expert base pointers (step 3)")` otherwise, because a
  mid-range split would need two base pointers inside one kernel. The column exists so step 3 changes no schema.
- **Guard.** `Runtime::init` throws unless `weights_.residency.all_primary()`, naming the peer-access measurement and
  the step-3 contract. `load_weights` itself stays callable with any table: that is the load-only entry tests use.
- **What stays primary, always**: `wproj`/`wo`/SEG views, norms, convs, router (fp32 + bf16 copy), loop gate stacks,
  queries, `embed`, `embed_norm`, `unembed`, `final_*`, `message_queries`, ablation, `State` (rings, full caches, conv),
  `Buffers`, drafter weights (`model/drafter.cpp:191`), vision tower (`model/vision.cpp:91`). Decode, prefill, head,
  spec and vision all run there and their graphs capture device-0 pointers.

`RuntimeOptions` gets `ExpertResidency expert_residency;` (default). No CLI in step 2: `--second-gpu` alone keeps every
expert on the primary (its help text is step 3's promise, and an inert flag beats a flag that loads bytes decode cannot
read). `ingine-serve`/`ingine-bench` stay step 1's files.

## 3. Files to edit in step 2

Wait for step 1's commit before editing `runtime.*` and `generator.h`; the API they add is already in the tree.

| File | Change |
|---|---|
| `native/src/model/model_data.h` | `ExpertResidency`; `Experts::{split,*_hi}`; `Weights::residency`, per-device byte counts |
| `native/src/model/weights.cpp` | `Loader` takes the second `DeviceContext`; per-device request buckets; split allocation; device-scoped binding; second uploader |
| `native/src/model/runtime.h` | `load_weights`/`load_layer` gain `const DeviceContext* second = nullptr, ExpertResidency = {}` |
| `native/src/model/runtime.cpp` | pass `second_device()`, set `weights_.residency`, add the `init()` guard |
| `native/include/ingine/generator.h` | `RuntimeOptions::expert_residency` (CUDA-free, defaulted) |
| `native/tests/model/test_residency.cpp`, `tests/model/CMakeLists.txt` | load-only two-device test (§4) |
| `native/DESIGN.md` | one paragraph: table, default, granule, guard |

Not touched: `core/device.*`, `core/safetensors.*`, `apps/serve/main.cpp` (step 1); `src/kernels/**`; `decoder.cpp`;
`prefill.cpp`; `state.cpp` (caches stay primary). If `DeviceBuffer` needs a device field, that is a join with step 1 —
default is no arena change: bind around `cudaMalloc`, and bind the owning device before `cudaFree` at teardown.

Loader mechanics: `alloc(bytes, device)` binds with an RAII scope and restores the previous device on exit; expert
tensors are the only allocations with `device == 1`. `Loader` holds `pending_[2]` and one optional
`DeviceUploader(opts, 1)` built with the small options (`{16 MiB, 4, 4}`, as `vision.cpp:77` does) — supra has 16 GB of
host RAM and the primary uploader already pins 384 MiB. `upload()` requires device 1 current
(`safetensors.cpp:429-437`), so the second flush runs inside the same RAII scope on `second->transfer_stream`; then
sync that device before `load_weights` returns. `load_weights` must end with the primary device current: the park
budget reads `cudaMemGetInfo` on whatever is current (`model/generator.cpp:82-87`).

## 4. Flag-off stays bit-identical

- The default is a value, not a branch: `split[l] == EXPERTS` ⇒ `alloc_hi` skipped, `pending_[1]` empty, uploader 1
  never constructed, one allocation per tensor in the same order on the same stream. `Weights::storage` has the same
  size, order and pointers as today, so `load_weights` returns byte-identical weights.
- `RuntimeOptions` gains a defaulted field; `load_layer`'s new parameters are defaulted, so `tests/model/test_layer.cpp:48`,
  `tests/prefill/bench_prefill.cu:103-105` and `tests/prefill/profile_prefill_model.cpp:56` compile unchanged.
- The `init()` guard compares an all-primary table: no-op. No kernel, graph or decode source changes.
- New test `model_residency` (MODEL + 2 GPUs, skip 77 otherwise, opt-in via `INGINE_RESIDENCY_TEST=1` so it never
  competes with a live server): loads with `split[l] = 0` for layers 20-23 (4.0 GiB on device 1) and checks per
  tensor that (a) the device-1 allocation's bytes equal the checkpoint range, (b) the device-0 bytes for those layers
  are absent, (c) `cudaMemGetInfo` on each device moved by the expected amount, (d) all-primary bytes equal a plain
  `load_layer` run. No decode.
- Locks to re-run: `model_layer`, `model_runtime`, `model_reuse`, `model_parity`, `model_parity_generate`,
  `model_parity_spec`, `model_path`, `model_path_spec`, `model_vision`, `spec_drafter`, `spec_forced`,
  `prefill_model_kv8`, `prefill_model_kv16`, `prefill_profile`, `kernels_bitlock`, `core_upload`,
  `core_safetensors`, `core_safetensors_model`.

## 5. VRAM

| Item | Bytes | Notes |
|---|---|---|
| One NVFP4 layer's experts (0-19) | 573,308,928 (546.75 MiB) | 4.27 MiB/expert, 4 allocations |
| One MXFP8 layer's experts (20-23) | 1,051,066,368 (1002.4 MiB) | 7.83 MiB/expert |
| All 24 layers' experts | 15,670,444,032 (14.59 GiB) | 20 × 546.75 + 4 × 1002.4 MiB |
| Dense fp8 `wproj`+`wo`, 24 layers | ≈ 0.62 GiB | 25.9 MB/layer (`model_data.h:85-98`) |
| `embed` bf16 + `unembed` fp8 | 1.29 GiB | 883 + 440 MiB |
| KV fp8, 8 full visits | 4.00 GiB | `KERNELS.md:108-110` |
| Sliding rings + conv state | ≈ 0.08 GiB | `model/state.cpp:37-67` |
| Drafter (raw `mtp`) | ≥ 0.59 GiB | plus derived fp8 copies, `drafter.cpp:191-250` |
| Vision tower | 0.59 GiB | `generator.cpp:80` |
| Parked conversations | ≤ 3 GiB | `generator.cpp:82-87` |

With every expert on the primary the weights alone are ≈ 16.5 GiB — over a 15.92 GiB card, before KV, drafter, vision or
park. With the maximal non-default table (`split[l] = 0` everywhere) device 1 holds 14.59 GiB of 15.92 (~1.3 GiB free)
and the primary holds ≈ 7.0 GiB before park, ≈ 10 GiB with a typical park budget. Both fit, so the table's first
useful setting is "all experts on the second card", not an even 64/64: capacity does not force a balance, bandwidth
does, and that is step 3's question (`todo/20261124_split-topology.md`). Layer granularity (546.75 MiB steps) is enough
to tune the leftover.

## 6. Open risks

1. **Current device leaks out of the loader.** Every `cudaSetDevice(1)` must be scoped; a stray current device makes
   `make_state`, the drafter, the vision tower, `Buffers` and the captured graphs allocate on device 1, and makes
   `cudaMemGetInfo` (`generator.cpp:84`) read the wrong card. Assert `cudaGetDevice(&cur); cur == device_` at the end of
   `load_weights` in debug builds.
2. **`DeviceBuffer` has no device.** `cudaFree` under a foreign current device is the failure mode; bind before free, or
   take step 1's lead if it adds a device field. Teardown today syncs only `stream_` (`runtime.cpp:139`); a
   device-1 shard needs its device quiet before its storage is destroyed.
3. **Two uploaders, two pinned pools.** 384 MiB (primary) + 64 MiB (second) pinned on a 16 GB host. Do not raise the
   second pool; the file reads are page-cache bound, not staging bound.
4. **Kernel single-base constraint.** Any split inside a layer needs `moe_up`/`moe_down`/prefill grouped GEMM to take
   `(base_lo, base_hi, split)`. That is a kernel change, has its own bitlock, and is step 3. Do not smuggle it in here.
5. **Peer access is unmeasured.** If `cudaDeviceCanAccessPeer(0,1) == 1` on supra, a GPU-0 kernel can read device-1
   memory and the table becomes usable for decode with no kernel change — slow, but the cheapest experiment available.
   Measure it before designing the step-3 pointer indirection; the `init()` guard is the single place to flip.
6. **Verify the expert size table against the checkpoint**, not this plan: `model_layer` already pins the format split
   (`test_layer.cpp:56`) and `hf_quant_config.json` is the authority (`weights.cpp:120-151`).
7. **Second 5070 Ti.** Nothing in the table assumes the 5060 Ti: it is a device index, and the granule is bytes. A
   later equal-bandwidth card changes only the choice of split, not the schema.
8. **Load time.** A second uploader serialises device-1 flushes behind device-0 ones. Full load is already disk-bound;
   measure the add before parallelising the two uploaders.
9. **Error path.** The OOM release hook (`arena.cu:98-126`, `generator.cpp:89`) frees parked conversations, which live
   on the primary; a device-1 allocation failure during load must not trigger it into a retry that hides the cause.

## 7. Step 3 handoff

Step 3 reads `Weights::residency` and `Experts::split`, adds `(base_lo, base_hi, split)` to the three expert entry
points, and executes each shard on its own device — the graph shape, PDL edges and arithmetic stay as they are. The
decision that comes before it is still `todo/20261124_split-topology.md`: EP (each card runs its own experts) versus
peer-memory reads from device 0. This plan's table serves both; nothing here presumes which one wins.
