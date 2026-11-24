# Step 3 — cross-device hop measured on supra (GPU0 + GPU1)

Measured, 2026-11-24, on the target box. Probe: `native/apps/hop/` → `ingine-hop`, compiled on turbo with CUDA
13.4 (`/usr/local/cuda/bin/nvcc`, `CMAKE_CUDA_ARCHITECTURES=120a`), copied to supra and run there. No llama-swap
model was loaded; both cards were at 2 MiB used. This is the gate that §7.3 of
`plans/20261124_ingine-vs-maya-ds.md` puts in front of the MoE split.

## 1. Verdict

| | Result |
|---|---|
| `cudaDeviceCanAccessPeer` | **0->1 = 0, 1->0 = 0** (never enabled; matches the plan) |
| Hop, mean | **27.1 µs** (runs: 27.10 / 27.27 / 27.10) |
| Hop, p50 / p99 | 26.66 / **39.6 µs** |
| 48-hop token | **1.30 ms** (48 × 27.1 µs) |
| Gate 40 µs/hop (mean) | **PASS** |
| **Step 4 (MoE split)** | **may proceed** |

27 µs is inside the plan's 10–40 µs uncertainty band and near its 20 µs working point, so no re-rank is needed:
EP-only stays ~85–89 tok/s at 85–90% DRAM, and the hybrid KV-head step stays at ~101–111 tok/s (§5). The hop is
stably under the gate on the mean, but the tail is real: ~1% of hops land at 40–55 µs (§4), which is why the
hybrid's wait budget must stay amortised rather than paid per activation.

## 2. Hardware as the probe sees it

| | GPU0 | GPU1 |
|---|---|---|
| Name | NVIDIA GeForce RTX 5070 Ti | NVIDIA GeForce RTX 5060 Ti |
| Compute capability | 12.0 | 12.0 |
| SMs | 70 | 36 |
| VRAM used at run | 2 MiB | 2 MiB |

The sm_120a cubin ran on CC 12.0 without a "no kernel image" failure — no fallback build was needed, and none was
made. Probe reports runtime CUDA 13.4.

## 3. What was measured

Per hop, the Maya path with no peer access: GPU0 kernel writes 4608 B (`bf16 [1, 2304]` hidden) → DtoH into a
portable pinned host buffer → `cudaStreamWaitEvent` → HtoD into GPU1 → GPU1 stub expands it to 8 × 2304 bf16 rows
(36864 B) → DtoH into a second pinned buffer → event → HtoD into GPU0 → GPU0 consume kernel. Streams are
non-blocking, the two cross-device dependencies are `cudaEventDisableTiming` events, and the hot path never calls
`cudaDeviceSynchronize` or a per-hop host sync. 48 hops per "visit" (one token-shaped number of index visits).

Instrumentation: one CUDA event pair on the GPU0 stream around every hop, one pair around every visit, read after a
single end-of-phase sync. Payload check after warmup: all 36864 B of the returned rows are bf16 1.0 — the path
moves data, not just timed events.

Run command on supra: `CUDA_VISIBLE_DEVICES=0,1 ./ingine-hop --visits 200 --seconds 2` (200 timed visits = 9600
hops, then an untimed stream until 2 s total). Wall tok/s is visits/wall of the timed phase; the hop-mean rate
(1e6 / token µs) is 764–769 tok/s and is the number §5 feeds into the model.

## 4. Numbers

| Run | hops | mean µs | p50 µs | p99 µs | min µs | max µs | visit µs | visit/48 | token ms | wall tok/s | extra tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9600 | 27.10 | 26.69 | 39.62 | 15.97 | 53.95 | 1324.7 | 27.60 | 1.301 | 755 | 814 |
| 2 | 9600 | 27.27 | 26.66 | 39.62 | 14.69 | 56.35 | 1332.8 | 27.77 | 1.309 | 750 | 811 |
| 3 | 9600 | 27.10 | 26.66 | 39.58 | 15.01 | 54.27 | 1324.5 | 27.59 | 1.301 | 755 | 809 |

Two cross-checks that the event instrumentation is not creating the latency:

- **Visit timer / 48 = hop mean.** The visit pairs record once per 48 hops, so they carry almost none of the
  per-hop event overhead, yet they agree to within 0.5 µs (27.6 vs 27.1). The per-hop events are not the cost.
- **Floor is ~15–16 µs.** The minimum hop is 15.0–16.0 µs across runs — that is the four-transit round trip with no
  queueing. The mean carries ~11 µs of scheduling jitter on top, and p99 carries ~24 µs.

The probe enqueues the hop as ~11 host API calls (3 launches, 4 copies, 2 records, 2 waits; 2 more records when
timed) with no CUDA graphs, so
27 µs includes host enqueue overlap and is an **upper bound** for a graph-replay path. It is also the honest number
for a per-hop `cudaStreamWaitEvent` design, which is exactly what step 4 will implement before graphs land.

## 5. Effect on the plan's model

Re-running §4.4's totals with 27 µs instead of the 20 µs working point (48 waits for EP, 56 for the hybrid):

| Topology | 20 µs | **27 µs (measured)** | 40 µs |
|---|---:|---:|---:|
| EP only, 90% DRAM | 91 | **89** | 84 |
| EP only, 85% DRAM | 87 | **85** | 80 |
| Hybrid 3/8 KV heads, 95% | 116 | **111** | 103 |
| Hybrid, 90% | 111 | **106** | 99 |
| Hybrid, 85% | 105 | **101** | 94 |

Read: the measurement does not move the rank and does not kill the 100 bar. EP alone is still short of it by ~11
tok/s (as planned); the hybrid still clears it, with ~1 tok/s of room at the pessimistic 85% end. That last margin
is the number to watch in step 5 — the hop is 1.52 ms (56 waits) of the hybrid's ~9.4 ms token, and it is the term
that grows fastest if per-visit waits are not amortised.

## 6. Friction, and what broke

- **sm_120a on CC 12.0: works.** `cuobjdump -lelf` shows `main.sm_120a.cubin`; it loaded and ran on both cards.
- **First two runs failed** with `cudaErrorInvalidResourceHandle` at `cudaEventRecord`. Cause: events are owned by
  the device that was current at `cudaEventCreate`, and the program left device 1 current, so the dependency and
  timing events belonged to GPU1's context and could not be recorded on GPU0's stream. Fix: set the recording
  device before creating each event. Not a platform or driver issue — no retry loop, no driver flag changed.
- Nothing else failed. No P2P was enabled at any point, no paid service was touched, nothing was committed.

## 7. Files

| File | Role |
|---|---|
| `native/apps/hop/main.cu` | probe: kernels, pinned staging, streams/events, stats |
| `native/apps/hop/CMakeLists.txt` | `ingine-hop`, `CUDA::cudart` only, inherits the 120a gate |
| `native/CMakeLists.txt` | `hop` added to the app list (two-line change) |
| built binary | `native/build-hop/apps/hop/ingine-hop`; ran on supra as `/tmp/ingine-hop` (copy) |

Build: `cmake -S native -B native/build-hop -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
-DCMAKE_CUDA_ARCHITECTURES=120a` then `cmake --build native/build-hop --target ingine-hop`.

## 8. Did not do

- No CUDA-graph or device-side doorbell variant; both would lower the number and neither is step 3.
- No separate forward/return transit timing, so the 4.6 KiB direction is not isolated; the 36 KiB return is 8× the
  payload and likely the larger half, but that is inference, not measurement.
- No PCIe counter or `nvidia-smi` bus-utilisation sampling, and no second topology (x8 vs x16) comparison.
- No multi-minute soak; three 2.3 s runs are the extent of the stability evidence.
