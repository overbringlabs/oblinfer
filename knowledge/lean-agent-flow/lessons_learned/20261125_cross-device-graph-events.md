---
id: L-20261125-01
date: 2026-11-25
areas: [cuda, graphs, ep]
touches: ["native/src/model/**"]
rule: Order two devices inside one step with cross-device child graph nodes, never with an event pair split across two graph launches.
---

# Cross-device event nodes do not pair across graph launches

Symptom: a plan for two-device decode as "two graphs per width, one per device, ordered by the hop's
`cudaEventRecord`/`cudaStreamWaitEvent` pair". Captures fine, replays fine, wrong answers.

Cause (measured on turbo, CUDA 13.4, two GPUs, sm_89; graph semantics are arch-independent):

- An external event **wait node binds to the event's most recent record when the node runs**, and a wait can
  never pair with a record that a graph launched *after* it will produce. Two graphs launched per step each
  wait on the other's previous-step record — deterministically stale, and a one-directional flow looks correct
  because the producer graph was launched first.
- Probe: `A_j` records `ready`; `B_j` waits and checks a per-index slot. `B` launched after `A` sees `A_j`
  (0 violations with a slow producer, 100% with no delay). Ping-pong (each graph waits on the other, `A` first)
  gave 10/10 violations: `A_j` consumed before `B_j` wrote. The `cudaEventRecordExternal` /
  `cudaEventWaitExternal` flags are what captures legally, not what pairs.
- What works: a graph captured on the second device (device bound **outside** the capture, no `cudaSetDevice`
  inside it) added to the first device's graph with `cudaGraphAddChildGraphNode`. Cross-device child nodes
  execute on their own device and their edges are the cross-device dependency. Splice them mid-capture with
  `cudaStreamGetCaptureInfo` + `cudaGraphAddChildGraphNode` (fork) and
  `cudaStreamUpdateCaptureDependencies(..., cudaStreamAddCaptureDependencies)` (join). Probe: 20 spliced children
  x 50 steps through pinned staging, 0 violations; the fork/join form keeps the local half running beside the
  hop.

Rule: capture the second device's work as its own graph and make the step graph hold it as a child node; keep
the event pair for the eager path only. Re-measure the hop cost: the child node is dispatch-bound, so the hop
does not get cheaper — the win, if any, is one launch per step.
