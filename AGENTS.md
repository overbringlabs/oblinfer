# AGENTS.md

Ingine: C++/CUDA OpenAI-compatible server for Inkling-2-Mini (`native/`). Single `DeviceContext`, kernels built for `sm_120a`. Serving profile in `runtime-manifest.json` (262144 ctx, dense-fp8, kv-fp8, spec).

This checkout's live goal is running that engine on **supra** (2× RTX 5060 Ti 16 GB). Do not implement a multi-GPU split until `knowledge/lean-agent-flow/plans/20261124_two-gpu-explore.md` exists and Isaak picks a topology. Depth bar: 256k context at 100–120 tok/s; Maya on the same box is ~80 tok/s.

## Knowledge map (lean-agent-flow)

Paths below are under `knowledge/lean-agent-flow/` unless noted. Long form: skill `lean-agent-flow`. This section carries zero status.

### Coordination kernel (always)

1. Collision signal is **in flight**: a live claim in `IN_FLIGHT.md`, or a write in the last minutes. `git status` answers uncommitted, not in progress.
2. Uncommitted, unclaimed, idle is **parked**: edit on top of it and commit the step. Operator hand-edits (secrets, `.env`, mid-rename, `.bak`) are his — reroute.
3. Before the first edit, write **your** block in `IN_FLIGHT.md`. Delete it when the step commits. Never rewrite another agent's block. A claim older than ~15 minutes with no fresh write is stale — delete it.
4. A blocked path is a blocked **step**: take the next ready step in the same plan and circle back. Halt only when no step is ready.
5. Stage only paths in your write-set. Never `git add -A`. Foreign live bytes are theirs to commit.

### Ledgers (read what the task needs)

- Principles — `MINDSET.md` (always)
- Claims — `IN_FLIGHT.md` (always, before an edit)
- Status head — `PROGRESS.md` (STATE + last two entries; journal in `progress/`)
- Forbids — `CONSTRAINTS.md`, `WONTDO.md`
- Indexes — `DECISIONS.md`, `TODO.md` (🟡/⚪), `LESSONS_LEARNED.md`
- Bodies, plans, cookbooks — by path, never wholesale
