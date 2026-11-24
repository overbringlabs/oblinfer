# TODO — generated index of pending and parked work

Built from the front-matter of each body in `todo/`, beside this index. Do not hand-edit a row:
amend the body and regenerate.

A decision log, not a task board. Status per row: 🟡 awaiting a decision (design
not free to assume) · ⚪ parked with a named revisit trigger · 🔴 in progress.
A finished item leaves this index the same turn; its record goes to
`PROGRESS.md`. Do not revive a parked item without a new decision entry.

Read at cold start: the 🔴 and 🟡 rows, one line each. Parked rows are read on
demand — they are the answer to "is anything already waiting on this?".

## Row shape

```
| **Title** | 🟡 | What decides it; the trigger that unparks it | `todo/YYYYMMDD_slug.md` |
```

## Body front-matter

```yaml
---
id: T-YYYYMMDD-NN
date: YYYY-MM-DD
status: parked          # open | parked | in-progress
trigger: The event that unparks this
areas: [money]
question: What is pending
---
```

## Active

| **Split topology** | 🟡 | PP vs TP vs Maya-like; unparks when `plans/20261124_two-gpu-explore.md` lands | `todo/20261124_split-topology.md` |
