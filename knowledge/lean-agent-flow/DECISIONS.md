# DECISIONS — generated index of rulings with reasons

Built from the front-matter of each body in `decisions/`, beside this index. Do not hand-edit a
row: amend the body and regenerate.

One line per ruling. Keep it under ~160 characters: date, status, title, areas,
the ruling in one sentence, path. Cross-cutting rulings get a row here;
feature-local ruling logs stay in their plan. Never delete a superseded entry —
its status says `superseded-by:D-…` and points at the replacement.

Budget check: the whole index is read at cold start, so ~500 chars per row is
the ceiling for a hundred rulings.

## Row shape

```
| **Title** (YYYY-MM-DD, standing) | areas | The ruling, one sentence. | `decisions/YYYYMMDD_slug.md` |
```

## Body front-matter

```yaml
---
id: D-YYYYMMDD-NN
date: YYYY-MM-DD
status: standing        # or superseded-by:D-YYYYMMDD-NN
areas: [money, offers]
question: What this ruling decides
---
```

## Standing

| **Explore first, then split** (2026-11-24, standing) | serving, process | No engine code until the two-GPU explore is in-tree and a split is approved. | `decisions/20261124_explore-first-two-gpu.md` |
| **Adopt lean-agent-flow** (2026-11-24, standing) | process | Ledgers live under `knowledge/lean-agent-flow/`; AGENTS.md is the kernel only. | `decisions/20261124_lean-agent-flow-adopted.md` |
