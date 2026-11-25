# LESSONS_LEARNED — generated index of re-diagnosis preventers

Built from the front-matter of each body in `lessons_learned/`, beside this index. Do not hand-edit
a row: amend the body and regenerate.

One lesson per body: symptom, cause, rule next time. If it only lives in one
agent's memory it is not a lesson. **The row is the rule, not the story** — the
body carries the evidence. A row that retells the defect turns the index into a
second copy of the corpus, and the cold start pays for both.

## Row shape

```
| **Title** (YYYY-MM-DD) | areas | Rule: the guard, in one sentence. | `lessons_learned/YYYYMMDD_slug.md` |
```

## Body front-matter

```yaml
---
id: L-YYYYMMDD-NN
date: YYYY-MM-DD
areas: [ecto, tests]
touches: ["lib/**", "test/**"]
rule: The guard, in one sentence, in the imperative.
---
```

## Rows

| **Cross-device event nodes do not pair across graph launches** (2026-11-25) | cuda, graphs, ep | Rule: order two devices inside one step with cross-device child graph nodes, never with an event pair split across two graph launches. | `lessons_learned/20261125_cross-device-graph-events.md` |
| **Self-consistent references prove nothing** (2026-11-26) | testing, numerics | An oracle from the same reading as the kernel only proves agreement; anchor interpretations to an external implementation. | `lessons_learned/20261126_self-consistent-oracles.md` |
