# Progress

Updated in the same turn as code. This file is the **status head**: the STATE
block below plus the last two dated entries. When an entry ages out of the head,
it moves to `progress/{YYYY-MM}.md` beside this file, unchanged — the journal is the
archive of why, and it is never read at cold start.

Cap: the STATE block stays at or under ~80 lines. If it grows, the oldest
bullets move into the journal; the head answers "where are we", not "how did we
get here".

## In flight

- (none) Waiting on Isaak: PP vs Maya-style EP, after parent check of the explore.

## Last completed

- 2026-11-24: Ornith explore landed `plans/20261124_two-gpu-explore.md`. Parent check: PDL/fusion/L2 stand; H1 half-wrong (`cvt.e2m1x2` is real); EP is wrong for 256k (KV stays on GPU0). Recommend PP.

## Needs a decision

- Split: pipeline-parallel (parent) vs Maya `IK2_EP` (Ornith). P2P on supra still unmeasured.

## Needs a decision

- Split topology (PP / TP / Maya-like) after the explore report. `todo/20261124_split-topology.md`.

## In-flight claims

- grok-parent: knowledge ledger files (this turn).
- turbo explore: will claim the plan file.

## Journal

- Older entries: `progress/{YYYY-MM}.md` beside this file (one file per month, newest first inside the month)
