# Progress

Updated in the same turn as code. This file is the **status head**: the STATE
block below plus the last two dated entries. When an entry ages out of the head,
it moves to `progress/{YYYY-MM}.md` beside this file, unchanged — the journal is the
archive of why, and it is never read at cold start.

Cap: the STATE block stays at or under ~80 lines. If it grows, the oldest
bullets move into the journal; the head answers "where are we", not "how did we
get here".

## In flight

- (none) Steps 1–2 in tree. Next is step 3 (stub PCIe hop on supra) — needs Maya unloaded.

## Last completed

- 2026-11-24: step 1 `--second-gpu`. Step 2 expert residency table (default all GPU0; split 0 or 128 only; decode guard if any experts on GPU1).

## Needs a decision

- Go on step 3 hop measurement (unload Maya), or hold.

## Needs a decision

- Split topology (PP / TP / Maya-like) after the explore report. `todo/20261124_split-topology.md`.

## In-flight claims

- grok-parent: knowledge ledger files (this turn).
- turbo explore: will claim the plan file.

## Journal

- Older entries: `progress/{YYYY-MM}.md` beside this file (one file per month, newest first inside the month)
