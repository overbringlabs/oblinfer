# Progress

Updated in the same turn as code. This file is the **status head**: the STATE
block below plus the last two dated entries. When an entry ages out of the head,
it moves to `progress/{YYYY-MM}.md` beside this file, unchanged — the journal is the
archive of why, and it is never read at cold start.

Cap: the STATE block stays at or under ~80 lines. If it grows, the oldest
bullets move into the journal; the head answers "where are we", not "how did we
get here".

## In flight

- DeepSeek Flash: rank EP vs PP (and a hybrid if it beats both) for 256k tok/s on 5070 Ti + 5060 Ti. Report: `plans/20261124_ingine-vs-maya-ds.md`.

## Last completed

- 2026-11-24: hardware = 5070 Ti + 5060 Ti, no P2P, 16 GB host RAM. Maya at depth ~60. PP back on the table (D-20261124-04).

## Needs a decision

- Which topology to implement first, after the DeepSeek ranking.

## Needs a decision

- Split topology (PP / TP / Maya-like) after the explore report. `todo/20261124_split-topology.md`.

## In-flight claims

- grok-parent: knowledge ledger files (this turn).
- turbo explore: will claim the plan file.

## Journal

- Older entries: `progress/{YYYY-MM}.md` beside this file (one file per month, newest first inside the month)
