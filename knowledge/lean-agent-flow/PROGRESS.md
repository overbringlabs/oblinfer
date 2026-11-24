# Progress

Updated in the same turn as code. This file is the **status head**: the STATE
block below plus the last two dated entries. When an entry ages out of the head,
it moves to `progress/{YYYY-MM}.md` beside this file, unchanged — the journal is the
archive of why, and it is never read at cold start.

Cap: the STATE block stays at or under ~80 lines. If it grows, the oldest
bullets move into the journal; the head answers "where are we", not "how did we
get here".

## In flight

- (none) Step 4 committed. EP decode is eager; spec and prefill refuse a split. Waiting on Isaak.

## Last completed

- 2026-11-24: step 4 `92d1688` — `split==0` hops moe to GPU1 (`moe_down_rows` + `moe_sum`). Default fused path unchanged.
- 2026-11-24: hop on supra 27.1 µs PASS.

## Needs a decision

- Next: EP+DSpark (standing bar), 64/64 split, or hold.

## In-flight claims

- (none)

## Journal

- Older entries: `progress/{YYYY-MM}.md` beside this file (one file per month, newest first inside the month)
