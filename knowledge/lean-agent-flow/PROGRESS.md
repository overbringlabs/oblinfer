# Progress

Updated in the same turn as code. This file is the **status head**: the STATE
block below plus the last two dated entries. When an entry ages out of the head,
it moves to `progress/{YYYY-MM}.md` beside this file, unchanged — the journal is the
archive of why, and it is never read at cold start.

Cap: the STATE block stays at or under ~80 lines. If it grows, the oldest
bullets move into the journal; the head answers "where are we", not "how did we
get here".

## In flight

- (none) Serve/bench take `--remote-experts` (working tree, uncommitted): every layer's experts on `--second-gpu`.
  `model_ep_spec` needs supra.

## Last completed

- 2026-11-24: EP+DSpark — spec runs on the eager step: the verify width is read back (`verify_width`) and
  `run_step_eager` runs it, the acceptance graph unchanged; a prompt is prefilled one token per step under EP
  (`prefill_steps`). One device / all-primary: byte-identical (captured graphs, captured verify switch).
- 2026-11-24: step 4 `1a5a47b` — hop moe to GPU1. Spec refused.

## Needs a decision

- Next: hop the prefill's grouped expert GEMM (prompt prefill is one token per step on a split), dual verify
  graphs, 64/64 split, or hold.

## In-flight claims

- (none)

## Journal

- Older entries: `progress/{YYYY-MM}.md` beside this file (one file per month, newest first inside the month)
