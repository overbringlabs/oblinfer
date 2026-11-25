# Progress

Updated in the same turn as code. This file is the **status head**: the STATE
block below plus the last two dated entries. When an entry ages out of the head,
it moves to `progress/{YYYY-MM}.md` beside this file, unchanged — the journal is the
archive of why, and it is never read at cold start.

Cap: the STATE block stays at or under ~80 lines. If it grows, the oldest
bullets move into the journal; the head answers "where are we", not "how did we
get here".

## In flight

- (none)

## Last completed

- 2026-11-25: dual-16 GB Blackwell serving on supra (5070 Ti + 5060 Ti, no P2P). Live llama-swap
  `inkling-2-mini-ingine`: prefix cache to ~237k, prefill ~1.1–2.8k tok/s, decode **108–122 tok/s**
  at that depth with DSpark (real chat). Bar 100–120 at 256k is met at ~237k. Installer and README
  match EP 64/64 + KV-split + `--no-vision --park-mb 0`. Timings on the last SSE chunk.
- 2026-11-24–25: two-GPU engine (second device, hop, EP decode, chunked prefill, KV-split 5/3,
  64/64 experts, child-graph splice, CS=8 cluster fallback).

## Needs a decision

- Second 5070 Ti when it lands: re-measure; even split stays honest. No code required for the bar.

## In-flight claims

- (none)

## Journal

- Older entries: `progress/{YYYY-MM}.md` beside this file (one file per month, newest first inside the month)
