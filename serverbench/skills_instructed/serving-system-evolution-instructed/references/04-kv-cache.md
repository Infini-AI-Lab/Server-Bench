# KV cache design

The cache is where serving systems win, and where they go quietly wrong. A
scheduling bug shows up as bad latency. A cache bug shows up as fluent, confident,
incorrect text.

## What it has to do

Hold each request's keys and values so a decode step attends over the whole
history without recomputing it — while requests of different lengths join and
leave between steps, memory is finite, and something is always in flight.

## Static slots or paged blocks

**Static slots.** Each request gets a contiguous region sized for the maximum
length. Indexing is trivial, there is no indirection, and the memory is
predictable.

- *Good when* the length envelope is narrow, concurrency is bounded, and you want
  the simplest thing that is correct.
- *Costs* internal fragmentation: a request generating 20 tokens holds a slot
  sized for 2,000. Concurrency is capped by the worst case, not the typical one.

**Paged blocks.** Memory is fixed-size blocks; a request holds a list of them and
grows by acquiring more.

- *Good when* lengths vary widely, or when you want prefix sharing — two requests
  can point at the same block.
- *Costs* an indirection on every access, a block table to maintain and keep on
  the device, and a genuinely harder lifetime problem.

**Start static IF your workload lets you.** Read the corpus first: where the prompt distribution has a median in the tens of thousands of tokens and a p90 in the hundreds of thousands -- which is this suite's case -- a static layout sized for the worst case holds one or two requests on a 48 GB card, and the task is chosen precisely because it forces eviction and reuse. Elsewhere, static first is still the right order.

Where a static layout fits the workload: get correctness with the simple layout, measure whether
fragmentation is actually your limit, and page only if it is. Paging a system
that was never memory-bound adds indirection and buys nothing — and you will not
know, because you will have changed two things at once.

## Lifetime is the hard part

Whatever the layout, the same questions decide whether it is correct:

**Reference counts.** A block shared between requests is freed when the last
holder releases it. Freeing on the first release corrupts the other holder;
never freeing leaks until the server dies.

**Generations.** Reusing a slot without invalidating references to it means a
stale reader gets the new occupant's data. Attach a generation number and check
it, or make reuse impossible until every reader is retired.

**Reserve, commit, roll back.** Allocation happens while a plan is being built,
and a plan can fail after several requests have been allocated. Without rollback
you leak on every failed schedule — a leak that appears only under load, which is
the worst kind.

**Deferred free.** A block belonging to a request that finished during step N may
still be read by step N until it retires. Free it immediately and, under overlap,
the next occupant reads live data. Defer frees until the reading step has retired.

## Prefix reuse

Two requests that begin identically can share the computed prefix. The gain is
real; the correctness conditions are strict.

**Identity must be exact.** Same tokens, same positions, same order, from the
beginning. Not "the same text" — the same *token ids*. A different tokenizer
setting, a different template, one extra leading space, and the prefixes are not
the same prefix.

**Hash the content, not the string.** Include enough that no two different
histories collide: the token ids, and the identity of the parent block so that
the same tokens at a different offset are a different block.

**A pinned prefix is a policy decision.** Keeping a hot system prompt resident
saves recomputation and permanently reduces the memory available for everything
else. That is a trade to measure, not to assume.

**Test the near-miss.** Two prompts that differ only in the last token of the
shared region must produce different answers. If they produce the same answer,
your matcher is matching too much — the single most damaging cache bug there is,
because the output is fluent and the tests you wrote all pass.

## Two costs that hide well

**Full-history repacking.** If a decode step copies or reallocates the whole
history to append one token, per-step cost grows with sequence length for reasons
that have nothing to do with attention. Symptom: step time rising with generated
length even at fixed batch size.

The diagnostic is precise and worth doing: **measure bytes moved per step at two
very different history lengths.** Cache maintenance should be roughly *constant*
per step — one new key and value per layer per row. If it grows with history, you
are repacking. Check that the authoritative storage addresses do not move.

Note what this does *not* cover: the attention read itself legitimately scales
with history, and so does any expansion of keys and values for grouped-query
attention. Those are real, measurable costs and often worth removing — but they
are a different problem from repacking, and conflating the two sends you
optimising the wrong thing.

**Stale reuse.** Reusing a slot whose previous contents were not invalidated, or
reading beyond the valid region of a slot. Both produce plausible text. Zero on
release, or track the valid extent per row and never read past it.

## Cancellation, preemption, capacity

- **Cancellation** can arrive at any moment, including mid-step. Release must be
  idempotent and must not disturb other rows.
- **Preemption** frees a running request's blocks. Whether it resumes by
  recompute or by retaining blocks is a policy choice; either way the blocks are
  not reusable until every in-flight reader has retired.
- **Capacity failure** must be a clean refusal, not a crash and not a silent
  truncation. "Ran out of cache" is an admission-control failure that reached
  execution.
- **Compaction**, if you do it, has to be atomic with respect to in-flight steps.
  Moving a block a running step is reading is the hardest bug in this file to
  reproduce and the easiest to cause.

## Tests that catch real cache bugs

- Same prompt, different batch positions, different batch sizes → identical
  output.
- Two prompts sharing a long prefix but differing in the final shared token →
  different outputs.
- Fill the cache, force preemption, resume → identical output to the unpreempted
  run.
- Cancel mid-generation, then run a new request into the freed slot → the new
  request's output is unaffected.
- Cancel the same request twice → no error, no double free.
- Run to capacity → clean refusal, and the server still serves afterwards.
- Bytes moved per step at history 8 and history 600 → constant for maintenance.
- Long run at steady state → memory flat once settled. Measure the *settled*
  baseline, not a point sample taken before the allocator warmed up; a pre-warmup
  baseline manufactures growth that is not there.
