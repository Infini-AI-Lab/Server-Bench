# Sampling and top-k over the vocabulary

*Applies to: `lm_head`, `topk`, `sampling`.*

The vocabulary is large — 151,936 for Qwen3 — and it is not a power of two. Both
facts drive the design, and the second one causes more bugs than the first.

## Never sort the row

A full sort is `O(V log V)` for a result that needs the top 50. Two approaches
beat it:

**Radix select.** Four passes over 256 bins, narrowing to a candidate pool of a
few hundred, then a small sort of the pool. Total work is a handful of streaming
passes plus `O(k log k)`.

For the bit ordering: floats sort correctly as unsigned integers if you flip the
sign bit for non-negatives and all bits for negatives. That is the whole trick,
and it lets the bins be plain integer arithmetic.

**Pivot bisection for top-p.** Binary-search the probability threshold such that
the mass above it reaches `p`, then inverse-CDF sample among the survivors. No
sort at all, converges in a fixed number of iterations, and is typically about
twice as fast as sort-based top-p at this vocabulary size.

## The truncation bug

151,936 is 593.5 × 256. A kernel that assumes a whole number of tiles drops the
last partial tile — 128 logits — and does it silently. The tokens lost are rare
ones, so the output stays fluent and the defect surfaces as a model that never
produces certain tokens.

Test at the real vocabulary size. A test at 151,936 rounded down to 151,552 does
not test the thing that breaks.

## Batch and the grid

At batch 1, one block per row leaves the machine idle. Split each row across
several blocks and reduce their partial results. At batch 128 the opposite is
true and one block per row is right. This is a per-signature tuning axis, not a
constant.

## Greedy is not exempt

Greedy decoding is `argmax`, which is a reduction, which has all the same tiling
and boundary issues plus one of its own: **ties**. Two logits with the same value
must resolve the same way every time or the output is not deterministic. Pick
lowest index and stick to it; a reduction whose tie-break depends on thread
scheduling produces a model that gives different answers to the same prompt.

## What the check has to cover

- the exact vocabulary size, not a rounded one;
- ties, resolved deterministically;
- the candidate's chosen token equal to its own logits' argmax — a sampler that
  is internally inconsistent is wrong even if its output looks reasonable;
- `-inf` and NaN in the logits, which is what a masked token looks like.
