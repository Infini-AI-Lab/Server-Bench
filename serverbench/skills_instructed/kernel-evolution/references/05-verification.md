# Before you claim a kernel is correct

Nothing in your workspace runs these for you -- they are checks you write.
This is the part you have to think
about rather than run.

## What "correct" means here

Not "close enough to look right". Correct means: within a dtype-aware tolerance
of a reference that is **independent of your implementation**, at every point of
the declared envelope, refusing the inputs you said you would refuse, with no
NaN, no out-of-bounds write, and no undeclared fallback.

If your reference calls the same library your kernel dispatches to, it cannot
falsify anything and the whole ladder above it is decoration.

## Tolerance

| dtype | relative floor |
|---|---|
| FP32 | 1e-5 |
| FP16 | 1e-3 |
| BF16 | 1e-2 |
| FP8 E4M3 | 6e-2 |

Plus a matched-element ratio around 0.99. A different-but-valid reduction order
flips rare elements; demanding every element match rejects correct kernels. A
ratio loose enough to admit a systematically wrong kernel is not a tolerance.

Freeze the tolerance before you have results. Widening it after a candidate misses
is how a check stops checking.

## Boundaries are where kernels are wrong

Test at the edges, not in the interior:

- the smallest and largest value of every axis;
- one past a tile boundary and one before it;
- the shape that is not a whole number of tiles (a 151,936-entry vocabulary is
  593.5 tiles of 256, and the half is where the rare tokens live);
- a sequence **with history**, not a fresh one — a stale KV offset is invisible on
  the first decode step;
- non-contiguous input, because you will get one;
- inputs you must refuse: wrong dtype, wrong device, wrong rank.

## Negative controls, and why they matter most

A check battery quietly stops checking: a tolerance slightly too wide, a boundary
case lost in a refactor, a fixture nobody regenerates. From then on every
candidate passes, including the broken ones, and it looks exactly like a healthy
battery.

So the battery must be shown to *kill* deliberately broken variants. Each of these
is a defect that has shipped in a real kernel:

| broken variant | which rung must catch it |
|---|---|
| attend past the current position | parity |
| skip the QK normalisation | parity |
| read the KV cache at the previous offset | boundaries |
| reduce over a truncated vocabulary | boundaries |
| sum where the reference takes a max | parity |
| leave the last output elements unwritten | numerics |
| dispatch to the library without saying so | dispatch |
| time without synchronising | microbenchmark |

Two failures matter equally. A mutant that **survives** is a hole the shape of a
real defect. A legitimate variant that is **rejected** means the battery is
over-sensitive and will reject real work from now on.

And a control that **cannot be applied** — because the code it patched has moved —
is neither: it is a broken tool, it qualified nothing, and it must be reported as
such rather than counted as "no mutant survived".

## Two failures no numerical check can see

**Silent fallback.** The answer is right. The measurement is fiction.

**Inaccurate timing.** Without synchronisation you time the launch. This produces
enormous, entirely plausible speedups, and it is the single most common way a
kernel result turns out to be nothing.
