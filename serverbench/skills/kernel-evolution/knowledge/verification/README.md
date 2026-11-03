# Verifying a kernel, in the order that wastes the least time

*Applies to: all. This is the kernel verification ladder; read it before writing a
check of your own.*

## The ladder

```
manifest → build → dispatch → parity → boundaries → numerics
        → determinism → negative controls → microbench
        → model integration → end-to-end serving
```

Cheapest first, and the ordering is the point: a candidate that does not compile
should learn that in two seconds, not at the end of a forty-minute benchmark.

## What each rung is actually for

**manifest** — the source is what it says it is and stays inside the candidate
root. Catches a build that reaches outside its own tree before anything runs.

**build** — compiles and imports *in a separate process*. Separate because a CUDA
extension that fails to load can take the process with it, and losing the harness
loses the evidence explaining why.

**dispatch** — a custom kernel actually ran. This is the anti-cheat rung, and it
is the only one that can catch a wrapper around the library the baseline already
uses. No numerical check can: the answer is correct. What is wrong is that the
library's speed is being reported as the candidate's.

**parity** — matches the reference on the nominal shape, at a dtype-aware
tolerance. A BF16 kernel checked at 1e-6 is misconfigured; one checked at 1e-1 is
not checked.

**boundaries** — every edge of the declared envelope, plus the inputs the kernel
must *refuse*. A kernel is wrong at boundaries or nowhere: the last partial tile,
the unaligned head dimension, the sequence one past a block.

**numerics** — finite values, poisoned guard regions intact, resources released
after a failure. A partial write is often plausible and occasionally zero, so this
has to be a poisoned buffer rather than a value comparison.

**determinism** — repeats, *if the candidate promised it would*. A kernel using
float atomics does not promise it and should not be failed for it.

**negative controls** — the rung that checks the rungs. See below.

**microbench** — may reject, may never accept.

**model integration** — still correct once wired into the forward path. An
operator correct in isolation and wrong in the model is the common case, not the
exotic one.

**end-to-end serving** — the only rung that can accept.

## Negative controls, and why they are the most valuable rung

A check battery quietly stops checking. A tolerance set slightly too wide, a
boundary case dropped during a refactor, a fixture that stopped being regenerated
— and from that moment every candidate passes, including the broken ones. It
looks exactly like a healthy battery.

The controls catch it: deliberately broken variants that the checks are *required*
to kill. If a mutant survives, the hole is exactly the shape of a defect that has
already shipped in a real kernel.

Two symmetric failures both fail the matrix:

- a mutant that **survives** — the checks cannot see that class;
- a legitimate variant that is **rejected** — the battery is over-sensitive and
  will now reject real work for the rest of the campaign.

And a control that **cannot be applied** — a moved target, a patch that no longer
fits — is neither. It is a broken tool, reported as `INVALID_CONTROL`, and it
qualified nothing. Controls therefore bind to a frozen source digest.

## Tolerance, per dtype

| dtype | relative floor |
|---|---|
| FP32 | 1e-5 |
| FP16 | 1e-3 |
| BF16 | 1e-2 |
| FP8 E4M3 | 6e-2 |
| FP8 E5M2 | 1.2e-1 |

Plus a matched-element ratio of about 0.99: a different-but-valid reduction order
genuinely flips a rare element, and demanding 100% rejects correct kernels. A
ratio low enough to let a systematically wrong kernel through is not a tolerance,
it is a formality.

## Two failures that no numerical check can see

**Silent fallback.** The kernel dispatches to the library on some inputs and does
not say so. Output is correct; the measurement is fiction. Only dispatch
telemetry catches it, which is why every capability must declare its fallback.

**Inaccurate timing.** Measuring without synchronising times kernel launch, which
produces enormous fictitious speedups. The timing protocol has to be able to
detect its own misuse; that is what the `inaccurate_timing` control is for.
