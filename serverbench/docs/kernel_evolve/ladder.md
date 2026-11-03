# The ladder

Eleven rungs, cheapest first.

```
 10  kernel_manifest         source is what it says it is, and stays in scope
 20  kernel_build            compiles and imports, in a separate process
 30  kernel_dispatch         a custom kernel actually ran
 40  kernel_parity           agrees with the reference on the nominal shape
 50  kernel_boundaries       ... and at every edge of the declared envelope
 60  kernel_numerics         finite, guards intact, no memory damage
 70  kernel_determinism      repeats, if it promised to
 80  kernel_negative_controls  the checks above can still kill a known defect
 90  kernel_microbenchmark   faster in isolation      (may reject, never accept)
100  kernel_model_integration  still correct inside the model
110  kernel_serving_endtoend   the server is correct and faster
```

## Why the order

Cost. A candidate that does not compile should learn that in two seconds, not at
the end of a forty-minute end-to-end benchmark. Every rung that would have spent
device time on a candidate an earlier rung could have killed is a rung in the
wrong place.

The stopping rule is fail-fast, and everything that consequently did not run is
still **recorded**, marked `SKIPPED` with the rung that stopped the ladder. "Nine
gates passed" reads very differently when the tenth is missing than when it is
present and says why.

## What each rung is for

**manifest** — declared source exists, hashes as declared, resolves inside the
candidate root. A build that reaches outside its own tree is refused before
anything executes, because a build script runs arbitrary code by design.

Also: a manifest with no device-language source fails here. Whatever that
candidate is doing, it is not evolving a kernel.

**build** — compiles and imports **in a child process**. Separate because a CUDA
extension that fails to load can take the process with it, and losing the harness
loses the evidence explaining why. Also where a broken toolchain is told apart
from broken code: `nvcc: command not found` is `INVALID`, an undefined identifier
in the candidate's own source is `FAIL`.

**dispatch** — a kernel the candidate wrote actually executed. The only rung that
can catch a wrapper around the library the baseline already uses, because no
numerical check can: the answer is correct. What is wrong is the attribution.

**parity** — matches the reference at a dtype-aware tolerance frozen before the
result. The reference must be independent of the candidate; one that calls the
same library cannot falsify anything, and every rung above it becomes decoration.

**boundaries** — every declared edge, plus the inputs the kernel must **refuse**.
A tiled kernel is right in the middle of its domain by construction and wrong at
the edge by omission.

**numerics** — finite output, poisoned guard regions intact, resources released
after a failure. A partial write is often plausible and occasionally zero, so
this has to poison the buffer rather than compare against a fresh allocation.

**determinism** — repeats, *if the candidate promised it would*. A kernel using
float atomics does not promise it and must not be failed for it.

**negative controls** — see [controls.md](controls.md). The most valuable rung.

**microbenchmark** — advisory. It exists to kill ideas cheaply, which it is very
good at.

**model integration** — correct once wired into the forward path. An operator
correct in isolation and wrong in the model is the common case.

**end-to-end serving** — the only rung that can accept.

## How skipped and advisory rungs are decided

**A skipped required rung judged nothing.** A required gate that is `SKIPPED`
makes the decision `INVALID`, naming the rung; skipping a rung is never a route
to acceptance without it.

**Only a required rung can contradict an acceptance.** An advisory
microbenchmark that could not decide, or a rung skipped and never required,
does not block one. The decision records which gates were required, so it can
be re-derived from the document rather than from the policy it names.

A rung that raises is `INVALID`: the evaluator failed, not the candidate.

## The smoke

`kernel_evolve.smoke` runs the whole ladder against a real Triton kernel on a
real device, requires every corruption class that applies to its operator to be
killed, and checks that no device memory is left allocated. That is a statement
about the harness, not a benchmark result, and it makes no performance claim.
