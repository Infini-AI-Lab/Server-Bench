# `knowledge/` — what a kernel writer needs to know before proposing

Part of the kernel-evolution skill, and kept apart from the serving skill on
purpose. The serving skill is about *serving systems*: schedulers, batching, KV
ownership, the shape of a request's life. This tree is about *kernels*: what the
hardware can do, what the operator has to compute, and what will make a
borrowed design silently wrong.

An agent asked to make a server faster that reads a pile of GEMM tiling advice
first tends to conclude the answer is a GEMM before anything has profiled the
scheduler. Keeping the two apart lets the routing decision — *is this even a
kernel problem?* — be made on evidence.

## How to read this

Not front to back. The corpus is bigger than any one candidate needs, and reading
all of it is how a context window gets spent on material that turns out to be
about a different operator.

1. **`hardware/`** — read the entry for the architecture you are on, once, at the
   start. It is the constraint set everything else is written against.
2. **`profiling/`** — read before proposing anything. The point of a profile is to
   tell you whether a kernel is the answer at all.
3. **The operator directory** for the thing your profile pointed at:
   `attention/`, `gemm/`, `normalization_rope/`, `sampling/`, `moe/`,
   `communication/`.
4. **`models/`** — the model you are serving. This is where the semantics live
   that make a fast kernel wrong.
5. **`verification/`** — before you believe your own result.

`catalog.yaml` is the machine-readable index.

## What is here and what is not

**Here:** invariants, decision processes, failure modes, shape and tolerance
advice, and small original examples written to explain a principle.

**Not here:** upstream serving-framework source, translated source, large
distinctive excerpts, or any implementation you could paste. That is not
squeamishness about licences — although it is that too. It is that a kernel
pasted from a production repository is not a kernel this agent designed, and the
question being asked is what the agent can do.

The rule the source material stated best, and which survives here unchanged: read
production code for **what algorithm works and why**, then write your own from
the idea. A candidate that `#include`s or imports its way to a result has answered
a different question.

## The one rule that is not advice

**A microbenchmark can reject a candidate and can never accept one.**

A kernel three times faster on an operator holding four per cent of the critical
path cannot move the served system by more than about three per cent, and if the
operator was not on the critical path it cannot move it at all. The harness
enforces this — the microbenchmark rung is `can_accept=False` — and it is repeated
here because it is the rule most often argued with by someone holding a very
good-looking speedup.
