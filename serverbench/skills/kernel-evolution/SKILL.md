---
name: kernel-evolution
description: Decide whether a GPU kernel is the right thing to change, then write, verify and measure one honestly. Use when a profile points at an operator, when choosing between Triton and CUDA, when deciding whether to fuse, when a kernel is fast but the server is not, or when checking that a kernel change is actually correct.
---

# Evolving a GPU kernel

You are deciding whether a kernel is the right thing to change, and if it is,
writing one that is faster **and still computes what the model needs**.

The second half of that sentence is where kernels differ from the rest of a
serving system. A scheduler bug tends to announce itself: a request hangs, an
ordering is wrong, something crashes. A kernel bug produces fluent, plausible,
subtly wrong output at a very good latency, and no measurement in your benchmark
can see it. Most of this skill is about that.

## The rule that outranks everything else

**A microbenchmark can reject a candidate and can never accept one.**

A kernel three times faster on an operator holding four per cent of the critical
path cannot move the served system by more than about three per cent. If the
operator was not on the critical path, it cannot move it at all.

Nothing enforces this for you. In a serving run the only measurement that
accepts anything is the full scored scenario, which applies the correctness
gates and scores every request against the task's p90 bars on the published
head; a microbenchmark is a cheap way to find out that an idea is dead before
you spend a full scored replay proving it. The rule is stated here because it is the one most often argued
with by someone holding a very good-looking speedup.

## Before anything else: is this a kernel problem?

In a serving system that has not been optimised, usually it is not.

Account for the whole step first — launch gaps, the scheduler, batch metadata
construction, host/device copies, output retirement — and only then the kernels.
If kernels are 40% of the step and gaps are 45%, the best kernel you can write
buys less than closing the gaps.

Then name the number your idea is bounded by. If the operator holds fraction `f`
of the critical path and you make it `s` times faster, the ceiling is
`1 / (1 − f + f/s)`. Write it down before you start. It is usually much smaller
than the local speedup, and it is what decides whether the work is worth doing.

→ [references/01-is-it-a-kernel.md](references/01-is-it-a-kernel.md)

## Where to go next

| you are… | read |
|---|---|
| deciding whether a kernel is the answer at all | [references/01-is-it-a-kernel.md](references/01-is-it-a-kernel.md) |
| choosing the exact shape and dtype to target | [references/02-signature.md](references/02-signature.md) |
| choosing Triton or CUDA, and the kernel's shape | [references/03-writing.md](references/03-writing.md) |
| deciding whether to fuse two operations | [references/04-fusion.md](references/04-fusion.md) |
| about to claim a kernel is correct | [references/05-verification.md](references/05-verification.md) |
| about to claim a kernel is faster | [references/06-measuring.md](references/06-measuring.md) |
| substituting a kernel into a model | [references/07-substitution.md](references/07-substitution.md) |
| asking which kernels you may write | [references/08-strategies.md](references/08-strategies.md) |
| reviewing a kernel change, yours or another's | [templates/review-checklist.md](templates/review-checklist.md) |

Operator-specific material — hardware, profiling, attention, GEMM,
normalisation, sampling, MoE, collectives, verification — is in
[knowledge/README.md](knowledge/README.md), indexed by `knowledge/catalog.yaml`.
Read the entry for the operator your profile pointed at and the entry for the
architecture you are actually on; reading all of it spends context on operators
you are not touching.

## How this tends to go well

None of it is checked. It is what keeps a kernel change attributable.

**One hypothesis at a time.** A change that does three things and gets faster
has taught you nothing about which of the three did it, and cannot be rolled
back to the part that worked.

**Profile the current head, not the one you remember.** The bottleneck moves
whenever anything lands.

**State the claim before writing code**: the profile evidence, the expected
local speedup, the end-to-end ceiling that implies, the correctness risks, and
the cheapest observation that would kill it.

**Order the checks by cost.** Compile, then dispatch, then parity on the shapes
the workload serves, then boundaries and numerics, then a microbenchmark. Each
can reject in seconds or minutes. None of them accepts — only the full scored
scenario does that, and a full test costs its replay window (300 to 1,800
seconds, chosen per test) plus a cold start and the probes.

**Keep the rejections.** A rejection with a conclusion is worth as much as an
acceptance; one nobody drew a conclusion from is device time spent for
nothing.

## What counts as a kernel

A kernel change counts when the served model's path executes a kernel **you
wrote**, on the workload being measured. A Python wrapper that dispatches to the library the baseline already
uses is not a candidate: it will pass every numerical check, because the answer
is correct, and it will report the library's speed as yours.

Memory-layout plumbing is not dispatch — allocations, views, reshapes,
transposes, concatenations, scalar arithmetic on small auxiliary tensors. The
line is the *primary compute path*, and a declared fallback for shapes you do
not support is fine as long as it is declared and announces itself.

→ [references/03-writing.md](references/03-writing.md) for the toolchain and
the levels you can write at.

## Scope

**Ordinary composable kernels.** Attention components, GEMMs, normalisation,
RoPE, sampling, quantisation, data movement, collective helpers.

A single persistent kernel that owns the whole model iteration is a different
research question: it needs a schedule compiler before it computes anything,
and it does not compose with an incremental loop. Nothing forbids one. It is
unlikely to fit inside a run whose full measurements each cost up to half an
hour of replay plus a cold start, which is the reason to weigh it rather than
a rule against it.
