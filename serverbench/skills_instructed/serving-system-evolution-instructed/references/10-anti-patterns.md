# Anti-patterns and a review checklist

Each of these is something that looked reasonable when it was written.

## Architecture

**The module that owns everything.** One file holding HTTP, scheduling, cache,
the model and sampling. Every change touches it, nothing can be tested alone, and
the first real refactor is a rewrite. Split by ownership — who is allowed to know
what — not by file size.

**Request objects reaching the runner.** The runner imports the HTTP type "just
for the request id". Now the model cannot run without a web framework, and the
batch is not really compiled. Test: could the runner be exercised from a unit
test with no server?

**A per-request cache treated as authoritative.** Fine while requests run one at
a time. The moment two share a step, two owners are writing memory neither fully
controls. The cache owner must be single.

**The scheduler indexing memory.** It computes a slot, or reads a block table, or
"just checks" free space by looking at the allocator's internals. Policy and
placement are now coupled and neither can change alone.

**Benchmark code in the serving path.** A metrics hook, a trace writer, an
evolution import. Now the thing being measured contains the measurement.

## Hot path

**Allocating in the step.** New lists, new tensors, new dicts, every step. Shows
up as per-step cost that scales with resident requests and reads as "slower at
high concurrency".

**Accidental synchronisation.** A `.item()`, a shape read, a Python branch on a
device value, a log line that formats a tensor. Each is a pipeline bubble, and
each is invisible until you look at a timeline.

**Rebuilding what barely changed.** Block tables, masks, position ids, sampling
parameters — reconstructed from scratch when one row joined and one left. Patch
instead.

**Full-history repacking.** Copying or reallocating the whole cache to append one
token. Per-step cost then grows with sequence length for reasons unrelated to
attention.

**Request threads touching the device.** Nondeterministic ordering,
nondeterministic reduction order, nondeterministic argmax, days lost. One owner.

## Correctness

**Silent fallback.** The fast path fails and a slow path answers, and nobody is
told. Every measurement afterwards describes a system that is not the one under
test. Fallbacks are allowed; hidden ones are not.

**Reusing a slot without invalidating it.** The new occupant attends over the old
one's history. The output is fluent, plausible and wrong — and only under load.

**Freeing while in flight.** A block released as soon as its request finished,
while a step still reads it. The same fluent, plausible, load-dependent wrongness.

**Matching a prefix too eagerly.** Two prompts differing only in the last shared
token return the same answer. Every test you wrote passes.

**Comparing free-running outputs across implementations.** They diverge after the
first different argmax, and the divergence count means nothing. Teacher-force.

**Trusting a fixture.** A reference built with a different model or revision fails
every case at once and looks exactly like a catastrophic regression.

**Narrowing the envelope.** A shape that used to work now returns "unsupported",
and the benchmark got faster. That is a workload change.

## Measurement

**Changing the workload after seeing results.** Almost never a decision; it
happens one reasonable adjustment at a time. Freeze it and put the digest in the
record.

**Shrinking the denominator.** Excluding failures, timeouts or cancellations from
a percentile. The most effective way to improve a p99 without improving anything.

**Shorter outputs.** Returning fewer tokens is the cheapest possible speedup and
the least honest one. Compare length per request, not on average.

**Timing submission instead of completion.** Measures how fast you can ask.

**One run, no spread.** Not a measurement. If p50 and p99 disagree wildly, that
disagreement is the finding.

**Treating requests as independent samples.** They share a machine, a cache and a
queue. The unit of independence is the run.

**"No significant difference" as a pass.** It is compatible with a large
difference and a small sample. Equivalence is a different test and needs a margin
declared in advance.

**A pre-warmup baseline.** Comparing a settled measurement against a baseline
taken while the allocator was still growing manufactures a trend that is not
there.

**Contaminated windows explained rather than invalidated.** A foreign process on
the device, a throttled clock, a busy host. Record the conditions and throw the
window away.

## Optimisation

**Kernels first.** The most interesting work, almost never the highest-ceiling
work, and worthless on a host-bound loop.

**Optimising the largest cost instead of the most removable one.** 40% of the
step that is 10% removable is worth less than 15% that is fully removable.

**Accepting on a microbenchmark.** It can reject. It cannot accept -- and none is supplied: the 1,800 s scenario is the only measurement that accepts anything.

**Bundling changes.** Two ideas, one measurement, no attribution.

**Not reprofiling after a win.** The bottleneck moved; the plan is now about the
wrong thing.

**A configuration set after construction.** A backend chosen when the object was
built does not change because you toggled a flag afterwards. Read back what is
actually in effect and fail loudly if it differs.

## Claims

**Framework-wide superiority from one narrow envelope.** One model, one hardware
configuration, one length distribution, one concurrency level — and the losing
arm untuned. State the envelope with the number, every time.

**A diagnostic presented as a result.** A development measurement is not a
performance claim until it has been produced under a frozen plan by a qualified
estimator. Label it.

**Two metrics adjacent so the reader multiplies them.** A wall-clock reduction
next to a concurrency factor invites the reader to believe the larger one. Define
both.

The checklist to run before calling a change done lives in
[../templates/review-checklist.md](../templates/review-checklist.md), so it can be
copied straight into a review.
