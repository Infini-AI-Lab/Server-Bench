# Profiling and the order of optimisation

## Amdahl, stated so it is usable

A change to something that is 20% of your time can never buy more than 20%. And
you rarely remove all of it. So every optimisation has a ceiling:

```
ceiling = fraction_of_critical_path x fraction_actually_removable
```

Two numbers, and the second is the one people skip. A cost that is 40% of the
step but only 10% removable has a ceiling of 4%. A cost that is 15% and fully
removable has a ceiling of 15%. **Rank by ceiling, not by size.**

Then divide by effort. The best next change is high ceiling, low cost, low risk —
and that is very often not the most interesting one.

## Find the critical path before you time anything

The first question is not "which kernel is slow". It is **is the loop waiting on
the host or on the device?**

- **Host-bound**: the device has gaps between kernels. Host work per step, launch
  overhead, or Python is the limit. Kernel optimisation buys nothing.
- **Device-bound**: kernels run back to back. Now kernel work matters.
- **Synchronisation-bound**: both sides idle, waiting on each other. Overlap or a
  removable synchronisation point is the win.

Get this wrong and you can spend weeks optimising a kernel on a host-bound loop
for a measured 0% improvement.

**A ratio of host time to device time does not establish this.** Self-time
accounting mixes launch gaps, implicit synchronisation, allocator behaviour,
warm-up, transfers and genuine dispatch. A high host-to-device ratio is
*consistent with* being host-bound; it does not demonstrate it. What demonstrates
it is a timeline showing device gaps, or removing host work and watching step time
fall.

Be honest about which one you have. "Host-dispatch-bound" from a ratio alone is a
hypothesis, and labelling it as a finding is how a team optimises the wrong thing
for a month.

## Measure end to end first

Profile the whole request path, not one function:

```
offered → admitted → prefill → decode steps → sampled → detokenized → sent
```

Time each segment. The surprises are usually outside the model: queueing,
tokenization, response construction, a lock, a log statement in the hot path.

Then, per step: host preparation, submission, device execution, output handling.

## Order of attack

Roughly in order of return per unit of effort and risk:

1. **Eliminate work.** The fastest code is the code that does not run. Rebuilt
   metadata, recomputed masks, redundant copies, per-request Python that could be
   vectorised, logging in the hot path.
2. **Eliminate synchronisation.** Every point where the host blocks on the device
   is a pipeline bubble. Some are necessary; most are accidental — a `.item()`, a
   shape read, a print.
3. **Fix layout and persistence.** Persistent batch, in-place cache growth,
   pre-allocated buffers. Removes per-step allocation and copying.
4. **Improve scheduling and cache utilisation.** Better batching means more work
   per step at the same cost. Often the single largest structural win.
5. **Overlap.** Hide the host behind the device once there is host work worth
   hiding.
6. **Specialise shapes.** Buckets, graph capture, compilation. Removes launch
   overhead. Costs flexibility.
7. **Kernels.** Where the profile puts one on the critical path -- CUDA C++ and PTX are both available and advised there, not only Triton. Highest effort, highest risk, and only pays once you are
   genuinely device-bound.

Most systems that "need a custom kernel" need steps 1 to 4.

## Causal evidence versus attractive hypotheses

A hypothesis becomes a finding when an intervention confirms it:

- **Correlation**: step time rises with batch size.
- **Hypothesis**: metadata rebuild is the cause.
- **Evidence**: allocation count per step scales with batch size; a host profile
  shows time in list and tensor construction; **replacing the rebuild removes the
  scaling**.

The third bullet is the one that matters. Until you have it, you have a plausible
story, and plausible stories are wrong often enough to be dangerous.

Attractive hypotheses that are wrong more often than not:

- "It must be the attention kernel." Usually it is host work.
- "It must be memory bandwidth." Usually there are gaps between kernels.
- "It must be Python overhead." Sometimes — measure how much.
- "Nondeterminism must be atomics in the kernel." It is very often something
  chosen at construction time and never actually changed by the toggle you set
  afterwards. Verify the configuration that is *in effect*, not the one you asked
  for.

That last one is worth an explicit habit: after configuring a backend, read back
what the object actually resolved, and fail loudly if it differs. A toggle applied
after construction changes nothing, and you can lose a week to it.

## Microbenchmarks: what they are for

None is supplied, and a short run of the real scenario is not one
either: `--duration` below the full window measures a different
distribution, because the tail grows with the window. Build your own
component tests to reject ideas quickly; only the full scenario
accepts anything.

A microbenchmark can **reject** an idea cheaply. It can never **accept** one.

An isolated function twice as fast tells you nothing about end-to-end time: the
function may be 3% of the path, the improvement may vanish under real memory
pressure, or the change may slow something else down. Use microbenchmarks as
fail-fast filters, and let only end-to-end measurement accept.

## Measuring without fooling yourself

- **Warm up.** First iterations include compilation, allocator growth, cache
  population. Exclude them explicitly, and say so.
- **Measure completion, not submission.** Asynchronous work timed at submission
  measures how fast you can ask.
- **Repeat, and report spread.** One number is not a measurement. If p50 and p99
  disagree wildly, that is the finding.
- **Check the machine.** A throttled clock, a foreign process, a busy host — each
  produces a real, reproducible, meaningless regression. Record device clocks and
  foreign occupancy alongside every timing, and invalidate contaminated windows
  rather than explaining them.
- **Percentiles over every offered request.** Excluding failures and timeouts
  from a tail measurement is the most effective way to improve a p99 without
  improving anything.
- **Distinguish a rate you paced from a rate the system sustained.** A closed-loop
  measurement at fixed concurrency is a capacity sighting, not a sustainable
  offered rate, and calling it throughput will mislead everyone including you.
