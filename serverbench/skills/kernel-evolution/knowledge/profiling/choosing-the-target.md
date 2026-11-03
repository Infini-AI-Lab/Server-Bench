# Choosing what to optimise, and the signature that defines it

*Applies to: all. Read before proposing any candidate.*

## First question: is this a kernel problem at all?

Most of the time, in a serving system that has not been optimised, it is not.
Before writing a kernel, account for the step:

- time between kernels (launch gaps, host synchronisation, Python)
- time in the scheduler and admission path
- time building batch metadata — index tensors, block tables, position ids
- host-to-device and device-to-host copies
- time in the kernels themselves
- time retiring output: detokenisation, streaming, serialisation

If the kernels are 40% of the step and the gaps are 45%, the best kernel in the
world buys you less than closing the gaps. Write that number down before
proposing anything, because it is also the ceiling on whatever you propose.

## Amdahl, stated as the number you must produce

A candidate's hypothesis has to name two figures: the local speedup it expects,
and the **end-to-end ceiling** that implies. If the operator holds fraction `f`
of the critical path and you make it `s` times faster, the ceiling is
`1 / (1 − f + f/s)`. At `f = 0.04` and `s = 3` that is 2.7%.

The contract refuses a hypothesis whose end-to-end ceiling exceeds its local
speedup, because that is arithmetically impossible. It cannot refuse a candidate
whose ceiling is real but tiny — that judgement is yours, and it is the main one
worth making well.

## The signature is the target

The unit of optimisation is one `(operator, shape, dtype, layout, hardware)`, not
"attention". A kernel rarely wins at every batch size: launch overhead dominates
small batches while tiling dominates large ones, and the optima are different
kernels.

So decide, and write down:

1. **Which regime dominates your workload.** For interactive chat serving, decode
   dominates the wall clock; prefill is a burst. For an offline batch job the
   opposite. Optimising the average of the two optimises for a case you never run.
2. **Which batch sizes you care about.** Each is its own signature with its own
   ceiling.
3. **What the sequence length distribution actually is**, not what the maximum is.

If you do not choose, you will silently optimise the mean of a distribution you
never serve.

## Reading a profile without fooling yourself

- **Total time is not critical-path time.** An operation overlapped with something
  longer costs nothing.
- **The gap between kernels is data.** A step that is 60% gaps is telling you the
  answer, and the answer is not a kernel.
- **Profile the accepted head, not the last one you remember.** The bottleneck
  moves every time something is accepted, and last week's profile is describing a
  program that no longer exists.
- **Profile after a warm-up.** The first iterations include compilation, cache
  population and clock ramp.
