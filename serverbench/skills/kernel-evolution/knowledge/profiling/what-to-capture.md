# What a profile has to capture to be worth acting on

*Applies to: all.*

A profile that says "attention is 30%" supports no decision. These are the fields
that turn an observation into a hypothesis.

| field | why it decides something |
|---|---|
| share of the **critical path** | overlapped time is not cost |
| **removable fraction** | some of that share is irreducible; the rest is the ceiling |
| **bound class** — memory, compute, latency, launch | picks the lever; a bandwidth-bound kernel does not get faster from better tiling |
| achieved vs achievable **bandwidth** | tells you how much of the ceiling is already taken |
| achieved vs achievable **flops** | same, for the compute-bound case |
| **occupancy** and the limiter — registers, shared memory, blocks | says *why* the machine is idle |
| **launch count and gap** per step | the fusion argument lives or dies here |
| **workspace bytes** | a kernel that is faster because it allocated 8 GB has not made the server faster |
| **variance across repetitions** | a wide spread means the number is not stable enough to act on |

## The bound classes, and what each implies

**Memory-bound.** Achieved bandwidth is near achievable, arithmetic intensity is
below the ridge. Levers: fuse to remove a round trip, widen the access, improve
coalescing, quantise. Not: better tiling.

**Compute-bound.** Tensor core utilisation is high. Levers: bigger tiles, deeper
pipeline, a lower-precision format the model tolerates. Not: fusion, which
usually costs registers here.

**Latency-bound.** Neither is near peak and occupancy is low. Typical of decode.
Levers: fill the grid — split-K, more parallel work — or reduce the number of
dependent steps.

**Launch-bound.** The gap between kernels is comparable to the kernels. Levers:
fuse adjacent operations, capture the step in a graph, remove host
synchronisation.

## Before you believe it

The profiler itself perturbs what it measures — instrumentation adds overhead,
and a kernel that is 3 µs long is mostly measurement. Cross-check the conclusion
with a wall-clock measurement of the whole step. When the profile and the clock
disagree, the clock is what the user experiences.
