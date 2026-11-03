# Fusing, and when not to

Fusion is worth it when it removes a **round trip through HBM** or a
**synchronisation**, and not when it merely removes a kernel launch you were not
paying much for.

## The test

Before fusing A into B, answer three questions:

1. **What traffic disappears?** If A writes an intermediate that B immediately
   reads, fusing removes a write and a read of that intermediate. That is the
   whole win, and you can compute it in bytes.
2. **What does it cost in registers or shared memory?** If the fused kernel drops
   a pipeline stage or halves occupancy, the fusion can be a net loss even though
   it removed traffic.
3. **Does it cross a synchronisation the model actually needs?** A reduction has
   to complete before the thing that consumes it. Fusing across one means
   re-reading the data to finish the reduction, which buys nothing.

## Fusions that usually pay

- **Activation into the producing GEMM's epilogue.** `silu(gate) * up` costs
  nothing where the accumulator already is and a full round trip where it does
  not.
- **Gate and up projections into one GEMM.** They share an input and their outputs
  are consumed together: one weight-stream pass instead of two.
- **Quantisation or scaling into the epilogue.** Same argument; the values are
  already in registers.
- **Bias, residual add, and the next norm's first pass** — anything elementwise
  that touches data the kernel already has.

## Fusions that usually do not

- **Two compute-bound kernels.** Neither was waiting on memory; the fused version
  is short of registers and slower than both.
- **Across a reduction boundary.** See question 3.
- **Anything that makes the tile too large for shared memory.** The pipeline stage
  you lose is usually worth more than the traffic you saved.
- **"Fewer kernels" as the goal.** Launch overhead is a real cost at decode, but
  it is a few microseconds; a round trip through HBM for a large intermediate is
  much more. Fuse for the traffic and take the launch saving as a bonus.

## Bandwidth-bound chains are the sweet spot

A sequence of elementwise and reduction operators — norm, RoPE, activation,
residual — reads and writes its data once per operator. Fusing the chain reads
once and writes once. That is often a 2–4× on that chain, and because these
operators are individually cheap, it is a chunk of the step nobody was looking at.

The same argument makes them poor standalone kernel targets: there is no
arithmetic to speed up, only traffic to remove.

## Measure the fused thing, not the pieces

A fused kernel's benefit is a property of the pair. Timing A and B separately and
adding them ignores exactly the effect you are trying to create, which is that B
starts with its input in registers. Time the chain before and the fused kernel
after, on the same device, interleaved.
