# The signature is the target

The unit of optimisation is one `(operator, shape, dtype, layout, hardware)`. Not
"attention". Not "the GEMM".

## Why the distinction is not pedantic

A kernel rarely wins at every shape. At batch 1 the output grid does not fill the
machine and the win comes from splitting the reduction; at batch 32 the grid is
full and the win comes from tiling and pipelining. These are different kernels
with different optima, and a tip tuned at one is routinely *slower* at the other.

Optimising the geomean across both optimises for a workload you never serve.

## Decide and write down

1. **Which regime dominates your workload.** Interactive chat serving spends most
   of its wall clock in decode; prefill is a burst at the start. An offline batch
   job is the opposite. If you do not choose, you optimise the mean of a
   distribution you do not run.
2. **Which batch sizes.** Each is its own signature with its own ceiling. Read
   them off the workload rather than assuming a spread: this suite fixes a small
   offered concurrency per task, so the batch shapes your scheduler actually
   forms are few and knowable.
3. **The real sequence-length distribution**, not the maximum -- and read it,
   because the intuition that the median is short is wrong here. On this corpus
   the median turn is tens of thousands of tokens and the p90 is in the hundreds
   of thousands, so long context IS the common case rather than the tail.
4. **The dtype and layout you actually get.** A kernel that assumes contiguity and
   receives a slice of a fused QKV projection reads the wrong elements, and the
   output is usually still plausible.

## Declare the envelope, including its edges

The capability contract makes you state the axes, their ranges, their alignment,
and what the kernel does **not** support. The last field is the one that earns its
place: a kernel with no declared limit has not been characterised, and the limit
will be found in production as fluent, wrong output.

Boundary values matter more than interior ones. A tiled kernel is right in the
middle of its domain by construction and wrong at the edge by omission — the last
partial tile, the sequence one past a block, the head dimension that is not a
multiple of the vector width.

## Do not encode the regime in a name

`decode`, `prefill`, `light`, `heavy` in a directory name or a workload key are a
label for a shape you should have written down. Write the axes: `bs1_S2048_bf16`
says what `light` was gesturing at, and it still says it in six months.
