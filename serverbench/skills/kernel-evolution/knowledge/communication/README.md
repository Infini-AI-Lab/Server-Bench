# Collectives, and what tensor parallelism actually costs

*Applies to: `all_reduce`, `all_gather`, `reduce_scatter`. Read before claiming
any multi-GPU result.*

## The claim that needs proving

"Tensor parallel" means one request's computation is **sharded across ranks**, and
that requires collectives at specific points in the layer. An implementation that
runs a whole model per GPU and routes requests between them is **data parallel** —
a different thing with different properties, and calling it TP does not make the
comparison valid.

Prove the defining behaviour rather than the API name: the collective is present,
it carries the volume the sharding implies, and removing it produces wrong output.

## Where the collectives are in a decoder layer

With weights sharded column-wise then row-wise:

- attention output projection → **all-reduce** over the hidden dimension
- MLP down projection → **all-reduce**

Two per layer. For a 36-layer model that is 72 collectives per forward, on the
critical path, every step.

## Why it dominates decode

At decode the message is small — one token's hidden state per rank — and the cost
is latency, not bandwidth. Small-message all-reduce latency does not shrink with
faster links the way bandwidth does, so the collective is a fixed cost per layer
that the compute cannot hide.

The consequences worth knowing before proposing anything:

- **TP2 at batch 1 can be slower than TP1**, and often is. Halving the compute
  does not halve a latency-bound step, and it adds 72 round trips.
- **Overlap is the lever.** Start the collective for layer *n* while layer *n*'s
  remaining independent work proceeds. This needs the collective on its own
  stream and a dependency the scheduler can actually see.
- **Fusing the collective into the producing GEMM's epilogue** removes a launch
  and a round trip through HBM. It is a real technique and a hard one: the
  epilogue now has a cross-device dependency inside it.

## Measuring honestly

The number that matters is the collective's contribution to the *critical path*,
not its standalone latency. A 30 µs all-reduce fully overlapped with 40 µs of
compute costs nothing; the same all-reduce unoverlapped costs 30 µs × 72 layers.
Profile the gap, not the operation.

And: every rank must be measured. A per-rank number from rank 0 with the others
unobserved cannot distinguish "fast" from "waiting for the straggler".
