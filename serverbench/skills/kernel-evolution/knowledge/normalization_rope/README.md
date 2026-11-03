# Normalisation, activation and position encoding

*Applies to: `rms_norm`, `qk_norm`, `rope`, `silu_mul`.*

These are bandwidth-bound. The design space is small and the ceiling is the
memory system, so the only question worth asking is whether you are reading and
writing each element exactly once at full width.

## The checklist

- **One pass.** Two-pass RMS norm (reduce, then scale) reads the row twice.
  Reduce into registers and keep the row there if it fits; use a block-wide
  reduction rather than a round trip through global memory.
- **Vectorised access.** 16-byte loads and stores. Eight BF16 per `uint4` is the
  difference between saturating HBM and reaching a third of it.
- **FP32 accumulation.** The sum of squares over a 4096-element row in BF16 loses
  precision that shows up as a systematically wrong scale. Accumulate in FP32,
  convert once.
- **Fuse into a neighbour, not into a loop.** These operators are worth fusing
  precisely because their arithmetic is free relative to their traffic — but only
  where the fusion removes a round trip.

## RoPE

Rotary embeddings pair dimensions and rotate by an angle that depends on the
token's absolute position.

Two conventions exist and they are not interchangeable. **NeoX** pairs
`(i, i + d/2)`; **GPT-J** pairs `(2i, 2i + 1)`. A kernel written for one and
applied to the other produces well-formed, wrong output. Declare it; do not infer
it from a shape.

The failure that matters more: **position zero for every token**. Correct for the
first token of every sequence, wrong for the rest, invisible in any test that
generates one token. This is why the boundary matrix carries a case with history.

The cos/sin tables are worth precomputing once and worth keeping in the layout the
kernel reads them in. Recomputing them per call is trigonometry in the inner loop.

## QK normalisation — the Qwen3 trap

Qwen3 applies a per-head RMS norm to Q and K after projection and **before**
RoPE. Qwen2 does not.

Take an attention kernel written for a model without it, apply it to Qwen3, and
everything works: shapes agree, output is fluent, the benchmark is faster. The
model is simply computing something slightly different from what it was trained
to compute, and no timing measurement anywhere can see it.

This is why `KernelCapability` declares `qk_norm` and why substitution across a
mismatch is refused rather than warned about.

## SwiGLU

`silu(gate) * up`, elementwise over the MLP's intermediate. Almost always the
epilogue of the gate/up GEMM rather than a kernel of its own — as a standalone
kernel it reads and writes the intermediate for arithmetic that costs nothing.

`silu(x) = x · sigmoid(x)`. Compute in FP32; the BF16 version of `sigmoid` near
zero is not accurate enough to leave alone.
