# GEMM and the linear layers

*Applies to: `qkv_projection`, `output_projection`, `gated_mlp`, `lm_head`.*

Every linear layer in a decoder is the same kernel API in a few quantisation
flavours. What differs is the shape, and the shape differs enormously between
prefill and decode.

## The two regimes

**Prefill** — `M` in the thousands. A classic tiled GEMM: fill the tensor cores,
keep the shared-memory pipeline full, and the epilogue out of the inner loop.

**Decode** — `M` equal to the batch size, often 1 to 32. This is a GEMV or a very
skinny GEMM. The output grid is smaller than the machine, so most SMs are idle
and the kernel is bound by streaming the weights out of HBM. The lever is
**split-K / stream-K**: partition the reduction dimension across blocks and
combine, so all 132 SMs are reading weights instead of eight.

Skinny-`M` split-K is routinely the single largest decode win available, and it
is invisible to a benchmark run only at large `M`. Pick your signature before you
tune (`../profiling/choosing-the-target.md`).

## What to internalise from production GEMM design

These are principles, not code. Write your own from them.

1. **Persistent blocks with a work queue.** Fix the grid at the SM count; each
   block pulls output tiles until the queue is empty. Removes per-tile launch
   overhead and lets the prologue stay out of the inner loop.
2. **Producer/consumer warp specialisation.** One set of warps issues the copies,
   another issues the MMAs, meeting at a barrier ring. On Hopper the producer is
   a TMA issue from a single thread, which is nearly free.
3. **Swizzled shared memory.** A 128-byte XOR swizzle removes the bank conflicts
   that otherwise dominate `ldmatrix`. Derive the pattern rather than copying it;
   the shape is `addr ^= ((row >> 3) & 7) << 4` for the common 16-byte-element
   case, and the derivation is what tells you when it changes.
4. **Enough pipeline stages to cover the latency, and no more.** Stages cost
   shared memory, and shared memory bounds the tile. Three is the usual answer for
   a 128×128×64 BF16 tile; measure rather than assume.
5. **Vectorised epilogue.** Accumulate in FP32 registers, apply scales there,
   convert once, store 16 bytes at a time. A sequence of eight scalar BF16 stores
   is eight times the store instructions and a bank-conflict pattern.
6. **For FP8: rescale in registers.** Cast the accumulator to FP32, multiply by
   the per-token and per-channel scales, convert, store. No shared-memory round
   trip for the epilogue.

## Fusions that are usually worth it

- **Gate and up projections in one GEMM.** They share an input and their outputs
  are consumed together; one kernel, one weight-stream pass, one launch.
- **Activation in the epilogue.** `silu(gate) * up` costs almost nothing where the
  accumulator already is, and a whole HBM round trip where it does not.
- **Norm into the following projection's prologue.** Only when the norm's
  reduction is already complete — a fused norm that has to re-read the row to
  finish its reduction has bought nothing.

## Fusions that usually are not

- Anything that turns two compute-bound kernels into one register-starved kernel.
- Anything crossing a synchronisation the model actually needs.
- Anything that makes a tile too large for shared memory and drops a pipeline
  stage: the pipeline is usually worth more than the fusion.
