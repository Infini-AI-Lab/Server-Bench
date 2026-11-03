# Patterns worth having in hand

*Applies to: all. Principles, with small original illustrations.*

## Choosing the language

**Triton** gets you a competitive bandwidth-bound kernel quickly, autotunes over
a block-size space for free, and handles boundary masking without a hand-written
predicate. It is the right first move for norms, activations, elementwise fusions
and many attention variants.

**CUDA C++** is what you reach for when you need an instruction Triton does not
expose, control over shared-memory layout and the pipeline, or warp
specialisation. On Hopper, `wgmma` and TMA scheduling is where the gap opens.

Neither is a badge. Write the Triton version first, measure it, and only drop to
CUDA where the profile says the missing control is the constraint.

## Autotuning without exploding

An autotune space of every plausible combination compiles for an hour to find
what a shaped search finds in five minutes.

- Vary the block sizes over the range where the tile still fits shared memory.
- Vary pipeline depth over 2–4; more is rarely better and always costs SMEM.
- Vary `num_warps` over 4 and 8 at most.
- Keep the space **per signature**. One space for every shape you serve is a
  compile for every shape you serve.

## Boundary handling

Masking at the tile edge is not an afterthought; it is where most correctness
bugs live. In Triton use a mask on both the load and the store, and give `other=`
a value that is neutral for the reduction being done — zero for a sum, `-inf` for
a max. A mask on the load and not the store writes garbage past the end.

## Vectorisation

Sixteen bytes per access is the target: eight BF16, four FP32. This requires the
base pointer to be aligned and the access pattern contiguous within the thread's
lane. A kernel that is bandwidth-bound and reaching a third of peak is usually
issuing scalar accesses.

## A worked illustration — one-pass RMS norm

The point of this snippet is the *shape* of the solution: one pass, FP32
accumulation, a block-wide reduction and no round trip through global memory
between the reduce and the scale.

```python
# Illustration, not a drop-in kernel: one row per program, one pass.
@triton.jit
def rms_norm(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    # FP32 accumulation: a BF16 sum of squares over 4096 elements loses enough
    # precision to shift the scale systematically.
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (x * scale * w).to(y_ptr.dtype.element_ty),
             mask=mask)
```

Three things to notice. `other=0.0` is neutral for the sum, so masked lanes do not
perturb it. The accumulation is FP32 even though the data is not. And the row is
loaded once — the reduction and the scaling share it.

The version that reaches peak on a long row splits it into several BLOCK-sized
chunks and reduces across them, which is the same structure with an outer loop.
