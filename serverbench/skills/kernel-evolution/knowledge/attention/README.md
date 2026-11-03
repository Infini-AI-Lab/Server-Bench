# Attention

*Applies to: `attention_prefill`, `attention_decode`, `kv_cache_append`.*

Prefill and decode are the same mathematics and different kernels. Writing one
and using it for both is the most common way to leave most of the win on the
table, and the most common way to be slower than the thing you replaced.

| | prefill | decode |
|---|---|---|
| query rows per step | thousands | one per active request |
| bound by | tensor cores | reading K/V out of HBM |
| the lever | tiling, pipelining, skipping masked blocks | filling the grid, not re-reading K/V |
| what fills the SMs | the Q tile | splitting the KV length across blocks |

## Streaming softmax — the reason any of this works

Never materialise the `[B, H, S, S]` score matrix. For each KV tile: compute
`s = q·kᵀ`, take the row max `m'`, rescale the running output by `exp(m − m')`,
accumulate `p = exp(s − m')` and `o += p·v`, rescaling the normaliser `l` the same
way. Divide once at the end.

Two details are worth stating because they are where implementations go wrong:

- **The rescale must be applied to `o` and `l` together.** Rescaling one and not
  the other is a bug that produces plausible, slightly wrong output — exactly the
  class no latency measurement can see.
- **`exp2` with `log2(e)·scale` folded into the QK rescale** saves an operation
  per element in the inner loop. Fold it into the scale you were multiplying by
  anyway, not into a separate pass.

## Causal masking at block granularity

For a query block `qi` and a key block `kj`, if the whole block is above the
diagonal the entire block is masked and you `continue`. That skips close to half
the K/V traffic in the causal case, and it is a loop-bound change rather than a
per-element predicate.

The diagonal block is the one to be careful with: it needs the element-wise mask,
and it is where an off-by-one lets a token attend to itself-plus-one. That
particular bug produces text that is fluent and subtly better at prediction than
it should be, which is why it survives casual inspection.

## GQA — the shape that changes the traffic

With `G = num_q_heads / num_kv_heads`, each KV head serves `G` query heads. Pack
those `G` query rows so a single K/V tile load serves all of them. That divides
K/V HBM traffic by `G`, and in decode — where K/V traffic *is* the kernel — it is
close to the whole optimisation.

The failure mode: broadcasting one KV head to every query head *in the group*
without actually indexing the right head. Shapes stay valid, output is
well-formed, values are wrong.

## Decode: the grid is the problem

At batch 1 with one query row, a block-per-(batch, head) grid launches a few
dozen blocks on a 132-SM machine. Split the KV length across blocks — each
computes a partial `(o, m, l)` — and combine with a second, tiny reduction
kernel. The combine is the same rescale as the inner loop, applied across
partials.

Watch the partial-count: too few and the SMs are still idle, too many and the
combine and its extra global round trip cost more than they save. It is a tuning
axis per (batch, sequence length), not a constant.

## The KV cache is part of the kernel

A paged cache means the K/V for one sequence is not contiguous. The kernel takes
a block table and gathers. Two things then matter more than they look:

- **Page size versus tile size.** A page smaller than the tile forces a gather
  per tile; a page much larger wastes memory on short sequences.
- **The write offset.** Appending at the previous step's offset is invisible on a
  single-token test and on the first decode step of a fresh sequence. It needs a
  sequence with history to show up, which is why the boundary matrix includes one.

## Qwen3 specifically

Qwen3 applies RMS normalisation to Q and K **inside the attention block**, before
RoPE. Qwen2 did not. A fused attention kernel written for the older shape will
run, will be faster, and will silently omit an operation the model needs. See
`../models/qwen3.md` — this is the canonical example of a capability mismatch that
only a semantic declaration can catch.
