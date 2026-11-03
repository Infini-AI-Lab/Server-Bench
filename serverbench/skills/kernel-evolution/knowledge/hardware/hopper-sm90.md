# Hopper, `sm_90` — H100 and H200

*Applies to: all operators on H100/H200. Read once at the start of a session.*

## The numbers that decide designs

| | H100 SXM (80 GB) | H200 NVL (141 GB) |
|---|---|---|
| SMs | 132 | 132 |
| HBM bandwidth | ~3.35 TB/s | ~4.8 TB/s |
| L2 | 50 MB | 50 MB |
| SMEM per SM | 228 KB (up to 227 KB per block) | same |
| Registers per SM | 65,536 × 32-bit | same |
| BF16 dense tensor core | ~990 TFLOP/s | ~990 TFLOP/s |
| FP8 dense tensor core | ~1,979 TFLOP/s | ~1,979 TFLOP/s |

Verify on the machine rather than trusting the table: clocks, power caps and
memory configuration vary, and a design tuned to a number you did not measure is
tuned to a different machine.

```
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm --format=csv
```

## What Hopper has that Ampere does not

**`wgmma` — warpgroup matrix multiply-accumulate.** The MMA unit is driven by a
*warpgroup* (four warps, 128 threads) rather than a single warp. Operands come
from shared memory, not registers, which is what frees the register file for the
accumulator and the epilogue. This is the single biggest change from `sm_80` and
the reason an Ampere kernel ported unchanged leaves most of the machine idle.

**TMA — `cp.async.bulk.tensor`.** A dedicated copy engine moves whole tiles
between global and shared memory, addressed by a descriptor rather than by 128
threads each computing an address. One thread issues it; the rest do arithmetic.
It also handles the boundary predication that otherwise costs a branch per
element at the edge of every tile.

**Thread-block clusters and distributed shared memory.** Up to 16 blocks can be
co-scheduled on a GPC and read each other's shared memory. Where a tile is needed
by several blocks — the K/V tile in attention, the B tile in a GEMM — one load
can serve the cluster instead of one load per block.

**Asynchronous transaction barriers (`mbarrier`).** The synchronisation primitive
that makes the above usable: a consumer waits on a byte count rather than on a
thread count, so a producer warp can run ahead without a `__syncthreads()` that
stalls everyone.

## The shape of a fast Hopper kernel

Warp-specialised and persistent. A fixed grid of `num_SMs` blocks; inside each
block, some warps are **producers** that issue TMA loads and some are
**consumers** that issue `wgmma`. They meet at an `mbarrier` ring. Nobody
computes an address in the inner loop and nobody waits on a full block barrier.

Register pressure is the usual limit. The accumulator for a 128×256 BF16 tile in
FP32 is 128 registers per thread on its own; the budget is 255. If occupancy
drops to one block per SM, the producer/consumer overlap is what keeps the tensor
cores fed — which is why the specialisation matters more here than on Ampere.

## What does *not* carry over

- `tcgen05` MMA and the associated tensor memory are **`sm_100` (Blackwell)
  only**. Code written against them will not compile for `sm_90`.
- FP4 and block-scaled FP8 formats are Blackwell.
- Ampere's register-operand `mma.sync` still works and is still the right tool
  for small tiles, but it will not reach peak.

## Where the time actually goes in decode

Decode is not compute-bound and treating it as though it is wastes the session.
At batch 1 every GEMM is a GEMV: the output grid is a handful of tiles, most SMs
are idle, and the kernel is bound by reading the weights out of HBM. The levers
that matter are split-K (or stream-K) to fill the grid, coalesced and vectorised
weight loads, and not launching more kernels than necessary.

At larger batch the same GEMM becomes tile-bound and the tiling choices start to
matter. **These are different signatures with different optima**, and a tip tuned
at batch 1 should not be assumed to hold at batch 32.

## Launch overhead is a real cost at decode

A decode step at batch 1 can be a few hundred microseconds of work spread over
dozens of kernel launches of a few microseconds each. Fusing adjacent
bandwidth-bound operations — norm into the following projection's prologue,
activation into the MLP epilogue — buys back launch overhead and a round trip
through HBM at the same time. Measure the gap between kernels before assuming the
kernels are the problem.
