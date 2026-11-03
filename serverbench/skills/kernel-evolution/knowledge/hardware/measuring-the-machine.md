# Measuring the machine you are actually on

*Applies to: all. Read before quoting any number from `hopper-sm90.md`.*

Every architecture note in this tree is a starting point, not a measurement of
your host. Clocks vary with power and thermal state, memory bandwidth varies with
the module, and a number copied from a datasheet has been wrong on real hardware
often enough to be worth ten minutes.

## Establish, in this order

1. **Identity.** `compute_cap` decides which instructions exist. Everything else
   is tuning; this one is compile-or-not.
2. **Achievable HBM bandwidth**, not the peak. A large strided copy gets you the
   number a bandwidth-bound kernel is actually competing with — typically 80–90%
   of the datasheet figure.
3. **Achievable tensor-core throughput** for your dtype, from a large square GEMM
   at a size that is a whole number of tiles.
4. **Empty-kernel launch latency.** This is the floor under every fusion argument
   you will make later.

## The roofline question, asked properly

For an operator moving `B` bytes and doing `F` flops, arithmetic intensity is
`F/B`. Compare it to the machine's ratio of achievable flops to achievable
bandwidth. Below that ridge point you are bandwidth-bound and no amount of tiling
will help; above it you are compute-bound and reducing memory traffic will not.

The mistake worth naming: computing intensity from the *algorithm* rather than
from the *implementation*. A kernel that re-reads a tile because it did not fit
in shared memory has the memory traffic of its implementation, not of its maths,
and the roofline of the maths will tell you it should be compute-bound while it
sits at 15% of peak.

## Guard against measuring the wrong thing

- **Synchronise around the region, not inside it.** Without a sync you time the
  launch, which is smaller and meaningless — and it produces enormous, entirely
  fictitious speedups.
- **Warm up until it stops changing.** The first iterations measure the clock
  ramp, not the kernel.
- **Interleave candidate and reference.** Thermal drift over minutes is real, and
  A-then-B attributes all of it to B.
- **Keep raw samples and report a median with a spread.** A mean is owned by its
  worst outlier, and a spread that is wide is telling you the number is not
  stable enough to act on whatever the median says.
- **Check the device is yours.** Another process on the GPU invalidates the
  timing; that is a contaminated measurement, not a slow candidate.
