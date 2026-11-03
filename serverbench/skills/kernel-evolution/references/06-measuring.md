# Before you claim a kernel is faster

## The protocol, and why each part is there

- **Warm up until it stops changing.** Early iterations measure clock ramp,
  cache population and JIT compilation.
- **Synchronise around the measured region, not inside it.** Inside, you serialise
  what you are measuring; without it entirely, you measure enqueue time.
- **Interleave candidate and reference within a repetition.** Thermal and clock
  drift over minutes is real, and A-then-B attributes all of it to B.
- **Several independent repetitions**, not one long run. One run measures one
  thermal state.
- **Keep the raw samples.** A summary cannot be re-analysed, and the first
  question anyone asks about a surprising number is about its distribution.
- **Report a median and a spread.** A mean belongs to its worst outlier. If the
  interquartile spread is wide, the number is not stable enough to act on, whatever
  the median says.
- **Record workspace bytes.** A kernel that is faster because it allocated eight
  gigabytes of scratch has not made the server faster.
- **Same device, same session, verified.** A ratio across two devices measures the
  machines. Another process on your GPU means the timing is contaminated — that is
  an invalid measurement, not a slow candidate.

## What the number means

It means your kernel is faster on that shape, on that device, in isolation. It
does not mean the server got faster, and nothing but the full scored
scenario can be the
reason a candidate is accepted.

Between the microbenchmark and the end-to-end result sit several things that
routinely eat the win: the operator was overlapped with something longer; the new
kernel needs metadata that costs more to build than the kernel saves; occupancy
changed and something else slowed down; the win was at a shape the workload does
not actually produce.

## The cheap falsifier

Name it before you write the kernel: the smallest observation that would tell you
the idea is wrong.

Good ones are concrete and take minutes. "If the fused version is not at least 1.5×
on the isolated chain, the end-to-end ceiling is under 2% and I stop." "If
occupancy does not go above two blocks per SM, the register reduction did not
land."

A candidate with no cheap falsifier will consume its entire budget before anyone
finds out it was never going to work.

## Reporting

State the shape, the dtype, the device, the protocol, the median, the spread, and
the ceiling the local number implies end to end. A speedup quoted without its
shape is not a result anyone can check, and a local speedup quoted without its
ceiling is the number most likely to be believed and least likely to matter.
