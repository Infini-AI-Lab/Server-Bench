# Is this a kernel problem?

Answer this before writing anything. The answer is often no, and finding that out
after a week of kernel work is expensive.

## Account for the whole step

Break one decode step into these, and make them sum to the wall clock:

| bucket | what it is |
|---|---|
| ingress | receiving, parsing, tokenising |
| scheduling | admission, batch selection, preemption |
| metadata | index tensors, block tables, position ids, mask construction |
| transfer | host↔device copies |
| kernels | the forward pass itself |
| gaps | time between kernels: launch overhead, host sync, Python |
| retirement | detokenisation, streaming, serialisation |

Two patterns show up constantly in an unoptimised server, and neither is fixed by
a kernel:

- **metadata is rebuilt from scratch every step**, so per-step host work grows
  with batch size and eventually dominates;
- **gaps are a third of the step**, because a few hundred small kernels are
  launched one at a time with a synchronisation somewhere in the middle.

## The ceiling, computed before the work

If the operator holds fraction `f` of the critical path and you make it `s` times
faster, the end-to-end ceiling is:

```
1 / (1 - f + f/s)
```

| f | s = 2 | s = 3 | s = ∞ |
|---|---|---|---|
| 0.05 | 1.026 | 1.034 | 1.053 |
| 0.15 | 1.081 | 1.111 | 1.176 |
| 0.30 | 1.176 | 1.250 | 1.429 |
| 0.50 | 1.333 | 1.500 | 2.000 |

Read the first row again. An operator at 5% of the step, made *infinitely* fast,
is a 5% system. That is often below the practical margin, which means the
candidate cannot be accepted no matter how well it is written.

## Overlapped time is not cost

An operation that runs concurrently with something longer contributes nothing to
the critical path. A profiler's total-time column will still show it, and it will
look like a target. Ask what the step time would be if the operation took zero,
not what the operation takes.

## When the answer is yes

- one operator holds a large, *unoverlapped* share of the step;
- it is far from its own roofline — bandwidth-bound at a third of achievable
  bandwidth, or compute-bound at a fraction of achievable flops;
- the reason it is far from the roofline is something you can name: the grid does
  not fill the machine, the access pattern is not coalesced, an unnecessary round
  trip through HBM, occupancy limited by a register count you can reduce.

"It feels slow" is not the third bullet. If you cannot name the mechanism, the
next step is more profiling, not a kernel.
