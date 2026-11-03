### Instructed loop reminder

Run the evolve loop -- plan, request, build, integrate, **verify**, decide --
with both subagents working in the background, each in its own git worktree,
and send each its next assignment when it returns; every hand-off names the
commit it measured. Keep the failure modes in view: correct alone but wrong
under load, a check that cannot fail, shipping what was not measured, numbers
explained away, a guessed offered load, latency lost to prefill between decode
steps, an engine that dies silently, and idle time.

A cycle ends only at a verified tree, cold-started through `serve.sh` (the
only thing the verdict starts):
- correct by
your own output-correctness checks
  against an independent reference of the checkpoint, through the HTTP
  endpoint, at the longest contexts, with prefix reuse, forks and requests
  joining and leaving a concurrent batch, long decodes, continuation past
  end-of-sequence under `ignore_eos` and a natural stop without it, plus the
  protocol probes (`--check-protocol-context`) and the lightweight replay gate;
- runnable and measured by
`flock /tmp/gpu.lock ./full-test --duration N --wait`
  at the concurrency in `concurrency.json`, with every request completing and
  every surprise in the server log and per-request export explained;
  performance decisions rest on repeated, long-enough windows, never the best
  one, and the offered load is chosen by measuring several.
Then `./checkpoint "verified: ..."`; otherwise repair or revert to the last
verified checkpoint. Profile real CUDA execution (torch.profiler, nsys or ncu),
bind evidence to the
immutable candidate SHA
it was measured on, and use design_review.py to update the affected axes.

Every GPU job runs under `flock /tmp/gpu.lock` for exactly its own lifetime,
one agent at a time, and you balance GPU time; CPU work overlaps freely, and
waiting is time for CPU work. In the last hour add nothing new: verify the
final tree with an 1800-second full test and the correctness suite, and use the
whole budget. Timebox selects the
last working tree,
with a concrete `./promote` only for rollback. Unmeasured is not a pass.

Use ./gpu-clock check for read-only diagnosis; low idle clocks are not proof
of a fault. Never recover another job's GPU or recover during measurement.
