# Asynchronous execution and overlap

## What overlap actually means

The device is idle while the host prepares a step, and the host is idle while the
device runs one. Overlap removes both by keeping a pipeline going:

```
step N-1 :  ................ retire outputs
step N   :  execute on device
step N+1 :  prepare on host
```

Three things happen at once: the host builds N+1, the device runs N, the output
path retires N-1. Steady state hides host time behind device time.

**This only pays if there is host work worth hiding.** Overlap on a device-bound
loop adds machinery and complexity for nothing. Profile first: if host time per
step is 5% of device time, overlap can win you at most 5%, and probably less.

## The one-batch-lag structure

The common shape, and the simplest correct one:

```
loop:
    receive new requests, admit them
    plan = schedule()                 # host work for step N
    if plan not empty:
        ticket = runner.submit(plan)  # returns immediately; device runs async
        pending.append((plan, ticket))
    if pending is deeper than the lag:
        older_plan, older_ticket = pending.popleft()
        older_ticket.wait()           # now block
        retire(older_plan, older_ticket.result)
```

The lag is what buys the overlap. A lag of one is usually enough; deeper lags buy
little and make every lifetime question harder.

**Know when to disable it.** Some steps cannot overlap — a step that must observe
the previous step's output to schedule at all, a structural change, a checkpoint.
Detect those and drain first. A loop that cannot express "not this one" will
eventually overlap something it must not.

## The hazard that is easy to miss

Under overlap, **host memory the device is still reading may not be mutated.**

If step N's inputs were copied to the device asynchronously and the scheduler
starts patching the host-side tables for N+1, it may be writing buffers the copy
has not finished reading. The failure is intermittent, load-dependent, and does
not look like a race — it looks like an occasional wrong answer.

Two defences, and you want both:

- **Wait for the input copy to complete before mutating anything it read.** Not
  for the compute — just the copy. That is a short wait and it is mandatory.
- **Double-buffer the host-side staging tensors** so step N+1 writes different
  memory from the one step N is copying out of.

## Letting the host run ahead of the output

For the host to plan step N+1 before step N's tokens exist, the request state has
to tolerate a gap: how many tokens are *expected* but not yet observed.

Reserve them as placeholders. The request's accounting advances so scheduling can
proceed; when the real tokens arrive, the placeholders are reconciled.

Two rules make this safe:

- **A placeholder is never an output.** It may not be streamed, counted as
  produced, or used to decide that a request finished. Only observed tokens end a
  request.
- **Reconciliation must handle "fewer than expected".** Speculative tokens get
  rejected, a request stops early, a step fails. The path where the placeholder
  count exceeds the real count is the one that gets skipped and then bites.

## Single device owner

**One thread owns the device context.** Not "usually one" — one.

Multiple threads issuing device work gives you nondeterministic ordering, and
therefore nondeterministic numerics: reduction order changes, and a different
reduction order eventually flips one argmax, and one flipped argmax diverges the
whole remainder of a sequence. You will spend days on it.

So: request-handling threads never touch the device. They enqueue. The owner
dequeues, submits, and hands results back. Bounded queues, so backpressure is
real rather than an unbounded memory growth curve.

## Faking it

The tempting shortcut is to wrap a synchronous call in a future, or to time
submission rather than completion. Both produce excellent numbers and no overlap.

The test that distinguishes real from fake:

1. Submit a step.
2. **Before waiting**, do a measurable amount of real host work — tokenize,
   admit, schedule — and count it.
3. Then wait.

If the ticket was still `device_pending` while that work happened, and the work
count is non-trivial, the overlap is real. If the ticket was already `completed`
before you did any of it, submission blocked and you have a wrapper.

Report the numbers: submission latency as a percentage of step time, and the
count of host work units executed while a ticket was pending. A serialized engine
shows zero.

## Failure, cancellation, quarantine

- **A failed ticket is quarantined.** If you do not know what the device did, you
  do not know whether the cache is consistent. Retrying blindly on top of unknown
  state turns one failure into corrupted output.
- **Cancellation is idempotent** and must work at every stage: before dispatch,
  while pending, after completion but before retirement.
- **Nothing is freed until the reading step retires.** This is the same invariant
  as in the cache and the batch, and overlap is exactly what makes it non-obvious.
- **Have a stall detector.** An asynchronous loop's usual failure is not a crash,
  it is a stop. A checkpoint per iteration and a deadline is cheap and turns a
  silent hang into a diagnosable event.

## Graph capture and compilation

Capturing a step as a replayable graph removes per-launch host overhead, which
can be most of the host cost on small models where kernels are short.

It is an *option chosen from a profile*, not a default:

- It needs stable shapes, hence buckets, hence padding waste.
- Captured memory addresses must stay valid, which constrains the allocator.
- Anything dynamic inside the step must move out of it.
- Capture and recapture cost real time and real memory.

If your profile does not show launch overhead as a dominant cost, capture buys
little and costs a lot of flexibility. Measure before, measure after, and keep
the uncaptured path working.
