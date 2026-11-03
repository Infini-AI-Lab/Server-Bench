# Persistent batch and metadata

## The cost nobody looks for

Per-step host work that scales with the number of resident requests. It does not
look like a bug. It looks like "the model is slower at high concurrency", which
is the one explanation everybody already believes.

It comes from rebuilding, every step, things that barely changed:

- Python lists of active requests, positions, lengths, sampling parameters.
- Tensors built from those lists and copied to the device.
- A block or slot table reconstructed from scratch.
- Masks and position ids recomputed for every row.

Between two consecutive decode steps, almost none of that changed. Typically one
request left, one joined, and every remaining row advanced by one position.

## Persist and patch

Allocate the batch metadata once, at maximum size, and mutate it:

```
rows            fixed-size arrays, indexed by row slot
row_of_request  request id -> row slot
free_slots      which rows are unoccupied
generation      bumped on every structural change
```

A step then does O(changes) host work instead of O(resident requests):

```
apply(delta):
    for request in delta.removed:  free the row, invalidate its entry
    for request in delta.added:    claim a free row, write its initial state
    generation += 1

advance():
    positions[active_rows] += 1     # one vectorised update, not a Python loop
```

The device-side tensors follow the same rule: allocate once, write only the
entries that changed, and never reallocate on the steady path.

## Generation counters

The counter is not decoration. It lets a consumer holding a view notice that the
view is stale rather than reading a slot whose occupant changed underneath it.
The moment anything is asynchronous, this is the difference between a detectable
bug and an undetectable one.

## The slot-reuse invariant

**A row slot freed in step N may not be refilled before every step that might
read it has retired.** Under overlap that is not the same as "before the next
step". Violating it gives the new occupant the previous one's history: fluent,
plausible, wrong, and load-dependent.

The cheap enforcement is a free list that only accepts a slot once its releasing
delta has committed, plus a test that cancels mid-stream and immediately admits a
new request into the freed slot.

## Padding and buckets

Heterogeneous lengths force a choice about shape:

- **Pad to the maximum in the batch.** Simple. Wasted compute grows with the
  spread of lengths.
- **Pad to a bucket.** Fewer distinct shapes, which matters if anything downstream
  specialises per shape (graph capture, compiled kernels). Costs a little waste
  and a bucket table.
- **Exact shapes.** No waste, but a new shape every step, so anything shape-
  specialised recompiles constantly — often a net loss.

Buckets are usually the right default once shapes matter at all. Pick them from
your measured length distribution, not from round numbers.

**Whatever you choose, an out-of-bucket request must still be served correctly.**
A fallback path that is slower is fine, and should be reported. A fallback path
that quietly refuses or truncates has narrowed your supported envelope, which is
a workload change dressed as an optimisation.

## Diagnosing it

The signature of rebuild cost is specific and easy to confirm:

1. **Host time per step rises with batch size, device time does not.**
2. **A host profile shows time in list, dict and tensor construction**, not in
   model operations.
3. **Allocation count per step scales with resident requests.**

The last one is the most decisive. Count allocations, or bytes moved, per step at
two very different batch sizes and history lengths. Maintenance work should be
roughly constant per row. If total per-step host work grows superlinearly in
rows, you are rebuilding something.

Do this measurement *before* rewriting anything. "Rebuild cost" is a satisfying
hypothesis and it is not always the right one.

## Tests

- Applying a delta then reading the batch gives the same state as rebuilding it
  from scratch would. This is the correctness backstop for the whole optimisation.
- Add and remove the same request repeatedly → no slot leak, no growth.
- Free a slot and refill it → the new occupant's output is unaffected by the old.
- Structural change bumps the generation; a stale view is detected.
- Output for a given prompt is identical across batch sizes and row positions.
