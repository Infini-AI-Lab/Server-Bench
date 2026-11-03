# From `generate()` to a serving loop

## Why `model.generate()` cannot own the server loop

`generate()` is a batch-oriented function: you hand it a set of prompts, it runs
until they are all finished, and it returns. Every property a server needs is
missing from that contract, and none of them can be added from outside it.

- **It owns the loop.** While it runs, no new request can join, no finished
  request can leave, and nothing can be cancelled. A request that arrives one
  token after the call started waits for every other request to finish.
- **It owns the cache.** The cache it builds belongs to that call. Two calls
  cannot share a prefix, and a request cannot be paused and resumed.
- **It hides the step.** You cannot see a step boundary, so you cannot schedule
  at one, cannot measure at one, and cannot overlap around one.
- **Its stopping rule is per-call.** It stops when the *set* is done, which means
  short requests wait for long ones — the single largest source of avoidable
  latency in a naive server.

So the loop moves into your engine. `generate()` stays exactly where it is
useful: **in the reference oracle**, where "run this prompt to completion,
correctly, slowly" is precisely the job.

## The smallest thing that is already a serving system

One step. Not one request — one *step*:

```
take a set of admitted requests
  decide which of them run this step, and how many tokens each
  gather their state into one forward
  run one forward
  turn logits into one new token per row
  append each token to its request, check stop conditions
  release anything that finished
```

Everything else — batching policy, paging, prefix reuse, overlap — is a
refinement of that loop. If your design cannot be described as one turn of it,
the design is not finished.

## Build it in this order

### 1. One request, one token, by hand

Prefill the prompt, take the argmax of the last position, then decode exactly one
more token using the cache from the prefill. Compare both tokens against the
reference.

This is where you find out that your position ids are off by one, your attention
mask is the wrong shape, your cache layout transposes two dimensions, or your
prefill returned logits for the wrong position. Every one of those is easy to see
with one request and nearly impossible to see inside a batch.

**Checks:** exact token match on 8 prompts; running the same prompt twice gives
the same tokens; the second token differs from what you would get by re-running
prefill on the extended prompt only in the ways you can explain.

### 2. One request, many tokens

Now you need the request to have a *state*: how many tokens it has produced,
what its limit is, whether it hit a stop string, why it ended.

**Checks:** length limits are exact; stop strings terminate at the right token;
a finished request reports why. Make terminal states terminal — a request that
can go from `finished` back to `decoding` will eventually do so.

### 3. Many requests, sequentially

No batching yet. Serve them one after another through the real request path.

**Checks:** each request's answer is identical to what it got alone; running the
same request three times in a row gives identical output. If this fails, state is
leaking through something you did not think of as state — a cached mask, a
position counter, a reused buffer.

### 4. Many requests, one batched forward

Now several rows go through one forward. This is the first change that can be
*fast* and the first that can be *subtly wrong*.

What breaks here, reliably:

- **Padding.** Rows have different lengths. Padded positions must be masked out
  of attention, and the pad token must not be sampled.
- **Position ids.** Per row, not per batch. A shared counter works until two rows
  are at different lengths, which is immediately.
- **Cache indexing.** Row *i* of the batch is not necessarily slot *i* of the
  cache. Get the mapping wrong and requests read each other's history — which
  often still produces *fluent* text, so it passes an eyeball test.

**Checks:** every prompt gives the same tokens batched as it did alone; try each
prompt in every position of the batch; try batch sizes 1, 2, 4, 8; interleave
long and short prompts.

### 5. Continuous batching

Requests join and leave *between steps* rather than at call boundaries. This is
the change that actually makes a server, and it is where the design stops being
obvious. See [03-scheduling.md](03-scheduling.md) and [04-kv-cache.md](04-kv-cache.md).

### 6. Overlap

Only after the synchronous version is correct and you have profiled it. See
[06-async-overlap.md](06-async-overlap.md).

## Two decisions to make early, and consciously

**Decode-time cache growth.** Naively, each step's attention needs the whole
history. If you rebuild or reallocate that history every step, per-step cost
grows with sequence length for reasons unrelated to the model. Deciding *how the
cache grows in place* is a design decision, not an implementation detail.

**Where the batch lives.** If the tensors describing the batch are rebuilt from
Python objects every step, per-step host cost grows with batch size — again for
reasons unrelated to the model. See [05-persistent-batch.md](05-persistent-batch.md).

Both are cheaper to decide now than to retrofit.

## What "done" looks like for the slice

- The engine, not the model library, controls the token loop.
- A request has a typed state with terminal states that are terminal.
- One step is a function you can call, time and test.
- The reference oracle still exists, still uses `generate()`, and is the thing
  you compare against.
- The task-declared correctness rule holds for every prompt, alone and batched.
  For changes intended to preserve arithmetic, exact token parity is also a
  useful local diagnostic; it is not a universal verdict rule.
