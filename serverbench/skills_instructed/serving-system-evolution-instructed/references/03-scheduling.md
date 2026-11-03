# Continuous batching and scheduling

## The idea in one paragraph

Static batching runs a fixed set of requests until they all finish. The batch is
as slow as its slowest member, and a request arriving one step late waits for the
whole batch. Continuous batching makes the *step* the unit instead of the *call*:
between any two steps, finished requests leave and waiting ones join. Utilisation
stops depending on your prompts happening to be the same length.

## Stop thinking in phases

The instinct is to write a prefill path and a decode path and alternate between
them. It works, and then it fights you: chunked prefill needs a request to be
"prefilling" across several steps, prefix reuse needs it to be partly prefilled
before it starts, speculative decoding needs several tokens per step. Each
becomes a special case, and the special cases interact.

The accounting that avoids all of it is per request rather than per phase:

```
computed  = how many of this request's tokens the model has already processed
target    = how many it needs processed before it can produce its next token
```

A step hands out tokens under one budget, letting each request's `computed` catch
up to its `target`:

```
budget = max_tokens_per_step
for request in priority_order:
    want = min(request.target - request.computed, budget)
    if want == 0: continue
    if not cache.can_fit(request, want): break-or-preempt
    plan.add(request, want)
    budget -= want
```

A request whose `want` is large is prefilling. One whose `want` is 1 is decoding.
Chunked prefill is a large `want` that did not fit in one step. Prefix reuse is a
request that starts with `computed` already non-zero. Speculative decoding is a
`want` above 1 during generation. None of them is a special case in the
scheduler; they are all the same subtraction.

This is the single highest-leverage structural decision in a scheduler.

## Token budget, not request count

Limit the step by **tokens**, not by number of requests. Eight decode rows are
eight tokens; one prefill of 2,000 tokens is 2,000. A request-count limit makes
step time swing by orders of magnitude depending on what happens to be admitted,
which destroys tail latency and makes every measurement noisy.

Choose the budget from what you are optimising for:

- **Latency-oriented**: a small budget. Steps are short and even, decode tokens
  come out at a steady rate, and a large prefill cannot stall everyone.
- **Throughput-oriented**: a large budget. Fewer, fatter steps amortise per-step
  overhead better.

There is no correct answer, only a correct *measurement*.

## Chunked prefill

A 4,000-token prefill run whole is one enormous step during which every decoding
request produces nothing. Users see it as a latency spike with no cause.

Split it: give the request part of the budget each step until it catches up. The
cost is bookkeeping (a partly-prefilled request has state) and a little
efficiency (several smaller matrix multiplications). The benefit is that a long
prompt stops being a stall for everyone else.

**Decode protection.** Reserve part of the budget for decoding rows, or a burst of
long prompts will starve every in-flight request. The reservation is a policy
knob and should be measured, not assumed.

## Admission and backpressure

Admission is where you decide whether a request enters at all. It is not
optional: without it, an overloaded server accepts everything, queues grow without
bound, and every request eventually times out. Refusing early is strictly kinder
than accepting and failing late.

Admit only if there is cache room for the request's *prompt plus a realistic
generation*. Admitting on prompt size alone means the request runs for a while and
then hits a wall — after you have already spent the prefill.

Signals worth watching: cache utilisation, queue depth, and whether queue depth
is *growing*. A queue that is deep but stable is a system at capacity; a queue
that is growing is a system already past it, and no amount of scheduling fixes
it.

## Fairness and aging

Pure shortest-job-first maximises throughput and starves long requests
indefinitely. Pure FIFO is fair and leaves the machine idle behind one huge
prompt.

Priority with aging is the usual compromise: rank by whatever you are optimising,
but let waiting time raise priority so nothing waits forever. Whatever you pick,
**measure the tail**, not the mean — starvation is invisible in an average and
obvious in the p90 this suite scores against.

## Preemption

When the cache fills, something has to give. A running request can be preempted
back to the waiting queue, freeing its blocks.

Two ways to resume it:

- **Recompute**: drop its cache, re-prefill later. Simple, and costs the prefill
  again.
- **Retain**: keep its blocks (or swap them out) and resume where it stopped.
  Faster to resume, but the blocks are not free while it waits.

Preempt the *newest* or *least-progressed* requests first. Preempting a request
that is 90% done throws away the most work.

**The invariant:** a preempted request's blocks may not be handed to anyone else
until every in-flight step that might read them has retired. Under overlap this
is not automatic.

## Prefix-aware scheduling

If requests share a long prefix, scheduling them together lets the shared work be
computed once. This is real and can be large.

It is also a trap when the lookup does not pay for itself. Prefix matching costs
host time on every admission, and if your workload's prefixes are short or
unique, you have added constant cost for no reuse. Measure the hit rate before
building the machinery, not after.

## Two policy bundles worth knowing

Neither is right; they are the ends of a dial you tune with measurements.

| | latency-oriented | throughput-oriented |
|---|---|---|
| token budget | small, even steps | large, fat steps |
| chunked prefill | aggressive, small chunks | off or coarse |
| decode protection | strong reservation | weak |
| admission | strict, refuse early | permissive, deep queue |
| priority | aging, tail-protective | shortest-job-first |
| batch size | capped for step-time evenness | as large as memory allows |

## Testing a scheduler without a device

The plan carries no memory, so the scheduler is testable in isolation. Do that:

- No request appears in both prefill and decode lists of one plan.
- The plan never exceeds its own budget.
- A request that cannot fit is deferred or preempts, never silently truncated.
- Long prompts do not starve decode rows under sustained load.
- Aging eventually schedules the oldest waiter.
- A cancelled request disappears from the next plan and never appears again.
- With a full cache, the scheduler preempts rather than over-allocating.

These are policy properties and they belong in unit tests, where they run in
milliseconds and fail readably.
