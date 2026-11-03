# Contracts and ownership

A serving system is mostly a question of *who is allowed to know what*. Get the
ownership boundaries right and you can replace the scheduler without touching the
cache. Get them wrong and every change touches everything.

Four values carry almost all the traffic between components. Name them whatever
suits your codebase; what matters is what each one is allowed to contain.

## Request state

Everything about one request that is not model memory.

Must carry: an id; a phase; how many prompt tokens; how many generated; the
limit; when it was **offered**, when it was **admitted**, when its first token
appeared, when it finished; why it finished; a handle to its cache allocation.

Two details that are load-bearing:

**Offered and admitted are different timestamps.** Time-to-first-token measured
from admission hides queueing entirely — which is exactly the delay a scheduling
change is meant to affect. Measure from *offered*, always. A system that looks
excellent on admission-relative TTFT and terrible to users is measuring its own
queue out of existence.

**Terminal states are terminal.** `finished`, `cancelled`, `failed`, `rejected`.
A request that can leave a terminal state will, under load, do so — usually by
emitting a token after its response was already sent.

Rejection is a terminal outcome, not an absence. A request refused at admission
still happened and still belongs in every denominator.

## Execution plan

What the scheduler decided for one step, immutable once made.

Must carry: which requests prefill, which decode, how many tokens each, the token
budget, and anything preempted.

Must **not** carry: device pointers, tensors, HTTP objects, or a reference to the
cache. The plan is a decision, not a memory layout. That is what lets you test
the scheduler with no device present — which you should, because scheduler bugs
are policy bugs and policy bugs are best found in unit tests.

A useful invariant: no request appears in both the prefill and decode lists of one
plan. If your design needs that, you actually have two steps.

## Batch delta

A transactional change to the resident batch: which rows were added, which were
removed, and a generation counter.

Deltas exist so the batch is **patched, not rebuilt**. If every step reconstructs
the batch from the current request set, per-step host cost scales with the number
of resident requests instead of the work actually done, and that cost shows up as
"the model got slower with more concurrency".

The generation counter lets a consumer notice it is holding a stale view rather
than reading torn state. That matters the moment anything is asynchronous.

**The invariant that bites:** a row slot freed in one delta may not be refilled in
the same delta. Between the two, a step may still be reading that slot. Reuse
waits for the freeing delta to commit. Getting this wrong produces a rare,
load-dependent, extremely convincing wrong answer — the new occupant attends over
the previous one's history and produces fluent, plausible, wrong text.

## Work ticket

One unit of device work, with an observable life:

```
queued → dispatching → device_pending → completed → retired
                    ↘ failed / cancelled / quarantined
```

The ticket is what makes overlap *checkable*. If the host prepares step N+1 while
the device executes step N, there is an interval where a ticket is
`device_pending` and real host work is happening. A system that wraps a
synchronous call in a future never shows that interval — and that is the test.

Two rules:

- **Cancellation is idempotent.** A request that disappears mid-flight must not
  leave its row half-freed or its cache half-released. Cancelling twice is normal
  under load, not an error.
- **A failed ticket is quarantined, not retried blindly.** If you do not know what
  the device did, you do not know whether the cache is still consistent.

## The ownership rules

**The cache owner is the sole writer of KV memory.** Everyone else asks it for an
allocation and gets a handle back. The moment a second component writes cache
memory, every bug in that memory becomes a whole-system bug.

**The scheduler owns policy, not memory.** It decides *which* requests and *how
many* tokens. It never decides *where*. It may ask "would this fit?" — it may not
answer that question itself.

**The runner consumes compiled batches.** If your runner imports your HTTP
request type, the boundary is decorative. The test: could you run the runner from
a unit test with no server, no sockets, no protocol?

**Output commits only completed tickets.** A token that exists because the host
assumed the device would produce it is not a token. Under overlap this is the
difference between a correct system and one that is usually correct.

## Dependencies run one way

```
protocol  →  engine  →  scheduler  →  cache owner
                    →  runner      →  device
```

Nothing points back. In particular, benchmark and evolution code must never be
importable from the serving hot path — that is how a measurement harness ends up
inside a latency-critical loop.

## A test that catches most boundary violations

For each component, ask: *what would I have to mock to unit-test this?*

- Scheduler needs a device → the plan is carrying memory.
- Cache owner needs an HTTP type → request state has leaked downward.
- Runner needs a socket → the batch is not compiled.
- Sampler needs the scheduler → sampling parameters are not travelling with the
  request.

Each of those is a boundary to fix, and fixing it is nearly always cheaper than
the debugging it prevents.
