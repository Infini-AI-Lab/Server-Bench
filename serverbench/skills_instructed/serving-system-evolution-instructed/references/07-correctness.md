# Correctness for serving systems

A serving system can be wrong in ways ordinary software cannot: it produces
fluent, confident, plausible output that is simply not what the model would have
said. Nothing crashes. Tests that check for a well-formed response pass. This
file is about catching that.

## The reference oracle

Keep the slow, obvious implementation — one request, `generate()`, no batching,
no cache reuse — as the definition of correct.

**Greedy decoding is what makes an exact-token reference legitimate.** With
sampling, two correct systems differ by design and no exact comparison means
anything.

**It is legitimate only against *this* reference.** Between two independent
engines, kernel choice and reduction order move logits by a hair, one different
argmax diverges everything after it, and the disagreement measures arithmetic
rather than correctness. Cross-engine agreement is a diagnostic; it is never a
gate.

Reference outputs are a *fixture*. Hash them, and check the fixture's model
identity before use. A stale fixture fails every case at once and looks exactly
like a catastrophic regression in the system under test — the most expensive
false alarm available.

## Identity comes before everything

Before comparing a single token, verify both sides are doing the same job:

- same model, same **pinned revision** (repositories move)
- same tokenizer and the same chat template
- same dtype
- same sampling parameters — greedy is greedy on both sides
- same stop rules and the same maximum length
- **identical prompt token ids**, not merely the same token *count*

That last one earns its place. Matching lengths while differing in a special
token, a BOS, an attention mask or a position id is a real defect that a
numerical tolerance would absorb silently. Record the full input identity and
assert equality; do not infer it from a length.

## Exact diagnostics for semantics-preserving changes

Most changes — batching, caching, scheduling, overlap — are not supposed to
change the arithmetic at all. For those, exactness is the strongest local
diagnostic:

- **Token parity.** Every reference prompt reproduces the reference token
  sequence exactly. Not similar. Equal.
- **Isolation.** The same prompt gives the same answer alone and batched, in any
  position, at any batch size.
- **Repeatability.** The same request twice in one process, and in two fresh
  processes, gives identical output. This catches nondeterminism that a single
  run hides.
- **Order invariance.** Permuting a batch permutes the outputs and changes
  nothing else.
- **Length.** Per request, not on average.

An exact comparison has no tolerance, so it has no threshold to argue about.
Prefer it as a development diagnostic wherever the change is supposed to
preserve arithmetic. The task contract remains authoritative for acceptance:
some tasks intentionally use answer matching, replay fidelity, or a declared
numerical tolerance, and this reference does not silently replace those rules.

## When the change really does change the math

Some changes legitimately alter arithmetic: a different attention implementation,
a different precision, a fused kernel. Exact parity is then the wrong bar.

**Teacher forcing** is the technique. Left free-running, two implementations
diverge after the first different argmax and every later comparison is between
different sequences. Instead, force both along the *same* history and compare one
step at a time:

```
for each step:
    ref_logits  = reference(history)
    sys_logits  = system(history)
    compare at this step
    history.append(argmax(ref_logits))     # the reference chooses, always
```

Now every comparison is at an identical input, and a divergence is attributable.

**Near-ties.** If the top two logits are nearly equal, a different argmax is
numerical noise rather than a defect. Define a gap threshold below which a
disagreement is inconclusive, and derive it from the *reference's own* run-to-run
variation — never choose it after seeing the system's results.

## The rule that actually decides

The endpoint `--check-protocol-context` command checks protocol, native token IDs and
context handling, NOT model probabilities. `--measure-only` measures performance
and replay fidelity, NOT model probabilities. Running both, including
`run_trial.py --through full`, still does not establish complete correctness.
Output correctness is verified privately by the host after the run; its method
is not disclosed. Design and run your own verification. Never substitute re-tokenized text,
replayed answers, canned output or workload-specific bypasses for model decoding.

Use the supplied `native_tokens.py` adapter: `delta.token_ids` must carry each
actually emitted token ID exactly once, in order, including special tokens;
text must be its decode. The task instruction specifies the request fields,
self-test, and host-only full-verifier boundary. Unknown/unmeasured is not a pass.
Checkpointing remains allowed before correctness or performance is established.

For suspected GPU clock locks, `./gpu-clock check` is read-only. Low idle MHz
alone is not evidence of a lock. Recovery requires an explicitly locked, idle,
exclusively assigned GPU; use the guarded command described in the task's GPU
clock section, never reset another job's GPU or reset during a measurement.

Actual-workload replay fidelity is also mandatory: every offered turn must be
served whole at the required output length. The short-token public replay
fixture is only a work-shape diagnostic, not a correctness check.

In a full measurement, read `osl_mismatch_count` and `error_request_count`
before any latency number. A server that returned fewer tokens than asked looks
faster on every gate.

## Proving the gate actually works

A gate nobody has tried to fool is a gate nobody should trust. Build deliberate
corruptions and check that each is caught:

- perturb logits by a defined amount
- permute position ids
- reuse a cache slot without clearing it
- drop the last token of the prompt
- swap two rows' cache slots
- return the previous request's answer

**A corruption that passes is a failure of the policy, not of the corruption.**

Two mistakes to avoid, both learned the hard way:

- **Make sure the corruption is mathematically capable of tripping the gate.**
  Applying a log-softmax to logits leaves softmax exactly unchanged — the
  divergence is 0.0, not "small". Scaling logits by 1.001 moves the distribution
  by about 1e-6, which will not trip a 1e-3 bound. Under a noisy backend both can
  *appear* to be detected, and then you are measuring noise rather than detection.
- **Score across a fixed, pre-committed set of cases.** Hand-picking the scoring
  prompt gives an unstable score that moves every run.

## Protocol and lifecycle

Correct tokens are not enough:

- health and readiness mean what they say, and readiness persists
- an unsupported input is a structured error, not a crash and not a wrong answer
- cancellation is honoured, idempotent, and frees state
- timeouts are enforced and reported
- the server survives clients disconnecting mid-stream
- shutdown drains rather than truncating
- **no hidden fallback**: if a request took a different path, the result says so
- **every offered request is in the denominator**, including failures, timeouts
  and cancellations

## Neighbouring the envelope

Test just inside every boundary you claim to support: minimum and maximum prompt
length, minimum and maximum output length, largest batch, highest concurrency.
Then test just outside, and check the refusal is clean.

Narrowing the envelope after seeing results is a workload change wearing an
optimisation's clothes.

## Under stress

Some defects only appear over time: slot leaks, reference-count drift, unbounded
queues, gradual memory growth.

Run a long window with churn — arrivals, cancellations, disconnects — and check
settled resource use, not point samples. **Take the baseline after the allocator
has settled.** A baseline sampled during warm-up manufactures growth that is not
there, and you will spend a day chasing it.
