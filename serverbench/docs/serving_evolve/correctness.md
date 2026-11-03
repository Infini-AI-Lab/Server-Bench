# Correctness

A serving system can be wrong in a way ordinary software cannot: fluent,
confident output that is not what the model would have said. Nothing crashes,
and a well-formed-response check passes.

## How a serving verdict decides it

This is repository-side documentation and is never delivered to a workspace
(`lifecycle/delivery.py` copies no part of `serverbench/docs/`). The agent is
told what correct output is (the checkpoint's own greedy tokens at its own
precision, including past an end-of-sequence token under `ignore_eos`) and
never how it is checked. The code is host-only
(`final_validate/private_correctness.py`, with `agentx_reference.py`), and
`package_grader` does not copy it into any task's `tests/final_verdict/`.

Three checks, all hard; failing any one makes the outcome `MISSES` regardless
of the objective. The settings are each task's private `correctness` block in
`reference.yaml` (stripped from the delivered copy); every live task uses rule
`replay_fidelity_and_held_out_answers_vs_plain_checkpoint` with
`answer_sample_size: 20`, `max_absolute_accuracy_drop: 0.2`,
`max_token_regret: 4.0` and `eos_audit: true`.

1. **The replay gate** (in `scenario.py`). Every timed request returns exactly
   the output length its trace asked for, with no errors or dropped turns
   (aiperf's own `osl_mismatch_count` and error counts). After the window, the
   streaming-protocol probe checks native token IDs, decoded text and usage.
   For a task whose module declares `REQUIRED_CONTEXT_LENGTH` (the Qwen3.5 9B
   tasks and the 27B INT4 task, at 1,048,576), a near-maximum context probe
   also checks prompt accounting.
2. **Held-out answer accuracy relative to the plain checkpoint.** After the
   timed replay, while the candidate is still up, a seeded draw of
   `answer_sample_size` held-out GSM8K questions with known numeric answers
   (`artifacts/eval_samples/gsm8k-final_validation.json`, gitignored,
   host-only) is put to it: greedy, a natural stop, at most 1,024 tokens. The
   candidate is then stopped and a plain SGLang reference of the same
   checkpoint is launched on the same cards (speculative decoding off,
   deterministic Triton attention, no radix cache, one request at a time) and
   answers the same questions. The candidate may lose at most
   `max_absolute_accuracy_drop` accuracy against the plain checkpoint: at most
   four fewer correct answers out of 20. Comparing answers rather than
   teacher-forcing the timed replies keeps an exact but differently-batched
   system from failing on legitimate numerical differences.
3. **The EOS gate** (tasks whose `reference.yaml` sets `eos_audit: true`).
   Under `ignore_eos` a server is told exactly how many tokens to return; one
   that stops at end-of-sequence and fills the rest with tokens it did not
   compute returns the right count for a fraction of the work, and replay
   fidelity cannot see it.
   - *Stop probe*: one short streamed request without `ignore_eos` must finish
     with `finish_reason == "stop"` before `max_tokens`. If the plain reference
     does not stop either, the probe is inconclusive rather than a failure.
   - *Padding audit*: every committed timed reply's native IDs are scanned. A
     reply is suspect when an EOS-class token is followed by any further token,
     or when 64 identical tokens run anywhere. Up to 16 suspects (a seeded
     choice) are teacher-forced on the plain checkpoint under
     `max_token_regret`; a model that genuinely continues past EOS, or a greedy
     continuation that really loops, passes, and fabricated tokens do not. A
     reply without native IDs, or whose native IDs do not decode to exactly the
     text it streamed, fails the gate.

Development evidence the agent sees (`--measure-only`, `--check-protocol-context`,
`./full-test`) covers the replay gate and the protocol (and declared context)
probes only. The final verdict's workload seed is drawn from a space no
development run uses.

## The gate ladder (kernel evolution)

`contracts/gates.py` and `evaluator/gates.py` implement a general gate ladder
(`identity`, `envelope`, `exact_tokens`, `output_length`, `isolation`,
`accounting`, `hidden_fallback`, `equal_work`) with `PASS`, `FAIL`,
`INCONCLUSIVE`, `SKIPPED` and absent kept distinct. The kernel-evolution
harness uses it; serving verdicts use the path above.

Exactness is legitimate only against a single reference under greedy decoding.
Between two independent engines, kernel choice moves logits and one different
argmax diverges everything after it, so cross-engine agreement is a diagnostic,
never a gate. For a change that alters the math, teacher forcing compares both
implementations along the same history, so a divergence is attributable.

## The golden cases

A clean case and sixteen deliberate corruptions, each with a known correct
verdict (`evaluator/golden.py`):

```
clean · wrong_model · wrong_revision · sampling_changed · one_token_differs
state_leaks · requests_dropped · outcomes_missing · shorter_output
hidden_fallback · undeclared_fallback · envelope_narrowed · unequal_work
stale_fixture · no_samples · zero_denominator · evaluator_tampered
```

**A corruption that passes its gate is a failure of the policy, not of the case.**
Two rules keep the cases honest:

- A corruption must be *mathematically capable* of tripping its gate. A
  log-softmax applied to logits leaves softmax unchanged, and scaling logits by
  1.001 moves the distribution by about 1e-6, far inside a 1e-3 bound; under a
  noisy backend either can appear detected, and the suite would then be
  measuring noise.
- Scoring runs over a fixed, pre-committed case set, never a hand-picked prompt.

Run them: `serverbench evolve golden`.

## What cannot be changed after the freeze

The workload digest is in every record, and the gates read it. So none of these
can be changed to rescue a losing result:

output length · tokenizer or template · precision · sampling · stop rules ·
the request trace · offered load · the failure denominator · fallback policy ·
the supported shape envelope

## Statistics

`evaluator/statistics.py` compares two arms over paired blocks. `INCONCLUSIVE`
is never a pass: "no significant difference" is compatible with a large
difference and a small sample, and equivalence needs a margin declared in
advance. Three rules:

- **the block is the unit of independence**, not the request: requests in one
  run share a machine, a cache, a queue and a thermal state, and treating them
  as independent inflates precision by roughly √(requests per block);
- **equivalence is the only route to PASS**: the whole interval inside the
  margin;
- **validity is checked before statistics**: an invalid block, an SLO violation
  or a shrunken denominator is decided first, in a fixed order.

Its golden populations, whose right answer is known by construction, run
deterministically under `serverbench evolve golden`.
