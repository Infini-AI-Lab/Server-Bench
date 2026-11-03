# How a p90 bar is derived, and what it guarantees

A task's p90 TTFT and TPOT bars (`service_levels` in its `contract.yaml`) are
frozen before any agent sees the task. This page states how they are measured,
what they do and do not promise, and how to check them. The bars themselves are
task data; [`serving_tasks/README.md`](../../../serving_tasks/README.md) lists
them, and the measuring commands are in
[`OPERATIONS.md` section 3](../../OPERATIONS.md#3-before-the-first-evolve-on-a-task).

## The two steps

**1. The reference operating point.** SGLang, at the task's pinned image and
launch line, is replayed at a set of offered concurrencies
(`python -m final_validate.concurrency`), each point a full 1,800-second window
on one fresh seed. `reference.yaml`'s `concurrency_search` records every point,
the `selection_rule` and the `adopted` point, which becomes the module's
`REFERENCE_CONCURRENCY` and is stated to the agent as a reference, never
imposed. Two rules are in use:

| `selection_rule` | picks the point that maximises |
|---|---|
| `max_gross_dollars_per_dollar_over_p90_request_latency` (also recorded as `..._over_p90_e2e_seconds`, the same computation) | gross dollars-per-dollar divided by p90 end-to-end request latency |
| `slo_constrained_max_throughput` | output tokens per second, subject to the SLO |

They optimise different things, so operating points chosen under one are not
comparable with points chosen under the other.
`tests/measurement/test_the_bar_is_a_max_of_noisy_quantiles.py` re-derives the
adopted point from the recorded points under the recorded rule and refuses a
rule it cannot reproduce.

**2. The bars.** Five more SGLang runs at the adopted point, each on its own
random seed. The rule, recorded under `concurrency_search.slo_derivation`, is

> the worst (highest) over 5 random seeds of SGLang's per-run p90 TTFT and p90
> TPOT, at the searched operating point

with `multiplier: 1.0`, so each bar *is* the worst of the five draws. The
derivation also records each run's seed and p90s and the worst values before
rounding. Where a bar is rounded for publication, the derivation records it
(`published_rounding`) and rounds up, so the bar is never tighter than what
SGLang measured.

A task that shares another task's bars names it in `derived_from` and carries
no `slo_derivation` of its own. A task not yet measured on its hardware (the
Modal L40S tasks) carries provisional bars and states no reference operating
point.

## The estimator

A run's p90 is the one aiperf reports: linear interpolation between adjacent
order statistics of the run's **profiling-phase** requests
(`metadata.benchmark_phase == "profiling"`); warmup requests are excluded.
Recomputing a p90 from `profile_export.jsonl` must use the same estimator and
the same filter, or it will not match.

The AgentX `inferencex-agentx-mvp` scenario refuses a window shorter than
900 s (it needs that long to reach steady state and trigger KV offloading), so
a short run is a different regime, not a noisier version of a long one. Every
reference and verdict window is 1,800 s.

## What a worst-of-S bar guarantees

Write `S = 5` for the seeds and `M = 1.0` for the multiplier:

```
bar = M * max_{s=1..S} q̂90(run s)
```

`q̂90(run s)` is an estimator over finitely many requests, so the maximum is
taken over noisy estimates.

**Lemma 1. The maximum of unbiased estimates is biased upward.** Let
`Ŷᵢ = yᵢ + εᵢ` with `E[εᵢ] = 0`. Then `E[max_i Ŷᵢ] ≥ max_i yᵢ`.
*Proof.* For each `j`, `max_i Ŷᵢ ≥ Ŷⱼ` pointwise; take expectations to get
`E[max_i Ŷᵢ] ≥ yⱼ`, for every `j`. ∎

**Lemma 2. The estimation noise shrinks with the request count.** For `F`
differentiable at its p-quantile with positive density `f`,
`√n (q̂p − qp) ⇝ N(0, p(1−p) / f(qp)²)` (van der Vaart, *Asymptotic
Statistics*, Corollary 21.5).

So the bar sits above a typical reference run by an amount that grows with `S`
and shrinks with `n`. A sample quantile is not exactly unbiased, and runs on
different seeds replay different samples of the corpus, so Lemma 1 describes
the idealised case; the size of the effect on a given task has to be measured
(below), not assumed.

**The rank statement.** If the run-level p90s of `S + 1` reference-equivalent
runs are exchangeable with no ties, a fresh run exceeds the worst of the other
`S` with probability exactly `1/(S+1)`: `1/6` at `S = 5`, `1/10` at `S = 9`.
Raising `S` is the lever that moves it. Three limits:

- it is a **marginal** probability over the joint draw of all `S + 1` runs, not
  the probability of exceeding a bar already fixed at a particular value;
- exchangeability is an assumption about the runs, not something the five
  recorded runs establish;
- it is **not a false-rejection rate**. The bars are not pass/fail gates: they
  qualify individual requests for billing (`service_levels.py`), so a missed
  bar lowers the objective and never disqualifies a run. The hard gates are
  replay fidelity, output correctness and the EOS gate
  ([`correctness.md`](../serving_evolve/correctness.md)).

The bar says nothing about a *candidate's* p90, which is measured once.

## Checking it

```bash
"$SERVERBENCH_PY" serverbench/scripts/audit/bar_is_a_max_of_noisy_quantiles.py
"$SERVERBENCH_PY" serverbench/scripts/audit/bar_is_a_max_of_noisy_quantiles.py <run>/profile_export.jsonl
```

With no argument it reads every live task's `reference.yaml`, re-derives each
bar from the runs the task records, simulates the max-of-quantiles inflation
under three tail families at the recorded request counts, names every task
whose bars trace to no recorded derivation, and reports bars stated to more
precision than their evidence. It exits non-zero if a bar stops reproducing.
Given a run's `profile_export.jsonl`, it bootstraps that run's own requests
instead of assuming a tail family, which is the measurement to prefer: the
inflation depends on the tail shape at each operating point and does not
extrapolate from the request count alone.
