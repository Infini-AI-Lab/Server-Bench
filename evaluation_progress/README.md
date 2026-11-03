# `evaluation_progress/` — results

Two kinds of results live here: the paper's, committed, and your own runs',
written by the harness as they finish. Nothing in this directory measures
anything.

| file | what it is |
|---|---|
| `leaderboard.csv` | the paper's main results: every agent model and both baselines on the three deployments, ranked by `dollars_per_dollar`, with the offered concurrency, p90 TTFT (s) and TPOT (ms), the value relative to the strongest baseline, and the agent's API cost at OpenRouter rates (Qwen3.5 9B and Gemma E4B) |
| `ablations.csv` | the paper's controlled comparisons, each replaying frozen servers on matched seeds: capacity (v1 to v6), draft length, replay feedback, successive versions and offered load; plus one server's development draw against its final draw |
| `<task id>.csv` | one row per finished evolve of that task, ranked by the scored objective within each agent budget |
| `record_run.py` | writes a run's row from its evidence; idempotent (re-running replaces the same row atomically) |
| `watch_evolve.py` | queues evolves of one agent configuration on free cards and records each as it ends; the launcher behind [`experiments/`](../experiments/README.md) |

Every agent in `leaderboard.csv` ran at maximum reasoning effort for one
8-hour development run per deployment; GPT models on Codex, the others on
Claude Code. `not passed` marks a submission that did not pass final
evaluation, which scores nothing. Qualified value in `ablations.csv` is
latency-qualified dollars per dollar; the ratios are after over before.

Your own runs' rows go to `<task id>.csv`, which is host-local output and is
not committed. Every number in such a row is copied from evidence the final
verdict already wrote.

## How a row gets here

The evolve driver calls `record_run.py --run <run root>` at the end of every
run (`harness/lifecycle/run.py`), and `watch_evolve.py` does the same for the
runs it queued. By hand:

```bash
"$SERVERBENCH_PY" evaluation_progress/record_run.py --run serverbench/evolve_space/evolve/<task, lower case>/<run-id>
"$SERVERBENCH_PY" evaluation_progress/record_run.py --all-runs      # every run under evolve_space
```

A row is final only once the lineage's `FINAL.json` exists. A run's
`SUMMARY.json` turns terminal earlier, when the verdict starts, and the recorder
refuses lifecycle states (`RUNNING`, `FINAL_VALIDATION_STARTED`, ...) as
results. Repeats of one configuration are separate rows: the key is the run id,
the arm, the system and the start time.

## Queueing several runs

```bash
"$SERVERBENCH_PY" evaluation_progress/watch_evolve.py --campaign <name> \
    --task gemma4_e4b_it_standard_1xL40 --task qwen3_5_9b_standard_2xL40 \
    --agent-cli claude --model claude-sonnet-5-5 --effort max --hours 8
```

Add `--agent-env serverbench/artifacts/secrets/<model>.env` for a model reached
through an API key, and `--gpus` to confine the campaign to some cards. The
watcher's log, `status.json` and one launcher log per job land under
`serverbench/artifacts/operations/<campaign>/`. A launch that loses a card to
another launcher before any agent work starts is requeued without a row.

## Columns

| column | meaning |
|---|---|
| `rank` | position by `dollars_per_dollar` within rows of the same `evolve_hours`, recomputed on every write; ties share a rank, unscored rows have none |
| `task`, `system`, `arm` | task id as run; `<model> <effort>` of the agent; `candidate` for an agent's server |
| `model`, `agent`, `effort` | the agent model's full id, the agent CLI, the reasoning effort |
| `concurrency` | the offered load the verdict actually used (read from its aiperf command) |
| `requests` | profiling requests the verdict's 1,800 s replay offered, errored and cancelled ones included |
| `correctness` | the private output check: held-out GSM8K answers, candidate vs the plain checkpoint on the same questions, under the accuracy rule stored with the result. Operator-facing: never delivered to an agent |
| `eos_gate` | the end-of-sequence gate: the stop probe plus the audit of padded-looking replies (`pass` / `FAIL`) |
| `replay` | replay fidelity: every reply exactly the requested length, no errors (`ok` / `FAILS`) |
| `ttft_p90_ms`, `tpot_p90_ms` | the verdict's p90 latencies; `not measured` for TPOT when the output was not streamed |
| `output_tokens_per_s`, `requests_per_s` | the verdict replay's throughput (aggregate output tokens/s and completed requests/s) |
| `gross_dollars_per_dollar` | token value over GPU cost, counting every request (a diagnostic) |
| `dollars_per_dollar` | the scored objective = gross x multiplier; empty unless the verdict is `HOLDS` |
| `gates_missed` | every bar or gate the run missed, in words |
| `verdict` | `HOLDS`, `MISSES`, `NOT_RUN`, or the terminal state of a run that never reached measurement |
| `evolve_hours`, `started_utc`, `measured_utc`, `run_id`, `version`, `versions_published`, `backend` | which run; the version the verdict judged (the run's final head) and how many versions the run published; where the agent ran: `container` (local Docker) or `modal` |
| `source` | `verdict` (read from `FINAL.json` + `scenario-*.json`) or `run_summary` (a run with no measurement) |
| `evidence` | the verdict's evidence directory under `server_output/<lineage>/FINAL_VALIDATION/` |
