# `serverbench/harness/final_validate/` — the verdict

Everything that decides what an agent's serving system scored. It runs on the
host, outside the agent's container, once per run (called by
`harness/lifecycle/run.py::_final_validation` through
`tools/validate_evolve.py`). The agent never sees its output.

Part of this package is also the agent's own measuring instrument: the
`SHARED_MODULES` listed in `package_grader.py` are copied byte for byte into
every task's `tests/final_verdict/`, so development measurements and the verdict
use the same code. `taskctl render` regenerates the copies; `taskctl check`
fails on drift.

## How a replay is run

`scenario.py` offers the candidate the load its `concurrency.json` declares (a
baseline arm is offered the task's stated SGLang reference point), over the
scenario's own loader on the pinned corpus, with `ignore_eos` required. The
seed is a fresh draw per run: even in development, odd in the verdict. The
verdict window is 1,800 s; a development measurement chooses 300-1,800 s
(`--duration`). The AgentX scenario refuses a window under 900 s, so a shorter
development `--measure-only` run passes `--unsafe-override` and is stamped
`submission_valid=false`; a development report accepts that stamp for such a
window and nothing else, and the verdict never scores an invalid submission.
`--measure-only` reports replay fidelity and the objective;
`--check-protocol-context` runs only the streaming and context probes; with
neither, the run is a final verdict.

## What is copied into each task (the measurement half)

| module | what it does |
|---|---|
| `scenario.py` | the verdict for an AgentX scenario task: launch an arm, replay the task's sessions at the declared load, aggregate, score |
| `spine.py`, `task_spec.py` | a task's frozen contract (`contract.yaml`, `reference.yaml`) and what a task module is |
| `service_levels.py` | the p90 TTFT/TPOT bars, stated once |
| `objective.py`, `token_prices.py` | `dollars_per_dollar`: served tokens valued at the task's prices over the GPU cost, scaled by the per-request score |
| `arms.py` | the pinned reference arms and the payload settings that make their work comparable |
| `agentx_pin.py` | verifies the AgentX installation before anything is measured |
| `stream_timing.py`, `native_tokens.py` | streamed-token timing and the native-token SSE contract |
| `replay_verify.py`, `warmup_watch.py` | the short replay-fidelity preflight, and the bound on a warmup that stops progressing |
| `candidate_bundle.py` | which files of a published tree are delivered to the verdict |
| `tasks/` | the registry, `_agentx_mvp.py` (the execution contract every live task shares) and one module per task, `<task id, lower case>.py` (model, devices, context rule, reference operating point); a task's copy carries only its own module |

## Host-only (not copied into any task)

| module | what it does |
|---|---|
| `private_correctness.py`, `agentx_reference.py` | the private correctness stage ([`docs/serving_evolve/correctness.md`](../../docs/serving_evolve/correctness.md)). `package_grader.NEVER_PACKAGED` keeps them out of every task copy, and `tests/integrity/` checks that nothing agent-visible describes them |
| `deploy.py` | materialises an immutable published version for measurement |
| `model_identity.py` | is the model on the card the one the task froze |
| `lease.py`, `gpu_reservations.py` | device ownership: one lock that honours both the claim files and `scripts/deploy/with-gpu`, and the container reservations that make a card busy (`SERVERBENCH_IGNORE_GPU_CONTAINER_IDS` exempts named containers) |
| `store.py` | where verdict evidence may be written (`artifacts/final_validation/`), and refusing anywhere an agent can reach |
| `modal_verdict.py`, `modal_worker.py` | the same verdict run in Modal sandboxes, for the Modal tasks |
| `package_grader.py` | writes and checks the per-task copies |
| `concurrency/` | `python -m final_validate.concurrency`: measures the SGLang operating point a task's bars come from, once per (task, model, hardware); writes to `artifacts/concurrency/<host>/<task>-<arm>/` unless `--out-dir` says otherwise. Method: [`docs/harness/bar-derivation.md`](../../docs/harness/bar-derivation.md) |
| `sglang_compat/` | a narrow `sitecustomize.py` fix loaded into the pinned SGLang container for Gemma 4 |

## `tools/` — operator entrypoints

| script | what it does |
|---|---|
| `validate_evolve.py` | take one finished run's accepted head to the verdict (what `run.py` calls) |
| `measure_published.py` | measure a published version, and optionally the reference arms, through the verdict by hand |
| `gpu_lease.py` | `--status`, or take a bounded lease on N cards, through `lease.py` |
| `prove_isolation.py` | probe, from inside the agent image, that none of this package is reachable; `--out` writes the evidence the `host` tests read (`artifacts/isolation/agent-container.json` by default) |
