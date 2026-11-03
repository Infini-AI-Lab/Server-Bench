# ServerBench

An **autonomous serving-system evolution harness**. A coding agent starts from a
minimal Hugging Face server, gets one frozen serving task, an offline container
and a wall clock, and evolves it into a specialised LLM serving system. What it
builds is judged by a verdict it never sees: a full AgentX replay of real coding
sessions, scored against the task's p90 bars, with output correctness verified
privately.

The live tasks are in [`serving_tasks/`](../serving_tasks/README.md). To set up a
machine and run something, read [`OPERATIONS.md`](OPERATIONS.md).

## The one boundary that matters

```
                HARNESS                    │              CANDIDATE
  ─────────────────────────────────────────┼──────────────────────────────────
  task definition                          │  serving architecture
  agent execution and its container        │  scheduler, batching, admission
  environment control + isolation          │  KV and state management
  measurement and the final verdict        │  kernels, optimisation choices
  versions and trajectory logging          │  the offered load (concurrency.json)
```

The harness contains no scheduler, no KV-cache design, no batching policy, and
no copy of vLLM, SGLang or TensorRT-LLM. Those are reference points measured on
the same hardware and workload, never templates.

## The loop

```
  create run ──► build workspace ──► prove it boots ──► start the agent
                                                              │
                                  the agent develops, measures itself with the
                                  packaged evaluator, and freezes versions with
                                  ./version "why"  (or ./checkpoint, instructed)
                                                              │
                 deadline ──► freeze the final working tree ──┘
                                     │
                                     ▼
          final verdict, once, on the final published head, outside the
          container: the task's 1,800 s AgentX replay at the tree's declared
          load and a fresh odd workload seed, then the private correctness
          stage, then per-request scoring against the p90 bars
```

An agent that stops with budget left is resumed into the same workspace. The
verdict is never fed back to the agent. Each stage is owned by one module of
`harness/lifecycle/`; [`harness/README.md`](harness/README.md) maps them.

## Layout

```
harness/             one package root, five packages: lifecycle/ (the evolve
                     driver), final_validate/ (the verdict), serving_evolve/
                     (contracts, the `serverbench evolve` CLI), kernel_evolve/
                     (kernel subsystem), serverbench/ (the console script)
scripts/             what an operator runs: evolve (the one entrypoint), env.sh,
                     correctness-gate.sh, deploy/, server_evolve/, agent/, evidence/
environments/        the agent-image Dockerfiles (hashed; see its README)
skills/              skill bundles delivered to standard tasks
skills_instructed/   skill bundles delivered to instructed tasks
tests/               the suite; no GPU, weights or network
docs/                design pages: serving_evolve/, harness/, kernel_evolve/
artifacts/           the AgentX pin and frozen wheelhouse resolutions (tracked),
                     plus host-local inputs and private verdicts; see its README
evolve_space/        runtime, gitignored: every run's workspace, evidence and journal
OPERATIONS.md        the runbook
```

A run lives at `evolve_space/evolve/<task, lower case>/<run-id>/`
(`workspace/`, `journal.jsonl`, `SUMMARY.json`, `trajectory.atif.json`), and its
published versions and verdict at `server_output/<lineage>/` in the
repository root, where a lineage is one `(task, model, agent)` ascent. Both are
gitignored, as are the per-task result files `evaluation_progress/<task>.csv`.

## Requirements and tests

Python 3.10+. The harness core is **stdlib-only by design** so it can verify a
candidate inside a bare image; torch, transformers and yaml are imported only
inside functions, and `tests/repo` checks it. Use the interpreters
`scripts/env.sh` resolves, never a bare `python3`:

```bash
. serverbench/scripts/env.sh
"$SERVERBENCH_PY" -m pytest serverbench/tests -q            # the suite, CPU only
bash serverbench/scripts/correctness-gate.sh                # lint + imports + suite
```

More in [`docs/harness/testing.md`](docs/harness/testing.md).

## Documentation

[`docs/`](docs/README.md) is organised by what owns the behaviour:
[`serving_evolve/`](docs/serving_evolve/) (the evolve harness),
[`harness/`](docs/harness/) (bar derivation, testing) and
[`kernel_evolve/`](docs/kernel_evolve/). How the verdict takes its numbers is
[`harness/final_validate/README.md`](harness/final_validate/README.md).
Reproducing the paper's experiments is
[`experiments/README.md`](../experiments/README.md).
