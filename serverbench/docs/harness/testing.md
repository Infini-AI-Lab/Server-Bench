# Testing

```bash
. serverbench/scripts/env.sh
"$SERVERBENCH_PY" -m pytest serverbench/tests -q               # the suite, CPU only
"$SERVERBENCH_PY" -m pytest serverbench/tests -q -m "not slow" # in-process only
"$SERVERBENCH_PY" -m pytest serverbench/tests -q -m host       # on a deployed host
bash serverbench/scripts/correctness-gate.sh                   # lint, imports, then the suite
```

The default suite needs no GPU, model weights or network. `slow` marks the
tests that shell out or drive docker. `host` marks the checks that read
evidence a deployed machine records (for example the isolation probe at
`serverbench/artifacts/isolation/agent-container.json`); they are deselected by
default (`addopts = "-m 'not host'"` in `serverbench/pyproject.toml`) and run with
`-m host` after [`deploy.sh`](../../scripts/deploy/README.md).
`scripts/ci-preflight.sh` runs CI's checks on exactly the files a push would
carry, so a test that depends on untracked host data fails before the push.
The suite's scope policy is [`tests/README.md`](../../tests/README.md).

`tests/` is organised by subject:

| package | what it pins |
|---|---|
| `contracts/` | the live tasks: every generated file in step, the p90 bars, the stated load, a brief that states only facts, instructed twins, `taskctl` and `new_task` |
| `measurement/` | how a replay is measured and scored, and the SGLang evidence the p90 bars are derived from |
| `verification/` | the verdict: request scoring and the EOS gate, private answer accuracy, the packaged evaluator, the frozen corpus, the results ledger |
| `integrity/` | what never reaches a workspace: held-out data, correctness code, other runs |
| `agent/`, `evolve/`, `driver/` | the agent's network and routes, its full context window, the brief, the run lifecycle and the driver |
| `kernels/` | the kernel-evolve contracts |
| `repo/` | repository rules: the core imports in a bare image, one production flow, deploy scripts, the gate runs what it names |
| `kernels/` | the kernel-evolution subsystem and its knowledge catalog |

The rules the suite holds itself to:

- **A failure is a failure.** Nothing rewrites a failing test into a skip; a
  test whose subject no longer exists is deleted, not silenced. A tracked file
  or a live task a test needs is asserted, not skipped when absent; a skip is
  only for what a host may lack (docker, a staged model, a wheelhouse) or for
  a task not yet measured on its hardware.
- **No test reads a run's workspace.** `evolve_space/` is runtime output; a
  check that needs a real export or stream builds one in `tmp_path`.
- **Tests read the live task list** (every `serving_tasks/*/` with a
  `contract.yaml`, `tests/paths.py::released_tasks`) rather than naming tasks,
  except where a test pins a value by hand on purpose, such as
  `contracts/test_the_p90_bars.py`.

`correctness-gate.sh` is the pre-merge check: ruff over
`harness/serving_evolve`, `scripts/server_evolve` and `tests/contracts`, an
import of every module under `serving_evolve`, `kernel_evolve` and `lifecycle`,
the `measurement/` + `verification/` suites, the `evolve/` (run spine) suite,
the `contracts/` suite, then the whole suite (`--quick` stops before the last
two). It prints `SERVING_EVOLVE_SPECIALIZED_CORRECTNESS_READY`, or
`NOT READY:` and the first step that broke.
