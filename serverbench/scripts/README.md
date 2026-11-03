# `scripts/` — programs an operator or the harness runs

Nothing here is library code. A script parses arguments and calls a package
(`harness/`, which includes `harness/final_validate/`); logic that another module
would import belongs in that package, not here.

## Top level

| file | what it is |
|---|---|
| `evolve` | **the one production entrypoint** for an evolve run; no logic, everything is in `harness/lifecycle/run.py` |
| `env.sh` | sourced, not run: resolves the two interpreters (below) and puts this checkout first on `PYTHONPATH` |
| `correctness-gate.sh` | lint, imports, the measurement path, the run spine and the full suite; ends `READY` or fails ([testing.md](../docs/harness/testing.md)) |
| `ci-preflight.sh` | CI's `harness` job (install, ruff, ty, pytest) run locally on exactly the files a push would carry, in a throwaway venv; catches a test that depends on untracked host data |
| `check_git_blob_sizes.py` | refuses tracked files GitHub would reject on push; run by `.github/workflows/cpu.yml` |
| `model_for_task.py` | which model a task pins (`--id`, `--revision`) and the local snapshot that holds exactly it |
| `gpu_holder.py` | makes a card a run owns look owned while it is idle; yields when the agent's server needs it. Started by `run.py`, under a different interpreter |

## Groups

| directory | what it holds |
|---|---|
| [`deploy/`](deploy/README.md) | standing up a machine: environments, preflight, the offline wheelhouses, the images, and `with-gpu` |
| [`server_evolve/`](server_evolve/README.md) | launching an evolve run on a local card or on Modal |
| `agent/` | the agent container: `harbor_env.py` (started by `run.py`), `egress_broker.py` (its only route out), `run_agent.py` (the boundary library: exposure audit, held-out hashing, model-link and device checks), `image_freshness.py` (refuses an image not built from this checkout's Dockerfiles), `gpu_clock.py` (the GPU-clock helper delivered into the workspace), the Modal backend (`modal_env.py`, `modal_exec.py`), and the wheelhouse builder (`build_env_bundle.py`) |
| `evidence/` | `collect_trajectory.py` (run by `run.py` at the end of a run) and `trajectory_query.py` (search a long agent trajectory without reading it) |
| `audit/` | `bar_is_a_max_of_noisy_quantiles.py`: re-derives every task's p90 bars from its recorded runs ([bar-derivation.md](../docs/harness/bar-derivation.md)) |

## The two interpreters

| variable | what it runs |
|---|---|
| `SERVERBENCH_PY` | the harness, the tests, the gates, `taskctl`. No torch or transformers: the harness core must import inside a bare image |
| `SERVERBENCH_BASELINE_PY` | the measurement path: transformers for the tokenizer, httpx for the client |

`env.sh` resolves them in this order: the two variables if set, then a
gitignored `scripts/env.local.sh`, then the checkout's own
`serverbench/.venv/bin/python` (built by `deploy/00-envs.sh`) for both. There is
no fallback to `PATH`; it resolves or refuses.
