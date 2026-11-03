# `serverbench/artifacts/` — pinned inputs and host-local state

Ignored by default (`.gitignore`: `serverbench/artifacts/*`), with a few files
re-included. So this directory holds two different kinds of thing.

## Tracked: pins every host rebuilds from

| path | what it pins | read by |
|---|---|---|
| `agentx/aiperf-pin.json` | the AgentX installation the verdict measures with | `final_validate/agentx_pin.py` (and the Modal verdict) |
| `env-input/resolved.requirements.txt` | the shared offline wheelhouse's frozen resolution | `scripts/deploy/02-build-offline-inputs.sh` |
| `env-input-int4/resolved.requirements.txt` | the same, for the INT4 task's wheelhouse | `scripts/deploy/02-build-offline-inputs.sh` |

## Ignored: per-host inputs and outputs

Produced on each host and never committed: some of it is large, some private
or secret, and greedy batched generation is not bit-reproducible across hosts.

| path | written by | read by |
|---|---|---|
| `secrets/` | the operator (one env file per agent model) | the evolve launcher (`AGENT_ENV=...`), read inside `scripts/agent/harbor_env.py` |
| `env-input/`, `env-input-int4/` (beyond the pin file) | `scripts/deploy/02-build-offline-inputs.sh` | every docker evolve and verdict, mounted at `/opt/env-input` |
| `agentx-runtime/` | `scripts/deploy/00-envs.sh` | `scripts/agent/harbor_env.py`, mounted read-only at `/opt/agentx` in the agent's container |
| `models/` | the operator, when weights are staged by hand (`MODEL_DIR=...`) | the evolve launcher |
| `concurrency/` | `python -m final_validate.concurrency` (default `--out-dir`) | the operator, when choosing a task's SGLang operating point |
| `final_validation/` | every final verdict (private) | `final_validate/store.py` |
| `eval_samples/gsm8k-final_validation.json` | `scripts/deploy/build_answer_pool.py` (run by `02-build-offline-inputs.sh`): the GSM8K test split | `final_validate/private_correctness.py` |
| `operations/` | `evaluation_progress/watch_evolve.py`, `record_run.py` | their own locks and logs |
| `isolation/agent-container.json` | `final_validate/tools/prove_isolation.py --out` | `tests/verification/test_private_final_validation_is_unreachable.py` (`-m host`) |

A fresh clone has none of the ignored half; `bash serverbench/scripts/deploy/deploy.sh`
builds the wheelhouses, and the `host` tests read the rest once a host has run.
