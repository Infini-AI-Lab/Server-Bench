# Launching an evolution run

```bash
bash serverbench/scripts/server_evolve/evolve-task.sh qwen3_5_2b_standard_1xL40
bash serverbench/scripts/server_evolve/evolve-task.sh 769bf633 --hours 4 --stop beat-one-baseline
bash serverbench/scripts/server_evolve/evolve-task.sh qwen3_5_9b_standard_2xL40 --dry-run
bash serverbench/scripts/server_evolve/evolve-task.sh --help          # every task, with its hash
```

| file | what it is |
|---|---|
| `evolve-task.sh` | resolve a task by full name (any case) or its hash (`task_hash` in `contract.yaml`), then `launch_task` |
| `evolve-common.sh` | the defaults, the device picker and `launch_task`, which execs `serverbench/scripts/evolve` |
| `modal-evolve.sh` | the same driver with the container on Modal instead of a local card |

Everything is overridable from the environment and every value used is recorded
in the run's journal:

| variable | default |
|---|---|
| `HOURS` | `4` |
| `GPU` | `auto` — a card with no compute process, under 1 GB, no claim, no flock, no container reservation, and still quiet after a minute |
| `EXCLUDE_GPUS` | `$SERVERBENCH_EXCLUDED_GPUS`: devices this host must never be given |
| `AGENT_CLI` / `AGENT_MODEL` / `AGENT_EFFORT` | `claude` / `claude-opus-5` / `xhigh`; `codex` defaults to `gpt-5.6-sol` / `max` |
| `AGENT_ENV` | a `KEY=VALUE` file naming another model backend; only its path is passed |
| `STOP` | `timebox` |
| `SEED` | the task's own `demo/` |
| `MODEL_DIR` | the local snapshot of the revision the task pins (`scripts/model_for_task.py`) |
| `ENV_INPUT` | `serverbench/artifacts/env-input`, built by `deploy/02-build-offline-inputs.sh` |
| `IMAGE` | `serverbench/evolve-agent:cu128`, built by `deploy/03-build-images.sh` |

A task that declares its own pair in `environment/agent-environment`
(`IMAGE=`, `ENV_INPUT=`; the INT4 task uses `serverbench/evolve-agent:cu128-int4`
and `serverbench/artifacts/env-input-int4`) gets that pair, and an explicit
`IMAGE` or `ENV_INPUT` that contradicts it is refused.

Launch it detached (`setsid nohup … < /dev/null &`) and record the launcher's
pid from `ps`; stop it only by that pid. `modal-evolve.sh` runs the same driver
with the container on Modal; see
[`serving_tasks/_modal/README.md`](../../../serving_tasks/_modal/README.md).

## What the driver refuses

* **A contended or held GPU.** A card with a foreign process, a live claim, a
  `with-gpu` flock or a container reserving it is declined, never reaped.
* **A device count the task was not measured on.** The task's `contract.yaml`
  says how many cards it is billed for.
* **A model it cannot identify.** The pinned revision must be present locally;
  a dangling Hugging Face symlink farm is refused before launch.
* **A workspace that leaks held-out inputs.** Every exposure is hashed against
  the task's held-out files and refused, not trimmed.
* **A directory that already holds run evidence.**

## What a run leaves behind

```
serverbench/evolve_space/evolve/<task id, lower case>/<run-id>/
  journal.jsonl     one line per event, append-only
  SUMMARY.json      written by the driver at finish()
  workspace/        the agent's tree, including MEMORY/
```

Published versions go to `server_prototypes/<lineage>/`. Nothing reclaims a
finished run's `workspace/.venv`, which holds several gigabytes; watch free disk
before a long series of runs. The wall clock, the journal and the verdict are
described in [`OPERATIONS.md`](../../OPERATIONS.md#2-running-an-evolve).

## When the driver itself dies

The deadline lives in the driver process, so a killed driver leaves the
container standing and nothing checking it. Starting again from its work is a
**new run** seeded from its workspace with `SEED=<path>`, never a reuse of its
directory.

```bash
ps -o pid,etimes,args -p <recorded driver pid>
docker ps --filter name=<task>
jq -r '.event' <run-dir>/journal.jsonl | tail -20
```

## One path from a workspace change to a published version

    the agent edits the workspace and asks with `./version "why"`
        -> `lifecycle/boundary.py` freezes the EXACT tree as a git tree object
           and registers it as an immutable version directory
        -> `lifecycle/publish.py` CASes the one head if the current head is an
           ancestor; otherwise the version is kept and the head stays

There is no other route, and nothing in the run judges a version: the verdict
is `final_validate`, outside Harbor, on a seed the agent never saw.
