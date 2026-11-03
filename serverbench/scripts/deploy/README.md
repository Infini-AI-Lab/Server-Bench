# Standing up a machine for serving-evolution runs

```bash
git submodule update --init --recursive
bash serverbench/scripts/deploy/deploy.sh
```

`deploy.sh` runs four stages in order; each can be run alone:

| stage | what it does | needs |
|---|---|---|
| `00-envs.sh` | the harness venv `serverbench/.venv` (`serverbench[dev]`, editable, plus the pinned AgentX, verified by `python -m final_validate.agentx_pin`) `harbor/.venv`, which composes every evolve container, and `serverbench/artifacts/agentx-runtime`, the same pinned AgentX mounted read-only at `/opt/agentx` in the agent's container (`SERVERBENCH_AGENTX_RUNTIME` overrides the location) | CPU, network, `uv` |
| `01-preflight.sh` | docker answers, the chosen `GPU` exists and is free, the runtime passes it through, `uv` and the `claude` CLI are on `PATH`, the model is staged, disk is free and off `/`, the interpreters and Harbor resolve | nothing |
| `02-build-offline-inputs.sh` | the offline wheelhouses runs mount at `/opt/env-input`: `serverbench/artifacts/env-input` and, for every task that declares its own, `serverbench/artifacts/env-input-int4`. Each replays its tracked `resolved.requirements.txt`. Then the verdict's held-out answer pool, `serverbench/artifacts/eval_samples/gsm8k-final_validation.json`, written by `build_answer_pool.py` from the GSM8K test split when missing. `--list` prints the plan | network |
| `03-build-images.sh` | `serverbench/serving-agent:cu128`, `serverbench/evolve-agent:cu128`, and each image a task declares (`serverbench/evolve-agent:cu128-int4`). Runs after `02`, because the evolve image installs a wheel from the wheelhouse | network |

`00-envs.sh` runs first because `config.sh` sources `scripts/env.sh`, which
refuses without the harness interpreter. A host may bind other environments in
the gitignored `scripts/env.local.sh` instead.

It stops there on purpose. What a task needs next is measured per
(task, model, hardware) and is not copyable between machines: the SGLang
reference operating point and, for a new task, its p90 bars
([`OPERATIONS.md` section 3](../../OPERATIONS.md#3-before-the-first-evolve-on-a-task)).
Then run an agent with [`server_evolve/evolve-task.sh`](../server_evolve/README.md).
The checks that read deployed-host evidence run with
`"$SERVERBENCH_PY" -m pytest serverbench/tests -q -m host`.

## Configuring for a different machine

Everything is in `config.sh` and overridable from the environment:

| variable | default |
|---|---|
| `TASK` | `qwen3_5_2b_standard_1xL40` |
| `GPU` | `0`, chosen explicitly; `nvidia-smi -L` lists what is here |
| `MODEL_ID` / `MODEL_REVISION` / `MODEL_SRC` | read from the task by `scripts/model_for_task.py` |
| `ENV_INPUT` | `serverbench/artifacts/env-input` |
| `BASE_IMAGE` / `TASK_IMAGE` | `serverbench/serving-agent:cu128` / `serverbench/evolve-agent:cu128` |
| `PROJECT_SCRATCH` | `/scratch/$USER/serverbench`: build contexts, temp and caches |
| `HF_HOME` | `~/.cache/huggingface` |

## Where large files go

Weights, wheelhouses and evidence stay under the repository and the Hugging Face
cache, because losing them costs GPU-hours. Large transients go to
`PROJECT_SCRATCH`: `config.sh` points `TMPDIR`, `DOCKER_TMPDIR` and `UV_CACHE_DIR`
there, since each defaults to the root filesystem if left alone, and
`01-preflight.sh` fails when any of them resolves to `/` and warns when docker's
layer store does. Docker extracts image layers into containerd's root (by default
`/var/lib/containerd`), which `docker system df` does not count; if that is on a
small root filesystem, moving it is root-owned daemon configuration for the
host's administrator.

On a shared host, never prune another user's images, volumes, containers or the
shared build cache (`docker system prune` does all of that).

**The model-staging trap:** a Hugging Face cache snapshot is a farm of relative
symlinks into `../../blobs/`. Bind-mounting the snapshot alone mounts the links
without their targets: no weights, no tokenizer, no chat template. Keep `blobs/`
adjacent to `snapshots/`, or copy real bytes; a run refuses a model whose links
dangle.

## `with-gpu`

```bash
bash serverbench/scripts/deploy/with-gpu 3 <command ...>
```

Runs a command holding an exclusive `flock` on each named card (`3` or `3,4`),
so two measurements can never share a device. It waits by default
(`GPU_LOCK_WAIT=0` fails at once), declines a card under a live run claim
(`/tmp/serverbench-card-claims/gpu<N>`), and exits 75 when it cannot get a card.
The evolve launcher does not take this lock: it declines a card whose lock or
claim is held (`evolve-common.sh::gpu_is_held`), and the driver writes a claim
for the cards it takes (`lifecycle/workspace.py::Ownership.took_device`).
`final_validate.lease` takes both the lock and the claim.
