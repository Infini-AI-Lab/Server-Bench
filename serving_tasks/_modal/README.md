# Running a serving task on Modal

**This directory is the Modal mode and nothing else.** Every command here
talks to Modal: the volumes it builds and deletes are Modal Volumes, the
sandboxes it stops are Modal sandboxes.

The docker mode is untouched by all of it. Its images, model cache and
wheelhouse are shared with every other run on the host, so nothing a Modal
cleanup does may reach them: no docker entrypoint mentions `modalctl`, `purge`
or `Volume.delete`.

| file | what it owns |
|---|---|
| `modalctl.py` | the commands below, and **the only place a run's sandbox is created** |
| `docker_to_modal.py` | composes the `serving-agent` and `evolve-agent` Dockerfiles into the Modal image: the same image, adapted, not a second one |
| `parity.py` | audits a sandbox against what the brief promises: paths, environment, commands |

Everything runs under an interpreter that can `import modal`:
`harbor/.venv/bin/python` (built by `serverbench/scripts/deploy/00-envs.sh`). The
harness interpreter deliberately cannot, so Modal's dependency tree never moves
packages underneath a running driver.

## Setting up

```bash
M=harbor/.venv/bin/python
$M serving_tasks/_modal/modalctl.py setup --task qwen3_5_9b_standard_modal_2xL40S
```

That builds, in order, and skips whatever is already correct:

1. **the checkpoint**: pulled from Hugging Face by its **pinned revision** into
   the `serverbench-models` volume, inside Modal; nothing is uploaded from the
   host.
2. **the replay corpus**: pulled by its pinned dataset id into a processed
   datasets cache, which is what an offline `load_dataset` reads.
3. **the wheelhouse**: the offline build inputs, into `serverbench-env-input`.
4. **the image**: built from the Dockerfiles, then proven by running `aiperf`,
   `nvidia-smi` and `torch.cuda.device_count()` inside a real sandbox.

Every stage verifies by **performing the real operation** (for the checkpoint,
resolving it offline from a different container than the one that wrote it),
never by listing files.

Then:

```bash
bash serverbench/scripts/server_evolve/modal-evolve.sh \
    qwen3_5_9b_standard_modal_2xL40S --hours 1
```

which runs `setup` and `check` itself first, so the command above is only needed
to build ahead of time or see the stages separately. The agent, its workspace,
its server and the cards are all in one Modal sandbox; the driver stays on the
host. The provider key goes to that sandbox as a per-sandbox Secret and nowhere
else.

## Tearing down

```bash
$M serving_tasks/_modal/modalctl.py purge --yes
```

Stops this app's sandboxes and deletes both volumes. `modal-evolve.sh` keeps the
volumes after a run by default; set `SERVERBENCH_KEEP_MODAL_STORAGE=0` to have it
purge when the run ends. Nothing in the volumes is authored here: every byte is
re-fetchable from a pinned revision by `setup`.

The run's evidence is never only in a volume. The agent's tree is pulled home at
every version boundary, and the verdict's evidence is pulled to the host before
its sandbox is terminated, both under `evolve_space/` and `server_prototypes/`.

`purge` **refuses while a sandbox is running** unless `--force`: deleting a
volume under a live run takes the checkpoint away from a server
mid-measurement, and the failure reads exactly like a candidate defect.

## Which GPUs a task gets

From the task id's suffix, cross-checked against `devices` in its
`contract.yaml`: `..._2xL40S` → `L40S:2`. A task whose suffix and device count
disagree is refused. Modal offers L40S, not L40, so a task frozen on L40 is a
different task with its own id, concurrency search and bars.

## The other commands

```bash
$M serving_tasks/_modal/modalctl.py check --task <TASK>     # refuse before a run, not during
$M serving_tasks/_modal/modalctl.py model --task <TASK>     # checkpoint only
$M serving_tasks/_modal/modalctl.py corpus --task <TASK>    # replay corpus only
$M serving_tasks/_modal/modalctl.py env-input               # wheelhouse only
$M serving_tasks/_modal/modalctl.py sandbox --task <TASK>   # one GPU sandbox, by hand
$M serving_tasks/_modal/modalctl.py teardown                # stop sandboxes, keep volumes
$M serving_tasks/_modal/parity.py --task <TASK>             # audit a throwaway sandbox
```

`check` and `setup` take `--allow-inherited-concurrency` to proceed when the
only thing missing is the task's own SGLang offered-load search.

## One factory

`create_sandbox(task, *, purpose, hours, workdir, env, network)` is the only
place a run's sandbox is created: the agent's, the verdict's and parity's
throwaway all come from it. It owns the image, the task's GPUs, both volumes,
the `/models/target` symlink and the offline environment, and verifies the
checkpoint is staged before it returns. A real difference between an agent's
sandbox and a verdict's (`/candidate` rather than `/workspace`, no egress at
all) is an argument to it, so it is chosen rather than drifted into.

The single exception is the provider-CIDR probe, which takes no GPU, uses none of the
image, and exists to *learn* the allowlist, so it cannot come from a factory
that requires one; it says so in its source.
