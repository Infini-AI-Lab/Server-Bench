# Running ServerBench

How to stand this up on a machine, run a serving-system evolve on a live task,
and read what it produced. The live tasks are listed in
[`serving_tasks/README.md`](../serving_tasks/README.md).

---

## 1. A machine, from nothing

### What has to be true

| | why |
|---|---|
| NVIDIA driver + `nvidia-smi` | every GPU path resolves the device at run time |
| Docker with the NVIDIA runtime | every framework and every agent runs in a container |
| `uv` and network access during deployment | `00-envs.sh` builds the Python environments; runs themselves are offline |
| ~200 GB free disk | framework images, model weights, per-run workspaces |
| an agent credential | `~/.claude/.credentials.json` (mounted read-only), or a per-model env file under `artifacts/secrets/` |

### Steps

```bash
git submodule update --init --recursive        # skill bundles, AgentX source
bash serverbench/scripts/deploy/deploy.sh      # the four stages below, in order
```

| stage | what it does |
|---|---|
| `00-envs.sh` | the harness venv `serverbench/.venv` (with the pinned AgentX), `harbor/.venv`, and the AgentX runtime the agent's container mounts at `/opt/agentx` (`serverbench/artifacts/agentx-runtime`); CPU and network only |
| `01-preflight.sh` | GPU, docker, disk, interpreters, Harbor, the model |
| `02-build-offline-inputs.sh` | the offline wheelhouses `artifacts/env-input` and `artifacts/env-input-int4`, rebuilt from their tracked `resolved.requirements.txt`, and the held-out answer pool `artifacts/eval_samples/gsm8k-final_validation.json` (`build_answer_pool.py`) |
| `03-build-images.sh` | the agent images `serverbench/serving-agent:cu128`, `serverbench/evolve-agent:cu128` and `serverbench/evolve-agent:cu128-int4` (needs the wheelhouse) |

Each stage runs alone too; [`scripts/deploy/README.md`](scripts/deploy/README.md)
has the configuration and the model-staging rules.

### Where the environment comes from

One place: **Harbor**. `harness/lifecycle/run.py::_start_through_harbor`
composes the container through `harbor.environments.docker.DockerEnvironment`,
driven by `scripts/agent/harbor_env.py` under `harbor/.venv`. A host without
that venv refuses to run. A run's `container_backend` journal event and its
`harbor_environment.json` record that the environment came from Harbor.

---

## 2. Running an evolve

### The one command

```bash
AGENT_ENV=serverbench/artifacts/secrets/<model>.env AGENT_MODEL=<model id> \
    GPU=auto setsid nohup \
    bash serverbench/scripts/server_evolve/evolve-task.sh <task> \
        --hours 8 --stop timebox < /dev/null > <log> 2>&1 &
```

`evolve-task.sh` takes the task by full name or hash (`--help` lists them),
resolves configuration and execs `serverbench/scripts/evolve`, the one
production entrypoint; everything else is in `harness/lifecycle/run.py`. Its
environment variables are in
[`scripts/server_evolve/README.md`](scripts/server_evolve/README.md).

| argument | meaning |
|---|---|
| `--hours` | the agent's wall-clock budget, enforced by the driver |
| `--stop` | `beat-all-baselines`, `beat-one-baseline`, `meet-slo` or `timebox`: a label recorded on the run (`run_created`, `SUMMARY.json`) and in the agent's `WORKSPACE.json`. It is not rendered into the brief, and `--hours` is the only bound the driver enforces |
| `--dry-run` | build and boot the workspace, launch no agent, in the `evolve-scratch` namespace |
| `GPU=<i>` / `GPU=auto` | a device, or the first genuinely free ones; a multi-card task gets that many or is refused |
| `SERVERBENCH_IGNORE_GPU_CONTAINER_IDS` | full container ids to disregard when judging a card busy (an idle buildkit builder that requests every GPU) |

**Launch it detached** (`setsid nohup … < /dev/null &`) and record the
launcher's pid from `ps`, not `$!` (setsid forks). The deadline lives in the
driver process, so a driver that dies mid-run leaves the container with nothing
enforcing it. Stop a run only by that pid. An agent that stops with budget left
is resumed into the same workspace and lineage by the driver itself.

**Plan a card's availability from the deadline plus the verdict.** After the
agent's deadline the driver freezes the final working tree and runs the final
verdict: a complete 1,800-second replay plus the private correctness stage, so
the card stays busy about an hour past `deadline_utc`.

### Watching it

```bash
R=serverbench/evolve_space/evolve/<task, lower case>/<run-id>
jq -r '.event' $R/journal.jsonl | tail -20        # what has happened, in order
cat $R/SUMMARY.json                               # terminal state, versions, verdict
(cd $R/workspace && "$SERVERBENCH_PY" measurements)   # the agent's full-window ledger
```

`journal.jsonl` runs `run_created`, `workspace_built`, `workspace_boot`,
`agent_launched`, a `version_requested` / `version_registered` pair per
published version, `final_validation_started` / `final_validation`, and the
summary. **An exit code is not a verdict**: the final verdict decides. Hitting
the deadline is `TIMEBOX_EXPIRED`, not `FAILED`.

The journal's timestamps say which phase the wall clock went to, and
`trajectory_query.py $R/trajectory.atif.json --timeline` (section 5) shows the
agent's own steps in order. Waiting inside the workspace belongs in `wait/`
(`wait_ready.sh`, `wait_for.sh`, `gpu_free.sh`), which return when their
condition holds, not in `sleep` loops.

### The final verdict

`run.py::_final_validation` runs `harness/final_validate/tools/validate_evolve.py`,
which measures the final published head with the host's `harness/final_validate/`:
`scenario.py --launch` on the run's own cards (Docker), or `modal_verdict.py` in
fresh sandboxes (Modal). The verdict offers the load the tree declares in
`concurrency.json`, draws an odd workload seed no development run could have
drawn, and runs the private correctness stage
([`docs/serving_evolve/correctness.md`](docs/serving_evolve/correctness.md)).
Evidence lands under `server_prototypes/<lineage>/FINAL_VALIDATION/<version>/`,
and `evaluation_progress/record_run.py --run $R` writes the run's row to
`evaluation_progress/<task>.csv` once the lineage's `FINAL.json` exists (see
[`evaluation_progress/README.md`](../evaluation_progress/README.md)).

---

## 3. Before the first evolve on a task

A new task (`taskctl new`) runs as is: the agent declares its own load. What it
lacks is the SGLang reference point the instruction states. Measure it on the
task's own hardware and record it:

```bash
"$SERVERBENCH_PY" -m final_validate.concurrency --task <task> --arm sglang \
    --model-dir <pinned snapshot> --gpu <ids> --points 1,2,4,8,16 --out-dir <dir>
"$SERVERBENCH_PY" serving_tasks/_manage/taskctl.py set <task> --reference-concurrency N
```

Every point is a full 1,800-second window on a fresh development (even) seed
unless `--random-seed` names one, so run points in parallel on separate idle
cards. `taskctl set --reference-concurrency` writes only the operating point
the instruction states. A task whose bars come from its own SGLang measurement
also needs five random-seed runs at that point; its bars are the worst
(highest) p90 TTFT and the worst p90 TPOT of the five, written with
`taskctl set <task> --p90-ttft ... --p90-tpot ... --reason "..."`, and the rule,
the worst p90s and the multiplier (1.0) are recorded in `reference.yaml` under
`concurrency_search.slo_derivation`. What those bars guarantee is
[`docs/harness/bar-derivation.md`](docs/harness/bar-derivation.md).

Keep the exports under `serving_tasks/_concurrency_search_archive/<task>/`
(`search-c<N>/` per search point, `slo-c<N>-s<K>/` per derivation seed); the
directory is host-local and gitignored. A dry run
(`evolve-task.sh <task> --dry-run`) then proves the workspace boots.

---

## 4. GPU discipline

- **Inspect occupancy before every GPU action**, with
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory`, not just
  `memory.used`: a card reads low for a moment while a neighbour's engine
  starts.
- **Never contend with another user's process, and never signal a process you
  did not start.** Stop your own work by recorded pid or exact container name.
  `pkill -f <task text>` matches the agent's own command line and kills the run.
- `scripts/deploy/with-gpu <i> <cmd>` takes an exclusive `flock` and refuses a
  busy card with exit 75, which means "try again". Evolve drivers claim cards
  under `/tmp/serverbench-card-claims/`; `final_validate.lease` takes both the
  claim and the flock, since neither mechanism sees the other.
- Reference measurements are taken with nothing else of ours on the card.
- A run keeps its idle cards visibly reserved with `scripts/gpu_holder.py`: it
  holds a small block after two minutes of emptiness and releases everything
  (block and CUDA context, same pid via `execv`) the moment anything else runs
  there. Its log is `<run>/gpu-holder.<index>.log`; stop it only by the pid in
  the journal's `card_held` event.

---

## 5. Trajectories

Every run writes `TRAJECTORY.json` (one point per published version, from the
agent's own AgentX runs, with evolve time and API cost) and
`trajectory.atif.json` (the agent's steps in ATIF v1.7, with the evolution
record in `extra.evolution`).

```bash
"$SERVERBENCH_PY" serverbench/scripts/evidence/collect_trajectory.py --run $R --validate
"$SERVERBENCH_PY" serverbench/scripts/evidence/trajectory_query.py $R/trajectory.atif.json --summary   # or --rounds, --search, --timeline
```

These are development measurements, not verdicts: the verdict is taken on a
workload seed the agent never saw.

---

## 6. Knowing what you are looking at

| file | what it says |
|---|---|
| `$R/SUMMARY.json` | how the run ended, and which version is the final head |
| `server_prototypes/<lineage>/<lineage>_v<N>/` | a published version: the tree, frozen, with `VERSION.json` |
| `server_prototypes/<lineage>/FINAL.json` | the verdict's summary for the final head |
| `…/FINAL_VALIDATION/<version>/scenario-candidate.json` | the verdict in full: replay fidelity, correctness, per-request scores, objective |
| `evaluation_progress/<task>.csv` | one row per finished run, read from the verdict |

The lineage layout is [`server_prototypes/README.md`](../server_prototypes/README.md).

---

## 7. Failure modes worth recognising

| what you see | what it usually is |
|---|---|
| the agent exits in seconds and the driver idle-restarts | the agent CLI refused the model (e.g. a CLI too old for it) or the credential is dead: preflight the endpoint with one request |
| a card reads busy though nothing of ours runs | a container requesting every GPU (an idle builder); exempt it by id |
| `exit_code: 143`, zero stdout | the agent was killed, e.g. by a `pkill -f` that matched its own argv |
| a verdict `MISSES` with `eos gate` in its why | the server did not stop at end-of-sequence without `ignore_eos`, or padded replies |
| `NOT_RUN` | the workload could not be offered at all: the tree did not start, or `concurrency.json` is missing or invalid |

A failure message names a subject, and the subject is often wrong: "X failed"
usually means the thing that measured X, or set X up, failed. Check the
instrument before the idea.
