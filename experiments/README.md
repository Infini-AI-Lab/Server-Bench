# Experiments

One bash entry point per experiment and ablation, plus the CPU-only analysis
that reads what they produce. Every script runs with no arguments, takes its
settings from environment variables, and queues work through the production
launchers: evolves through `evaluation_progress/watch_evolve.py` (which calls
`serverbench/scripts/server_evolve/evolve-task.sh`), reference measurements
through `final_validate.concurrency`, and re-measurements through
`final_validate/tools/measure_published.py`. Nothing here contains serving or
measurement logic of its own.

| script | experiment |
|---|---|
| `main_sweep.sh` | every agent model on every sweep task |
| `reference_bars.sh` | baseline frameworks' operating points and seed runs |
| `ablation_repeats.sh` | A1: repeated construction runs and reasoning effort |
| `ablation_guidance.sh` | A2: task guidance, standard vs instructed brief |
| `ablation_budget_retrospective.sh` | A3a: re-measure the version each run had at 2, 4 and 6 h |
| `ablation_budget_prospective.sh` | A3b: fresh evolves with a shorter budget |
| `analyse.sh` | every analysis table, from the runs on this machine |
| `config.sh` | the agent roster and shared settings |

## Before the first run

1. Deploy the machine ([`serverbench/OPERATIONS.md`](../serverbench/OPERATIONS.md) section 1): images
   built, offline inputs staged, every task's model in the local Hugging Face cache, `harbor/.venv`
   synced, and `serverbench/scripts/env.sh` finding both interpreters.
2. Check the roster in [`config.sh`](config.sh). Each entry is
   `label|agent CLI|model id|credentials file|efforts`. Claude and Codex models run on the operator's own
   login (`~/.claude/.credentials.json`, `~/.codex/auth.json`); every other model needs a `KEY=VALUE` file
   under `serverbench/artifacts/secrets/` (gitignored) naming its endpoint and key. To use other models,
   edit the roster or point `ROSTER_FILE` at a file that sets `ROSTER`.
3. Print what a script would do: `DRY_RUN=1 bash experiments/<script>.sh`. A dry run starts nothing,
   samples no card and writes no file; it prints each command, shell-quoted, one per line, and notes
   (lines starting with `#`) for anything missing, such as a credentials file.

## How the scripts behave

- **Selection.** `MODELS` (roster labels), `TASKS`, `EFFORT` / `EFFORTS`, `REPLICATES` and `HOURS`
  narrow or widen any evolve experiment; each script's header lists its own variables and defaults.
- **Queueing.** An evolve experiment starts one detached watcher per agent configuration (a
  "campaign") and returns. The watcher waits for enough free cards for each task, launches the run,
  and records its row in `evaluation_progress/<task>.csv` when the verdict lands. Its log and
  `status.json` are in `serverbench/artifacts/operations/<campaign>/`. Campaign names carry the
  experiment, model, effort and a UTC stamp, so re-running a script queues new runs and never
  touches old ones.
- **Cards.** `GPUS=0,1,2,3` confines every campaign to those cards and `EXCLUDE_GPUS` keeps cards out;
  otherwise any card the launcher finds idle (no process, no memory, no claim, no container reservation,
  sampled over a minute) is used. Check occupancy yourself before starting a large batch
  (`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`).
- **Stopping.** Stop a campaign by the watcher pid in `ps` (the launch is `setsid`, so `$!` is not it)
  and a run by the pid in its `status.json`. Never `pkill -f` a task name: it matches the agent.
- **Foreground measurements.** `reference_bars.sh` and `ablation_budget_retrospective.sh` measure one
  point after another in the foreground; launch them detached
  (`setsid nohup bash experiments/<script>.sh < /dev/null > <log> 2>&1 &`) and run disjoint `TASKS` in
  separate invocations, each with its own `GPU=`, to go faster.
- **Seeds.** Every final verdict and re-measurement draws a fresh odd workload seed, and every
  development and reference measurement a fresh even one. No script sets a seed.

GPU-hours below count the cards a task is billed for: an 8 h evolve holds them for the budget plus
about one hour of final verdict, so about 9 GPU-hours on a 1-card task and 18 on a 2-card task. API
spend depends on the model and is not estimated here.

## Main sweep

**Question.** How much value per GPU dollar does each agent model build, on tasks that stress different
resources?

**Arms.** Every roster model at its first listed effort (`max` in the default roster), on
`qwen3_5_9b_standard_2xL40`, `gemma4_e4b_it_standard_1xL40` and `qwen3_8_27b_int4_standard_2xL40`
(`SWEEP_TASKS`), 8 h each, `--stop timebox`.

```bash
bash experiments/main_sweep.sh
```

**Cost.** About 45 GPU-hours per model (18 + 9 + 18), so about 500 for the default 11 models.

## Reference arms and p90 bars

**Question.** Where does each baseline framework sit on a task's hardware, and what are its p90s and
value at its operating point across seeds? The worst p90 TTFT and p90 TPOT of five SGLang seed runs at the
adopted operating point is the rule a task's SGLang-derived bars follow.

**Arms.** For each task and each arm in `ARMS` (default `sglang vllm`, at the images pinned in
`final_validate/arms.py`): a search over `SEARCH_POINTS` (default `1,2,4,8,16`), then `SEEDS` (default 5)
runs at the operating point. The point is `CONCURRENCY` if set, else the task's adopted SGLang point in
`reference.yaml` for SGLang, else the search's recommendation. `STAGE=search` or `STAGE=seeds` runs one
half. Every point is a full 1,800 s window.

```bash
bash experiments/reference_bars.sh
"$SERVERBENCH_PY" experiments/analysis/reference_bars.py      # the worst p90s and the taskctl command
```

Evidence lands in `serverbench/artifacts/concurrency/reference/<task>/<arm>/` (`search/`,
`slo-c<N>-s<K>/`). Adopting a bar or an operating point is a task change and goes through `taskctl set`
(`--p90-ttft`, `--p90-tpot`, `--reference-concurrency`, with `--reason`), never a hand edit; record
the SGLang exports as [`OPERATIONS.md`](../serverbench/OPERATIONS.md) section 3 describes.

**Cost.** About 40 minutes per point including server start, 10 points per (task, arm): about 7 hours
per card-pair, about 85 GPU-hours for the four default tasks and two arms.

## A1: repeated construction runs and reasoning effort

**Question.** Is a model's score robust across independent construction runs, or can a run get lucky?
Which reasoning effort is the right default?

**Arms.** Every roster model at every effort its entry lists (Claude `max xhigh high`, Codex
`max xhigh ultra`, API-key models `max`), `REPLICATES` times (default 1), on the 9B and Gemma standard
tasks. Together with the main sweep this gives two independent runs at each model's first effort and
one at each other effort. `EFFORTS=max REPLICATES=2` gives same-effort replicates only.

```bash
bash experiments/ablation_repeats.sh
```

**Cost.** About 27 GPU-hours per (model, effort, replicate); about 570 for the default roster.

## A2: task guidance

**Question.** Does richer task guidance change what agents build and how often it is eligible, holding
model, budget, hardware and workload fixed?

**Arms.** `gemma4_e4b_it_standard_1xL40` (control) and `gemma4_e4b_it_instructed_1xL40`. They share one
contract (bars, workload, prices, window, correctness); only the brief differs, the instructed one adding a
failure taxonomy, an evolve loop, long-context correctness checks, a full-test routine, time guidance and
two coordinated subagents. One strong and one weaker agent (`MODELS`, default `sonnet-5-5 glm-5.3`),
`REPLICATES=2` per arm, max effort, 8 h. `ARMS=instructed` runs the instructed arm only, reusing the main
sweep's standard runs as the control; `STANDARD_TASK` / `INSTRUCTED_TASK` select the 9B pair instead.

```bash
bash experiments/ablation_guidance.sh
```

**Cost.** 8 runs of about 9 GPU-hours: about 72 GPU-hours (36 with `ARMS=instructed`).

## A3a: development budget, retrospective

**Question.** How good would each run's system have been had its budget ended at 2, 4 or 6 hours? No
agent runs: each existing run's accepted head at each cutoff is measured again through the final verdict
(full 1,800 s replay on a fresh odd seed plus the private correctness stage).

**Arms.** By default every run of the 9B and Gemma standard tasks at effort `max` whose lineage has a
final verdict (`TASKS`, `EFFORTS`, or name run roots in `RUNS`); cutoffs in `CUTOFFS` (default `2,4,6`).
A cutoff before the first version is a recorded zero; a version that is the head at several cutoffs is
measured once; the run's final head keeps its final verdict. The version is measured from a copy under
`serverbench/artifacts/budget-versions/`, so the lineage's own `FINAL.json` (which `record_run.py`
reads) is never replaced; the evidence lands in `serverbench/artifacts/final_validation/budget/`.
This answers what the artifact would have scored, not what an agent planning for a shorter budget would
have built; A3b asks that.

```bash
bash experiments/ablation_budget_retrospective.sh
```

**Cost.** About one hour per version on the task's cards: roughly 1-2 GPU-hours per Gemma version and 2-4
per 9B version, no API.

## A3b: development budget, prospective

**Question.** What does an agent build when it is given less time from the start?

**Arms.** Fresh evolves on `gemma4_e4b_it_standard_1xL40` for `MODELS` (default `sonnet-5-5 glm-5.3`),
compared with their main-sweep 8 h runs. `BUDGETS` lists `hours:replicates` (default `4:2 2:1`).

```bash
bash experiments/ablation_budget_prospective.sh
```

**Cost.** Per model 2 x (4 + 1) + (2 + 1) = 13 GPU-hours; 26 for the default two models.

## Analysing your runs

Everything reads evidence the runs already wrote, on CPU:

```bash
bash experiments/analyse.sh                    # every table, into experiments/analysis/out/
DRY_RUN=1 bash experiments/analyse.sh          # the commands, in order
```

`OUT`, `EVOLVE_ROOT` (run roots, default `serverbench/evolve_space/evolve`), `PROTOTYPES` (lineages,
default `server_prototypes`), `REMEASURED`, `REFERENCE_DIR` and `CUTOFFS` point it elsewhere. Each script
in `analysis/` also runs on its own, with `--help` listing its inputs and their repository defaults. The
result ledger itself, one row per run, is `evaluation_progress/<task>.csv`.

| script | writes | what it is |
|---|---|---|
| `rescore_finals.py` | `final_rescored.csv` | every verdict (final and re-measured), its decision and operating point, joined to its run's model, CLI, effort and budget, and re-scored per request with the task's rule (`rescore.py`) |
| `dev_trajectories.py` | `dev_trajectories.json` | every development replay an agent ran, scored on the same rule, with its fidelity, the published versions, and the cumulative API cost |
| `run_features.py` | `run_features.csv` | per final verdict: first measurement and first version, versions, measurements, full-length rehearsals, development fidelity failures, API cost |
| `component_timeline.py` | `component_timeline.csv` | the hour each serving component first appears in the agent's own version notes |
| `design_scan.py` | `design_features.csv` | what each final tree is built from: Triton or CUDA kernels, CUDA graphs, speculation, prefix and state reuse, paged KV, tensor parallelism, a host KV tier |
| `dev_vs_verdict.py` | `window_head_effect.csv`, `dev_dips.csv` | how the head of a verdict's window compares with the whole, and each drop between consecutive development replays split into latency and workload draw |
| `budget.py curve` | `budget_curve.csv` | A3a: $/$ against budget per re-measured run |
| `reference_bars.py` | `reference_seed_runs.csv`, `reference_bars.csv` | the reference arms' seed runs and the worst p90s per operating point |

A3b, A1 and A2 need no script of their own: their verdicts are rows of
`final_rescored.csv` and `evaluation_progress/<task>.csv`, told apart by model, effort, task and
`evolve_hours`.
