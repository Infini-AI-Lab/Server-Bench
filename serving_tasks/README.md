# `serving_tasks/` — the serving problems agents are judged on

One directory per task. A task is the frozen definition of one serving problem:
the model and hardware, the workload, the p90 bars and prices it is scored
against, the starter an agent begins from, and the evaluator that judges it.
What agents built for a task lives in `server_prototypes/`; the harness that
runs them is `serverbench/`.

## Live tasks

Every live task replays the AgentX `inferencex-agentx-mvp` scenario over the
`semianalysisai/cc-traces-weka-062126` coding-session corpus for one
1,800-second window.

| task | hash | model | hardware | profile | mode | p90 TTFT / TPOT (ms) | SGLang reference load |
|---|---|---|---|---|---|---|---|
| `qwen3_5_2b_standard_1xL40` | `769bf633` | `Qwen/Qwen3.5-2B` | 1xL40 | standard | docker | 2707.8 / 12.19 | 16 |
| `qwen3_5_2b_instructed_1xL40` | `6b27c0c9` | `Qwen/Qwen3.5-2B` | 1xL40 | instructed | docker | 2707.8 / 12.19 | 16 |
| `qwen3_5_9b_standard_2xL40` | `9191bc67` | `Qwen/Qwen3.5-9B` | 2xL40 | standard | docker | 4100.2848398999995 / 23.497353185237426 | 1 |
| `qwen3_5_9b_instructed_2xL40` | `e0da1134` | `Qwen/Qwen3.5-9B` | 2xL40 | instructed | docker | 4100.2848398999995 / 23.497353185237426 | 1 |
| `qwen3_8_27b_int4_standard_2xL40` | `88583199` | `RedHatAI/Qwen3.8-27B-INT4` | 2xL40 | standard | docker | 10298.8 / 48.3769 | 4 |
| `gemma4_e4b_it_standard_1xL40` | `3b7461b0` | `google/gemma-4-E4B-it` | 1xL40 | standard | docker | 2526.9273697000003 / 121.70566402153604 | 2 |
| `gemma4_e4b_it_instructed_1xL40` | `c163203d` | `google/gemma-4-E4B-it` | 1xL40 | instructed | docker | 2526.9273697000003 / 121.70566402153604 | 2 |
| `qwen3_5_9b_standard_modal_2xL40S` | `200fa6bd` | `Qwen/Qwen3.5-9B` | 2xL40S | standard | modal | 10000 / 60 | — |
| `gemma4_e4b_it_standard_modal_1xL40S` | `19e6d585` | `google/gemma-4-E4B-it` | 1xL40S | standard | modal | 3000 / 80 | — |

A task's id names its model, profile and hardware and carries no ordinal;
its hash (`task_hash` in `contract.yaml`, the first 8 hex digits of the id's
SHA-256) is its short handle, and `evolve-task.sh` accepts either.

Where the bars come from: the Qwen3.5 2B, Qwen3.5 9B and 27B INT4 tasks record
an SGLang derivation in `reference.yaml` (`concurrency_search.slo_derivation`:
the worst p90 of five random-seed SGLang runs at the searched operating point,
multiplier 1.0); an instructed task shares its standard twin's bars. The Gemma
tasks record their SGLang concurrency search in `reference.yaml`, and the
instructed Gemma task names its standard twin in `derived_from` (in
`contract.yaml`), whose bars it shares. The two Modal L40S tasks have not been
measured on their hardware, so they carry provisional bars and state no SGLang
reference point. The method, and what a bar
guarantees, is
[`serverbench/docs/harness/bar-derivation.md`](../serverbench/docs/harness/bar-derivation.md).

## Profiles

A **standard** task gives the agent the brief and the skill bundles in
`serverbench/skills/`. An **instructed** task is the same task with coaching
added: the instructed curriculum, a subagent team (`agents/`) and the bundles in
`serverbench/skills_instructed/`, with `./checkpoint` in place of `./version`.
`tests/contracts/test_instructed_variants_only_change_guidance.py` holds that an
instructed task's payload is byte-identical to its standard twin's once identity
and guidance are removed, so the profile is the only variable between them.

## The policy every live task follows

- **Request scoring** (`request_p90_billable_tokens_v4`). There is one latency
  row, p90. A profiling request scores 1 when its TTFT and TPOT both meet the
  p90 bars, and 0 otherwise (errors and cancellations included). The objective
  is `dollars_per_dollar` = qualifying requests' input/output token value
  divided by whole-window GPU cost. Gross revenue and request-count attainment
  are diagnostics, not the billing formula. There is no hard latency gate and
  no p95 row. Every live task prices cached input at the full input rate.
- **Hard eligibility gates**: replay fidelity (every request returns the
  output length its trace asked for), output correctness and the EOS gate,
  verified privately after the run. The agent is told what correct means and
  never how it is checked.
- **Candidate-declared load.** The published tree's `concurrency.json`
  (`{"offered_concurrency": N}`, 1–256) sets the offered session concurrency.
  The SGLang operating point on the task's hardware is stated in the
  instruction as a reference, never imposed.
- **Fresh workload seeds.** Development measurements draw even seeds and the
  final verdict an odd one, so the verdict never replays a request mix the
  agent measured. No task has a fixed seed.

## Anatomy

| path | what it is |
|---|---|
| `contract.yaml` | the p90 bars, token prices, devices and hourly rate, window, request scoring, the hashed id, and `model_config`: the model's Hugging Face `config.json` at the pinned revision, verbatim (written by `taskctl render`) |
| `reference.yaml` | the SGLang reference arms, their launch flags one per key (`sglang_settings`, generated from the SGLang arm), the operating point (`concurrency_search`), and the private correctness settings |
| `task.toml` | the Harbor task definition for the container |
| `demo/` | the correctness-first starter the agent begins from: `serve.sh`, `server.py` and helpers. It declares no offered concurrency: the agent writes `concurrency.json` |
| `instruction.md` | the one document the agent is given — generated |
| `README.md` | the repository-side overview — generated |
| `tests/final_verdict/` | the packaged evaluator (measurement only) — generated from `serverbench/harness/final_validate/` |
| `environment/Dockerfile`, `MODE.md` | the task image and where each part of its container comes from |
| `environment/agent-environment` | only where a task needs a non-default agent image and wheelhouse (`IMAGE=`, `ENV_INPUT=`, `IMAGE_DOCKERFILE=`); the INT4 task has one |
| `agents/kernel-engineer.md`, `agents/runtime-engineer.md` | instructed tasks only: the team's two subagent definitions, rendered from `_template/instructed/agents/` and delivered to the workspace's `.claude/agents/` |

Each task also has a harness module,
`serverbench/harness/final_validate/tasks/<task id, lower case>.py`, carrying its
model identity, context rule, seed policy and `REFERENCE_CONCURRENCY`.

## Creating and changing tasks

Use [`_manage/`](_manage/README.md); never hand-edit a task. Generated files
are regenerated and `taskctl check` refuses a task that is out of step.

```bash
. serverbench/scripts/env.sh
CTL="$SERVERBENCH_PY serving_tasks/_manage/taskctl.py"
$CTL check                                  # every live task consistent
$CTL set <task> --price-in 0.03             # change a parameter, re-render, read back
$CTL set <task> --p90-tpot 7.5              # tighten a bar (raising one needs --reason)
$CTL new <task> --mode docker ...           # create a task (see _manage/README.md)
$CTL rename <task> --to <new id>            # rename, moving its evidence with it
```

A new task's SGLang operating point and bars are measured on its own hardware
([`serverbench/OPERATIONS.md` section 3](../serverbench/OPERATIONS.md#3-before-the-first-evolve-on-a-task)).

## Directory map

```text
<model>_<profile>_<hardware>/     the live tasks, one directory each
_manage/                           taskctl, new_task and the renderers
_template/                         shared task data, per-mode files, guidance profiles
_modal/                            Modal backend tooling
_concurrency_search_archive/       host-local, gitignored: SGLang search and derivation exports
```
