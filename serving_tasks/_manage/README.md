# `_manage/` — the task tooling

A task's data is `contract.yaml`, `reference.yaml` and its harness module
under `serverbench/harness/final_validate/tasks/`. Everything else in a task directory
is generated from them. These tools are the only writers of task parameters.

| file | role |
|---|---|
| `taskctl.py` | the command line: `show`, `check`, `render`, `set`, `new`, `rename` |
| `new_task.py` | creates a task from `_template/` (called by `taskctl new`) |
| `instruction_render.py` | renders `instruction.md` and `README.md` from the parsed task |
| `sites.py` | task paths, the scalar registry `set` writes through, packaged-evaluator drift |
| `blocks.py` | edits one YAML key in place, without reserialising the file |
| `profiles.py` | the guidance profile (`standard` / `instructed`) a task id names |
| `instructed_appendix.md`, `instructed_session_prompt.md`, `instructed_claude/` | the instructed curriculum, session prompt and Claude Code settings for instructed tasks |
| `gpu_clock.md` | the GPU-clock section every instruction carries |

```bash
. serverbench/scripts/env.sh
CTL="$SERVERBENCH_PY serving_tasks/_manage/taskctl.py"

$CTL show  <task>                   # the task as task_spec parses it
$CTL check [task]                   # generated files in step, spec valid, evaluator copy identical
$CTL render [task] [--profile P]    # regenerate documents, starter header, evaluator copy
$CTL set   <task> [--p90-ttft MS] [--p90-tpot MS] [--reference-concurrency N]
                  [--price-in|--price-out|--price-cached|--cache-fraction ...]
                  [--usd-per-gpu-hour X] [--model-id ID --model-revision SHA]
                  [--answer-sample-size N] [--max-absolute-accuracy-drop X]
                  [--max-token-regret X] [--reason WHY]
$CTL rename <task> --to <new id>    # rename; moves the module and SGLang evidence too
```

`$CTL set --help` lists every parameter. `set` writes each parameter to its one
registered site (`sites.SCALARS`), regenerates every generated file, repackages
the evaluator, and then reads the task back and refuses if anything did not
land. Raising a p90 bar loosens it and needs `--reason`; the reason belongs in
the commit message, which is where a bar's history lives.

## Creating a task

```bash
$CTL new qwen3_5_9b_standard_2xH200 --mode docker \
    --model-id Qwen/Qwen3.5-9B --model-revision <sha> \
    --title "Qwen3.5 9B Standard" --question "<one sentence>" \
    --usd-per-gpu-hour 3.5 --price-in 0.1044 --price-out 0.1855 \
    --p90-ttft 10000 --p90-tpot 60 [--reference-concurrency N] \
    [--sglang-serve-extra "..."] [--from-demo <task>] [--dry-run]
```

`--mode docker|modal` picks the execution files in `_template/<mode>/`; the
profile (`standard` / `instructed`) is read from the id. The id must end in its
hardware, and a Modal task's card must be one Modal offers. `--sglang-serve-extra`,
`--vllm-serve-extra` and `--trtllm-serve-extra` set the reference arms' launch
lines. The new task is request-scored (`dollars_per_dollar`) with a
candidate-declared load and fresh seeds; nothing is measured, so measure SGLang's
operating point and the bars afterwards
([`serverbench/OPERATIONS.md` section 3](../../serverbench/OPERATIONS.md#3-before-the-first-evolve-on-a-task)).
