# `_template/` — what `new_task.py` renders a task from

```text
shared/contract.yaml       the p90 bars, prices, devices, window, request scoring
shared/reference.yaml      reference arms and private correctness settings (eos_audit on)
docker/  modal/            per-mode task.toml, Dockerfile.in and MODE.md
instructed/agents/         the instructed team's subagent definitions (kernel-engineer.md,
                           runtime-engineer.md), rendered into every instructed task's agents/
```

`{{PLACEHOLDER}}`s are substituted by `serving_tasks/_manage/new_task.py`,
which refuses to write a file that still carries one. The two modes differ in
where the container runs (a local docker host versus a Modal sandbox), the
hardware vocabulary (a Modal task's card must be one Modal offers), and
`task.toml`'s `[environment]`; the task data is identical.

`MODE.md` and, for instructed tasks, `agents/*.md` are regenerated into each task by `taskctl render`,
so edits here reach every live task on the next render.
