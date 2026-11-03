# `serverbench/harness/` — the evolve driver and its libraries

Five top-level packages under one root (`pyproject.toml`: `where = ["harness"]`; `scripts/env.sh`
puts it first on `PYTHONPATH`):

| package | what it is |
|---|---|
| `lifecycle/` | **the production driver.** `serverbench/scripts/evolve` hands its arguments to `lifecycle/run.py`, which builds a workspace, launches the agent, freezes versions, and runs the final verdict. |
| `serving_evolve/` | libraries the driver and `final_validate` share: the task contract reader (`contracts/`), pinned runtimes and images (`runtime.py`), the agent skill bundles (`skill_bundle.py`), the workspace denylist (`environment/`), and the `serverbench evolve` CLI (`cli.py`: `doctor`, `curve`, `golden`, `skill`, `kernels`). |
| `kernel_evolve/` | a separate kernel-evolution subsystem, driven by `serverbench evolve kernels`. It uses the gate ladder in `serving_evolve/contracts/gates.py` and `serving_evolve/evaluator/gates.py`; no serving run does. Its knowledge corpus lives in the kernel skill (`serverbench/skills/kernel-evolution/knowledge/`, indexed by `catalog.yaml`) and is checked by `serverbench evolve kernels knowledge`. |
| `final_validate/` | **the final verdict**: `scenario.py` (AgentX replay and per-request scoring), the host-only `private_correctness.py`, one module per task under `tasks/`, `package_grader.py` (the agent-visible copy in each task's `tests/final_verdict/`) and the operator tools under `tools/` (`validate_evolve.py` is what `lifecycle/run.py` calls). Importable like the others, but excluded from any built distribution. See its [README](final_validate/README.md). |
| `serverbench/` | the `serverbench` console script: `serverbench evolve <verb>` delegates to `serving_evolve/cli.py`. Nothing else. |

## `lifecycle/`, by phase

| phase | modules |
|---|---|
| run identity and record | `run.py`, `record.py`, `utils.py` |
| workspace | `workspace.py` (build and boot checks), `delivery.py` (inputs copied in), `backend.py` (Harbor docker or Modal), `credentials.py` |
| agent | `agent.py` (launch, resume, idle restart), `instructions.py` (loads the task's own `instruction.md`), `billing.py`, `api_cost.py` |
| versions | `boundary.py` (freeze a version), `boundary_checks.py` (portability checks), `publish.py` (lineage head), `basic_memory.py` |
| verdict and evidence | `full_test.py` (agent-requested development full test), `run.py::_final_validation` (calls `final_validate/tools/validate_evolve.py`), `verdict.py` (failure attribution), `report.py`, `trajectory.py`, `trajectory_capture.py`, `decisions.py`, `model_audit.py` (which models and provider hosts the agent used) |
| cleanup | `reclaim.py` |
| delivered to the agent | `waiting/` — blocking shell waiters, copied into every workspace as `wait/` by `delivery.py` |

The final verdict lives in `final_validate/`; `lifecycle/` drives it and
re-implements none of it, and `final_validate/` imports nothing from
`lifecycle/`.

## What the agent gets

No harness code is copied into a workspace; the one exception from this
directory is `lifecycle/waiting/`, delivered as `wait/`. The agent receives its
task's `instruction.md`, the starter (`demo/`), the evaluator's measurement copy
(`tests/final_verdict/`), `task_context/`, the tools named in the instruction,
and the skill bundles from `serverbench/skills/` (standard tasks) or
`serverbench/skills_instructed/` (instructed tasks).
