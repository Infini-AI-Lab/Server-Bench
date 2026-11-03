# docs/ — organised by what owns the behaviour

Three areas. How the verdict takes its numbers (replay, scoring and the
correctness stage) is
[`../harness/final_validate/README.md`](../harness/final_validate/README.md).

## [serving_evolve/](serving_evolve/) — the serving-evolution harness

| file | what it answers |
|---|---|
| [contracts.md](serving_evolve/contracts.md) | a task's contract files, and the typed contracts behind the gate ladder |
| [containers.md](serving_evolve/containers.md) | Harbor, isolation, offline enforcement, the images, Modal |
| [correctness.md](serving_evolve/correctness.md) | how a verdict decides that output is correct |
| [skills.md](serving_evolve/skills.md) | the skills delivered into the workspace |

## [harness/](harness/) — measurement method and the test suite

| file | what it answers |
|---|---|
| [bar-derivation.md](harness/bar-derivation.md) | how a p90 bar and its reference operating point are derived, and what the bar guarantees |
| [testing.md](harness/testing.md) | the suite, its subjects, and the pre-merge gate |

## [kernel_evolve/](kernel_evolve/) — the kernel-evolution harness

| file | what it answers |
|---|---|
| [README.md](kernel_evolve/README.md) | what this harness is for |
| [scope.md](kernel_evolve/scope.md) | what it does and does not cover |
| [ladder.md](kernel_evolve/ladder.md) | the verification ladder |
| [controls.md](kernel_evolve/controls.md) | the negative controls a battery must kill |
| [knowledge.md](kernel_evolve/knowledge.md) | what a kernel writer is given, and the `knowledge/` corpus |

## Not here

- **Running something** is [`../OPERATIONS.md`](../OPERATIONS.md).
- **Which module owns each stage of a run** is [`../harness/README.md`](../harness/README.md).
- **The tasks** are [`../../serving_tasks/README.md`](../../serving_tasks/README.md).
