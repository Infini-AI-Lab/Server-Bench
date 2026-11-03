# Serving evolution

The harness that runs a coding agent against one serving task and judges what
it builds, and the skills that help the agent build it.

```
serving_tasks/                            the tasks: what a serving system has to do
serverbench/harness/lifecycle/            the evolve driver
serverbench/harness/serving_evolve/       task and workspace contracts
serverbench/harness/final_validate/       the verdict (measurement, scoring, private correctness)
serverbench/skills/, skills_instructed/   what the agent is given (standard / instructed tasks)
server_prototypes/                        the serving systems that come out
```

| | |
|---|---|
| [contracts.md](contracts.md) | a task's contract files, and the typed contracts behind the gate ladder |
| [correctness.md](correctness.md) | how a verdict decides that output is correct |
| [containers.md](containers.md) | Harbor, Modal, the images, and the isolation around the agent |
| [skills.md](skills.md) | authoring, validating and delivering a skill bundle |
| [`harness/final_validate/README.md`](../../harness/final_validate/README.md) | how the numbers are taken |
| [bar-derivation.md](../harness/bar-derivation.md), [testing.md](../harness/testing.md) | the bars' method, and the test suite |

## Non-goals

- **It is not a serving system.** The harness contains no scheduler, KV-cache
  design, batching policy or kernels; those belong to the candidate.
- **It does not hand the agent a reference implementation.** SGLang is measured
  on the task's hardware as a reference point, and its operating point is
  stated to the agent; its code is never in a workspace.
- **It does not tell the agent how correctness is checked.** The agent is told
  what correct output is and builds its own verification.
