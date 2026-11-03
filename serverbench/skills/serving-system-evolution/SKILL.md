---
name: serving-system-evolution
description: Reference material for implementing or debugging an LLM serving system when protocol correctness, workload-specific performance, scheduling, KV state, or GPU execution is relevant. Use the task's own brief and evaluator as the authority; this skill does not prescribe an architecture or development sequence.
---

# Serving-system evolution

Use this skill as a reference, not as a required workflow. The task brief,
frozen task context, and delivered evaluator define the experiment. They take
precedence over generic serving advice.

How you organise the work is your call. Nothing in this skill requires a
particular architecture, file layout, implementation language, testing order,
measurement cadence, or note-taking system.

## Read the experiment from its sources

The workspace tells you where the authoritative material is. In particular:

- `AGENT_TASK.md` states the current objective, p90 bars, time budget, and
  exact evaluator commands.
- `task_context/` contains the frozen public task contract.
- `tests/final_verdict/` is the runnable development copy of the evaluator.
- `./clock`, `./measurements`, and `./version` expose the live budget,
  completed comparable measurements, and publication mechanism.

For scenario tasks, correctness is checked separately and each request is
scored against the p90 bars within one task-declared full window. A shorter run can be
useful as a diagnostic, but it is not evidence about a longer declared window.
Use the duration, scenario, corpus and bars from the task, and the offered
load your tree declares in `concurrency.json`; do not substitute generic values
from this skill.

The evaluator copy documents its own command line. `./tests/replay-verify`
runs a short-token public AgentX multi-turn fixture on the correlation-ID
session path and checks only request/error/OSL
replay fidelity. `--check-protocol-context` runs the public protocol and native-token probes,
and a context probe where the task declares one, without the timed replay,
while `--measure-only` runs an AgentX replay of the length you choose (the
task's full window by default) and checks actual-workload replay fidelity. Neither cheap
check establishes a score. `./measurements` summarizes completed
comparable runs. `./version` records a checkpoint and does not evaluate it.
The lightweight report samples only bounded metadata from AgentX's existing
post-run export; it adds no serving requests and omits prompt/response bodies.
Use a passing current-candidate replay gate before finalizing changes to
request/session state, cache or memory lifetime, scheduling/admission, or
streaming. A full replay can satisfy that gate, but a full window is not
required for this selection check, and the small fixture does not establish
long-context capacity.

## Questions that may help

These are diagnostic prompts, not a checklist or an order of work:

- Does the server implement the exact request, response, streaming, usage, and
  stop contracts exercised by the task?
- Is a discrepancy caused by the candidate, its input, the evaluator setup, or
  an artifact from a different attempt?
- Which part of the representative workload is on the critical path: prefill,
  decode, host scheduling, memory management, communication, or admission?
- Is state ownership clear enough to prevent one request, batch, or cache slot
  from corrupting another?
- Does an optimization preserve the task-declared correctness rule and the
  work represented by each request?
- What small experiment would distinguish the competing explanations for the
  observed result?

Choose whichever questions fit the evidence. A task may be served correctly by
many different architectures.

## Correctness

The task contract names the shipping gate. Exact token or logit parity can be a
useful local diagnostic for a supposedly semantics-preserving change, but it
becomes a shipping gate only when the task says so.

Check the externally visible behavior that the task actually consumes: output
text, streaming text, finish reasons, token accounting, request isolation,
and failures. Do not replace the task's correctness rule with a stricter
generic rule from this skill.

## Measurement and profiling

Use the delivered evaluator for claims about the task. Short requests,
component benchmarks, and profilers can help explain a result or reject an
idea cheaply; they do not replace the task's full workload.

Both NVIDIA profilers and the CUDA toolchain are available. `nsys` is useful
for locating time across kernels and host activity; `ncu` is useful for
examining one kernel. The image also provides `nvcc`, NVRTC, CUDA C++, direct
loading of hand-written PTX, and Triton. These are options rather than a
required implementation path. Select them only when they answer a question
raised by the workload or a profile.

When comparing artifacts, confirm that they belong to the current candidate
and completed successfully before interpreting missing or partial metrics.
