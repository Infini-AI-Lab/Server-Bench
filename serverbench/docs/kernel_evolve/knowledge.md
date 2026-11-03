# What a kernel writer is given

Three things reach an agent working under the `evolve` kernel strategy
(`kernel_evolve/contracts/task.py::KernelTaskSpec.strategy`, a field of the
kernel-evolution subsystem's own task specs; no serving task's contract has
one, and every serving task lets the agent write kernels), and one thing does
not.

| | where |
|---|---|
| how to work on a kernel | `serverbench/skills/kernel-evolution/` |
| what the hardware and the operators are | `serverbench/skills/kernel-evolution/knowledge/` |
| what may be dispatched to | the strategy catalog, `kernel_evolve.shapes` |
| **not** given | upstream serving-framework source, in any form |

## Why kernel knowledge lives in the kernel skill

The serving skills (`serverbench/skills*/serving-system-evolution*`) are about
serving systems: schedulers, batching, KV ownership, the shape of a request's
life. `serverbench/skills/kernel-evolution/knowledge/` is about kernels, so it is
part of the kernel-evolution skill and travels with it: a standard task's
workspace receives it at `.claude/skills/kernel-evolution/knowledge/`, routed to
from that skill's `SKILL.md`. There is one copy. Instructed tasks carry their
own kernel skill (`serverbench/skills_instructed/kernel-evolution/`) and the much
larger KernelWiki corpus, which covers the same ground.

Mixing kernel and serving material misroutes an agent: asked to make a server
faster, it reads a pile of GEMM tiling advice and concludes the answer is a GEMM
before anything has profiled the scheduler. Keeping them apart lets the routing decision — *is this
even a kernel problem?* — be made on evidence.

## Progressive disclosure, not a corpus dump

`knowledge/catalog.yaml` carries an `applies_to` field per entry. An agent
that has profiled its way to one operator reads that operator's entries plus the
hardware entry, and nothing else. Reading the whole tree spends a context window
on material about operators nobody is touching.

`serverbench evolve kernels knowledge` checks that every catalog entry resolves
**and** that no file exists which the catalog does not list — because an entry
nothing routes to is a file nobody will ever read.

## The two treatment arms

`KernelTaskSpec.strategy` (`KernelStrategy`) is frozen on every kernel-evolution
task spec, and that subsystem builds its workspace to match, so the difference
is enforced rather than requested. Serving tasks carry no strategy field.

**`evolve`** mounts the kernel harness, this skill, the knowledge tree, a
candidate workspace, reference implementations and the verification tools. Kernel
modification is *permitted and supported* — not required. A candidate that changes
a kernel because kernel work was allowed, rather than because a profile pointed
there, has no profile evidence and the contract refuses it.

**`provided_sota`** mounts the qualified catalog and its usage documentation.
Kernel source paths are absent or read-only, and the implementations behind the
entries are **withheld** — otherwise the arm is a slower version of the other one
rather than a contrast to it. What remains is still real optimisation: which entry
to dispatch to per shape, when to dispatch, and everything above the kernel.

Fallback telemetry is identical in both arms. A dispatch that silently falls back
is the same defect whether the agent wrote the kernel or was handed it.

## What a catalog entry is, and is not

**A permission with an ABI.** Package, symbol, version, digest, licence, the
capability it satisfies, the tolerance it is checked at, the fixtures behind that,
the conditions under which it must refuse, and what it falls back to.

**Not a measurement.** `MEASURED_HERE` is not an evidence level a catalog may use,
and the contract raises rather than warning. A number measured here belongs to a
run, under a frozen plan, with a device receipt. Letting a pointer carry one is
how a vendor's marketing figure ends up quoted as this project's result.

This is a different object from `serving_evolve.kernels.contracts.KernelCatalog`,
which records **what exists in the world** — a researched pointer with provenance
and a claim, written outside the sandbox. A researched pointer is a lead; a
catalog entry here is a permission.
